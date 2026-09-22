#!/usr/bin/env python3
"""
calc_tool_vllm.py —— 计算器 harness 的后端层版本（HF / vLLM 二选一）。vLLM 线专用：baseline_countdown_vllm / chat_countdown_vllm / vllm_check / 以后的 grpo_tool_mp_vllm。
原 calc_tool.py 不动，正在跑的 RL 和旧评测继续用它；这一版验证过、RL 切过来之后再合并。

协议
    模型写  <calc>42 + 12 - 18 - 6</calc>           → 生成在 </calc> 停
    harness 抠出式子 → countdown.safe_eval → 接上  <result>30</result>  → 模型继续写
    算不了 → <result>error</result>；每条最多 max_calls 次（默认 48：老师的 4 数轨迹一行一次、乘积表 6 次、第 2 层每行 2 次，16 不够）
    模型自己写出 <result>（不是 harness 注入的）= 假结果 → 立刻停，标 fake（训练器罚分）

后端（2026-09-07 抽出来的一层）：「一批序列各自生成到停止串 / EOS / 预算」这件事交给后端做，停注入续 mask 在后端之上，跟后端无关
    HFBackend    transformers 的 generate：左填充成矩阵，每轮从头 prefill 整批前缀（慢：一条 40 次调用 = 41 轮重算）
    VLLMBackend  vLLM 离线引擎：喂 prompt_token_ids（注入段的 id 跟训练端一致），每条自己的 max_tokens，前缀缓存 + 连续批处理
    generate_with_tools(model_or_backend, …)：传 HF 模型 = 自动包成 HFBackend，老调用方式不变

返回每条：
    ids     响应 token（含注入的 result 段），末尾带 EOS（如果停得干净）
    txt     响应文本
    spans   [(起点, 长度)] 注入段在 ids 里的位置 —— 训练时 mask 掉，不进梯度
    n_calls / n_err / fake / cont / cut
"""
import os, re
import countdown as CD
# ★ vLLM 0.8 的 V1 引擎默认另起子进程跑引擎核心；本进程已初始化 CUDA（HF 模型在同一进程）时只能 spawn，
#   spawn 会把主脚本重新执行一遍 → 「start a new process before bootstrapping」。让引擎核心在本进程跑，评测和以后的同卡共存都要这样
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

CALC_OPEN, CALC_CLOSE = "<calc>", "</calc>"
RES_OPEN, RES_CLOSE = "<result>", "</result>"
ASK_OPEN, ASK_CLOSE = "<ask>", "</ask>"                      # ★ 求助工具（臂 A）：问专家（DeepSeek reasoner）
REPLY_OPEN, REPLY_CLOSE = "<reply>", "</reply>"
CONT = "\nUser:"
STOPS = [CALC_CLOSE, RES_OPEN, CONT]
STOPS_ASK = [CALC_CLOSE, RES_OPEN, ASK_CLOSE, REPLY_OPEN, CONT]
CALC_RE = re.compile(r"<calc>(.*?)</calc>", re.S)
ASK_RE = re.compile(r"<ask>(.*?)</ask>", re.S)

ASK_HINT = ("You may also ask an expert at most twice: write <ask>QUESTION</ask> and the expert's answer will be returned as "
            "<reply>ANSWER</reply>. Always verify the expert's equation with the calculator before answering. ")


def ask_hint(n=2):
    """求助说明，次数可变（prompts_cd 的措辞 0）"""
    return ASK_HINT.replace("at most twice", "at most " + {1: "once", 2: "twice", 3: "three times", 4: "four times"}[n])


def _ask_retry(ask_fn, q):
    """★ 专家没答（正文空 / API 错）→ harness 自己再问一次，不算模型的求助次数：API 抽风不是模型的错，不该让它学"""
    a = ask_fn(q)
    return ask_fn(q) if a == "error" else a


def ask_prompt(nums, target):
    """工具版提示词 + 求助说明（插在 Show your work 之前，跟 TOOL_HINT 连着）"""
    return tool_prompt(nums, target).replace("Show your work in", ASK_HINT + "Show your work in")


def ask_expert(question, model="deepseek-reasoner", max_tokens=8000):
    """★ 求助工具的执行器：问 DeepSeek，只回一行 EQUATION: …；出错回 error。缓存、温度 0 在 deepseek_tool 里
    max_tokens 8000：推理段偶尔 >4000（[1,2,11,26]→28 一次 4000 被截、一次 453 答出），截断 = 正文空 = error"""
    try:
        import deepseek_tool as DS
        r = DS.ask(question.strip(), model=model, system=DS.CD_SYS, max_tokens=max_tokens)
        c = r["content"].strip()
        return c.split("\n")[0][:120] if c else "error"
    except Exception as e:
        return "error"

TOOL_HINT = ("You have a calculator: write <calc>EXPRESSION</calc> and the exact value will be returned as "
             "<result>VALUE</result>. Use it for every calculation. ")


def tool_prompt(nums, target):
    """R1 模板 + 工具说明（插在 Show your work 之前）"""
    return CD.make_prompt(nums, target).replace("Show your work in", TOOL_HINT + "Show your work in")


def evaluate(expr):
    try:
        v = CD.safe_eval(CD.clean_expr(expr))
    except Exception:
        return "error"
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.4g}"


def response_mask(n, spans):
    """长度 n 的 1/0 列表，注入段置 0"""
    m = [1] * n
    for s, l in spans:
        for i in range(s, min(n, s + l)):
            m[i] = 0
    return m


def vocab_limit(tok):
    """tokenizer 真实词表大小（含 added tokens）；假 tokenizer 没有 __len__ 就不限"""
    try:
        return len(tok)
    except TypeError:
        return 1 << 30


def drop_oov(ids, vocab, eos):
    """生成的 id 里第一个 ≥ vocab 的位置起截断，补 EOS（模型采到了输出层里没用过的占位行 = 写崩了）"""
    for i, t in enumerate(ids):
        if t >= vocab:
            return ids[:i] + [eos]
    return ids


def cut_at_stop(tok, ids, stops=STOPS):
    """ids 的文本里若出现停止串，截到「刚好含最早那个停止串」的最短 token 前缀；两个后端都过这一步，语义才一致"""
    text = tok.decode(ids, skip_special_tokens=True)
    found = [(text.find(s), s) for s in stops if s in text]
    if not found:
        return ids
    _, s = min(found)
    lo, hi = 1, len(ids)                       # 最小的 n：s in decode(ids[:n])（单调）
    while lo < hi:
        mid = (lo + hi) // 2
        if s in tok.decode(ids[:mid], skip_special_tokens=True):
            hi = mid
        else:
            lo = mid + 1
    return ids[:lo]


# ══════════════════════════════════════════════════════════════
#  后端：rows（每条 prompt+已生成 的 token id）+ budgets（每条还能写几个）→ 每条新生成的 id（停止串 / EOS 处截断）
# ══════════════════════════════════════════════════════════════
class HFBackend:
    def __init__(self, model, tok, dev=None, gen_bs=32):
        import torch
        self.torch, self.model, self.tok, self.gen_bs = torch, model, tok, gen_bs
        self.dev = dev or next(model.parameters()).device
        self.EOS = tok.eos_token_id
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        if pad == self.EOS:                      # ★ Qwen：pad == eos → 剥尾部 pad 会把真 EOS 一起剥掉（原 calc_tool.py 的 bug，臂 T 训练时响应 id 从没带过 EOS）
            cands = [i for i in (tok.all_special_ids or []) if i != self.EOS]    # 换一个模型不会生成的特殊 token 当 pad（<|im_start|>）
            pad = cands[0] if cands else self.EOS
        self.PAD = pad

    def generate(self, rows, budgets, temp, stops=STOPS):
        torch, tok, PAD, EOS = self.torch, self.tok, self.PAD, self.EOS
        outs = [None] * len(rows)
        with torch.no_grad():
            for c0 in range(0, len(rows), self.gen_bs):
                idx = list(range(c0, min(len(rows), c0 + self.gen_bs)))
                L = max(len(rows[i]) for i in idx)
                ids = torch.full((len(idx), L), PAD, dtype=torch.long)
                att = torch.zeros((len(idx), L), dtype=torch.long)
                for r, i in enumerate(idx):                              # 左填充
                    ids[r, L - len(rows[i]):] = torch.tensor(rows[i]); att[r, L - len(rows[i]):] = 1
                ids, att = ids.to(self.dev), att.to(self.dev)
                kw = dict(max_new_tokens=max(1, max(budgets[i] for i in idx)), do_sample=temp > 0, top_p=1.0, pad_token_id=PAD)
                if temp > 0:
                    kw["temperature"] = temp
                out = self.model.generate(input_ids=ids, attention_mask=att, stop_strings=stops, tokenizer=tok, **kw)
                for r, i in enumerate(idx):
                    new = out[r, L:].tolist()
                    while new and new[-1] == PAD:                         # 批里别人还在写时自己被填的 pad
                        new.pop()
                    if EOS in new:
                        new = new[: new.index(EOS) + 1]
                    outs[i] = new
        return outs


class VLLMBackend:
    """vLLM 离线引擎。hf_dir = export_hf.py 导出的目录（或 HF 仓库名）。tok 用同一个 tokenizer（注入段的 id 由它编）
    external=True：在 torchrun 起的训练进程里共存（distributed_executor_backend="external_launcher"，复用已有的进程组，每个 rank 自己一份）
    sleep=True：enable_sleep_mode，训练时 sleep() 让出显存，采样前 wake_up()；load_weights(state_dict) 把训练端权重灌进来（同卡本地拷贝）"""
    def __init__(self, hf_dir, tok, gpu_mem=0.6, max_model_len=2304, seed=0, external=False, sleep=False):
        from vllm import LLM, SamplingParams
        self.SP, self.tok = SamplingParams, tok
        self.EOS = tok.eos_token_id
        kw = dict(model=hf_dir, dtype="bfloat16", gpu_memory_utilization=gpu_mem, enable_prefix_caching=True,
                  max_model_len=max_model_len, seed=seed, disable_log_stats=True)
        if external:
            kw.update(distributed_executor_backend="external_launcher", tensor_parallel_size=1)
        if sleep:
            kw["enable_sleep_mode"] = True
        self.llm = LLM(**kw)
        self.asleep = False

    def sleep(self, level=2):
        """训练前让出显存。level 1：权重挪到 CPU、丢 KV；level 2：权重也直接丢（反正醒来要灌新的），只把非参数的缓冲（RoPE 表等）留在 CPU。
        level 2 的「缓冲没恢复」bug（issue #16564）由 PR #16889 修掉，0.8.5 起已包含 → 可以放心用 2"""
        if not self.asleep:
            self.llm.sleep(level=level); self.asleep = True

    def wake_up(self, tags=None):
        """★ 唤醒前先把 PyTorch 缓存着没用的显存块还给驱动：vLLM 唤醒是向驱动直接要物理显存，
        训练完 PyTorch 的分配器会留着几 GB 空闲块不还，驱动看到「还在用」就报 cumem out of memory。
        tags=["weights"] 只醒权重、["kv_cache"] 只醒 KV：官方给 RLHF 的分段唤醒，先醒权重灌完再醒 KV，峰值更低"""
        if self.asleep or tags:
            import torch, gc
            gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
            try:
                self.llm.wake_up(tags=tags) if tags else self.llm.wake_up()
            except TypeError:                                          # 老版没有 tags
                self.llm.wake_up()
            if not tags or "kv_cache" in tags:
                self.asleep = False

    def load_weights(self, state_dict):
        """把训练端的 state_dict（HF 命名）灌进 vLLM 的模型。同进程同卡，张量按引用传过去，device-to-device 拷贝"""
        items = list(state_dict.items())
        def _load(worker, items):
            return worker.model_runner.model.load_weights(items)
        self.llm.collective_rpc(_load, args=(items,))

    # ── ★ 逐步接口（级别二异步 harness 用）：往引擎塞请求 / 推一步 / 收完成的。就是 LLM.generate 内部那套，搬到外面来让每条轨迹自己走 ──
    def _engine(self):
        return self.llm.llm_engine

    def engine_add(self, rid, ids, budget, temp, stops=STOPS, logprobs=False):
        kw = dict(logprobs=0) if logprobs else {}                          # ★ logprobs=0：只回采到的那个 token 的 logp（量 vLLM 和 HF 的概率错位用）
        p = self.SP(max_tokens=max(1, budget), temperature=temp, top_p=1.0, stop=list(stops), include_stop_str_in_output=True, skip_special_tokens=False, **kw)
        self._engine().add_request(str(rid), {"prompt_token_ids": list(ids)}, p)

    def engine_step(self):
        """推一步 → [(rid, 新 token 列表, 每个 token 的 logp 或 None)]，只回这一步完成的请求；后处理和 generate 一样（因 EOS 停但没带 EOS 的补上）"""
        done = []
        nan = float("nan")
        for o in self._engine().step():
            if not o.finished:
                continue
            c = o.outputs[0]; new = list(c.token_ids)
            lps = None
            if getattr(c, "logprobs", None):
                lps = []
                for t, d in zip(c.token_ids, c.logprobs):
                    v = d.get(t) if isinstance(d, dict) else None
                    lps.append(float(v.logprob) if v is not None else nan)
            if c.finish_reason == "stop" and c.stop_reason is None and (not new or new[-1] != self.EOS):
                new.append(self.EOS)
            if self.EOS in new:
                new = new[: new.index(self.EOS) + 1]
            if lps is not None:
                lps = lps[: len(new)] + [nan] * (len(new) - len(lps[: len(new)]))
            done.append((o.request_id, new, lps))
        return done

    def engine_busy(self):
        return self._engine().has_unfinished_requests()

    def generate(self, rows, budgets, temp, stops=STOPS):
        params = [self.SP(max_tokens=max(1, b), temperature=temp, top_p=1.0, stop=list(stops),
                          include_stop_str_in_output=True, skip_special_tokens=False) for b in budgets]
        try:
            res = self.llm.generate([{"prompt_token_ids": r} for r in rows], params, use_tqdm=False)
        except TypeError:                                                  # 老版接口
            res = self.llm.generate(prompts=None, sampling_params=params, prompt_token_ids=rows, use_tqdm=False)
        outs = []
        for o in res:
            c = o.outputs[0]
            new = list(c.token_ids)
            if c.finish_reason == "stop" and c.stop_reason is None and (not new or new[-1] != self.EOS):
                new.append(self.EOS)                                       # 因 EOS 停但 id 里没带 EOS
            if self.EOS in new:
                new = new[: new.index(self.EOS) + 1]
            outs.append(new)
        return outs


def make_backend(engine, model=None, tok=None, hf_dir=None, dev=None, gen_bs=32, gpu_mem=0.6, max_model_len=2304):
    if engine == "vllm":
        return VLLMBackend(hf_dir, tok, gpu_mem=gpu_mem, max_model_len=max_model_len)
    return HFBackend(model, tok, dev=dev, gen_bs=gen_bs)


def _as_backend(model, tok, dev, gen_bs):
    """传进来的是 HF 模型（有 parameters）就包成 HFBackend；否则当作后端对象直接用（鸭子类型，测试用假后端也行）"""
    if hasattr(model, "parameters"):
        return HFBackend(model, tok, dev=dev, gen_bs=gen_bs)
    return model


# ══════════════════════════════════════════════════════════════
#  停、注入、续（后端无关）
# ══════════════════════════════════════════════════════════════
def generate_with_tools(model, tok, prompts, max_new=1024, temp=1.0, max_calls=48, gen_bs=32, dev=None, progress=False,
                        ask=False, max_asks=2, ask_fn=None, ask_workers=16):
    """progress=True：每 5 轮打一行进度。ask=True：多认 <ask>…</ask>（求助工具），一轮里所有求助并发发出去，注入 <reply>…</reply>；
    每条最多 max_asks 次；ask_fn(question) → 回答字符串（默认 ask_expert = DeepSeek reasoner；单测传假函数）"""
    import time
    from concurrent.futures import ThreadPoolExecutor
    be = _as_backend(model, tok, dev, gen_bs)
    EOS = tok.eos_token_id
    VOCAB = vocab_limit(tok)
    stops = STOPS_ASK if ask else STOPS
    ask_fn = ask_fn or ask_expert
    seqs = [dict(prompt=tok.encode(p, add_special_tokens=False), gen=[], spans=[], n_calls=0, n_err=0, n_asks=0,
                 fake=False, cont=False, cut=False, done=False) for p in prompts]
    rounds, t0 = 0, time.time()

    def inject(s, text):
        inj = tok.encode(text, add_special_tokens=False)
        if len(s["gen"]) + len(inj) > max_new:
            s["cut"] = True; s["done"] = True; return False
        s["spans"].append((len(s["gen"]), len(inj))); s["gen"] += inj; return True

    while True:
        active = [s for s in seqs if not s["done"]]
        if progress and (rounds % 5 == 0 or not active):
            done_n = len(seqs) - len(active)
            print(f"    第 {rounds:>2} 轮   还在写 {len(active):>5} 条   已完成 {done_n:>5}/{len(seqs)}   {time.time()-t0:>5.0f} s", flush=True)
        if not active:
            break
        rounds += 1
        outs = be.generate([s["prompt"] + s["gen"] for s in active], [max_new - len(s["gen"]) for s in active], temp, stops=stops)
        pending_ask = []                                       # (seq, question)：这一轮要问专家的，收齐后并发
        for s, new in zip(active, outs):
            new = cut_at_stop(tok, new, stops)
            new = drop_oov(new, VOCAB, EOS)
            remain = max_new - len(s["gen"])
            if len(new) > remain:
                new = new[:remain]
            s["gen"] += new
            text_new = tok.decode(new, skip_special_tokens=True)
            full = tok.decode(s["gen"], skip_special_tokens=True)
            if new and new[-1] == EOS:
                s["done"] = True
            elif CONT in full:                                # 幻觉出下一轮：截掉，补 EOS
                s["cont"] = True
                keep = tok.encode(full.split(CONT)[0], add_special_tokens=False)
                s["gen"] = keep + [EOS]; s["spans"] = [sp for sp in s["spans"] if sp[0] + sp[1] <= len(keep)]
                s["done"] = True
            elif RES_OPEN in text_new or (ask and REPLY_OPEN in text_new):   # ★ 自己编结果 / 编专家回复：立刻停
                s["fake"] = True; s["gen"].append(EOS); s["done"] = True
            elif text_new.rstrip().endswith(CALC_CLOSE) and s["n_calls"] < max_calls:
                found = CALC_RE.findall(full)
                v = evaluate(found[-1]) if found else "error"     # ★ 写了 </calc> 却没有 <calc>（SFT-1 在陌生模板上真发生过）→ 当算不了，别崩
                if inject(s, f"{RES_OPEN}{v}{RES_CLOSE}"):
                    s["n_calls"] += 1; s["n_err"] += (v == "error")
            elif ask and text_new.rstrip().endswith(ASK_CLOSE) and s["n_asks"] < max_asks:
                found = ASK_RE.findall(full)
                if found:
                    pending_ask.append((s, found[-1]))
                elif inject(s, f"{REPLY_OPEN}error{REPLY_CLOSE}"):   # </ask> 没有 <ask>：不问专家，直接回 error
                    s["n_asks"] += 1; s["n_err"] += 1
            elif len(s["gen"]) >= max_new:
                s["cut"] = True; s["done"] = True
            else:                                              # 没停在标签上又没 EOS：预算用完或调用满了
                s["cut"] = len(s["gen"]) >= max_new; s["done"] = True
        if pending_ask:                                        # ★ 并发问专家，回来按序注入
            with ThreadPoolExecutor(max_workers=ask_workers) as ex:
                answers = list(ex.map(lambda q: _ask_retry(ask_fn, q), [q for _, q in pending_ask]))
            for (s, _), a in zip(pending_ask, answers):
                if inject(s, f"{REPLY_OPEN}{a}{REPLY_CLOSE}"):
                    s["n_asks"] += 1; s["n_err"] += (a == "error")
    return [dict(ids=s["gen"], txt=tok.decode(s["gen"], skip_special_tokens=True), spans=s["spans"],
                 n_calls=s["n_calls"], n_err=s["n_err"], n_asks=s["n_asks"], fake=s["fake"], cont=s["cont"], cut=s["cut"],
                 rounds=rounds) for s in seqs]


def plain_generate(backend, tok, prompts, max_new=1024, temp=1.0):
    """不带工具的普通生成（评测非工具模型用），停在 \\nUser:。→ 每条 dict(ids, txt, cont, L, L_eff)"""
    EOS = tok.eos_token_id
    rows = [tok.encode(p, add_special_tokens=False) for p in prompts]
    outs = backend.generate(rows, [max_new] * len(rows), temp, stops=[CONT])
    res = []
    for new in outs:
        new = new[:max_new]
        t = tok.decode(new, skip_special_tokens=True)
        cont = CONT in t
        te = t.split(CONT)[0]
        res.append(dict(ids=new, txt=te, cont=cont, L=len(new), L_eff=len(tok(te).input_ids) if cont else len(new)))
    return res
