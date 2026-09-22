#!/usr/bin/env python3
"""
harness.py —— 通用的「停 → 环境回一段 → 注入 → 续」循环（E 猜数字起用；以后编程题的 <run> 也走这里）。
后端层（HFBackend / VLLMBackend / make_backend / cut_at_stop / response_mask）复用 calc_tool_vllm，那个文件原样不动。

环境接口（鸭子类型，见 guess_env.GuessEnv）：
    env.stops       list[str]   生成停在这些串上（如 ["</guess>"]）
    env.fake_tags   list[str]   模型自己写出这些 = 假观测 → 立刻停、标 fake（训练器罚分）
    env.step(state, full_text) → (inject_text | None, done)   停下后调用；inject_text 贴回文本（记 spans，mask 掉）；done=True 补 EOS 结束
返回每条：ids / txt / spans / fake / cont / cut / rounds / state（环境的每条状态，打分用）
"""
import os, time
import calc_tool_vllm as T

CONT = T.CONT


def generate_with_env(backend, tok, prompts, env, states, max_new=512, temp=1.0, gen_bs=32, dev=None, progress=False, step_workers=1, max_total=None, sync=None, inflight=None, want_lp=False):
    """入口。后端有逐步接口（VLLMBackend.engine_step）且没被 HARNESS_ASYNC=0 关掉 → 走级别二异步（每条轨迹自己走，步末汇合一次）；否则走同步轮次。
    inflight：异步时同时在引擎里的轨迹数上限（HARNESS_INFLIGHT，默认 128）—— 超过 KV cache 能装的条数，每次从沙盒回来前缀就被挤出去、整段重算 prefill，GPU 时间反而涨"""
    if sync is None:
        sync = os.environ.get("HARNESS_ASYNC", "1") == "0" or not hasattr(backend, "engine_step")
    if not sync:
        try:
            return generate_with_env_async(backend, tok, prompts, env, states, max_new, temp, progress, step_workers, max_total,
                                           inflight or int(os.environ.get("HARNESS_INFLIGHT", "128")), want_lp=want_lp)
        except (AttributeError, TypeError) as e:                       # 引擎接口对不上（版本差）→ 退回同步，打警告，别把训练搞崩
            print(f"    ⚠ 异步 harness 不可用（{e.__class__.__name__}: {e}），退回同步轮次", flush=True)
    return generate_with_env_sync(backend, tok, prompts, env, states, max_new, temp, gen_bs, dev, progress, step_workers, max_total)


def generate_with_env_sync(backend, tok, prompts, env, states, max_new=512, temp=1.0, gen_bs=32, dev=None, progress=False, step_workers=1, max_total=None):
    """step_workers > 1：一轮里停在标签上的各条并发调 env.step（环境要线程安全：code_env 每条一个临时目录，可以；猜数字纯计算也可以）—— 沙盒跑测试是 I/O，并发快 4~8 倍"""
    from concurrent.futures import ThreadPoolExecutor
    be = T._as_backend(backend, tok, dev, gen_bs)
    EOS = tok.eos_token_id
    VOCAB = T.vocab_limit(tok)
    stops = list(env.stops) + list(env.fake_tags) + [CONT]       # 假观测的开标签也是停止串：一写出来就停，下面才能判 fake（同 calc_tool 的 <result>）
    seqs = [dict(prompt=tok.encode(p, add_special_tokens=False), gen=[], spans=[], fake=False, cont=False, cut=False, done=False, state=st)
            for p, st in zip(prompts, states)]
    if max_total:                                                  # ★ 引擎的上下文上限：每条的预算 = min(max_new, 上限 − 题面长)，超长的条直接截，不让 vLLM 报错
        for s in seqs:
            s["budget"] = max(1, min(max_new, max_total - len(s["prompt"]) - 8))
    else:
        for s in seqs:
            s["budget"] = max_new
    rounds, t0 = 0, time.time()
    T_GEN = T_ENV = 0.0; N_STEP = 0; MAX_PEND = 0                  # ★ 计时：生成 / 环境各花多少，最多同时几条等环境

    def inject(s, text):
        inj = tok.encode(text, add_special_tokens=False)
        if len(s["gen"]) + len(inj) > s["budget"]:
            s["cut"] = True; s["done"] = True; return False
        s["spans"].append((len(s["gen"]), len(inj))); s["gen"] += inj; return True

    while True:
        active = [s for s in seqs if not s["done"]]
        if progress and (rounds % 5 == 0 or not active):
            print(f"    第 {rounds:>2} 轮   还在写 {len(active):>5} 条   已完成 {len(seqs)-len(active):>5}/{len(seqs)}   {time.time()-t0:>5.0f} s", flush=True)
        if not active:
            break
        rounds += 1
        _t = time.time()
        outs = be.generate([s["prompt"] + s["gen"] for s in active], [s["budget"] - len(s["gen"]) for s in active], temp, stops=stops)
        T_GEN += time.time() - _t
        pending = []                                            # (seq, full)：这一轮停在环境标签上的，收齐后（并发）调 env.step 再注入
        for s, new in zip(active, outs):
            new = T.cut_at_stop(tok, new, stops)
            new = T.drop_oov(new, VOCAB, EOS)                  # ★ 采到词表外的占位 id（Qwen 输出层多 270 行）→ 截断补 EOS，否则下一轮喂回去 vLLM 报 out of vocabulary
            remain = s["budget"] - len(s["gen"])
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
            elif any(tag in text_new for tag in env.fake_tags):   # ★ 自己编观测：立刻停
                s["fake"] = True; s["gen"].append(EOS); s["done"] = True
            elif any(text_new.rstrip().endswith(st) for st in env.stops):
                pending.append((s, full))
            elif len(s["gen"]) >= s["budget"]:
                s["cut"] = True; s["done"] = True
            else:                                              # 没停在标签上又没 EOS：预算用完
                s["cut"] = len(s["gen"]) >= s["budget"]; s["done"] = True
        if pending:
            _t = time.time(); N_STEP += len(pending); MAX_PEND = max(MAX_PEND, len(pending))
            if step_workers > 1 and len(pending) > 1:
                with ThreadPoolExecutor(max_workers=step_workers) as ex:
                    results = list(ex.map(lambda sf: env.step(sf[0]["state"], sf[1]), pending))
            else:
                results = [env.step(s["state"], full) for s, full in pending]
            for (s, _), (inj, done) in zip(pending, results):
                if inj is not None and not inject(s, inj):
                    continue                                   # 注入放不下 → 已标 cut/done
                if done:
                    s["gen"].append(EOS); s["done"] = True
            T_ENV += time.time() - _t
    generate_with_env.last_timing = dict(gen=T_GEN, env=T_ENV, rounds=rounds, env_calls=N_STEP, max_pending=MAX_PEND)
    if progress:
        print(f"    计时：生成 {T_GEN:.1f} s  环境 {T_ENV:.1f} s（{N_STEP} 次调用，{rounds} 轮，一轮最多 {MAX_PEND} 条同时等）", flush=True)
    return [dict(ids=s["gen"], txt=tok.decode(s["gen"], skip_special_tokens=True), spans=s["spans"], fake=s["fake"], cont=s["cont"],
                 cut=s["cut"], rounds=rounds, state=s["state"]) for s in seqs]


def generate_with_env_async(be, tok, prompts, env, states, max_new=512, temp=1.0, progress=False, step_workers=8, max_total=None, inflight=128, want_lp=False):
    """★ 级别二异步：每条轨迹自己走 —— 写到标签停下 → 自己去跑沙盒（线程池）→ 跑完自己回引擎接着写；引擎连续批处理，谁来了就带上谁。
    只在全部轨迹结束时汇合一次；仍然严格 on-policy（权重整步不变）。输出和同步版完全一样（同一套截断 / 假观测 / 预算规则）"""
    import time as _time
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
    EOS, CONT, VOCAB = tok.eos_token_id, "\nUser:", T.vocab_limit(tok)
    stops = list(env.stops) + list(env.fake_tags)
    seqs = [dict(prompt=tok.encode(p, add_special_tokens=False), gen=[], spans=[], fake=False, cont=False, cut=False, done=False, state=st, n_env=0, lp=[])
            for p, st in zip(prompts, states)]
    NAN = float("nan")
    for s in seqs:
        s["budget"] = max(1, min(max_new, max_total - len(s["prompt"]) - 8)) if max_total else max_new
    T_GEN = T_ENV = 0.0; N_STEP = 0; MAX_PEND = 0; n_sub = 0; t0 = _time.time(); last_pr = t0
    active = {}                                                     # rid → seq（在引擎里写的）
    futs = {}                                                       # future → seq（在沙盒里跑的）
    pool = ThreadPoolExecutor(max_workers=max(1, step_workers))

    def inject(s, text):
        inj = tok.encode(text, add_special_tokens=False)
        if len(s["gen"]) + len(inj) > s["budget"]:
            s["cut"] = True; s["done"] = True; return False
        s["spans"].append((len(s["gen"]), len(inj))); s["gen"] += inj; s["lp"] += [NAN] * len(inj); return True

    def submit(s):
        nonlocal n_sub
        rid = f"{id(s)}-{n_sub}"; n_sub += 1
        active[rid] = s
        if want_lp:
            try:
                be.engine_add(rid, s["prompt"] + s["gen"], s["budget"] - len(s["gen"]), temp, stops, logprobs=True); return
            except TypeError:
                pass                                                # 后端不认 logprobs 参数：退回老接口
        be.engine_add(rid, s["prompt"] + s["gen"], s["budget"] - len(s["gen"]), temp, stops)

    def env_call(s, full):
        t = _time.time(); r = env.step(s["state"], full); return r, _time.time() - t

    def on_generated(s, new, lps=None):
        new = T.cut_at_stop(tok, new, stops)
        new = T.drop_oov(new, VOCAB, EOS)
        remain = s["budget"] - len(s["gen"])
        if len(new) > remain:
            new = new[:remain]
        lps = list(lps or [])[: len(new)]; lps += [NAN] * (len(new) - len(lps))     # 对齐到截断后的 new；没要 logp 就全是 nan
        s["gen"] += new; s["lp"] += lps
        text_new = tok.decode(new, skip_special_tokens=True)
        full = tok.decode(s["gen"], skip_special_tokens=True)
        if new and new[-1] == EOS:
            s["done"] = True
        elif CONT in full:
            s["cont"] = True
            keep = tok.encode(full.split(CONT)[0], add_special_tokens=False)
            s["gen"] = keep + [EOS]; s["lp"] = s["lp"][: len(keep)] + [NAN]; s["spans"] = [sp for sp in s["spans"] if sp[0] + sp[1] <= len(keep)]
            s["done"] = True
        elif any(tag in text_new for tag in env.fake_tags):
            s["fake"] = True; s["gen"].append(EOS); s["lp"].append(NAN); s["done"] = True
        elif any(text_new.rstrip().endswith(st) for st in env.stops):
            s["n_env"] += 1
            futs[pool.submit(env_call, s, full)] = s
        elif len(s["gen"]) >= s["budget"]:
            s["cut"] = True; s["done"] = True
        else:
            s["cut"] = len(s["gen"]) >= s["budget"]; s["done"] = True

    def on_env(s, res):
        (inj, done), dt = res
        nonlocal T_ENV
        T_ENV += dt
        if inj is not None and not inject(s, inj):
            return                                                  # 注入放不下 → 已标 cut/done
        if done:
            s["gen"].append(EOS); s["lp"].append(NAN); s["done"] = True
        else:
            submit(s)

    waiting = list(seqs)                                            # 还没进引擎的（滑动窗口：一条结束再放一条进来，让每条的前缀在 KV cache 里活到它回来）
    try:
        while waiting and len(active) + len(futs) < inflight:
            submit(waiting.pop(0))
        while active or futs or waiting:
            while waiting and len(active) + len(futs) < inflight:
                submit(waiting.pop(0))
            if active:
                t = _time.time()
                for item in be.engine_step():
                    rid, new = item[0], item[1]; lps = item[2] if len(item) > 2 else None
                    s = active.pop(rid, None)
                    if s is not None:
                        on_generated(s, new, lps)
                T_GEN += _time.time() - t
            if futs:
                MAX_PEND = max(MAX_PEND, len(futs))
                done_f, _ = wait(list(futs), timeout=0 if active else 0.05, return_when=FIRST_COMPLETED)
                for f in done_f:
                    s = futs.pop(f); N_STEP += 1
                    on_env(s, f.result())
            if progress and _time.time() - last_pr > 30:
                last_pr = _time.time()
                print(f"    已完成 {sum(s['done'] for s in seqs):>5}/{len(seqs)}   在写 {len(active):>4}   在跑沙盒 {len(futs):>4}   排队 {len(waiting):>4}   {last_pr - t0:>5.0f} s", flush=True)
    finally:
        pool.shutdown(wait=True)
    generate_with_env.last_timing = dict(gen=T_GEN, env=T_ENV, rounds=max((s["n_env"] for s in seqs), default=0), env_calls=N_STEP, max_pending=MAX_PEND,
                                         wall=_time.time() - t0, mode="async", inflight=inflight)
    if progress:
        print(f"    计时（异步，窗口 {inflight}）：墙钟 {_time.time()-t0:.1f} s  引擎 {T_GEN:.1f} s  沙盒累计 {T_ENV:.1f} s（{N_STEP} 次调用，最多一条 {generate_with_env.last_timing['rounds']} 次，最多 {MAX_PEND} 条同时在跑沙盒）", flush=True)
    return [dict(ids=s["gen"], txt=tok.decode(s["gen"], skip_special_tokens=True), spans=s["spans"], fake=s["fake"], cont=s["cont"],
                 cut=s["cut"], state=s["state"], lp=(s["lp"] if want_lp else None)) for s in seqs]
