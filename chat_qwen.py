#!/usr/bin/env python3
"""
跟 Qwen 对话，顺便预演 GRPO 的前半段。

    python3 chat_qwen.py
    python3 chat_qwen.py --model Qwen/Qwen2.5-1.5B      # Base 版（第二次实验用）

  普通输入          多轮对话，流式输出
  /gsm              随机抽一道 GSM8K 题，答完自动判分
  ★ /grpo           对当前问题采样 8 次，判分，算优势 —— GRPO 的第 2~4 步
  /help             全部命令
"""
import re, time, random, argparse
from threading import Thread
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

ap = argparse.ArgumentParser()
ap.add_argument("--model",   default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--tokens",  type=int,   default=768)  # ★ 留余量：RL 会让回答变长
ap.add_argument("--temp",    type=float, default=0.7)
ap.add_argument("--dtype",   default="bf16", choices=["bf16", "fp32"])
ap.add_argument("--ckpt",    default=None,
                help="★ 加载 GRPO 训练出的权重，例如 out_grpo_lr5e6/ckpt_latest.pt")
ap.add_argument("--cmp-ckpt", default=None,
                help="★ /cmp 用的对照模型的 checkpoint（不给就用原始 Qwen）")
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
TD = torch.bfloat16 if (args.dtype == "bf16" and dev == "cuda") else torch.float32

def load_model(path, dtype, device):
    """
    不用 device_map —— 模型只有 3 GB，单卡装得下，device_map 需要 accelerate 还没好处。
    dtype 这个参数名在新版 transformers 里从 torch_dtype 改成了 dtype，两个都试一遍。
    """
    try:
        m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
    return m.to(device).eval()


print(f"载入 {args.model} …", flush=True)
tok = AutoTokenizer.from_pretrained(args.model)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = load_model(args.model, TD, dev)
CKINFO = "原始 Qwen（未训练）"
if args.ckpt:
    _c = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(_c["model"]); model.to(dev).eval()
    CKINFO = f"★ {args.ckpt}（step {_c.get('step','?')}）"
    del _c; torch.cuda.empty_cache()

n = sum(p.numel() for p in model.parameters())
cfg = model.config
print("=" * 72)
print(f"  {args.model}\n  {CKINFO}")
print(f"  {n/1e9:.2f}B 参数   {cfg.num_hidden_layers} 层   C={cfg.hidden_size}   "
      f"{cfg.num_attention_heads} 头 / ★ {getattr(cfg,'num_key_value_heads',cfg.num_attention_heads)} KV头(GQA)")
print(f"  词表 {cfg.vocab_size:,}   上下文 {cfg.max_position_embeddings:,}   "
      f"显存 {torch.cuda.memory_allocated()/1024**3:.1f} GB")
print("=" * 72)

P = dict(temperature=args.temp, top_p=0.9, n=args.tokens, k=8)
HIST = []
GSM = None                                  # 懒加载
CMP = None                                  # /cmp 的对照模型，懒加载
cur_q = None                                # /grpo 用的当前问题
last = None
LASTG = None                                # 上一轮 /grpo 的 (题目, 答案, 8 条文本, 8 个分数)

HELP = """
  /gsm               随机抽一道 GSM8K 题（会自动判分）
  ★ /grpo             ★ 直接抽一道新题，采样 8 次 + 判分 + 算优势
  /grpo same         重跑上一道（看同一道题的方差）
  /grpo <问题>        用你自己的问题（没标准答案，判分会全 0）
  /set temp 1.0      温度（★ GRPO 训练时用 1.0，聊天用 0.7）
  /set n 512         最多生成几个 token
  /set k 8           /grpo 采样几次（组大小）
  /clear             清空对话历史
  ★ /cmp <问题>       ★ 当前模型 vs 对照模型，同一问题并排（看训练前后的差别）
  ★ /show [n]         打印上一轮 /grpo 的完整输出（不带 n 就全打）
                     ← 用来查「判分说错了但其实答对了」的情况
  /stats             上次生成的速度
  /help  /quit
"""

from _reward import reward, extract, has_format    # ★ 判分器抽成独立模块，见 _reward.py


def load_gsm():
    global GSM
    if GSM is None:
        print("  首次使用，载入 GSM8K …", end="", flush=True)
        from datasets import load_dataset
        d = load_dataset("openai/gsm8k", "main", split="test")
        GSM = [(x["question"], x["answer"].split("####")[-1].strip().replace(",", "")) for x in d]
        print(f" {len(GSM):,} 题")
    return GSM


def prompt_of(msgs):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def stream(msgs):
    """流式生成一条"""
    enc = tok(prompt_of(msgs), return_tensors="pt").to(dev)
    st = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    kw = dict(**enc, streamer=st, max_new_tokens=P["n"], do_sample=P["temperature"] > 0,
              temperature=max(P["temperature"], 1e-5), top_p=P["top_p"],
              pad_token_id=tok.pad_token_id)
    Thread(target=model.generate, kwargs=kw).start()
    t0, out, n_new = time.time(), "", 0
    for piece in st:
        print(piece, end="", flush=True)
        out += piece
    dt = time.time() - t0
    n_new = len(tok.encode(out))
    return out, dict(n_prompt=enc.input_ids.size(1), n_new=n_new, t=dt,
                     tps=n_new / max(dt, 1e-6))


@torch.no_grad()
def sample_k(question, k):
    """一次采 k 条 —— GRPO 的第 2 步"""
    INSTR = "\n\nSolve this step by step. End your answer with '#### <number>'."
    p = prompt_of([{"role": "user", "content": question + INSTR}])
    enc = tok([p] * k, return_tensors="pt", padding=True).to(dev)
    t0 = time.time()
    out = model.generate(**enc, max_new_tokens=P["n"], do_sample=True,
                         temperature=1.0, top_p=1.0, pad_token_id=tok.pad_token_id)
    new = out[:, enc.input_ids.size(1):]
    txts = [tok.decode(o, skip_special_tokens=True) for o in new]
    ntok = int((new != tok.pad_token_id).sum())
    return txts, time.time() - t0, ntok


print(HELP)

while True:
    try:
        line = input("\n>>> ").strip()
    except (EOFError, KeyboardInterrupt):
        print(); break
    if not line:
        continue
    if line in ("/quit", "/q", "/exit"):
        break
    if line == "/help":
        print(HELP); continue
    if line in ("/clear", "/reset"):
        HIST.clear(); print("  历史已清空"); continue
    if line.startswith("/show"):
        if LASTG is None:
            print("  先跑一次 /grpo"); continue
        q_, g_, ts_, rs_ = LASTG
        arg = line[5:].strip()
        idx = range(len(ts_)) if not arg else [int(arg) - 1]
        for i in idx:
            a, f = extract(ts_[i])
            print(f"\n{'━'*72}\n  第 {i+1} 条   {'✅' if rs_[i] else '❌'}   "
                  f"抽出 {a!r}   标准答案 {g_}   {'有 ####' if f else '★ 无 ####'}\n{'━'*72}")
            print(ts_[i].strip())
            print(f"\n  ── 末尾 80 字符（抽取器就看这段）──\n  …{ts_[i].strip()[-80:]!r}")
        continue
    if line == "/stats":
        if last:
            print(f"\n  prompt {last['n_prompt']} token   生成 {last['n_new']} token"
                  f"   {last['t']:.1f} 秒   ★ {last['tps']:.1f} tok/s")
        else:
            print("  还没生成过")
        continue
    if line.startswith("/set"):
        try:
            _, kk, v = line.split()
            key = {"temp": "temperature", "top_p": "top_p", "n": "n", "k": "k"}[kk]
            P[key] = int(v) if key in ("n", "k") else float(v)
            print(f"  {key} = {P[key]}")
        except Exception:
            print("  用法: /set temp 1.0 | /set n 512 | /set k 8")
        continue

    # ── /gsm：抽一道题，答一次，判分 ──
    if line.startswith("/gsm"):
        g = load_gsm()
        q, gold = random.choice(g)
        cur_q = (q, gold)
        print(f"\n【GSM8K】{q}\n  标准答案 {gold}\n{'─'*72}")
        INSTR = "\n\nSolve this step by step. End your answer with '#### <number>'."
        txt, last = stream([{"role": "user", "content": q + INSTR}])
        r = reward(txt, gold)
        print(f"\n\n  {'✅ 答对' if r else '❌ 答错'}   reward = {r}   "
              f"{last['n_new']} token   {last['tps']:.1f} tok/s")
        print("  ★ 想看 GRPO 怎么处理这道题，输入 /grpo")
        continue

    # ── ★ /cmp：当前模型 vs 对照模型，同一个问题并排 ──
    if line.startswith("/cmp"):
        q = line[4:].strip()
        if not q:
            if cur_q is None:
                print("  用法: /cmp <问题>   或先 /grpo 抽一道题再 /cmp"); continue
            q, gold = cur_q
        else:
            gold = None
        if CMP is None:
            tagb = args.cmp_ckpt or "原始 Qwen（未训练）"
            print(f"  首次使用，载入对照模型 {tagb} …", end="", flush=True)
            CMP = load_model(args.model, TD, dev)
            if args.cmp_ckpt:
                _c = torch.load(args.cmp_ckpt, map_location="cpu", weights_only=False)
                CMP.load_state_dict(_c["model"]); CMP.to(dev).eval(); del _c
                torch.cuda.empty_cache()
            print(f" 完成   显存 {torch.cuda.memory_allocated()/1024**3:.1f} GB")

        INSTR2 = "\n\nSolve this step by step. End your answer with '#### <number>'."
        msgs = [{"role": "user", "content": q + (INSTR2 if gold else "")}]
        print(f"\n【问题】{q}" + (f"\n  标准答案 {gold}" if gold else ""))

        for tag, m in ((f"对照：{args.cmp_ckpt or '原始 Qwen'}", CMP),
                       (f"当前：{CKINFO}", model)):
            print(f"\n  ── {tag} ──")
            _save = model
            globals()["model"] = m                  # stream() 里用的是全局 model
            txt, st = stream(msgs)
            globals()["model"] = _save
            r = reward(txt, gold) if gold else None
            has = "有 ####" if "####" in txt else "★ 无 ####"
            print(f"\n     [{st['n_new']} token  {st['tps']:.1f} tok/s  {has}"
                  + (f"  {'✅ 答对' if r else '❌ 答错'}" if gold else "") + "]")
        continue

    # ── ★ /grpo：采样 k 次 + 判分 + 算优势 ──
    if line.startswith("/grpo"):
        rest = line[5:].strip()
        if rest == "same" and cur_q is not None:
            pass                                    # 重跑上一道，看方差
        elif rest and rest not in ("new", "same"):
            cur_q = (rest, None)                    # 自定义问题（没标准答案）
        else:
            cur_q = random.choice(load_gsm())       # ★ 默认每次抽新题
        q, gold = cur_q
        if gold is None:
            print("  （自定义问题没有标准答案，判分会全 0 —— 用 /gsm 抽的题才有意义）")
        k = P["k"]
        print(f"\n【问题】{q}\n  标准答案 {gold}")
        print(f"\n  ── 第 2 步：采样 {k} 次（T=1.0，要多样性）──", flush=True)
        txts, dt, ntok = sample_k(q, k)

        print(f"\n  ── 第 3 步：判分（这就是全部的『环境』）──")
        rs = [reward(t, gold) if gold else 0.0 for t in txts]
        for i, (t, r) in enumerate(zip(txts, rs)):
            a, has_fmt = extract(t)                      # ★ 显示判分器真正抽出的数
            nt = len(tok.encode(t))
            cut = " ✂️截断" if nt >= P["n"] - 2 else ""   # ★ 撞上限=假阴性，要盯
            print(f"    {i+1}  {'✅' if r else '❌'} {r:.1f}   抽出 {str(a):>10}  "
                  f"{'####' if has_fmt else ' -- '}  {nt:>4}tok{cut}   {t.strip()[:56]}…")

        mean = sum(rs) / len(rs)
        print(f"\n  ── 第 4 步：算优势 ──")
        print(f"    平均分 = {sum(rs):.0f} / {k} = {mean:.3f}")
        for i, r in enumerate(rs):
            a = r - mean
            arrow = "★ 推大" if a > 0 else ("★ 压小" if a < 0 else "  不动")
            print(f"    {i+1}  A = {r:.1f} − {mean:.3f} = {a:+.3f}   {arrow}"
                  f"   （这条回答的每个 token 都用这个 A）")

        LASTG = (q, gold, txts, rs)
        n_ok = sum(rs)
        n_cut = sum(1 for t in txts if len(tok.encode(t)) >= P["n"] - 2)
        print(f"\n  ── 结论 ──")
        if n_cut:
            print(f"    ⚠️ {n_cut}/{k} 条撞上 {P['n']} token 上限被截断 → ★ 假阴性")
            print(f"       RL 会让回答越训越长，上限要留余量：/set n {P['n']*2}")
        if n_ok == 0 or n_ok == k:
            print(f"    ⚠️ {k} 条{'全对' if n_ok else '全错'} → 优势全 0 → "
                  f"★ 这道题产生不了梯度，白跑")
        else:
            print(f"    ✅ {int(n_ok)} 对 {k-int(n_ok)} 错 → ★ 有梯度信号，这道题有用")
        print(f"\n    耗时 {dt:.1f} 秒   {ntok} token   {ntok/dt:.0f} tok/s"
              f"   → ★ 训练时每步大概就这个开销")
        continue

    # ── 普通对话 ──
    HIST.append({"role": "user", "content": line})
    print()
    txt, last = stream(HIST)
    HIST.append({"role": "assistant", "content": txt})
    print(f"\n\n  [{last['n_new']} token  {last['tps']:.1f} tok/s  "
          f"T={P['temperature']}  历史 {len(HIST)//2} 轮]")
