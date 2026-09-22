#!/usr/bin/env python3
"""
跟 SFT 后的模型对话（chat.py 是跟预训练的 base 模型续写，两回事）。

    python3 chat_sft.py                              # out_sft/ckpt_final.pt（过训版）
    python3 chat_sft.py --ckpt out_sft/ckpt.pt       # 欠训版
    python3 chat_sft.py --raw                        # 不套 Alpaca 格式（看格式依赖）

跟 chat.py 的四点不同：
  ① 自动套上训练时用的 Alpaca 格式（### Instruction / ### Response）
  ② 生成到 <|endoftext|> 就停 —— SFT 教会的就是这个
  ③ 流式输出，边生成边打
  ④ /cmp 把同一个问题同时喂给 base 模型，直接看差别
"""
import time, argparse
import torch
import torch.nn.functional as F
import tiktoken

from config import Config
from model import GPT

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt",   default="out_sft/ckpt_final.pt")
ap.add_argument("--base",   default="out_run2/ckpt_final.pt", help="/cmp 用的对照模型")
ap.add_argument("--tokens", type=int,   default=200)
ap.add_argument("--temp",   type=float, default=0.7)
ap.add_argument("--dtype",  default="bf16", choices=["bf16", "fp32"])
ap.add_argument("--raw",    action="store_true", help="启动时就关掉 Alpaca 格式")
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token
TD = torch.bfloat16 if (args.dtype == "bf16" and dev == "cuda") else torch.float32


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config(**{k: v for k, v in ck["cfg"].items() if k in Config.__dataclass_fields__})
    m = GPT(cfg); m.load_state_dict(ck["model"])
    return m.to(device=dev, dtype=TD).eval(), cfg, ck


model, cfg, ck = load(args.ckpt)
base = None                                   # /cmp 第一次用到时再载，省显存

P = dict(temperature=args.temp, top_p=0.92, top_k=None,
         repetition_penalty=1.15, n=args.tokens, cache=True, alpaca=not args.raw,
         multi=False)

BYTES = 2 if TD == torch.bfloat16 else 4
W_MB = model.num_params() * BYTES / 1024**2
HBM = 1008.0 if dev == "cuda" else 50.0

print("=" * 68)
print(f"  {args.ckpt}   step {ck['step']:,}   val {ck['best_val']:.4f}")
print(f"  {model.num_params()/1e6:.1f}M 参数   {dev}   {'bf16' if TD==torch.bfloat16 else 'fp32'}"
      f"   权重 {W_MB:.0f} MB")
print(f"  格式 {'Alpaca（### Instruction / ### Response）' if P['alpaca'] else '★ 关闭（原始问题直接喂）'}"
      f"   T={P['temperature']}")
print("=" * 68)

if dev == "cuda":                             # 预热，别让首次的 JIT 混进计时
    print("  预热中…", end="", flush=True)
    with torch.no_grad():
        w = torch.zeros(1, 8, dtype=torch.long, device=dev)
        c = [(None, None) for _ in range(cfg.n_layer)]
        model(w, kv_cache=c, pos=0); model(w[:, :1], kv_cache=c, pos=8)
    torch.cuda.synchronize()
    print(" 完成")

print("""
  直接输入问题，回车。模型会按训练时的格式回答，答完自己停。

  问题 ||| 素材        给它一段素材去处理（对应 Alpaca 的 Input 字段）
                       例：Summarize this ||| The Amazon rainforest produces…
  /help 看全部命令     /quit 退出""")

HELP = """
  /raw on|off        Alpaca 格式开关 —— off 之后它多半"会停但不会答"
  /cmp <问题>        同一问题同时跑 base 和 SFT，两个输出并排
  /multi on|off      ★ 把对话历史拼进上下文（默认关 —— Alpaca 全是单轮数据）
  /clear             清空历史
  /show              打出上一次的完整 prompt + 生成的 token（含不可见的 EOT）
  /set temp 0.5      温度（SFT 模型建议 0.5~0.8）
  /set top_p 0.9   /set rep 1.2   /set n 300
  /cache on|off      KV cache 开关
  /stats             上一次生成的速度拆解
  /help  /quit
"""


HIST = []        # [(问题, 回答), …]，只有 /multi 开着时才用


def block(q, resp=None):
    """一轮的 Alpaca 块。||| 前后拆成 instruction / input。"""
    instr, _, inp = q.partition("|||")
    p = f"### Instruction:\n{instr.strip()}\n"
    if inp.strip():
        p += f"\n### Input:\n{inp.strip()}\n"
    p += "\n### Response:\n"
    return p + (resp.strip() + "\n\n" if resp is not None else "")


def wrap(q):
    """套成训练时的样子。/multi 开着就把历史一起拼进去。"""
    if not P["alpaca"]:
        return q
    if not P["multi"]:
        return block(q)
    # 历史 + 当前，超长就从最旧的一轮开始丢
    hist = list(HIST)
    while True:
        s = "".join(block(hq, ha) for hq, ha in hist) + block(q)
        if len(enc.encode_ordinary(s)) + P["n"] < cfg.block_size or not hist:
            return s
        hist.pop(0)


@torch.no_grad()
def gen(m, prompt, stream=True):
    """流式生成，遇到 EOT 就停。返回 (文本, 统计)。"""
    ids = torch.tensor([enc.encode_ordinary(prompt)], dtype=torch.long, device=dev)
    if ids.size(1) >= cfg.block_size:
        ids = ids[:, -cfg.block_size + 1:]
    n_prompt = ids.size(1)

    def sync():
        if dev == "cuda":
            torch.cuda.synchronize()

    seen = torch.zeros(cfg.vocab_size, dtype=torch.bool, device=dev)
    seen[ids[0]] = True

    def sample(logits):
        lg = logits[:, -1, :].float()
        rp = P["repetition_penalty"]
        if rp != 1.0:
            l = lg[0]
            lg[0] = torch.where(seen, torch.where(l > 0, l / rp, l * rp), l)
        lg = lg / max(P["temperature"], 1e-5)
        if P["top_k"]:
            v, _ = torch.topk(lg, min(P["top_k"], lg.size(-1)))
            lg[lg < v[:, [-1]]] = -float("inf")
        if P["top_p"] and P["top_p"] < 1.0:
            sl, si = torch.sort(lg, descending=True, dim=-1)
            cum = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
            drop = cum > P["top_p"]
            drop[..., 1:] = drop[..., :-1].clone(); drop[..., 0] = False
            lg = torch.full_like(lg, -float("inf")).scatter_(-1, si, sl.masked_fill(drop, -float("inf")))
        return torch.multinomial(F.softmax(lg, dim=-1), 1)

    cache = [(None, None) for _ in range(cfg.n_layer)] if P["cache"] else None
    sync(); t0 = time.time()
    logits, _ = m(ids, kv_cache=cache, pos=0) if P["cache"] else m(ids)
    sync(); t_pre = time.time() - t0

    new, printed, pos, stopped = [], "", n_prompt, False
    sync(); t0 = time.time()
    for _ in range(P["n"]):
        if pos >= cfg.block_size:
            break
        nxt = sample(logits)
        tid = int(nxt[0, 0])
        if tid == EOT:                                   # ★ SFT 教会的：答完就停
            stopped = True
            break
        new.append(tid); seen[nxt[0]] = True

        if stream:
            full = enc.decode(new)
            if not full.endswith("�"):              # 多字节字符没拼完就先不打
                print(full[len(printed):], end="", flush=True)
                printed = full

        if P["cache"]:
            logits, _ = m(nxt, kv_cache=cache, pos=pos); pos += 1
        else:
            ids = torch.cat((ids, nxt), 1)
            logits, _ = m(ids[:, -cfg.block_size:]); pos += 1
    sync(); t_dec = time.time() - t0

    txt = enc.decode(new)
    if stream and txt != printed:
        print(txt[len(printed):], end="", flush=True)
    return txt, dict(n_prompt=n_prompt, n_new=len(new), stopped=stopped,
                     prompt=prompt, ids=new + ([EOT] if stopped else []),
                     t_pre=t_pre, t_dec=t_dec,
                     pre_tps=n_prompt / max(t_pre, 1e-6),
                     dec_tps=len(new) / max(t_dec, 1e-6))


def show_stats(s):
    print(f"\n  ── 速度拆解 ──")
    print(f"    prefill  读入 {s['n_prompt']:>4} token   {s['t_pre']*1000:>7.1f} ms"
          f"   {s['pre_tps']:>8.0f} tok/s   ← 并行，compute-bound")
    print(f"    decode   生成 {s['n_new']:>4} token   {s['t_dec']*1000:>7.1f} ms"
          f"   {s['dec_tps']:>8.1f} tok/s   ← 串行，memory-bound")
    lim = HBM * 1024 / W_MB
    print(f"    decode 理论上限 {lim:.0f} tok/s，实测利用率 {100*s['dec_tps']/lim:.0f}%")
    print(f"    ★ 推理的瓶颈在访存，不在算力")


last = None
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
    if line == "/stats":
        show_stats(last) if last else print("  还没生成过"); continue
    if line.startswith("/raw"):
        a = line.split()[1:]
        P["alpaca"] = (a[0] == "off") if a else (not P["alpaca"])   # /raw on = 关格式
        print(f"  Alpaca 格式 = {'开' if P['alpaca'] else '★ 关（原始问题直接喂，预期会停但不会答）'}")
        continue
    if line.startswith("/multi"):
        a = line.split()[1:]
        P["multi"] = (a[0] != "off") if a else (not P["multi"])
        print(f"  多轮历史 = {'★ 开（Alpaca 全是单轮数据，预期它会犯糊涂）' if P['multi'] else '关'}"
              f"   当前 {len(HIST)} 轮")
        continue
    if line in ("/clear", "/reset"):
        HIST.clear(); print("  历史已清空"); continue
    if line == "/show":
        if last is None:
            print("  还没生成过"); continue
        pr, ids = last["prompt"], last["ids"]
        pt = enc.encode_ordinary(pr)
        print(f"\n  模型实际收到的 prompt（{len(pt)} token）:\n   {pr!r}\n")
        print(f"  {'#':>4} {'id':>6}  解码")
        for i, t in enumerate(pt):
            print(f"  {i:>4} {t:>6}  {enc.decode([t])!r}")
        print(f"  ─── 以上喂进去 {len(pt)} 个，以下模型生成 {len(ids)} 个 ───")
        for j, t in enumerate(ids):
            d = "'<|endoftext|>'  ★ SFT 教会的就是这一个" if t == EOT else repr(enc.decode([t]))
            print(f"  {len(pt)+j:>4} {t:>6}  {d}")
        continue
    if line.startswith("/cache"):
        P["cache"] = "off" not in line
        print(f"  KV cache = {'开' if P['cache'] else '关'}"); continue
    if line.startswith("/cmp"):
        q = line[4:].strip()
        if not q:
            print("  用法: /cmp What is the capital of France?"); continue
        if base is None:
            print(f"  首次使用，载入对照模型 {args.base} …", end="", flush=True)
            base, _, bck = load(args.base)
            print(f" 完成（step {bck['step']:,}, val {bck['best_val']:.4f}）")
        print(f"\n  ── BASE（预训练，没做过 SFT）──")
        _, sb = gen(base, wrap(q))
        print(f"\n     [{sb['n_new']} token, {'⏹ 自己停' if sb['stopped'] else '✂️ 写满被砍'}]")
        print(f"\n  ── SFT ──")
        _, ss = gen(model, wrap(q))
        print(f"\n     [{ss['n_new']} token, {'⏹ 自己停' if ss['stopped'] else '✂️ 写满被砍'}]")
        last = ss; continue
    if line.startswith("/set"):
        try:
            _, k, v = line.split()
            key = {"temp": "temperature", "top_p": "top_p", "top_k": "top_k",
                   "rep": "repetition_penalty", "n": "n"}[k]
            P[key] = int(v) if key in ("n", "top_k") else float(v)
            print(f"  {key} = {P[key]}")
        except Exception:
            print("  用法: /set temp 0.7 | /set top_p 0.9 | /set rep 1.2 | /set n 300")
        continue

    print()
    txt, last = gen(model, wrap(line))
    if P["multi"]:
        HIST.append((line, txt))
    tag = "⏹ 自己停下" if last["stopped"] else f"✂️ 写满 {P['n']} 被砍断"
    print(f"\n\n  [{last['n_new']} token  {tag}  {last['dec_tps']:.1f} tok/s"
          f"  T={P['temperature']}  格式={'Alpaca' if P['alpaca'] else '关'}"
          f"{f'  历史 {len(HIST)} 轮' if P['multi'] else ''}]   /show 看 token")
