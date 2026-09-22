#!/usr/bin/env python3
"""
第 8 步：交互式推理。

    python3 chat.py                              # 默认 out_run1/ckpt_final.pt
    python3 chat.py --ckpt out_run1/ckpt.pt

输入一段英文，模型续写。顺便把 L10 的核心现象量给你看：
  prefill（读 prompt）  compute-bound，快
  decode（逐 token 生成）memory-bound，慢十几倍
"""
import os, time, argparse
import torch
import torch.nn.functional as F
import tiktoken

from config import Config
from model import GPT

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="out_run2/ckpt_final.pt")
ap.add_argument("--tokens", type=int, default=120)
ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"],
                help="bf16 权重只有一半大，decode 的带宽上限直接翻倍")
ap.add_argument("--compile", action="store_true", help="torch.compile（首次编译约 1 分钟）")
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
enc = tiktoken.get_encoding("gpt2")

ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
cfg = Config(**{k: v for k, v in ck["cfg"].items() if k in Config.__dataclass_fields__})
model = GPT(cfg); model.load_state_dict(ck["model"])
torch_dtype = torch.bfloat16 if (args.dtype == "bf16" and dev == "cuda") else torch.float32
model.to(device=dev, dtype=torch_dtype).eval()
if args.compile and dev == "cuda":
    print("  torch.compile 编译中（约 1 分钟）…", flush=True)
    model = torch.compile(model)

# 生成参数，可在会话中用 /set 改
P = dict(temperature=0.8, top_p=0.92, top_k=None,
         repetition_penalty=1.15, n=args.tokens, cache=True)

kv_bytes_per_token = 2 * cfg.n_layer * cfg.n_head * cfg.head_dim * 2   # k+v, bf16
BYTES = 2 if torch_dtype == torch.bfloat16 else 4
HBM_GBPS = 1008.0 if dev == "cuda" else 50.0   # 4090 显存带宽 / CPU 内存带宽（粗估）

print("=" * 66)
print(f"  {args.ckpt}   step {ck['step']:,}")
W_MB = model.num_params() * BYTES / 1024**2 if not args.compile else \
       sum(p.numel() for p in model.parameters()) * BYTES / 1024**2
DT = "bf16" if torch_dtype == torch.bfloat16 else "fp32"
print(f"  {model.num_params()/1e6:.1f}M 参数   {dev}   {DT}   上下文上限 {cfg.block_size}")
print(f"  权重 {W_MB:.0f} MB   decode 理论上限 = {HBM_GBPS*1024/W_MB:.0f} tok/s "
      f"（{HBM_GBPS:.0f} GB/s ÷ 权重大小）")
print(f"  KV cache 每 token 占 {kv_bytes_per_token} 字节 "
      f"→ 满上下文 {kv_bytes_per_token*cfg.block_size/1024**2:.1f} MB")
print("=" * 66)
# ── 预热：第一次前向包含 cuDNN 算法选择 + kernel JIT，会慢十几倍。
#    第一轮实测：冷启动 286 ms，预热后个位数毫秒。不预热的话首次 /stats 全是假数字。
if dev == "cuda":
    print("  预热中…", end="", flush=True)
    with torch.no_grad():
        warm = torch.zeros(1, 8, dtype=torch.long, device=dev)
        c = [(None, None) for _ in range(cfg.n_layer)]
        model(warm, kv_cache=c, pos=0)                       # prefill 路径
        model(warm[:, :1], kv_cache=c, pos=8)                # decode 路径
        model(warm)                                          # 无 cache 路径
    torch.cuda.synchronize()
    print(" 完成\n")

print("  直接输入英文开头，回车让模型续写。")
print("  /help 看命令   /quit 退出\n")

HELP = """
  /set temp 0.9          温度（0.1 保守 ~ 1.5 发散）
  /set top_p 0.95        nucleus 截断
  /set rep 1.2           重复惩罚（1.0 = 关）
  /set n 200             生成多少 token
  /cache on|off          开关 KV cache，用来对比速度
  /stats                 上一次生成的速度拆解
  /help  /quit
"""

last = None


@torch.no_grad()
def gen(prompt: str):
    """自己写生成循环，为了把 prefill 和 decode 分开计时。"""
    ids = torch.tensor([enc.encode_ordinary(prompt)], dtype=torch.long, device=dev)
    n_prompt = ids.size(1)
    if n_prompt >= cfg.block_size:
        ids = ids[:, -cfg.block_size + 1:]; n_prompt = ids.size(1)

    def sync():
        if dev == "cuda":
            torch.cuda.synchronize()

    # 增量维护「哪些 token 出现过」的布尔掩码。
    # 原来用 torch.unique：输出大小依赖数据 → 每步强制 GPU↔CPU 同步 → 慢一大截
    seen_mask = torch.zeros(cfg.vocab_size, dtype=torch.bool, device=dev)
    seen_mask[ids[0]] = True

    def sample(logits, seen):
        logits = logits[:, -1, :].float()
        rp = P["repetition_penalty"]
        if rp != 1.0:
            l = logits[0]
            pen = torch.where(l > 0, l / rp, l * rp)
            logits[0] = torch.where(seen_mask, pen, l)      # 纯张量运算，零同步
        logits = logits / max(P["temperature"], 1e-5)
        if P["top_k"]:
            v, _ = torch.topk(logits, min(P["top_k"], logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        if P["top_p"] and P["top_p"] < 1.0:
            sl, si = torch.sort(logits, descending=True, dim=-1)
            cum = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
            drop = cum > P["top_p"]; drop[..., 1:] = drop[..., :-1].clone(); drop[..., 0] = False
            sl = sl.masked_fill(drop, -float("inf"))
            logits = torch.full_like(logits, -float("inf")).scatter_(-1, si, sl)
        return torch.multinomial(F.softmax(logits, dim=-1), 1)

    out = ids
    if P["cache"]:
        cache = [(None, None) for _ in range(cfg.n_layer)]
        sync(); t0 = time.time()
        logits, _ = model(ids, kv_cache=cache, pos=0)          # ── prefill ──
        sync(); t_pre = time.time() - t0

        pos = n_prompt
        sync(); t0 = time.time()
        for _ in range(P["n"]):                                # ── decode ──
            if pos >= cfg.block_size:
                break
            nxt = sample(logits, out)
            seen_mask[nxt[0]] = True
            out = torch.cat((out, nxt), 1)
            logits, _ = model(nxt, kv_cache=cache, pos=pos)
            pos += 1
        sync(); t_dec = time.time() - t0
    else:
        sync(); t0 = time.time()
        logits, _ = model(ids)
        sync(); t_pre = time.time() - t0
        sync(); t0 = time.time()
        for _ in range(P["n"]):
            nxt = sample(logits, out)
            seen_mask[nxt[0]] = True
            out = torch.cat((out, nxt), 1)
            if out.size(1) >= cfg.block_size:
                break
            logits, _ = model(out[:, -cfg.block_size:])        # 每步重算全部
        sync(); t_dec = time.time() - t0

    n_new = out.size(1) - n_prompt
    return enc.decode(out[0].tolist()), dict(
        n_prompt=n_prompt, n_new=n_new, t_pre=t_pre, t_dec=t_dec,
        pre_tps=n_prompt / max(t_pre, 1e-6), dec_tps=n_new / max(t_dec, 1e-6),
        kv_mb=kv_bytes_per_token * (n_prompt + n_new) / 1024**2)


def show_stats(s):
    print(f"\n  ── 速度拆解 ──")
    print(f"    prefill  读入 {s['n_prompt']:>4} token   {s['t_pre']*1000:>7.0f} ms"
          f"   {s['pre_tps']:>8.0f} tok/s   ← 并行，compute-bound")
    print(f"    decode   生成 {s['n_new']:>4} token   {s['t_dec']*1000:>7.0f} ms"
          f"   {s['dec_tps']:>8.1f} tok/s   ← 串行，memory-bound")
    r = s['pre_tps'] / max(s['dec_tps'], 1e-6)
    print(f"\n    prefill 比 decode 快 {r:.0f} 倍")
    lim = HBM_GBPS * 1024 / W_MB
    print(f"    decode 理论上限 {lim:.0f} tok/s（受显存带宽限制），"
          f"实测利用率 {100*s['dec_tps']/lim:.0f}%")
    print(f"    KV cache 占用 {s['kv_mb']:.1f} MB   （cache={'开' if P['cache'] else '关'}）")
    print(f"    ★ 这就是 L10 的核心：推理的瓶颈在访存，不在算力")


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
        show_stats(last) if last else print("  还没生成过")
        continue
    if line.startswith("/cache"):
        P["cache"] = "off" not in line
        print(f"  KV cache = {'开' if P['cache'] else '关'}"); continue
    if line.startswith("/set"):
        try:
            _, k, v = line.split()
            key = {"temp": "temperature", "top_p": "top_p", "top_k": "top_k",
                   "rep": "repetition_penalty", "n": "n"}[k]
            P[key] = int(v) if key in ("n", "top_k") else float(v)
            print(f"  {key} = {P[key]}")
        except Exception:
            print("  用法: /set temp 0.9 | /set top_p 0.95 | /set rep 1.2 | /set n 200")
        continue

    txt, last = gen(line)
    print(f"\n{txt}")
    print(f"\n  [{last['n_new']} token, {last['dec_tps']:.1f} tok/s"
          f"  T={P['temperature']} top_p={P['top_p']} rep={P['repetition_penalty']}"
          f"  cache={'开' if P['cache'] else '关'}]   /stats 看详细")
