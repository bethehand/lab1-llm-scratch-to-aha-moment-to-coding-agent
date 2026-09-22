#!/usr/bin/env python3
"""
第 7 步：评估。

    python3 evaluate.py                          # 用 out_run1/ckpt.pt
    python3 evaluate.py --ckpt out_run1/ckpt_final.pt
    python3 evaluate.py --gpt2                   # 额外跟 GPT-2 small 比困惑度（需 transformers）

做五件事：
  ① 完整 val 集上的困惑度（不是训练时那 20 个 batch 的抽样）
  ② 跟 L9 缩放律公式的预测对比
  ③ 一组固定 prompt 的生成，多个温度横向看
  ④ 从训练日志里提取所有样本，按 step 排列 —— 看它怎么一步步学会说话
  ⑤ [可选] 跟 GPT-2 small 在同一份 val 数据上比困惑度
"""
import os, re, math, time, argparse
import numpy as np
import torch
import tiktoken

from config import Config
from model import GPT

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="out_run2/ckpt_final.pt")
ap.add_argument("--val",  default=None, help="默认用 config.data_dir/val.bin")
ap.add_argument("--log",  default="run2.log")
ap.add_argument("--gpt2", nargs="?", const="gpt2", default=None,
                metavar="MODEL",
                help="跟 GPT-2 对比。裸 --gpt2 = gpt2(124M)；"
                     "也可写 gpt2-medium / gpt2-large / gpt2-xl，"
                     "或逗号分隔多个：--gpt2 gpt2,gpt2-medium")
ap.add_argument("--max-batches", type=int, default=0, help="0=跑完整个 val 集")
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
enc = tiktoken.get_encoding("gpt2")

# ════════════════ 载入模型 ════════════════
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
cfg = Config(**{k: v for k, v in ck["cfg"].items()
                if k in Config.__dataclass_fields__})
model = GPT(cfg)
model.load_state_dict(ck["model"])
model.to(dev).eval()

print("=" * 68)
print(f"  {args.ckpt}")
print("=" * 68)
print(f"  训练到 step {ck['step']:,}   训练时记录的 best_val {ck['best_val']:.4f}")
print(f"  {model.num_params():,} 参数  "
      f"(C={cfg.C} L={cfg.n_layer} H={cfg.n_head} T={cfg.block_size} V={cfg.vocab_size})")


# ════════════════ ① 完整 val 集困惑度 ════════════════
@torch.no_grad()
def full_val_loss(m, path, bs=16, label=""):
    data = np.memmap(path, dtype=np.uint16, mode="r")
    T = cfg.block_size
    n_win = (len(data) - 1) // T                 # 不重叠地铺满整个 val 集
    tot, cnt = 0.0, 0
    t0 = time.time()
    for start in range(0, n_win, bs):
        idx = range(start, min(start + bs, n_win))
        x = torch.from_numpy(np.stack([data[i*T:(i+1)*T].astype(np.int64) for i in idx])).to(dev)
        y = torch.from_numpy(np.stack([data[i*T+1:(i+1)*T+1].astype(np.int64) for i in idx])).to(dev)
        ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
               if dev == "cuda" else torch.no_grad())
        with ctx:
            _, loss = m(x, y)
        tot += loss.item() * len(idx); cnt += len(idx)
        if args.max_batches and cnt >= args.max_batches * bs:
            break
    print(f"    {label}扫过 {cnt:,} 个窗口 × {T} token = {cnt*T:,} token"
          f"   ({time.time()-t0:.0f} 秒)")
    return tot / cnt


val_path = args.val or os.path.join(cfg.data_dir, "val.bin")
print(f"\n{'─'*68}\n① 完整验证集困惑度\n{'─'*68}")
vl = full_val_loss(model, val_path)
print(f"\n    val loss    {vl:.4f}")
print(f"    困惑度       {math.exp(vl):.2f}")
print(f"    每 token 比特 {vl/math.log(2):.3f} bits")

# ════════════════ ② 跟 L9 的缩放律公式对比 ════════════════
N, D = model.num_params(), cfg.train_tokens
pred = 1.69 + 406.4 / N**0.34 + 410.7 / D**0.28
print(f"\n{'─'*68}\n② 对比 L9 的 Chinchilla 公式\n{'─'*68}")
print(f"    L = 1.69 + 406.4/N^0.34 + 410.7/D^0.28")
print(f"      = 1.69 + 406.4/{N:.3e}^0.34 + 410.7/{D:.3e}^0.28")
print(f"      = 1.69 + {406.4/N**0.34:.3f} + {410.7/D**0.28:.3f}")
print(f"      = {pred:.4f}")
print(f"\n    公式预测  {pred:.4f}")
print(f"    实测      {vl:.4f}   ({vl-pred:+.4f})")
if D < 1e8 or N < 1e7 or pred > 8:
    print(f"\n    ⚠️ 这个规模远超 Chinchilla 的拟合范围（他们的实验在 70M~16B 参数、")
    print(f"       5B~500B token 之间），公式外推到这里没有意义，上面的数字忽略。")
elif vl < pred:
    print(f"    ✅ 跑赢公式 {pred-vl:.3f}")
    print(f"       Chinchilla 的常数是在 MassiveText 上拟合的；")
    print(f"       FineWeb-Edu 经过教育性筛选，信息密度更高 → 同样 token 数学到更多。")
    print(f"       这正是 L11 说的：缩放律的『形式』通用，『常数』依赖数据。")
else:
    print(f"    落后公式 {vl-pred:.3f}（常数来自不同数据集，有偏差属正常）")

# ════════════════ ③ 批量生成 ════════════════
PROMPTS = [
    "The meaning of life is",
    "The capital of France is",
    "In 1969, humans first",
    "Water boils at a temperature of",
    "The main difference between a virus and a bacterium is",
    "Photosynthesis is the process by which",
    "Once upon a time, there was",
]
print(f"\n{'─'*68}\n③ 生成质量（T=0.8, top_p=0.92, 重复惩罚 1.15）\n{'─'*68}")
for p in PROMPTS:
    ids = torch.tensor([enc.encode_ordinary(p)], dtype=torch.long, device=dev)
    out = model.generate(ids, 60, temperature=0.8, top_p=0.92, repetition_penalty=1.15)
    txt = enc.decode(out[0].tolist())
    print(f"\n  ▸ {p}")
    print(f"    {txt[len(p):].strip()[:340].replace(chr(10), ' ⏎ ')}")

print(f"\n{'─'*68}\n③b 解码策略对比 —— 同一个模型，同一个 prompt\n{'─'*68}")
P = "The meaning of life is"
SETTINGS = [
    ("训练时用的（易重复循环）", dict(temperature=0.8, top_k=50, top_p=None,
                                     repetition_penalty=1.0)),
    ("加重复惩罚",              dict(temperature=0.8, top_k=50, top_p=None,
                                     repetition_penalty=1.15)),
    ("top-p + 重复惩罚 ★推荐",  dict(temperature=0.8, top_k=None, top_p=0.92,
                                     repetition_penalty=1.15)),
    ("低温（保守）",            dict(temperature=0.4, top_k=None, top_p=0.92,
                                     repetition_penalty=1.15)),
    ("高温（发散）",            dict(temperature=1.2, top_k=None, top_p=0.95,
                                     repetition_penalty=1.15)),
]
for name, kw in SETTINGS:
    ids = torch.tensor([enc.encode_ordinary(P)], dtype=torch.long, device=dev)
    txt = enc.decode(model.generate(ids, 70, **kw)[0].tolist())
    body = txt[len(P):].strip()[:300].replace(chr(10), " ⏎ ")
    # 粗略统计重复：出现 3 次以上的 4-gram
    w = body.split()
    grams = [" ".join(w[i:i+4]) for i in range(max(0, len(w)-3))]
    rep = sum(1 for g in set(grams) if grams.count(g) >= 3)
    flag = f"  ⚠️ {rep} 个 4-gram 重复≥3 次" if rep else "  ✅ 无明显重复"
    print(f"\n  ▸ {name}{flag}")
    print(f"    {body}")

# ════════════════ ④ 训练过程中的样本演化 ════════════════
if os.path.exists(args.log):
    print(f"\n{'─'*68}\n④ 学会说话的全过程（从 {args.log} 提取）\n{'─'*68}")
    txt = open(args.log, errors="ignore").read()
    blocks = re.split(r"── 生成 @ (\d+) ──", txt)
    if len(blocks) < 3:
        print(f"    （{args.log} 里没找到生成样本 —— 日志可能没抓全）")
    for i in range(1, len(blocks) - 1, 2):
        step = int(blocks[i])
        body = blocks[i+1].split("step ")[0].strip()
        print(f"\n  ── step {step:,} ──")
        print(f"    {body[:300].replace(chr(10), ' ⏎ ')}")

# ════════════════ ⑤ 跟 GPT-2 small 对比 ════════════════
if args.gpt2:
    names = [x.strip() for x in args.gpt2.split(",") if x.strip()]
    print(f"\n{'─'*68}\n⑤ 对比 GPT-2 系列（同一份 val 数据）\n{'─'*68}")
    print("    注意：GPT-2 在 WebText 上训练，这里是在 FineWeb-Edu 上评估 —— 它在客场。")
    try:
        from transformers import GPT2LMHeadModel

        @torch.no_grad()
        def hf_val_loss(m, path, bs=8):
            data = np.memmap(path, dtype=np.uint16, mode="r")
            T = min(cfg.block_size, 1024)
            n_win = (len(data) - 1) // T
            tot, cnt = 0.0, 0
            for st in range(0, n_win, bs):
                idx = range(st, min(st + bs, n_win))
                x = torch.from_numpy(np.stack([data[i*T:(i+1)*T].astype(np.int64)
                                               for i in idx])).to(dev)
                tot += m(x, labels=x).loss.item() * len(idx); cnt += len(idx)
            return tot / cnt

        rows = [("你的模型", model.num_params(), cfg.train_tokens/1e9, vl)]
        for name in names:
            print(f"\n    载入 {name} …（首次要下载）", flush=True)
            g = GPT2LMHeadModel.from_pretrained(name).to(dev).eval()
            npar = sum(p.numel() for p in g.parameters())
            t0 = time.time()
            gl = hf_val_loss(g, val_path)
            print(f"    完成，用时 {time.time()-t0:.0f} 秒", flush=True)
            rows.append((name, npar, 9.0, gl))
            del g
            if dev == "cuda":
                torch.cuda.empty_cache()

        print(f"\n    {'模型':<14} {'参数':>9} {'数据':>8} {'loss':>8} {'困惑度':>8} {'vs 你':>8}")
        print("    " + "-" * 60)
        for name, npar, dtok, l in rows:
            gap = "" if name == "你的模型" else f"{l - vl:+8.4f}"
            print(f"    {name:<14} {npar/1e6:>7.1f}M {dtok:>7.1f}B {l:>8.4f} "
                  f"{math.exp(l):>8.2f} {gap:>8}")

        best = min(rows[1:], key=lambda r: r[3]) if len(rows) > 1 else None
        if best:
            d = vl - best[3]
            print()
            if d < 0:
                print(f"    ✅ 你赢了最强的对手 {best[0]}（{best[1]/1e6:.0f}M）{-d:.4f}")
                print(f"       但记住：GPT-2 是客场（WebText → FineWeb-Edu 有分布偏移惩罚），")
                print(f"       第一轮实测那个惩罚约 +0.18。扣掉之后差距会缩小。")
            else:
                print(f"    落后 {best[0]} {d:.4f} —— 差距在 0.3 以内即训练成功。")
    except ImportError:
        print("    需要 transformers：pip install transformers")
    except Exception as e:
        print(f"    对比失败: {e}")

print()
