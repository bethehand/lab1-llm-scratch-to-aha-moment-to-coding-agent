#!/usr/bin/env python3
"""
监督微调（SFT）—— 把 base model 从「续写网页」变成「回答问题」。

    python3 sft.py                                   # 默认 out_run2/ckpt_final.pt
    python3 sft.py --ckpt out_run1/ckpt_final.pt     # 也能微调第一轮的模型
    python3 sft.py --epochs 3 --lr 3e-5

约 10 分钟（单卡）。数据用 Alpaca 52K 条指令。

★ 这一步验证的是我们讨论过的「第③层」：
    ① 语言能力（语法/文体）  十亿级 token 才能建立
    ② 知识（事实）           万亿级 token + 大参数才能记住
    ③ 行为（怎么回答）       几万条就能「激活」   ← 就是这一步
  所以：SFT 能让它变成助手，但不能让它变对。
"""
import os, math, time, json, argparse, random
import numpy as np
import torch
import torch.nn.functional as F
import tiktoken

from config import Config
from model import GPT

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt",    default="out_run2/ckpt_final.pt")
ap.add_argument("--out",     default="out_sft")
ap.add_argument("--epochs",  type=int,   default=3)
ap.add_argument("--lr",      type=float, default=3e-5,
                help="远低于预训练的 5e-4 —— SFT 只是微调行为，不是重新学语言")
ap.add_argument("--batch",   type=int,   default=8)
ap.add_argument("--max-len", type=int,   default=512)
ap.add_argument("--limit",   type=int,   default=0, help="只用前 N 条（调试用）")
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev != "cuda":
    raise SystemExit("需要 GPU。CUDA 不可用 —— 检查 venv 里的 torch 是否匹配驱动。")
enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token
torch.manual_seed(1337)

# ══════════════════════════════════════════════════════════
#  数据：Alpaca 52K
# ══════════════════════════════════════════════════════════
os.makedirs("data/sft", exist_ok=True)
raw_path = "data/sft/alpaca.json"
if not os.path.exists(raw_path):
    print("下载 Alpaca 数据集（约 24 MB）…", flush=True)
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    json.dump([dict(r) for r in ds], open(raw_path, "w"))
    print(f"  已保存 {len(ds):,} 条 → {raw_path}", flush=True)

data = json.load(open(raw_path))
if args.limit:
    data = data[: args.limit]
print(f"载入 {len(data):,} 条指令数据")


def build(ex):
    """
    格式化成「指令 → 回答」，并构造 loss mask。

    ★ 关键：只对「回答」算 loss，不对「指令」算。
      跟 agent RL 里「只对模型生成的 token 算 loss」是同一个道理 ——
      你要教它「怎么回答」，不是「怎么出题」。
    """
    instr = ex["instruction"].strip()
    inp   = (ex.get("input") or "").strip()
    out   = ex["output"].strip()

    prompt = f"### Instruction:\n{instr}\n"
    if inp:
        prompt += f"\n### Input:\n{inp}\n"
    prompt += "\n### Response:\n"

    p_ids = enc.encode_ordinary(prompt)
    r_ids = enc.encode_ordinary(out) + [EOT]      # 回答末尾加 EOT，教它什么时候停

    ids  = p_ids + r_ids
    mask = [0] * len(p_ids) + [1] * len(r_ids)    # ← 只有回答部分参与 loss
    return ids[: args.max_len], mask[: args.max_len]


samples = []
for ex in data:
    ids, mask = build(ex)
    if sum(mask) >= 4:                            # 回答太短的丢掉
        samples.append((ids, mask))
random.Random(0).shuffle(samples)

n_val = min(500, len(samples) // 20)
val_s, train_s = samples[:n_val], samples[n_val:]

tot_tok  = sum(len(s[0]) for s in train_s)
loss_tok = sum(sum(s[1]) for s in train_s)
print(f"训练 {len(train_s):,} 条   验证 {n_val:,} 条")
print(f"总 token {tot_tok:,}   其中参与 loss 的 {loss_tok:,} "
      f"（{100*loss_tok/tot_tok:.0f}%）  ← 指令部分只当上下文")


def make_batch(pool, bs):
    """取 bs 条，右侧补齐到本批最长，padding 位置 mask=0。"""
    picks = [pool[i] for i in np.random.randint(0, len(pool), bs)]
    L = max(len(p[0]) for p in picks)
    x = torch.full((bs, L), EOT, dtype=torch.long)
    m = torch.zeros((bs, L), dtype=torch.float)
    for i, (ids, mask) in enumerate(picks):
        x[i, :len(ids)] = torch.tensor(ids)
        m[i, :len(mask)] = torch.tensor(mask, dtype=torch.float)
    return x.to(dev), m.to(dev)


# ══════════════════════════════════════════════════════════
#  模型
# ══════════════════════════════════════════════════════════
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
cfg = Config(**{k: v for k, v in ck["cfg"].items() if k in Config.__dataclass_fields__})
model = GPT(cfg)
model.load_state_dict(ck["model"])
model.to(dev)
opt, *_ = model.configure_optimizers(0.0, args.lr, (0.9, 0.95), "cuda")   # SFT 通常不加 wd

steps_per_epoch = len(train_s) // args.batch
max_steps = steps_per_epoch * args.epochs
warmup = max(20, max_steps // 50)

print(f"\n{'='*62}")
print(f"  基座 {args.ckpt}   (step {ck['step']:,}, val {ck['best_val']:.4f})")
print(f"  {model.num_params()/1e6:.1f}M 参数   lr {args.lr:.1e}（预训练是 5e-4，低 17 倍）")
print(f"  {args.epochs} epoch × {steps_per_epoch:,} 步 = {max_steps:,} 步，batch {args.batch}")
print(f"{'='*62}\n")


def get_lr(it):
    if it < warmup:
        return args.lr * (it + 1) / (warmup + 1)
    r = (it - warmup) / max(1, max_steps - warmup)
    return args.lr * 0.5 * (1 + math.cos(math.pi * r))


def masked_loss(x, m):
    """
    只在 mask=1 的位置算交叉熵。

    ★ 不能只传 idx —— model.forward 在 targets=None 时走推理分支，
      只返回最后一个位置的 logits。必须传 targets 才走训练分支。
      把不算 loss 的位置的 target 设成 -1，交给 ignore_index=-1 处理。
    """
    tgt = x[:, 1:].clone()
    tgt[m[:, 1:] == 0] = -1            # ← mask=0 的位置标成 -1
    _, loss = model(x[:, :-1], tgt)    # model 里的 cross_entropy(ignore_index=-1)
    return loss                        #   只对非 -1 的位置求平均，跟加权求和等价


@torch.no_grad()
def evaluate(n=20):
    model.eval()
    tot = 0.0
    for _ in range(n):
        x, m = make_batch(val_s, args.batch)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            tot += masked_loss(x, m).item()
    model.train()
    return tot / n


os.makedirs(args.out, exist_ok=True)
model.train()
t0 = time.time()
best = 1e9
print(f"  微调前 val loss  {evaluate():.4f}\n")

for step in range(max_steps):
    for g in opt.param_groups:
        g["lr"] = get_lr(step)
    x, m = make_batch(train_s, args.batch)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = masked_loss(x, m)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)

    if step % 200 == 0 or step == max_steps - 1:
        el = time.time() - t0
        print(f"  step {step:>5,}/{max_steps:,} | loss {loss.item():.4f} | "
              f"lr {get_lr(step):.2e} | {el/60:.1f} 分 | "
              f"剩余 {(max_steps-step-1)*el/max(step+1,1)/60:.1f} 分", flush=True)

    if step > 0 and step % 500 == 0:
        v = evaluate()
        print(f"    ── val {v:.4f} {'✅ 存盘' if v < best else ''}")
        if v < best:
            best = v
            tmp = os.path.join(args.out, "ckpt.pt.tmp")
            torch.save({"model": model.state_dict(), "cfg": vars(cfg),
                        "step": step, "best_val": v, "sft": True}, tmp)
            os.replace(tmp, os.path.join(args.out, "ckpt.pt"))

v = evaluate()
tmp = os.path.join(args.out, "ckpt_final.pt.tmp")
torch.save({"model": model.state_dict(), "cfg": vars(cfg),
            "step": max_steps, "best_val": v, "sft": True}, tmp)
os.replace(tmp, os.path.join(args.out, "ckpt_final.pt"))

print(f"\n{'='*62}")
print(f"  SFT 完成   用时 {(time.time()-t0)/60:.1f} 分钟")
print(f"  val loss {v:.4f}   权重 → {args.out}/ckpt_final.pt")
print(f"{'='*62}")
print(f"\n  下一步：python3 compare_sft.py   —— 同一批问题跑 base 和 SFT 两个模型")
