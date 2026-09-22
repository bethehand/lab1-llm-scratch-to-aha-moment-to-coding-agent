#!/usr/bin/env python3
"""
看看 SFT 数据长什么样 —— 训练前先看清楚。

    python3 inspect_sft.py            # 看 3 条样本 + 统计
    python3 inspect_sft.py -n 6       # 看 6 条
"""
import os, json, argparse
import numpy as np
import tiktoken

ap = argparse.ArgumentParser()
ap.add_argument("-n", type=int, default=3, help="看几条样本")
ap.add_argument("--max-len", type=int, default=512)
args = ap.parse_args()

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token

os.makedirs("data/sft", exist_ok=True)
raw = "data/sft/alpaca.json"
if not os.path.exists(raw):
    print("下载 Alpaca（约 24 MB）…", flush=True)
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    json.dump([dict(r) for r in ds], open(raw, "w"))
data = json.load(open(raw))

print("=" * 74)
print(f"  Alpaca 指令数据集   {len(data):,} 条")
print("=" * 74)


def fmt(ex):
    instr = ex["instruction"].strip()
    inp = (ex.get("input") or "").strip()
    out = ex["output"].strip()
    prompt = f"### Instruction:\n{instr}\n"
    if inp:
        prompt += f"\n### Input:\n{inp}\n"
    prompt += "\n### Response:\n"
    return prompt, out


# ── ① 原始 JSON ──
print(f"\n{'─'*74}\n① 原始数据（一条 JSON）\n{'─'*74}")
print(json.dumps(data[1], ensure_ascii=False, indent=2)[:700])

# ── ② 拼成模型看到的样子 + loss mask ──
print(f"\n{'─'*74}\n② 拼成模型看到的样子（★ = 参与 loss）\n{'─'*74}")
withinput = [d for d in data if (d.get("input") or "").strip()]
picks = [data[1], data[5], withinput[0]][: args.n]
for k, ex in enumerate(picks):
    prompt, out = fmt(ex)
    print(f"\n  ══ 样本 {k+1} ══")
    for line in prompt.split("\n"):
        print(f"     │ {line}")
    for line in (out + "<|endoftext|>").split("\n"):
        print(f"   ★ │ {line}")

# ── ③ token 级别 ──
print(f"\n{'─'*74}\n③ token 级别（第 1 条的前后各几个）\n{'─'*74}")
prompt, out = fmt(picks[0])
p_ids = enc.encode_ordinary(prompt)
r_ids = enc.encode_ordinary(out) + [EOT]
print(f"   指令 {len(p_ids)} 个 token（mask=0）  回答 {len(r_ids)} 个 token（mask=1）")
print(f"\n   {'位置':>5} {'token id':>9} {'mask':>5}   解码")
for i in list(range(4)) + [-1]:
    t = p_ids[i]
    print(f"   {i if i>=0 else len(p_ids)-1:>5} {t:>9} {0:>5}   {enc.decode([t])!r}")
print(f"   {'...':>5}")
for i in range(3):
    t = r_ids[i]
    print(f"   {len(p_ids)+i:>5} {t:>9} {1:>5}   {enc.decode([t])!r}")
print(f"   {'...':>5}")
print(f"   {len(p_ids)+len(r_ids)-1:>5} {EOT:>9} {1:>5}   '<|endoftext|>'  ← 教它「答完了」")

# ── ④ 统计 ──
print(f"\n{'─'*74}\n④ 统计\n{'─'*74}")
pl, rl, kept = [], [], 0
for ex in data:
    prompt, out = fmt(ex)
    a, b = len(enc.encode_ordinary(prompt)), len(enc.encode_ordinary(out)) + 1
    if a + b <= args.max_len:
        kept += 1
    pl.append(a); rl.append(b)
pl, rl = np.array(pl), np.array(rl)
tot = pl.sum() + rl.sum()
print(f"   有 input 字段的         {len(withinput):,} 条 ({100*len(withinput)/len(data):.0f}%)")
print(f"   指令长度   中位 {np.median(pl):.0f}  均值 {pl.mean():.0f}  P95 {np.percentile(pl,95):.0f} token")
print(f"   回答长度   中位 {np.median(rl):.0f}  均值 {rl.mean():.0f}  P95 {np.percentile(rl,95):.0f} token")
print(f"   总长 ≤ {args.max_len} 的            {kept:,} 条 ({100*kept/len(data):.0f}%)  ← 超长的会被截断")
print(f"\n   总 token          {tot:,}")
print(f"   其中参与 loss     {rl.sum():,}  ({100*rl.sum()/tot:.0f}%)  ← 只有回答部分")
print(f"   3 个 epoch 共训   {3*tot/1e6:.1f}M token"
      f"   = 预训练 10,000M 的 {100*3*tot/1e10:.2f}%")
print()
