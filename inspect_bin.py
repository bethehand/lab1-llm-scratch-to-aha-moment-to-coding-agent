#!/usr/bin/env python3
"""
看看 train.bin 里到底是什么。

用法:
    python3 inspect_bin.py                          # 看 train.bin
    python3 inspect_bin.py data/fineweb_edu/val.bin # 看 val.bin
    python3 inspect_bin.py --pos 123456789          # 看指定位置
"""
import os, sys, argparse
import numpy as np
import tiktoken

ap = argparse.ArgumentParser()
ap.add_argument("path", nargs="?", default="data/fineweb_edu/train.bin")
ap.add_argument("--pos", type=int, default=None, help="从哪个 token 位置开始看")
args = ap.parse_args()

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token
data = np.memmap(args.path, dtype=np.uint16, mode="r")

print("=" * 70)
print(f"  {args.path}")
print("=" * 70)
print(f"  文件大小   {os.path.getsize(args.path):>15,} 字节")
print(f"  token 数   {len(data):>15,}   （= 字节数 ÷ 2，因为每个 token 占 uint16）")
print(f"  dtype      uint16          取值范围 0~65535，够装下 vocab 50257")
print(f"  最大 id    {int(data[:10_000_000].max()):>15,}   （前1000万内）")

pos = args.pos if args.pos is not None else 0

# ── ① 裸字节 ──
print(f"\n{'─'*70}\n① 从 token {pos:,} 开始的原始字节（十六进制，小端序）\n{'─'*70}")
raw = data[pos:pos+16].tobytes()
for r in range(2):
    line = raw[r*16:(r+1)*16]
    print("   ", " ".join(f"{b:02x}" for b in line))

# ── ② 字节 → 整数 → 文本 ──
print(f"\n{'─'*70}\n② 每 2 字节 = 1 个 token\n{'─'*70}")
print(f"   {'字节偏移':>10} {'十六进制':>10} {'token id':>10}   解码")
for i in range(12):
    t = int(data[pos+i])
    b = data[pos+i].tobytes()
    tok = "<|endoftext|>" if t == EOT else repr(enc.decode([t]))
    print(f"   {(pos+i)*2:>10,} {b.hex():>10} {t:>10,}   {tok}")

# ── ③ 连起来读 ──
print(f"\n{'─'*70}\n③ 连续 120 个 token 解码回文本\n{'─'*70}")
txt = enc.decode([int(t) for t in data[pos:pos+120]])
print("   " + txt.replace("\n", " ⏎ ")[:600])

# ── ④ 文档边界长什么样 ──
print(f"\n{'─'*70}\n④ 文档分隔符 <|endoftext|> 处的样子\n{'─'*70}")
window = np.asarray(data[pos:pos+2_000_000])
hits = np.flatnonzero(window == EOT)
if len(hits):
    e = pos + int(hits[len(hits)//2])
    print(f"   在 token {e:,} 处找到一个 EOT")
    print(f"\n   前一篇的结尾:")
    print("   ..." + enc.decode([int(t) for t in data[e-40:e]]).replace("\n", " ⏎ "))
    print(f"\n   ── EOT (id {EOT}) ──")
    print(f"\n   后一篇的开头:")
    print("   " + enc.decode([int(t) for t in data[e+1:e+41]]).replace("\n", " ⏎ ") + "...")
    print(f"\n   附近 EOT 密度: 每 {2_000_000//max(len(hits),1):,} 个 token 一篇文档")

# ── ⑤ 训练时模型实际看到什么 ──
print(f"\n{'─'*70}\n⑤ 训练时的一个样本：x 和 y 错开一位\n{'─'*70}")
T = 8
x = [int(t) for t in data[pos:pos+T]]
y = [int(t) for t in data[pos+1:pos+1+T]]
print(f"   {'':6} " + " ".join(f"{i:>7}" for i in range(T)))
print(f"   x (输入) " + " ".join(f"{v:>7,}" for v in x))
print(f"   y (标签) " + " ".join(f"{v:>7,}" for v in y))
print()
for i in range(T):
    xi = "<EOT>" if x[i] == EOT else enc.decode([x[i]])
    yi = "<EOT>" if y[i] == EOT else enc.decode([y[i]])
    print(f"   第{i}行:  看到 {xi!r:<14} → 要预测 {yi!r}")
print(f"\n   ★ 一个长度 {T} 的窗口 = {T} 个训练样本（L2 讲的「T 行 = T 个样本」）")
print(f"   ★ 实际训练用 T={1024}，所以一个窗口就是 1024 个样本")
print()
