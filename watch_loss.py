#!/usr/bin/env python3
"""
训练监控：从日志里解析 loss，在终端画 ASCII 曲线。

用法:
    python3 watch_loss.py                      # 读 run1.log，画一次
    watch -n 30 'python3 watch_loss.py'        # 每 30 秒刷新
    python3 watch_loss.py out_run1/train.log   # 指定日志
"""
import re, sys, os, time

LOG = sys.argv[1] if len(sys.argv) > 1 else "run1.log"
H, W = 18, 62      # 图的高和宽

if not os.path.exists(LOG):
    sys.exit(f"找不到 {LOG}")

pat  = re.compile(r"step\s+(\d+)/(\d+)\s+\|\s+loss\s+([\d.]+).*?\|\s+(\d+)ms\s+\|\s+MFU\s+([\d.]+)%")
evp  = re.compile(r"eval @ (\d+): train ([\d.]+)\s+val ([\d.]+)")

steps, losses, mfus, dts = [], [], [], []
evals = []
for line in open(LOG, errors="ignore"):
    m = pat.search(line)
    if m:
        steps.append(int(m.group(1))); total = int(m.group(2))
        losses.append(float(m.group(3)))
        dts.append(int(m.group(4))); mfus.append(float(m.group(5)))
    e = evp.search(line)
    if e:
        evals.append((int(e.group(1)), float(e.group(2)), float(e.group(3))))

if not steps:
    sys.exit(f"{LOG} 里还没有 step 行 —— 可能还在 torch.compile 编译中")

cur, tot = steps[-1], total
# 用最近 50 步的耗时估 ETA，比全程平均准（跳过第 0 步的编译时间）
recent = dts[-50:] if len(dts) > 50 else dts[1:] or dts
ms = sum(recent) / len(recent)
eta = (tot - cur) * ms / 1000
# 日志每 log_every 步打一行，所以每行的 dt 代表 log_every 步的耗时
log_every = (steps[1] - steps[0]) if len(steps) > 1 else 1
elapsed = sum(dts[1:]) * log_every / 1000 if len(dts) > 1 else 0

print(f"\n  {LOG}   step {cur:,}/{tot:,}  ({100*cur/tot:.1f}%)")
print(f"  loss {losses[-1]:.4f}   MFU {mfus[-1]:.1f}%   {ms:.0f}ms/步")
print(f"  已用 {elapsed/60:.0f} 分   预计还要 {eta/60:.0f} 分\n")

# ── ASCII 曲线（loss 用对数轴更能看清后期的缓慢下降）──
lo, hi = min(losses), max(losses)
pad = (hi - lo) * 0.05 or 0.1
lo, hi = lo - pad, hi + pad

grid = [[" "] * W for _ in range(H)]
for s, l in zip(steps, losses):
    x = int((W - 1) * s / max(tot, 1))
    y = int((H - 1) * (hi - l) / (hi - lo))
    grid[max(0, min(H-1, y))][max(0, min(W-1, x))] = "●"

for r in range(H):
    v = hi - (hi - lo) * r / (H - 1)
    axis = f"{v:6.2f} ┤" if r % 3 == 0 else "       │"
    print("  " + axis + "".join(grid[r]))
print("  " + " " * 7 + "└" + "─" * W)
print(f"  {'':8}0{' ' * (W - 12)}{tot:,} 步")

if evals:
    print(f"\n  ── eval 记录 ──")
    for st, tr, va in evals[-6:]:
        gap = va - tr
        flag = "⚠️ val 远低于 train，查污染" if gap < -0.3 else ""
        print(f"    step {st:>6,}   train {tr:.4f}   val {va:.4f}   (差 {gap:+.4f}) {flag}")

# ── 健康检查 ──
warn = []
if len(losses) > 20 and losses[-1] > max(losses[-20:-10]):
    warn.append("⚠️ loss 近期在上升")
if any(l != l for l in losses[-5:]):
    warn.append("❌ 出现 NaN，立刻停")
if len(mfus) > 5 and mfus[-1] < 40:
    warn.append(f"⚠️ MFU 掉到 {mfus[-1]:.0f}%（正常约 55%）")
print("\n  " + ("  ".join(warn) if warn else "✅ 一切正常"))
print()
