#!/usr/bin/env python3
"""
slim_ckpt.py —— 把 ckpt 里的优化器状态剥掉，只留权重和 step。评测只用权重；带优化器的 ckpt 约 15 GB，剥完约 3 GB。
★ 剥掉之后这份 ckpt 不能再 --resume（没有优化器状态），只能 --init 或评测。跑完的臂才剥，正在跑的别碰。

    python3 slim_ckpt.py out_arm0/ckpt_latest.pt out_arm1/ckpt_latest.pt …      # 原地替换（先写临时文件再改名，中途断电不丢）
    python3 slim_ckpt.py --dry out_*/ckpt_latest.pt                             # 只看每份多大、有没有优化器，不动
"""
import os, sys, torch

dry = "--dry" in sys.argv
paths = [p for p in sys.argv[1:] if p != "--dry"]
total_before = total_after = 0
for p in paths:
    if not os.path.exists(p):
        print(f"  {p}: 不存在"); continue
    sz = os.path.getsize(p) / 1024**3
    ck = torch.load(p, map_location="cpu", weights_only=False)
    has_opt = "opt" in ck
    print(f"  {p}: {sz:.1f} GB  step {ck.get('step','?')}  {'带优化器' if has_opt else '只有权重'}", end="")
    total_before += sz
    if not has_opt or dry:
        print(); total_after += sz; continue
    slim = {k: v for k, v in ck.items() if k != "opt"}
    tmp = p + ".tmp"
    torch.save(slim, tmp); os.replace(tmp, p)
    sz2 = os.path.getsize(p) / 1024**3; total_after += sz2
    print(f"  → {sz2:.1f} GB")
    del ck, slim
print(f"  合计 {total_before:.1f} GB → {total_after:.1f} GB" + ("（--dry，没动）" if dry else ""))
