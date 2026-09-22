#!/usr/bin/env python3
"""
把所有 baseline*.json 排成一张表 —— 温度 1.0 下的 pass@1 / pass@8 对比。

    python3 baseline_gsm8k.py -n 200                                        # ① 原始（已跑过）
    python3 baseline_gsm8k.py -n 200 --ckpt out_grpo_lr5e6/ckpt_latest.pt   # ③ 训练后
    python3 baseline_gsm8k.py -n 200 --ckpt out_grpo/ckpt_latest.pt         # ② 对照组
    python3 compare_passk.py                                                # 汇总
"""
import json, glob, argparse

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="*")
args = ap.parse_args()

files = args.files or [f for f in (sorted(glob.glob("pk_*.json")) or
                                   sorted(glob.glob("baseline*.json")))
                      if not f.endswith("_raw.json")]     # ★ 排除原始数据文件
if not files:
    raise SystemExit("没找到 baseline*.json —— 先跑 baseline_gsm8k.py")

D = []
for f in files:
    d = json.load(open(f))
    if "k" not in d and "n_samples" in d:      # 兼容不同版本的字段名
        d["k"] = d["n_samples"]
    if "k" not in d or "pass1" not in d:
        print(f"  ⚠️ {f} 缺统计字段（可能是原始数据文件），跳过")
        continue
    name = (f.replace("pk_", "").replace("baseline_", "").replace(".json", "")
             .replace("_ckpt_latest", " 末值").replace("_ckpt_best", " ★峰值"))
    if d.get("ckpt") in (None, ""):
        name = "★ 原始 Qwen（未训练）" + name[name.find("_k"):] if "_k" in name else "★ 原始 Qwen"
    if d.get("offset", 0) >= 200:
        name += " [holdout]"
    D.append((name, d))

print("=" * 96)
print("  pass@k 对比   温度 1.0   （★ 跟贪心评估不是一回事）")
print("=" * 96)
KS = sorted({d["k"] for _, d in D})
if len(KS) > 1:
    print(f"\n  ⚠️ 检测到多个 k 值 {KS} —— ★ 不同 k 的 pass@k 不可比，已分组\n")
W = max(len(n) for n, _ in D) + 2
for k in KS:
    G = [(n, d) for n, d in D if d["k"] == k]
    off = G[0][1].get("offset", 0)
    print(f"\n  ── k = {k}   TEST[{off}:{off+G[0][1]['n']}]   {G[0][1]['n']} 道题"
          f"   {'★ 无偏' if off >= 200 else '⚠️ 训练时用过'} ──")
    print(f"  {'模型':<{W}}{'pass@1':>9}{f'pass@{k}':>10}{'有信号':>9}{'格式率':>9}"
          f"{'平均长度':>10}{'截断':>8}")
    print("  " + "─" * (W + 55))
    for n, d in G:
        print(f"  {n:<{W}}{d['pass1']:>9.4f}{d['passk']:>10.4f}{d['useful_frac']:>9.4f}"
              f"{d.get('fmt_rate', float('nan')):>9.4f}{d['avg_tokens']:>10.0f}"
              f"{100*d.get('trunc_frac', 0):>7.1f}%")
    if len(G) >= 2:
        b0 = G[0][1]
        print(f"\n  {'相对第一行的变化':<{W}}{'Δpass@1':>9}{f'★ Δpass@{k}':>12}"
              f"{'Δ有信号':>10}{'Δ长度':>9}")
        print("  " + "─" * (W + 40))
        for n, d in G[1:]:
            print(f"  {n:<{W}}{d['pass1']-b0['pass1']:>+9.4f}"
                  f"{d['passk']-b0['passk']:>+12.4f}"
                  f"{d['useful_frac']-b0['useful_frac']:>+10.4f}"
                  f"{d['avg_tokens']-b0['avg_tokens']:>+9.0f}")

if len(D) >= 2:
    b = D[0][1]
    # ★★ 完整曲线（新版脚本才有 curve 字段）
    HAVE = [(n, d) for n, d in D if d.get("curve")]
    if len(HAVE) >= 2:
        by_off = {}
        for n, d in HAVE:
            by_off.setdefault((d.get("offset", 0), d["n_samples"], d["n"]), []).append((n, d))
        for (off, ns, nq), G in by_off.items():
            if len(G) < 2:
                continue
            KS = sorted(int(k) for k in G[0][1]["curve"])
            NW = max(len(n) for n, _ in G) + 2
            print(f"\n{'='*96}")
            print(f"  ★★ pass@k 完整曲线   TEST[{off}:{off+nq}]   {nq} 题   "
                  f"每题 n={ns} 次采样   {'★ 无偏' if off >= 200 else '⚠️ 训练时用过'}")
            print("=" * 96)
            print(f"\n  {'k':>4}", end="")
            for n, _ in G:
                print(f"{n[:16]:>18}", end="")
            for n, _ in G[1:]:
                print(f"{('Δ ' + n[:10]):>14}", end="")
            print()
            print("  " + "─" * (4 + 18 * len(G) + 14 * (len(G) - 1)))
            b0 = G[0][1]["curve"]
            for k in KS:
                print(f"  {k:>4}", end="")
                for _, d in G:
                    print(f"{d['curve'][str(k)]:>18.4f}", end="")
                for _, d in G[1:]:
                    print(f"{d['curve'][str(k)] - b0[str(k)]:>+14.4f}", end="")
                print()
            print(f"""
  ★★ 看最右边那几列（Δ）：应该随 k 增大【单调收缩到 0】

     k 小   测的是「概率质量集中在哪」  →  ★ RL 大幅改变了它
     k 大   测的是「分布的支撑集多大」  →  ★ RL 完全没动它

     如果 Δ 在大 k 处仍然明显 > 0  →  ⚠️ RL 真的扩展了能力，理论要修
""")

    print(f"""
  ★★ Δpass@k 是这次实验的核心指标（★ k 越大，测量越灵敏）：

     ≈ 0 (±0.02)   ✅ RL 只重排概率，不创造能力
                      → pass@1 涨、pass@k 不变，教科书式的结果
                      ★ k=8 时涨了不算数 —— p 一涨探测灵敏度就跟着涨
                        要看 k=64 才有区分力

     明显 +        ⚠️ 打破理论 —— RL 真的学到了新东西？值得深挖
                      （也要先排除：是不是判分器被作弊了）

     ★ 明显 −      ⚠️ 熵坍缩的代价
                      8 次采样越来越像 → 碰到正确解法的机会变少
                      → 可提升空间变小 → 解释了训练为什么会见顶回落
""")
