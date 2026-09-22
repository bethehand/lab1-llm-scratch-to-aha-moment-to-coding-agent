#!/usr/bin/env python3
"""
对比多次 GRPO 实验 —— 读各自的 hist.json，把关键曲线并排画出来。

    python3 compare_runs.py                              # 自动找 out_grpo*
    python3 compare_runs.py out_grpo out_grpo_lr5e6      # 指定
    python3 compare_runs.py --png cmp.png                # 另存图片
    python3 compare_runs.py --smooth 20                  # 平滑窗口
"""
import os, json, glob, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="*", help="实验目录；不给就自动找 out_grpo*")
ap.add_argument("--smooth", type=int, default=10, help="训练曲线的滑动平均窗口")
ap.add_argument("--width",  type=int, default=64, help="ASCII 图宽度")
ap.add_argument("--png",    default=None, help="另存 matplotlib 图片")
args = ap.parse_args()

runs = args.runs or sorted(d for d in glob.glob("out_grpo*") if os.path.isdir(d))
DATA = {}
for d in runs:
    f = os.path.join(d, "hist.json")
    if not os.path.exists(f):
        print(f"  ⚠️ {d} 没有 hist.json，跳过"); continue
    DATA[d] = json.load(open(f))
if not DATA:
    raise SystemExit("没找到任何实验记录")

BARS = "▁▂▃▄▅▆▇█"


def smooth(v, w):
    if w <= 1 or len(v) < w:
        return v
    return list(np.convolve(v, np.ones(w) / w, mode="valid"))


def spark(vals, lo, hi, width):
    if not vals:
        return ""
    if len(vals) > width:
        idx = np.linspace(0, len(vals) - 1, width).astype(int)
        vals = [vals[i] for i in idx]
    if hi - lo < 1e-12:
        return BARS[0] * len(vals)
    return "".join(BARS[min(7, max(0, int((v - lo) / (hi - lo) * 8)))] for v in vals)


def series(h, key, only_eval=False):
    if only_eval:
        return [(x["step"], x[key]) for x in h if key in x]
    return [(x["step"], x[key]) for x in h if key in x and x.get("step", -1) >= 0]


NAME = max(len(d) for d in DATA) + 2

# ══════════════════════════════════════════════════════════════════
#  ① 汇总表
# ══════════════════════════════════════════════════════════════════
print("=" * 78)
print("  GRPO 实验对比")
print("=" * 78)
print(f"\n  {'实验':<{NAME}}{'步数':>6}{'lr':>10}{'起点':>9}{'最终':>9}{'变化':>10}{'最好':>9}")
print("  " + "─" * (NAME + 53))
SUM = {}
for d, h in DATA.items():
    ev = series(h, "test_acc", only_eval=True)
    lrs = [x["lr"] for x in h if x.get("lr")]
    lr = max(lrs) if lrs else None          # ★ 取最大值 —— step 0 的 lr 被 warmup 打了 1/10
    steps = max([x.get("step", -1) for x in h]) + 1
    if ev:
        st, fi = ev[0][1], ev[-1][1]
        best = max(v for _, v in ev)
        print(f"  {d:<{NAME}}{steps:>6}{lr if lr else float('nan'):>10.1e}"
              f"{st:>9.4f}{fi:>9.4f}{fi-st:>+10.4f}{best:>9.4f}")
        SUM[d] = dict(start=st, final=fi, best=best, ev=ev)
    else:
        print(f"  {d:<{NAME}}{steps:>6}{lr if lr else float('nan'):>10.1e}"
              f"{'—':>9}{'—':>9}{'—':>10}{'—':>9}")

# ══════════════════════════════════════════════════════════════════
#  ② 曲线
# ══════════════════════════════════════════════════════════════════
PANELS = [
    ("test_acc", "★ test 正确率（主指标，贪心解码，无采样噪声）", True,  "{:.4f}"),
    ("kl",       "KL —— 离原模型多远（★ 停滞 = 学习率太低）",      False, "{:.4f}"),
    ("acc",      "训练时正确率（温度 1.0，8 题，噪声大）",          False, "{:.3f}"),
    ("len",      "平均回答长度（★ RL 会让它自己变长）",             False, "{:.0f}"),
    ("logp",     "平均 logP（★ 快速趋近 0 = 熵坍缩预警）",          False, "{:.3f}"),
    ("useful",   "有信号的题目比例（全对/全错的都白跑）",           False, "{:.2f}"),
]
for key, title, is_ev, fmt in PANELS:
    got = {d: series(h, key, is_ev) for d, h in DATA.items()}
    got = {d: v for d, v in got.items() if v}
    if not got:
        continue
    allv = [v for s in got.values() for _, v in s]
    lo, hi = min(allv), max(allv)
    print(f"\n{'─'*78}\n  {title}\n  范围 {fmt.format(lo)} ~ {fmt.format(hi)}\n{'─'*78}")
    for d, s in got.items():
        vals = [v for _, v in s]
        if not is_ev:
            vals = smooth(vals, args.smooth)
        print(f"  {d:<{NAME}}{fmt.format(vals[0]):>9} ┤{spark(vals, lo, hi, args.width)}│"
              f" {fmt.format(vals[-1])}")

# ══════════════════════════════════════════════════════════════════
#  ③ ★ 自动诊断
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*78}\n  ★ 自动诊断\n{'='*78}")
for d, h in DATA.items():
    print(f"\n  【{d}】")
    kl = [x["kl"] for x in h if "kl" in x]
    ln = [x["len"] for x in h if "len" in x]
    us = [x["useful"] for x in h if "useful" in x]
    ev = SUM.get(d, {}).get("ev", [])
    msgs = []

    if kl:
        k_end = np.mean(kl[-20:])
        if k_end < 0.005:
            msgs.append(f"❌ KL 只有 {k_end:.4f} —— ★ 模型几乎没动，学习率太低"
                        f"（有意义的变化应在 0.01~0.1）")
        elif k_end > 0.5:
            msgs.append(f"❌ KL 高达 {k_end:.3f} —— ★ 可能跑飞了，去看实际输出")
        else:
            msgs.append(f"✅ KL {k_end:.4f} 在合理区间")

    if ev and len(ev) >= 3:
        vals = [v for _, v in ev]
        st, fi = vals[0], vals[-1]
        best = max(vals); bi = vals.index(best); bstep = ev[bi][0]
        gain, drop = best - st, best - fi
        if gain >= 0.02:
            msgs.append(f"✅ ★ 峰值 {best:.4f} 在 step {bstep}（比起点 +{gain:.4f}）")
            if drop >= 0.02:
                msgs.append(f"⚠️ 之后退化到 {fi:.4f}（−{drop:.4f}）"
                            f" —— ★ 峰值的 checkpoint 才是你要的，别用最后那个")
            else:
                msgs.append(f"✅ 末值 {fi:.4f}，没有明显退化")
        elif abs(fi - st) < 0.02:
            msgs.append(f"❌ 全程在 {min(vals):.3f}~{max(vals):.3f} 震荡，"
                        f"峰值只比起点高 {gain:+.4f} —— ★ 等于没学")
        else:
            msgs.append(f"⚠️ test 正确率 {st:.4f} → {fi:.4f}（{fi-st:+.4f}）在下降")

    if ln and len(ln) > 40:
        a, b = np.mean(ln[:20]), np.mean(ln[-20:])
        if b > a * 1.5:
            msgs.append(f"⚠️ 长度 {a:.0f} → {b:.0f}（+{100*(b/a-1):.0f}%）"
                        f" —— ★ 可能在靠写长蒙分，考虑加长度惩罚")
        elif b > a * 1.15:
            msgs.append(f"✅ 长度 {a:.0f} → {b:.0f} 在变长 —— ★ 这是 RL 起效的典型信号")
        else:
            msgs.append(f"   长度 {a:.0f} → {b:.0f}，基本没变")

    lp = [x["logp"] for x in h if "logp" in x]
    if lp and len(lp) > 40:
        a, b = np.mean(lp[:20]), np.mean(lp[-20:])
        if b - a > 0.30:
            msgs.append(f"⚠️ ★ logP {a:.3f} → {b:.3f}（e^{b:.2f}={np.exp(b):.0%} 把握）"
                        f" —— ★ 熵坍缩：模型过度自信，采样多样性下降")
        elif b - a > 0.10:
            msgs.append(f"   logP {a:.3f} → {b:.3f}，在变自信（正常范围）")

    if us and len(us) > 40:
        a, b = np.mean(us[:20]), np.mean(us[-20:])
        if b < a * 0.75:
            msgs.append(f"⚠️ ★ 有信号率 {a:.2f} → {b:.2f}（−{100*(1-b/a):.0f}%）"
                        f" —— 8 条采样越来越像，梯度来源在枯竭")

    tr = [x["acc"] for x in h if "acc" in x]
    if tr and ev and len(tr) > 40:
        ta, tb = np.mean(tr[:20]), np.mean(tr[-20:])
        te = ev[-1][1] - ev[0][1]
        if tb - ta > 0.10 and te < (tb - ta) * 0.5:
            msgs.append(f"⚠️ ★ 训练正确率 {ta:.3f}→{tb:.3f}（+{tb-ta:.3f}）"
                        f"但 test 只 {te:+.3f} —— ★ 泛化崩塌")

    if us:
        u = np.mean(us)
        tag = "✅" if u >= 0.6 else "⚠️"
        msgs.append(f"{tag} 平均 {100*u:.0f}% 的题有梯度信号"
                    f"{'' if u>=0.6 else ' —— 低于 60%，考虑筛掉太简单/太难的题'}")

    for m in msgs:
        print(f"    {m}")

# ══════════════════════════════════════════════════════════════════
#  ④ 可选：matplotlib
# ══════════════════════════════════════════════════════════════════
if args.png:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(17, 8))
        for ax, (key, title, is_ev, _f) in zip(axes.ravel(), PANELS):
            for d, h in DATA.items():
                s = series(h, key, is_ev)
                if not s:
                    continue
                xs = [x for x, _ in s]; ys = [v for _, v in s]
                if not is_ev:
                    ys = smooth(ys, args.smooth); xs = xs[len(xs) - len(ys):]
                ax.plot(xs, ys, label=d, marker="o" if is_ev else None, ms=3)
            ax.set_title(title.replace("★ ", "")); ax.grid(alpha=.3); ax.legend(fontsize=7)
            ax.set_xlabel("step")
        plt.tight_layout(); plt.savefig(args.png, dpi=110)
        print(f"\n  → 图片已存 {args.png}")
    except ImportError:
        print("\n  （没装 matplotlib，跳过绘图：pip install matplotlib --no-deps）")
print()
