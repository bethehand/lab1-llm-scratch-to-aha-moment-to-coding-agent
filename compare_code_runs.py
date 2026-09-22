#!/usr/bin/env python3
"""并排读两次（或多次）修 bug RL 的 hist.json：评测曲线、训练曲线、语法错突变、计时。
    python3 compare_code_runs.py out_codeRL_w32 out_codeRL_dyn
    python3 compare_code_runs.py out_codeRL_w32 out_codeRL_dyn --smooth 5 --last 20
注意：开了 --dyn-sample 的那趟，训练行的「修好」是丢掉无差异组之后的残余组，天然偏低，两趟不可比；
      可比的只有每 25 步那次 80 道贪心评测（ev）。"""
import os, json, argparse, statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--smooth", type=int, default=5, help="训练曲线滑动平均窗口")
ap.add_argument("--last", type=int, default=12, help="尾部逐步表打几行")
ap.add_argument("--errs-hi", type=float, default=0.10, help="语法错高于它就算异常步")
ap.add_argument("--max-new", type=int, default=1024, help="训练时的 --max-new，用来判断异常步是不是被预算截断")
args = ap.parse_args()

BARS = "▁▂▃▄▅▆▇█"
D = {}
for d in args.runs:
    f = os.path.join(d, "hist.json")
    if not os.path.exists(f):
        print(f"  ⚠️ {d}/hist.json 不在，跳过"); continue
    h = json.load(open(f))
    D[d] = h if isinstance(h, list) else h.get("hist", h)
if not D: raise SystemExit("没读到任何 hist.json")
NAME = max(len(d) for d in D) + 2


def col(h, k):
    return [(x["step"], x[k]) for x in h if k in x and x[k] == x[k]]


def evcol(h, k):
    return [(x["step"] + 1, x["ev"][k]) for x in h if "ev" in x and k in x["ev"]]


def smooth(v, w):
    if w <= 1 or len(v) < w: return v
    return [sum(v[max(0, i - w + 1): i + 1]) / len(v[max(0, i - w + 1): i + 1]) for i in range(len(v))]


def spark(vals, lo, hi, width=60):
    if not vals: return ""
    if len(vals) > width:
        step = (len(vals) - 1) / (width - 1)
        vals = [vals[round(i * step)] for i in range(width)]
    if hi - lo < 1e-12: return BARS[0] * len(vals)
    return "".join(BARS[min(7, max(0, int((v - lo) / (hi - lo) * 8)))] for v in vals)


print("=" * 100)
print("  修 bug RL 多趟对比")
print("=" * 100)

# ── ① 评测（贪心 80 道，唯一两趟可比的口径）──
print(f"\n{'─'*100}\n  ① 仪表盘评测（贪心，--eval-n 道，两趟可比）\n{'─'*100}")
steps = sorted({s for h in D.values() for s, _ in evcol(h, "code_correct")})
print(f"  {'步':>5} " + "".join(f"{d[:22]:>24}" for d in D))
for s in steps:
    row = f"  {s:>5} "
    for d, h in D.items():
        m = {k: v for k, v in evcol(h, "code_correct")}
        g = {k: v for k, v in evcol(h, "score")}
        row += f"{('修好 %.3f 分 %.3f' % (m[s], g[s])) if s in m else '—':>24}"
    print(row)
for d, h in D.items():
    c = evcol(h, "code_correct")
    if c:
        bi, bv = max(c, key=lambda x: x[1])
        print(f"  {d:<{NAME}} 起 {c[0][1]:.3f} → 终 {c[-1][1]:.3f}（{c[-1][1]-c[0][1]:+.3f}）  峰 {bv:.3f} @ step {bi}"
              f"  |  调用 {dict(evcol(h,'code_calls')).get(c[-1][0], float('nan')):.2f}  长 {dict(evcol(h,'code_length')).get(c[-1][0], float('nan')):.0f}"
              f"  语法错 {dict(evcol(h,'code_errs')).get(c[-1][0], float('nan')):.2f}")

# ── ② 训练曲线 ──
PANELS = [("code_correct", "训练行 修好（温度 1；开了动态采样的趟数值天然偏低）", "{:.3f}"),
          ("nvar",         "有效组 / 每卡题数（动态采样要顶到满）",                "{:.2f}"),
          ("code_errs",    "★ 语法错 / 条",                                        "{:.3f}"),
          ("code_length",  "平均长度",                                             "{:.0f}"),
          ("code_calls",   "调用 / 条",                                            "{:.2f}"),
          ("code_adv",     "|A| 组内奖励差（缩小 = 组在两极化）",                  "{:.3f}"),
          ("kl",           "KL 离 SFT 多远",                                       "{:.4f}"),
          ("code_fake",    "假观测",                                               "{:.3f}")]
for k, title, fmt in PANELS:
    got = {d: col(h, k) for d, h in D.items()}
    got = {d: v for d, v in got.items() if v}
    if not got: continue
    allv = [v for s in got.values() for _, v in s]
    lo, hi = min(allv), max(allv)
    print(f"\n{'─'*100}\n  {title}   范围 {fmt.format(lo)} ~ {fmt.format(hi)}\n{'─'*100}")
    for d, s in got.items():
        v = smooth([x for _, x in s], args.smooth)
        print(f"  {d:<{NAME}}{fmt.format(v[0]):>9} ┤{spark(v, lo, hi)}│{fmt.format(v[-1]):>9}")

# ── ③ 语法错异常步 ──
print(f"\n{'─'*100}\n  ③ 语法错 > {args.errs_hi}：什么时候开始、占多少步\n{'─'*100}")
for d, h in D.items():
    e = col(h, "code_errs")
    bad = [(s, v) for s, v in e if v > args.errs_hi]
    if not bad:
        print(f"  {d:<{NAME}} 没有异常步（最大 {max(v for _, v in e):.3f} @ step {max(e, key=lambda x: x[1])[0]}）"); continue
    first, last20 = bad[0], [x for x in bad if x[0] >= e[-1][0] - 19]
    print(f"  {d:<{NAME}} {len(bad)}/{len(e)} 步异常（{len(bad)/len(e):.0%}）  首次 step {first[0]} = {first[1]:.3f}"
          f"  最后 20 步里 {len(last20)} 步  最大 {max(v for _, v in bad):.3f} @ step {max(bad, key=lambda x: x[1])[0]}")
    lb = st.median([dict(col(h, "code_length"))[s] for s, _ in bad])
    lg = st.median([v for s, v in col(h, "code_length") if s not in dict(bad)])
    hint = "← 贴近 max_new，像是被预算截断" if lb > 0.8 * args.max_new else \
           ("← 只比正常步长一点，更像「这步抽到的题偏长偏难」，不是截断" if lb > lg else "← 不比正常步长，与长度无关")
    print(f"  {'':<{NAME}} 异常步的长度中位 {lb:.0f} vs 正常步 {lg:.0f}  {hint}")

# ── ④ 计时 ──
print(f"\n{'─'*100}\n  ④ 计时（rank 0 自己的；合梯度等最慢卡的时间不在里面）\n{'─'*100}")
for d, h in D.items():
    g = [v for _, v in col(h, "t_gen")]; t = [v for _, v in col(h, "t_train")]
    if not g: continue
    tot = (sum(g) + sum(t)) / 60
    print(f"  {d:<{NAME}} rollout 中位 {st.median(g):>5.1f} s（末 20 步 {st.median(g[-20:]):>5.1f}）"
          f" 训练 {st.median(t):>4.1f} s  步数 {len(g)}  合计 {tot:>5.1f} 分（不含评测与等最慢卡）")

# ── ⑤ 尾部逐步 ──
print(f"\n{'─'*100}\n  ⑤ 最后 {args.last} 步\n{'─'*100}")
for d, h in D.items():
    print(f"\n  【{d}】 步 | 修好 有效组 语法错 长 调用 |A| KL")
    for x in h[-args.last:]:
        if "code_correct" not in x: continue
        print(f"   {x['step']:>5} | {x['code_correct']:.2f} {x.get('nvar', float('nan')):.1f} {x['code_errs']:.2f}"
              f" {x['code_length']:>4.0f} {x['code_calls']:.2f} {x['code_adv']:.2f} {x['kl']:.4f}"
              + (f"   ← eval 修好 {x['ev']['code_correct']:.3f}" if "ev" in x else ""))
