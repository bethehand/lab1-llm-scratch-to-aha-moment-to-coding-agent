#!/usr/bin/env python3
"""
★ 硬题诊断：模型在全错的题上，到底试没试过乘除？

  6+ 桶 43% 不动、全错 32 道不动 —— 猜测是「只在加减里绕」。这里直接数：
    ① 全错的题里，哪些【必须】用乘除才有解（暴力枚举 +− 子空间）
    ② 模型在这些题的 8 条轨迹里，写过 × 或 ÷ 的比例
    ③ 对照：全对 / 有对有错的题里，用 × ÷ 的比例
  ② 是零 → p = 0，RL 抽不到，加 k 没用，得换任务或冷启动
  ② 是 2~5% → 半径之下但非零，k 加到 32~64 能抽到

    python3 scan_ops.py cd_out_mix_mp_ckpt_step600_s8_off95000_raw.json
"""
import sys, json, re, itertools
from collections import defaultdict
import countdown as CD

MUL = re.compile(r"[\d)]\s*(?:[\*×]|\\times|\\cdot)\s*[\d(]")   # ★ 要求两边有数字/括号：排掉开头复述的「+, -, *, /」
DIV = re.compile(r"[\d)]\s*(?:/|÷|\\div)\s*[\d(]")             # 同上，也排掉 </think> 里的 /


def solvable(nums, target, ops):
    """3 或 4 个数，给定运算符集合，暴力找有没有解（枚举全部括号形态）"""
    for perm in itertools.permutations(nums):
        for opseq in itertools.product(ops, repeat=len(nums) - 1):
            if len(nums) == 3:
                a, b, c = perm; o1, o2 = opseq
                exprs = (f"(({a}{o1}{b}){o2}{c})", f"({a}{o1}({b}{o2}{c}))")
            else:
                a, b, c, d = perm; o1, o2, o3 = opseq
                exprs = (f"((({a}{o1}{b}){o2}{c}){o3}{d})", f"(({a}{o1}{b}){o2}({c}{o3}{d}))",
                         f"(({a}{o1}({b}{o2}{c})){o3}{d})", f"({a}{o1}(({b}{o2}{c}){o3}{d}))",
                         f"({a}{o1}({b}{o2}({c}{o3}{d})))")
            for expr in exprs:
                try:
                    if abs(CD.safe_eval(expr) - target) < 1e-9:
                        return True
                except Exception:
                    pass
    return False

TOOLCALL = re.compile(r"<calc>(.*?)</calc>\s*<result>(.*?)</result>", re.S)


def report(path):
    d = json.load(open(path))
    P = json.load(open(d.get("data", "data/countdown.json")))[d["offset"]: d["offset"] + d["n"]]
    G = defaultdict(list)
    for r in d["records"]:
        if "<calc>" in r["txt"]:                  # ★ 工具版轨迹归一成「E = V」，探针照用
            r["txt"] = TOOLCALL.sub(r"\1 = \2", r["txt"])
        G[r["pi"]].append(r)
    print("=" * 78 + f"\n  {d.get('ckpt') or d['model']}   {d['n']} 题 × {d['n_samples']}\n" + "=" * 78)

    cats = {"全错": [], "有对有错": [], "全对": []}
    for pi, rs in G.items():
        c = sum(r["d"]["correct"] for r in rs)
        cats["全错" if c == 0 else "全对" if c == len(rs) else "有对有错"].append(pi)

    print(f"\n  ── ① 全错的 {len(cats['全错'])} 道题：解需要什么运算符 ──")
    need_mul = only_pm = 0
    for pi in cats["全错"]:
        nums, t = P[pi]["nums"], P[pi]["target"]
        pm = solvable(nums, t, "+-")
        if pm: only_pm += 1
        else:  need_mul += 1
    print(f"    只用 +− 就有解        {only_pm:3d} 道   ← 模型在自己的子空间里也没找到")
    print(f"    ★ 必须用 × 或 ÷      {need_mul:3d} 道   ← 不试乘除永远解不了")

    print(f"\n  ── ② 各类题的轨迹里，写过 × / ÷ 的比例 ──")
    print(f"    {'类别':<10}{'题数':>6}{'轨迹':>8}{'含 ×':>10}{'含 ÷':>10}{'× 或 ÷':>10}")
    for name, pis in cats.items():
        rs = [r for pi in pis for r in G[pi]]
        if not rs:
            continue
        m = sum(1 for r in rs if MUL.search(r["txt"]))
        v = sum(1 for r in rs if DIV.search(r["txt"]))
        e = sum(1 for r in rs if MUL.search(r["txt"]) or DIV.search(r["txt"]))
        print(f"    {name:<10}{len(pis):>6}{len(rs):>8}{m/len(rs):>10.3f}{v/len(rs):>10.3f}{e/len(rs):>10.3f}")

    print(f"\n  ── ③ 必须用乘除的那些题，模型的 8 条里试过乘除的比例 ──")
    hard = [pi for pi in cats["全错"] if not solvable(P[pi]["nums"], P[pi]["target"], "+-")]
    if hard:
        rs = [r for pi in hard for r in G[pi]]
        e = sum(1 for r in rs if MUL.search(r["txt"]) or DIV.search(r["txt"]))
        print(f"    {len(hard)} 道题 × 8 = {len(rs)} 条，试过 × 或 ÷ 的 {e} 条 = {e/len(rs):.3f}")
        print(f"    ★ 这个数是「p」：0 → RL 抽不到；0.02~0.05 → k=32~64 能抽到；>0.05 → 现在的 k 就够")
        print(f"\n    前 3 道必须用乘除的题：")
        for pi in hard[:3]:
            print(f"      nums {P[pi]['nums']}  target {P[pi]['target']}")
            for r in G[pi][:2]:
                one = r["txt"].replace("\n", " ⏎ ")[:220]
                print(f"        · {one}")
    else:
        print("    没有必须用乘除的全错题")

    # ── ④ ★ 乘除的【形态】：直接乘 a×b±c，还是先组合再乘 (a±b)×c ──
    #    三道样例的解全是后者：(13−10)×15、(28−19)×11、48÷2+12
    #    如果模型只会写前者，那不是「不试乘法」，是「乘法只试了一种形态」
    # ★ 形态分三种：
    #   纯直接乘   一行里只有 ×÷ 没有 +−          12 × 48 = 554         两个大数相乘，几乎必爆
    #   混合形态   一行里既有 ×÷ 又有 +−          48 / 2 + 12、(13−10)×15、37 + 39/13
    #              ← 硬题的解几乎全是这种，正则不限括号位置
    DIRECT = re.compile(r"^[^\n]*?\d+\s*[\*×/÷]\s*\d+[^\n+\-]*=", re.M)         # 一行：有乘除、等号前没加减
    MIXED  = re.compile(r"^[^\n]*?(?=[^\n]*[\*×/÷])(?=[^\n]*\d\s*[+\-]\s*\d)[^\n]*=", re.M)  # 一行：乘除和加减都有
    print(f"\n  ── ④ 乘除的形态：纯直接乘 vs 混合（乘除 + 加减同一行）──")
    print(f"    {'类别':<10}{'轨迹':>8}{'有纯直接乘':>12}{'有混合形态':>12}{'混合/条':>10}")
    for name, pis in cats.items():
        rs = [r for pi in pis for r in G[pi]]
        if not rs: continue
        dm = sum(1 for r in rs if DIRECT.search(r["txt"]))
        cm = sum(1 for r in rs if MIXED.search(r["txt"]))
        cn = sum(len(MIXED.findall(r["txt"])) for r in rs) / len(rs)
        print(f"    {name:<10}{len(rs):>8}{dm/len(rs):>12.3f}{cm/len(rs):>12.3f}{cn:>10.2f}")
    allr = d["records"]
    cm_all = sum(1 for r in allr if MIXED.search(r["txt"]))
    print(f"    ★ 全部 {len(allr)} 条里出现过混合形态的：{cm_all} 条 = {cm_all/len(allr):.3f}  ← 这才是那个 p")

    # ── ⑤ 看两条硬题的【完整】轨迹，别截断 ──
    if hard:
        pi = hard[0]
        print(f"\n  ── ⑤ 硬题 nums {P[pi]['nums']} target {P[pi]['target']} 的两条完整轨迹 ──")
        for r in G[pi][:2]:
            print("    " + "─" * 70)
            print("    " + r["txt"].strip().replace("\n", "\n    "))
    print()


for p in sys.argv[1:]:
    report(p)
