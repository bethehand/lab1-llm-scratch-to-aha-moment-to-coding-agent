#!/usr/bin/env python3
"""
enum_traces.py —— 穷举器：给一道 Countdown 题，用模型现在的措辞写一条【系统搜索】轨迹。臂 1 的「种」。

它写的轨迹有四样模型自己没有的东西（scan_ops 查出来的四样）：
    ① 不重复：+− 的符号形态各写一次
    ② 算对：每一行的值由程序算，方向标注必对
    ③ 上下界推理：+− 穷尽后写一句「最大 X 最小 Y，需要 × 或 ÷」再换运算符
    ④ 换运算符有章法：先列两数乘积/整除商并用上下界剪枝，按离目标近的先试；再「三数成链 + 剩数」

    python3 enum_traces.py --show                          # 看 5 条样例（内置题）
    python3 enum_traces.py --data data/countdown.json --n 2000 --out sft_cd3.jsonl
    python3 enum_traces.py --data data/countdown4.json --n 2000 --out sft_cd4.jsonl
    python3 enum_traces.py --data data/countdown4.json --n 500 --stats    # 只看能解率和长度分布

输出 jsonl 每行 {"prompt": R1 模板到 <think>, "response": 轨迹到 </answer>, "nums", "target", "lines", "level"}
每条 response 生成后都过一遍 countdown.reward，不是 1.0 的不落盘。
"""
import os, re, sys, json, random, argparse, itertools
from fractions import Fraction

CAP = 40            # 一条轨迹最多写几行算式；超了这道题不要（记进统计）
TOOL = False        # --tool：每个算出来的值都写成 <calc>expr</calc><result>V</result>（计算器版教材）
V2 = False          # --v2：老师 v2（见 trace_v2 的说明）
SHUFFLE = False     # --shuffle：v2 里第 1 层块的尝试顺序、第 2 层行序按题随机（给 RL 留可选的顺序）
MAX_CHARS = 2700    # v2：整条轨迹的字符上限（≈ 1250 token，给 --max-new 1400 留头），超了这道题不要


def calc(expr, v):
    """工具版：<calc>expr</calc><result>v</result>；普通版：expr = v"""
    return f"<calc>{expr}</calc><result>{v}</result>" if TOOL else f"{expr} = {v}"

# ══════════════════════════════════════════════════════════════
#  表达式树：值用 Fraction，渲染时按优先级加括号
# ══════════════════════════════════════════════════════════════
PREC = {"+": 1, "-": 1, "*": 2, "/": 2}

class E:
    __slots__ = ("op", "l", "r", "v", "n")
    def __init__(self, op, l=None, r=None, v=None):
        self.op, self.l, self.r = op, l, r
        if op == "num":
            self.v, self.n = Fraction(v), 0
        else:
            a, b = l.v, r.v
            self.v = {"+": a + b, "-": a - b, "*": a * b, "/": (a / b if b != 0 else None)}[op]
            self.n = l.n + r.n + (op in "*/")
    def s(self, parent=None, right=False):
        if self.op == "num":
            return str(self.v)
        t = f"{self.l.s(self.op)} {self.op} {self.r.s(self.op, True)}"
        if parent is None:
            return t
        p, q = PREC[parent], PREC[self.op]
        if q < p or (q == p and right and parent in "-/"):
            return f"({t})"
        return t


def num(x): return E("num", v=x)


# ══════════════════════════════════════════════════════════════
#  第 0 层：+− 符号形态。最大的数固定为正（负的镜像对正目标没意义），其余各取 ±
# ══════════════════════════════════════════════════════════════
def sign_lines(terms):
    """terms: [(value, expr_str)] → [(value, line_str)]，按减号个数升序、值降序"""
    terms = sorted(terms, key=lambda t: -t[0])
    big, rest = terms[0], terms[1:]
    out = []
    for signs in itertools.product([1, -1], repeat=len(rest)):
        v = big[0] + sum(s * t[0] for s, t in zip(signs, rest))
        pos = [big[1]] + [t[1] for s, t in zip(signs, rest) if s > 0]
        neg = [t[1] for s, t in zip(signs, rest) if s < 0]
        line = " + ".join(pos) + "".join(f" - {x}" for x in neg)
        out.append((signs.count(-1), -v, v, line))
    out.sort()
    seen, res = set(), []                      # ★ 题里有相同的数（4, 4）时符号互换会写出同一行，只留第一次
    for _, _, v, line in out:
        if line not in seen:
            seen.add(line); res.append((v, line))
    return res


def fmt_v(v):
    return str(int(v)) if v.denominator == 1 else f"{float(v):g}"


def tag(v, target):
    if v == target:
        return "(perfect match)"
    return "(too high)" if v > target else "(too low)"


# ══════════════════════════════════════════════════════════════
#  第 1 层：两数乘积 / 整除商，按离目标近的先试，各配剩下的数做 ±
# ══════════════════════════════════════════════════════════════
def blocks1(nums):
    out = []
    for i, j in itertools.combinations(range(len(nums)), 2):
        a, b = nums[i], nums[j]
        rest = [nums[k] for k in range(len(nums)) if k not in (i, j)]
        out.append((a * b, f"{max(a,b)} * {min(a,b)}", rest))
        for x, y in ((a, b), (b, a)):
            if y not in (0, 1) and x % y == 0 and x != y:
                out.append((x // y, f"{x} / {y}", rest))
    # 去重（同值同剩余）
    seen, uniq = set(), []
    for v, e, rest in out:
        key = (v, tuple(sorted(rest)))
        if key not in seen:
            seen.add(key); uniq.append((v, e, rest))
    return uniq


# ══════════════════════════════════════════════════════════════
#  第 2 层：n−1 个数先组成一条链（任意运算、整数值），再跟剩下的那个数配一次
#  按「链值离目标最近」排序 —— 老师的启发式；层内按 (链值, 剩数) 去重
# ══════════════════════════════════════════════════════════════
def sub_exprs(vals):
    def rec(items):
        if len(items) == 1:
            yield items[0]; return
        n = len(items)
        for i in range(n):
            for j in range(n):
                if i == j: continue
                a, b = items[i], items[j]
                rest = [items[k] for k in range(n) if k not in (i, j)]
                cands = []
                if i < j: cands += [E("+", a, b), E("*", a, b)]
                cands.append(E("-", a, b))
                if b.v != 0: cands.append(E("/", a, b))
                for c in cands:
                    if c.v is None or c.v.denominator != 1 or c.v < 0:   # ★ 只走非负整数，1.5B 心算不碰小数
                        continue
                    yield from rec(rest + [c])
    return rec([num(x) for x in vals])


ORDER = "natural"

def level2(nums, target):
    """→ [(closeness, line, hit_expr_or_None)]。ORDER=closest 按最近排（泄题：命中总在第一行）；natural 按枚举顺序"""
    T = Fraction(target)
    seen, out = set(), []
    for keep in itertools.combinations(range(len(nums)), len(nums) - 1):
        d = [nums[k] for k in range(len(nums)) if k not in keep][0]
        for ex in sub_exprs([nums[k] for k in keep]):
            v = ex.v
            if (v, d) in seen: continue
            seen.add((v, d))
            s = ex.s()
            sp = f"({s})" if ex.op in "+-" else s
            cands = [(v + d, f"{s} + {d}", f"{fmt_v(v)} + {d}"),
                     (v - d, f"{s} - {d}", f"{fmt_v(v)} - {d}"),
                     (d - v, f"{d} - {sp}", f"{d} - {fmt_v(v)}"),
                     (v * d, f"{sp} * {d}", f"{fmt_v(v)} * {d}")]
            if d != 0 and v % d == 0: cands.append((v / d, f"{sp} / {d}", f"{fmt_v(v)} / {d}"))
            if v != 0 and d % v == 0: cands.append((Fraction(d) / v, f"{d} / ({s})", f"{d} / {fmt_v(v)}"))
            # 链里没有乘除、且跟剩数也只用加减 → 第 0 层已经覆盖，跳过
            cands = [c for c in cands if ex.n > 0 or c[1].split(" ")[-2] in "*/"]
            if not cands: continue
            best = min(cands, key=lambda c: abs(c[0] - T))
            hit = best[0] == T
            line = f"{calc(s, fmt_v(v))}, then {calc(best[2], fmt_v(best[0]))} {tag(best[0], target)}"
            out.append((abs(best[0] - T), line, best[1] if hit else None, best[1], best[0]))
    # ── 2b：两两成块，再把两块合一次（a*b + c*d、(a+b)/(c/d) 这种 (2,2) 形态，链覆盖不了）──
    if len(nums) == 4:
        def pair_vals(a, b):
            hi, lo = max(a, b), min(a, b)
            r = [(hi + lo, f"{hi} + {lo}"), (hi - lo, f"{hi} - {lo}"), (hi * lo, f"{hi} * {lo}")]
            if lo not in (0, 1) and hi % lo == 0: r.append((hi // lo, f"{hi} / {lo}"))
            return r
        seen2 = set()
        for i in range(1, 4):
            A = [nums[0], nums[i]]; B = [nums[k] for k in range(1, 4) if k != i]
            for va, ea in pair_vals(*A):
                for vb, eb in pair_vals(*B):
                    (v1, e1), (v2, e2) = sorted([(va, ea), (vb, eb)], reverse=True)
                    if (v1, v2, e1, e2) in seen2: continue
                    seen2.add((v1, v2, e1, e2))
                    p1 = f"({e1})" if " + " in e1 or " - " in e1 else e1
                    p2 = f"({e2})" if " + " in e2 or " - " in e2 else e2
                    cands = [(v1 + v2, f"{e1} + {e2}", f"{v1} + {v2}"),
                             (v1 - v2, f"{e1} - {p2}", f"{v1} - {v2}"),
                             (v1 * v2, f"{p1} * {p2}", f"{v1} * {v2}")]
                    if v2 not in (0,) and v1 % v2 == 0: cands.append((v1 // v2, f"{p1} / ({e2})", f"{v1} / {v2}"))
                    md = ("*" in e1 or "/" in e1 or "*" in e2 or "/" in e2)
                    cands = [c for c in cands if md or c[1].split(" ")[-2] in "*/"]   # 全加减的第 0 层已覆盖
                    if not cands: continue
                    best = min(cands, key=lambda c: abs(c[0] - T))
                    hit = best[0] == T
                    line = f"{calc(e1, v1)} and {calc(e2, v2)}, then {calc(best[2], best[0])} {tag(Fraction(best[0]), target)}"
                    out.append((abs(best[0] - T), line, best[1] if hit else None, best[1], Fraction(best[0])))
    if ORDER == "closest":
        out.sort(key=lambda t: t[0])
    return out


# ══════════════════════════════════════════════════════════════
#  主函数：写一条轨迹
# ══════════════════════════════════════════════════════════════
def trace(nums, target):
    """→ (response_text, n_lines, level) 或 None（CAP 内没解出）"""
    T = Fraction(target)
    L, n_formula = [], 0
    def emit(v, line):
        nonlocal n_formula
        L.append(f"{calc(line, fmt_v(v))} {tag(v, target)}"); n_formula += 1
        return v == T

    head = (f" We need to use the numbers {nums} exactly once, combined with the operations "
            f"+, -, *, / to get a result of {target}.\nLet's try different combinations:\n")

    # ── 第 0 层：符号形态 ──
    L.append("First, only + and -:")
    lv0 = sign_lines([(Fraction(x), str(x)) for x in nums])
    hit = None
    for v, line in lv0:
        if emit(v, line):
            hit = (line, 0); break
    if hit is None:
        vals = [v for v, _ in lv0]
        L.append(f"None of the {len(lv0)} sign patterns gives {target} (they range from "
                 f"{fmt_v(min(vals))} to {fmt_v(max(vals))}), so we need * or /.")
        # ── 第 1 层：两数乘积/商 + 其余做 ±，先用上下界剪掉够不着的块 ──
        b1 = sorted(blocks1(nums), key=lambda t: (abs(t[0] - target), t[0]))
        L.append("Pair products and quotients: " + ", ".join(calc(e, v) for v, e, _ in b1) + ".")
        usable, far = [], []
        for v, e, rest in b1:
            span = sum(rest)
            if (v - span <= target <= v + span) or (-v - span <= target <= -v + span):
                usable.append((v, e, rest))
            else:
                far.append(str(v))
        if far:
            L.append(f"{', '.join(far)} cannot reach {target} even with the remaining numbers added or subtracted, skip them.")
        if usable:
            L.append("Try the rest with the remaining numbers, closest to the target first:")
        for v, e, rest in usable:
            for vv, line in sign_lines([(Fraction(v), e)] + [(Fraction(x), str(x)) for x in rest]):
                if emit(vv, line):
                    hit = (line, 1); break
            if hit or n_formula >= CAP: break
    if hit is None and n_formula < CAP:
        # ── 第 2 层：链 + 剩数 ──
        L.append("Next, build a bigger piece first (three numbers, or two pairs), then finish:")
        for _, line, hexpr, *_ in level2(nums, target):
            L.append(line); n_formula += 1
            if hexpr:
                hit = (hexpr, 2); break
            if n_formula >= CAP: break
    if hit is None:
        return None
    body = head + "\n".join(L) + f"\n</think>\n<answer> {hit[0]} </answer>"
    return body, n_formula, hit[1]


# ══════════════════════════════════════════════════════════════
#  老师 v2（2026-09-07，臂 T SFT 后 scan_ops 的证据）
#    ① 镜像符号行：最大数为负、其余之和更大时值为正的那族（「22 + 25 - 30 - 20」），v1 永远不写 → 全错 21 道里 7 道就是它
#    ② 第 1 层乘除纯机械：乘积表按降序对的顺序列全 → 一个上界（目标 + 全部数之和，多一次调用）→ 超上界的块按数值读出来跳过
#       → 剩下的块按表的顺序（或 --shuffle 随机）逐个配剩余数写符号行。没有「closest first」：那个顺序要先知道值才能排，模型学不到只能编
#    ③ 不复用数、不整除不写、去重：全部由程序保证
#    ④ 第 2 层同样的候选（链 + 剩数 / 两对），但一行一次调用：整条表达式交给工具，不再「先算链再收尾」两次调用；加 --shuffle
#    ⑤ 上限按字符（≈ token）而不是按行：MAX_CHARS 3000 ≈ 1300 token
# ══════════════════════════════════════════════════════════════
def sign_lines_v2(terms):
    """v1 的 2^(n−1) 行（最大项为正）+ 镜像族里值为正的行（最大项为负）"""
    res = list(sign_lines(terms))
    terms = sorted(terms, key=lambda t: -t[0])
    big, rest = terms[0], terms[1:]
    if sum(t[0] for t in rest) <= big[0]:
        return res
    extra = []
    for signs in itertools.product([1, -1], repeat=len(rest)):
        v = -big[0] + sum(s * t[0] for s, t in zip(signs, rest))
        if v <= 0:
            continue
        pos = [t[1] for s, t in zip(signs, rest) if s > 0]
        neg = [big[1]] + [t[1] for s, t in zip(signs, rest) if s < 0]
        extra.append((signs.count(-1), -v, v, " + ".join(pos) + "".join(f" - {x}" for x in neg)))
    extra.sort()
    seen = {line for _, line in res}
    for _, _, v, line in extra:
        if line not in seen:
            seen.add(line); res.append((v, line))
    return res


def blocks1_v2(ns):
    """ns 降序。→ [(值, 式子, 剩余数)]，按降序对的顺序：乘积，然后整除商（除数不是 0/1、两数不等）"""
    out, seen = [], set()
    for i, j in itertools.combinations(range(len(ns)), 2):
        a, b = ns[i], ns[j]
        rest = [ns[k] for k in range(len(ns)) if k not in (i, j)]
        # ×1 是「把 1 吸收掉」的用法（生成器常这么用），但 27*1+22+6 = 27+22*1+6 = 27+22+6*1 同值 → 只在最大数上写一次
        cands = ([(a, f"{a} * 1")] if i == 0 else []) if b == 1 else [(a * b, f"{a} * {b}")]
        if b not in (0, 1) and a != b and a % b == 0:
            cands.append((a // b, f"{a} / {b}"))
        for v, e in cands:
            key = (v, tuple(sorted(rest)))
            if key not in seen:                       # 题里有相同的数时同值同剩余只留一次
                seen.add(key); out.append((v, e, rest))
    return out


L2_ORDER = "natural"

def l2_key(fexpr):
    """整式最外层最后一个运算符：/ → 0，* → 1，± → 2"""
    depth, last = 0, "+"
    for ch in fexpr:
        if ch == "(": depth += 1
        elif ch == ")": depth -= 1
        elif depth == 0 and ch in "+-*/": last = ch
    return {"/": 0, "*": 1}.get(last, 2)


def trace_v2(nums, target, rng=None):
    """→ (response_text, n_lines, level) 或 None（字符上限内没解出）"""
    T = Fraction(target)
    ns = sorted(nums, reverse=True)
    L, n_formula = [], 0
    def emit(v, line):
        nonlocal n_formula
        L.append(f"{calc(line, fmt_v(v))} {tag(v, target)}"); n_formula += 1
        return v == T
    def chars():
        return sum(len(x) + 1 for x in L)

    head = (f" We need to use the numbers {nums} exactly once, combined with the operations "
            f"+, -, *, / to get a result of {target}.\nLet's try different combinations:\n")
    # ── 第 0 层：符号形态 + 镜像 ──
    L.append("First, only + and -:")
    lv0 = sign_lines_v2([(Fraction(x), str(x)) for x in ns])
    total = lv0[0][0]                                   # 全加那行 = 所有数之和，第 1 层的上界要用
    hit = None
    for v, line in lv0:
        if emit(v, line):
            hit = (line, 0); break
    if hit is None:
        L.append(f"None of the {len(lv0)} sign patterns gives {target}, so we need * or /.")
        # ── 第 1 层：乘积表 → 上界 → 按序逐块配剩余数 ──
        b1 = blocks1_v2(ns)
        L.append("Pair products and quotients: " + ", ".join(calc(e, v) for v, e, _ in b1) + ".")
        bound = target + total
        L.append(f"Upper bound: {calc(f'{target} + {fmt_v(total)}', fmt_v(bound))}; blocks above it cannot come back down"
                 + (": " + ", ".join(str(v) for v, _, _ in b1 if v > bound) + ", skip them." if any(v > bound for v, _, _ in b1) else "."))
        usable = [(v, e, rest) for v, e, rest in b1 if v <= bound]
        if SHUFFLE and rng is not None:
            rng.shuffle(usable)
        if usable:
            L.append("Try each of the others with the remaining numbers:")
        for v, e, rest in usable:
            for vv, line in sign_lines_v2([(Fraction(v), e)] + [(Fraction(x), str(x)) for x in rest]):
                if emit(vv, line):
                    hit = (line, 1); break
            if hit or chars() > MAX_CHARS: break
    if hit is None and chars() <= MAX_CHARS:
        # ── 第 2 层：链 + 剩数 / 两对 ──
        L.append("Next, build a bigger piece first (three numbers, or two pairs), then finish:")
        rows = level2(ns, target)
        if L2_ORDER == "divfirst":                        # ★ 按【形态】排，模型读得出来：整式最后一步是 ÷ 的先试，然后 ×，最后 ±
            rows.sort(key=lambda r: l2_key(r[3]))        #   目标 ≤ 100 而乘积动辄几百，把大数「除回来」的形态命中率最高（贪心覆盖量出来的）
        if SHUFFLE and rng is not None:
            rng.shuffle(rows)
        for _, _line, hexpr, fexpr, fval in rows:          # ★ v2：整条表达式一次调用（工具算整式，不用先算链再收尾）
            L.append(f"{calc(fexpr, fmt_v(fval))} {tag(fval, target)}"); n_formula += 1
            if hexpr:
                hit = (hexpr, 2); break
            if chars() > MAX_CHARS: break
    if hit is None or chars() > MAX_CHARS:
        return None
    body = head + "\n".join(L) + f"\n</think>\n<answer> {hit[0]} </answer>"
    return body, n_formula, hit[1]


_trace_v1 = trace
def trace(nums, target, rng=None):
    return trace_v2(nums, target, rng) if V2 else _trace_v1(nums, target)


# ══════════════════════════════════════════════════════════════
DEMO = [([1, 13, 42], 56), ([3, 7, 25], 46), ([6, 12, 18, 42], 30), ([5, 7, 14, 33], 43), ([2, 3, 5, 7], 31)]

def main():
    global ORDER, CAP, TOOL, V2, SHUFFLE, MAX_CHARS, L2_ORDER
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--data", default=None)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--lo", type=int, default=0, help="题号范围下限")
    ap.add_argument("--hi", type=int, default=80000, help="★ 题号范围上限，别碰评测片 [95000:95100]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--order", choices=["natural", "closest"], default="natural", help="第 2 层的顺序")
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--tool", action="store_true", help="★ 计算器版教材：所有值写成 <calc>expr</calc><result>V</result>")
    ap.add_argument("--ids", default=None, help="★ 按这份 jsonl 里的 idx 重生成（同题号），不再随机抽")
    ap.add_argument("--v2", action="store_true", help="★ 老师 v2：镜像符号行 + 第 1 层纯机械 + 字符上限（见 trace_v2）")
    ap.add_argument("--shuffle", action="store_true", help="v2：第 1 层块序、第 2 层行序按题随机")
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS, help="v2：整条轨迹字符上限")
    ap.add_argument("--l2-order", choices=["natural", "divfirst"], default="divfirst", help="v2 第 2 层顺序：按最后一步运算符 ÷ → × → ±")
    ap.add_argument("--only-level", type=int, default=None, help="只留在这一层命中的题（过采样深层用）")
    ap.add_argument("--exclude", default=None, help="这份 jsonl 里出现过的 idx 不要（跟主文件去重）")
    args = ap.parse_args()
    ORDER, CAP, TOOL, V2, SHUFFLE, MAX_CHARS = args.order, args.cap, args.tool, args.v2, args.shuffle, args.max_chars
    L2_ORDER = args.l2_order

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import countdown as CD

    if args.show:
        for nums, target in DEMO + ([([20, 22, 25, 30], 37), ([1, 2, 11, 26], 28)] if V2 else []):
            r = trace(nums, target, random.Random(str(nums)))
            print("=" * 74 + f"\n  nums {nums}  target {target}")
            if r is None:
                print("  （CAP 内没解出）"); continue
            body, n, lvl = r
            score, d = CD.reward(body, nums, target)
            print(f"  {n} 行算式，第 {lvl} 层命中，判分器 {score:.2f}\n" + "-" * 74)
            print("<think>" + body)
        return

    D = json.load(open(args.data))
    rng = random.Random(args.seed)
    if args.ids:
        idx = [json.loads(l)["idx"] for l in open(args.ids)]; args.n = len(idx)
        print(f"  ★ 按 {args.ids} 的 {len(idx)} 个题号重生成")
    else:
        idx = rng.sample(range(args.lo, min(args.hi, len(D))), min(args.n * (20 if args.only_level else 2), args.hi - args.lo))
    if args.exclude:
        ex = {json.loads(l)["idx"] for l in open(args.exclude)}
        idx = [i for i in idx if i not in ex]
    rows, drop_cap, drop_score, lines, levels = [], 0, 0, [], {0: 0, 1: 0, 2: 0, 3: 0}
    for i in idx:
        if len(rows) >= args.n: break
        p = D[i]
        r = trace(p["nums"], p["target"], random.Random(i))
        if r is None:
            drop_cap += 1; continue
        body, n, lvl = r
        if args.only_level is not None and lvl != args.only_level:
            continue
        score, _ = CD.reward(body, p["nums"], p["target"])
        if score < 0.999:
            drop_score += 1; continue
        prompt = CD.make_prompt(p["nums"], p["target"])
        if TOOL:
            import calc_tool
            prompt = calc_tool.tool_prompt(p["nums"], p["target"])
        rows.append(dict(prompt=prompt, response=body,
                         nums=p["nums"], target=p["target"], lines=n, level=lvl, idx=i))
        lines.append(n); levels[lvl] += 1
    tried = len(rows) + drop_cap + drop_score
    lines.sort()
    q = lambda f: lines[int(f * (len(lines) - 1))] if lines else 0
    print(f"  {args.data}  题号 [{args.lo}:{args.hi}]  试了 {tried}  留 {len(rows)}"
          f"  CAP 内没解 {drop_cap} ({100*drop_cap/max(1,tried):.1f}%)  判分器不认 {drop_score}")
    print(f"  算式行数  中位 {q(.5)}  P90 {q(.9)}  最长 {lines[-1] if lines else 0}"
          f"   命中层 0/1/2/3 = {levels[0]}/{levels[1]}/{levels[2]}/{levels[3]}")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
        tl = sorted(len(tok(r["response"]).input_ids) for r in rows[:500])
        print(f"  token 长度（前 500 条）中位 {tl[len(tl)//2]}  P90 {tl[int(.9*(len(tl)-1))]}  最长 {tl[-1]}")
    except Exception as e:
        print(f"  （token 长度没算：{e}）")
    if args.out:
        with open(args.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"  → {args.out}")


if __name__ == "__main__":
    main()
