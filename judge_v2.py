#!/usr/bin/env python3
"""
judge_v2.py —— 判分器 v2 的过程项。纯函数，不依赖 torch，grpo_mix_mp.py --judge v2 时调用；也能单独测。

    r = 0.9·对 + 0.02·格式 + 0.02·解析 + 0.06·数字比例          ← 阶梯 0.25 → 0.10，错式子的吸引力降下来
        − 0.2 · [假宣告]                                          ← 「expr = V (perfect/works/…)」而 expr ≠ 目标
        − 0.1 · 错步率                                            ← 标注行里算错或标反的比例
        − 0.05 · min(重复行, 4)                                    ← 同一条里同形态第二次出现（不记账）
        − 0.05 · min(违规行, 4)                                    ← 一次尝试里某个数用的次数超过题里给的
        − 长度：对的只扣 0.05·len/max_new；错的照旧 soft 线性到 len_pen（在 grpo_mix_mp 里做）
    底线：最差的对 0.9 − 0.05 − 0.1 − 0.2 − 0.2 = 0.35  >  最好的错 0.10 + 0.05 = 0.15  → 任何对 > 任何错
    v2.1（臂 3 前 126 步被钻后加的三条）：尝试行认任意短标签（换说法躲不掉）；答错且尝试 < 3 行错步率按 1 算；每条算对的尝试 +0.01 封顶 0.05

    python3 judge_v2.py        # 跑自检
"""
import re
from collections import Counter
import countdown as CD

W2 = {"fmt": 0.02, "parse": 0.02, "nums_frac": 0.06, "correct": 0.90}
PEN = {"false_claim": 0.20, "step_err": 0.10, "dup": 0.05, "viol": 0.05, "len_ok": 0.05, "ok_line": 0.01}
CAP_LINES = 4
MIN_ATTEMPTS = 3      # v2.1：答错且尝试行少于这个数 → 错步率按 1 算（不搜就猜是最差的错法）
CAP_OK_BONUS = 0.05   # v2.1：每条算对的尝试行 +0.01，封顶

# 尝试行：「expr = V (任意短标签)」都算 —— v2.1：只认 too high/low 会被换说法绕开（「(not 30)」「(no)」）
#   算术每行都验；方向只在标签里写了 high/low 时验
ATTEMPT = re.compile(r"((?:\(|\d)[\d\s\+\-\*/\(\)]*?)\s*=\s*(-?\d+(?:\.\d+)?)\s*\(([^()\n]{1,40})\)", re.I)
HIGH = re.compile(r"\b(too )?(high|big|large|much|more)\b", re.I)
LOW = re.compile(r"\b(too )?(low|small|little|less)\b", re.I)
ANNOT = re.compile(r"((?:\(|\d)[\d\s\+\-\*/\(\)]*?)\s*=\s*(-?\d+(?:\.\d+)?)\s*\(?\s*(too (?:high|low|big|small|large))", re.I)   # 旧探针，analyze_search 还在用
CLAIM_WORDS = re.compile(r"\b(perfect|works|correct|success|yes|found|matches|got it|that'?s it)\b", re.I)
CLAIM = re.compile(r"((?:\(|\d)[\d\s\+\-\*/\(\)]*?)\s*=\s*(-?\d+(?:\.\d+)?)\s*\(?\s*[A-Za-z ,'!]{0,16}?\b(perfect|works|correct|success|yes|found|matches|got it|that'?s it)\b", re.I)   # 值和宣告词之间只许字母/空格，不许跨过下一个等式
NUM = re.compile(r"\d+")


def _eval(expr):
    try:
        return CD.safe_eval(CD.clean_expr(expr))
    except Exception:
        return None


def canon(expr):
    """纯加减按项集合归一（25+7−3 = 7+25−3）；含 ×÷ 或括号按原样（去空格）"""
    e = expr.replace(" ", "")
    if "*" in e or "/" in e or "(" in e:
        return ("md", e)
    return tuple(sorted(re.findall(r"[+-]?\d+", e if e[:1] in "+-" else "+" + e)))


def terms(txt, nums, target):
    """→ dict(false_claim 0/1, step_err 率, dup 行数, viol 行数, n_annot, n_ok, n_claim)"""
    tgt = float(target)
    need = Counter(int(x) for x in nums)
    n = ok = 0
    exprs = []
    for m in ATTEMPT.finditer(txt):
        label = m.group(3)
        if CLAIM_WORDS.search(label):            # perfect / works … 归宣告那边算
            continue
        real = _eval(m.group(1))
        if real is None:
            continue
        n += 1
        exprs.append(m.group(1))
        arith_ok = abs(real - float(m.group(2))) <= 1e-3 * max(1.0, abs(real))    # ★ 相对容差：工具版非整数只印 4 位有效数字（51.71）
        h, l = bool(HIGH.search(label)), bool(LOW.search(label))
        dir_ok = True if h == l else (h == (real > tgt))     # 没写方向（或写乱了）就只验算术
        if arith_ok and dir_ok:
            ok += 1
    false_claim, n_claim = 0, 0
    for m in CLAIM.finditer(txt):
        real = _eval(m.group(1))
        if real is None:
            continue
        n_claim += 1
        exprs.append(m.group(1))
        if abs(real - tgt) > 1e-6:
            false_claim = 1                        # 说了 perfect / works，式子却不等于目标
    # 重复：同形态第二次出现
    seen, dup = set(), 0
    for e in exprs:
        c = canon(e)
        if c in seen:
            dup += 1
        else:
            seen.add(c)
    # 违规：一次尝试里某个题目里的数用的次数超过给的次数（题外数不算：第 2 层的中间值「10 + 33」是合法的）
    viol = 0
    for e in exprs:
        used = Counter(int(x) for x in NUM.findall(e))
        if any(used[k] > need.get(k, 0) for k in used if k in need):
            viol += 1
    return dict(false_claim=false_claim, step_err=(1 - ok / n) if n else 0.0,
                dup=dup, viol=viol, n_annot=n, n_ok=ok, n_claim=n_claim)


def score(txt, nums, target):
    """完整的 v2 判分（不含长度项）→ (r, d)。d 里带各过程项，训练日志用"""
    r, d = CD.reward(txt, nums, target, w=W2)
    t = terms(txt, nums, target)
    if d["correct"] <= 0 and t["n_annot"] < MIN_ATTEMPTS:     # ★ v2.1 b：答错又不搜 → 按缺几行分级罚，0 行最重
        t["step_err"] = max(t["step_err"], (MIN_ATTEMPTS - t["n_annot"]) / MIN_ATTEMPTS)
    r -= PEN["false_claim"] * t["false_claim"]
    r -= PEN["step_err"] * t["step_err"]
    r += min(PEN["ok_line"] * t["n_ok"], CAP_OK_BONUS)        # ★ v2.1 c：算对的尝试行有正回报
    r -= PEN["dup"] * min(t["dup"], CAP_LINES)
    r -= PEN["viol"] * min(t["viol"], CAP_LINES)
    d.update(t)
    return r, d


if __name__ == "__main__":
    cases = [
        ("对 + 全干净", [3, 7, 25], 46,
         " 25 + 7 + 3 = 35 (too low)\n25 + 7 - 3 = 29 (too low)\n25 + 7 * 3 = 46 (perfect match)\n</think>\n<answer> 25 + 7 * 3 </answer>"),
        ("假宣告", [6, 12, 18, 42], 30,
         " 42 + 12 - 18 - 6 = 28 (too low)\n42 * 12 / 6 - 18 = 30 (perfect match)\n</think>\n<answer> 42 * 12 / 6 - 18 </answer>"),
        ("重复 + 违规 + 算错", [2, 3, 5, 7], 31,
         " 7 + 5 + 3 + 2 = 17 (too low)\n2 + 3 + 5 + 7 = 17 (too low)\n7 * 5 * 3 * 2 / 2 = 105 (too high)\n7 + 5 + 3 - 2 = 12 (too low)\n<answer> 7 * 5 - 3 + 2 </answer>"),
        ("第 2 层中间值不算违规", [5, 7, 14, 33], 43,
         " 33 + 14 + 7 + 5 = 59 (too high)\n5 * 14 / 7 = 10, then 10 + 33 = 43 (perfect match)\n</think>\n<answer> 5 * 14 / 7 + 33 </answer>"),
        ("诚实放弃", [2, 3, 5, 7], 31,
         " 7 + 5 + 3 + 2 = 17 (too low)\n7 + 5 + 3 - 2 = 13 (too low)\nNone of these work.\n<answer>None</answer>"),
        ("换说法躲罚 (not 30)", [6, 12, 18, 42], 30,
         " 6 + 12 + 18 - 42 = 0 (not 30)\n6 + 12 + 42 - 18 = 40 (not 30)\n6 + 42 - 12 - 18 = 18 (no)\n<answer> 6 + 42 - 12 - 18 </answer>"),
        ("不搜直接猜错", [6, 12, 18, 42], 30,
         " Let's see.\n<answer> 42 + 18 - 12 - 6 </answer>"),
        ("搜得对但没解出", [5, 7, 14, 33], 43,
         " 33 + 14 + 7 + 5 = 59 (too high)\n33 + 14 + 7 - 5 = 49 (too high)\n33 + 14 - 7 - 5 = 35 (too low)\n33 + 7 - 14 - 5 = 21 (too low)\n<answer> 33 + 14 - 7 + 5 </answer>"),
    ]
    print(f"{'情形':18} {'r':>6} {'对':>3} {'假宣告':>4} {'错步率':>6} {'重复':>4} {'违规':>4}")
    for name, nums, tgt, txt in cases:
        r, d = score(txt, nums, tgt)
        print(f"{name:18} {r:>6.3f} {int(d['correct']):>3} {d['false_claim']:>4} {d['step_err']:>6.2f} {d['dup']:>4} {d['viol']:>4}")
    print("\n期望：对+干净 1.02；假宣告 −0.20；重复+违规 ≈ 0.005；第 2 层 1.01 不误判；"
          "\n      诚实放弃（2 行）≈ 0.02 − 0.033 + 0.02 = 0.007；(not 30)：3 行都算进去，2 行算错 → 错步 0.67；"
          "\n      不搜直接猜错：0.10 − 0.10 = 0.00（最差的错）；搜得对没解出：0.10 + 0.04 = 0.14（最好的错）"
          "\n      顺序：任何对 ≥ 0.35 > 搜得对没解出 0.14 > 诚实放弃 ≈ 0.01 ≈ 不搜猜错 0.00 > 假宣告 −0.20")
