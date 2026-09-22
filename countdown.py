#!/usr/bin/env python3
"""
Countdown 数字游戏 —— R1-Zero 复现用的任务模块。

  给几个数和一个目标值，用 + − × ÷ 各数各用一次，凑出目标。
    [44, 19, 35]  目标 98   →   44 + 19 + 35

★ 为什么选它而不是 GSM8K：
  它本质是【搜索】—— 一条路走不通就得回头
  → 天然逼出「wait, that doesn't work, let me try...」
  → ★ TinyZero 就是用它复现出顿悟时刻的

自检：
    python3 countdown.py              # 生成样例 + 判分器测试
    python3 countdown.py --gen 100000 --out data/countdown.json
"""
import ast, json, random, operator, argparse, re
from collections import Counter

# ══════════════════════════════════════════════════════════════
#  ① 数据生成 —— ★ 反向生成，保证有解
# ══════════════════════════════════════════════════════════════
OPS = "+-*/"


def gen_one(k, lo, hi, tmax, rng):
    """随机取 k 个数，随机加运算符算出目标 → ★ 保证至少有一个解"""
    for _ in range(200):
        nums = [rng.randint(lo, hi) for _ in range(k)]
        order = nums[:]; rng.shuffle(order)
        cur = order[0]
        ok = True
        for v in order[1:]:
            op = rng.choice(OPS)
            if op == "+":   cur = cur + v
            elif op == "-": cur = cur - v
            elif op == "*": cur = cur * v
            else:
                if v == 0 or cur % v != 0:      # ★ 只保留整除
                    ok = False; break
                cur = cur // v
            if not (0 < cur <= tmax * 10):      # 中间值别爆掉
                ok = False; break
        if ok and 1 <= cur <= tmax:
            return sorted(nums), cur
    return None


def generate(n, k=3, lo=1, hi=50, tmax=100, seed=0):
    rng = random.Random(seed)
    out, seen = [], set()
    while len(out) < n:
        r = gen_one(k, lo, hi, tmax, rng)
        if r is None:
            continue
        nums, t = r
        key = (tuple(nums), t)
        if key in seen:            # 去重
            continue
        seen.add(key)
        out.append({"nums": nums, "target": t})
    return out


# ══════════════════════════════════════════════════════════════
#  ② 提示模板 —— ★ R1-Zero 的原版，Base 模型也能被它带出结构
# ══════════════════════════════════════════════════════════════
SYS = ("A conversation between User and Assistant. The user asks a question, "
       "and the Assistant solves it. The Assistant first thinks about the reasoning "
       "process in the mind and then provides the user with the answer. The reasoning "
       "process and answer are enclosed within <think> </think> and <answer> </answer> "
       "tags, respectively.")


def make_prompt(nums, target):
    """★ 结尾直接给出 '<think>'，硬把 Base 模型推进思考模式"""
    return (f"{SYS}\nUser: Using the numbers {nums}, create an equation that equals "
            f"{target}. You must use ALL of the numbers, each EXACTLY once, combined with "
            f"the operations +, -, *, /. Show your work in <think> </think> tags. "
            f"Return the final equation in <answer> </answer> tags, "
            f"for example <answer> (1 + 2) / 3 </answer>.\n"
            f"Assistant: Let me solve this step by step.\n<think>")


# ══════════════════════════════════════════════════════════════
#  ③ ★ 判分器 —— 全部的「环境」
# ══════════════════════════════════════════════════════════════
ANS = re.compile(r"<answer>(.*?)</answer>", re.S)
NUM = re.compile(r"\d+")
_OP = {ast.Add: operator.add, ast.Sub: operator.sub,
       ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expr):
    """
    ★ 只允许数字和四则运算 —— 模型生成的字符串绝不能直接 eval()。
      走 AST 白名单，任何函数调用/属性访问/名字都直接拒绝。
    """
    def ev(n):
        if isinstance(n, ast.Expression):  return ev(n.body)
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)): return n.value
            raise ValueError("非数字常量")
        if isinstance(n, ast.BinOp) and type(n.op) in _OP:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Div) and b == 0: raise ZeroDivisionError
            return _OP[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OP:
            return _OP[type(n.op)](ev(n.operand))
        raise ValueError(f"不允许的节点 {type(n).__name__}")
    return ev(ast.parse(expr.strip(), mode="eval"))


def clean_expr(e):
    """
    ★ 模型的实际写法五花八门，规范化后再解析：
        "44 + 19 + 35 = 98"     → 等号后面是结果，不是表达式  ★ 最常见的坑
        "\\(44 \\times 19\\)"      → LaTeX
        "`44 + 19`"             → markdown 代码块
        "44 × 19 ÷ 3"           → Unicode 运算符
    """
    e = e.strip().strip("`$ \n")
    for a, b in (("\\times", "*"), ("\\div", "/"), ("\\cdot", "*"),
                 ("×", "*"), ("÷", "/"), ("−", "-"), ("\\(", ""), ("\\)", ""),
                 ("\\[", ""), ("\\]", ""), ("$", ""), ("`", "")):
        e = e.replace(a, b)
    if "=" in e:                       # ★ 只要等号左边那段
        e = e.split("=")[0]
    return e.strip()


def extract(text):
    """→ (规范化后的表达式或 None, 是否有 <answer> 标签)"""
    m = ANS.search(text)
    if not m:
        return None, False
    e = clean_expr(m.group(1))
    return (e if e else None), True


# ★★ 阶梯式部分分 —— 把 1 bit 的稀疏奖励变成 N bit 的密集奖励
#    稀疏版（0.1 格式 + 0.9 正确）在基线上 99% 的组全错 → 优势全 0 → 训不起来
#    密集版让同一组里的 8 条分出高下 → ★ 有优势 → 有梯度
W = {"fmt": 0.05, "parse": 0.05, "nums_frac": 0.15, "correct": 0.75}


def reward(text, nums, target, w=None):
    """
    ★ 四级阶梯，每一级都是【程序能验证的中间信号】：
        有 <answer> 标签       0.05
        表达式能解析           0.05
        ★★ 数字用对的比例      0.15   ← ★ 连续值，不是 0/1
                                        实测这一层落差最大（0.63 → 0.13）
        ★ 答案正确             0.75

    → (总分, 明细dict)
    """
    w = w or W
    d = {"fmt": 0.0, "parse": 0.0, "nums_frac": 0.0, "nums_ok": 0.0, "correct": 0.0}
    expr, has = extract(text)
    if has:
        d["fmt"] = 1.0
    if expr is not None:
        try:
            val = safe_eval(expr)
            d["parse"] = 1.0
            used = [int(x) for x in NUM.findall(expr)]

            # ★★ 部分分：用对了几个数字（多重集合交集，多用的要扣）
            #    基线上 "能解析 0.63 → 数字对 0.13" 掉 4.8 倍，是最大的落差
            #    只给 0/1 的话，「用对 2 个」和「一个都不对」同分 → 学不到方向
            cu, cn = Counter(used), Counter(nums)
            match = sum((cu & cn).values())
            extra = sum(cu.values()) - match
            d["nums_frac"] = max(0.0, (match - extra) / len(nums))

            if cu == cn:                            # ★ 恰好各用一次
                d["nums_ok"] = 1.0
                if abs(val - target) < 1e-6:
                    d["correct"] = 1.0
        except Exception:
            pass
    return sum(w.get(k, 0.0) * v for k, v in d.items()), d


# ══════════════════════════════════════════════════════════════
#  ④ 自检
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, default=0, help="生成多少道并存盘")
    ap.add_argument("--k",   type=int, default=3, help="★ 每题几个数（4 个搜索空间几千种，太难）")
    ap.add_argument("--lo",  type=int, default=1)
    ap.add_argument("--hi",  type=int, default=50,  help="★ 操作数上限（原来 99 太大）")
    ap.add_argument("--tmax", type=int, default=100, help="★ 目标值上限（原来 999 太大）")
    ap.add_argument("--out", default="data/countdown.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.gen:
        import os
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        D = generate(a.gen, k=a.k, lo=a.lo, hi=a.hi, tmax=a.tmax, seed=a.seed)
        json.dump(D, open(a.out, "w"))
        import numpy as np
        T = np.array([d["target"] for d in D])
        print(f"✅ 生成 {len(D):,} 道 → {a.out}")
        print(f"   每题 {a.k} 个数，目标值 中位 {np.median(T):.0f} "
              f"均值 {T.mean():.0f} 范围 {T.min()}~{T.max()}")
        raise SystemExit

    print("=" * 74 + "\n  ① 样例（★ 3 个数，操作数 1~50，目标 ≤100）\n" + "=" * 74)
    for d in generate(4, k=3, seed=1):
        print(f"   nums {d['nums']}   target {d['target']}")

    print("\n" + "=" * 74 + "\n  ② 提示模板（Base 模型看到的全部内容）\n" + "=" * 74)
    p = make_prompt([44, 19, 35], 98)
    for line in p.split("\n"):
        print(f"   │ {line}")
    print(f"\n   ★ 结尾是 '<think>' —— 硬把 Base 模型推进思考模式")

    print("\n" + "=" * 74 + "\n  ③ ★ 判分器测试\n" + "=" * 74)
    NUMS, TGT = [44, 19, 35], 98
    CASES = [
        ("<think>...</think><answer>44 + 19</answer>",                   "★★ 用对 2/3 → 部分分 0.10（旧版是 0）"),
        ("<think>44+19=63, +35=98</think><answer>44 + 19 + 35</answer>", "★ 完全正确"),
        ("<think>...</think><answer>(44 + 19) + 35</answer>",            "带括号，也对"),
        ("<think>...</think><answer>44 + 19 + 34</answer>",              "用了 34，不在给定数里"),
        ("<think>...</think><answer>44 + 19</answer>",                   "★ 没用完所有数 → 只拿格式+解析分"),
        ("<think>...</think><answer>44 * 19 * 35</answer>",              "数对了但结果不对"),
        ("<think>44+19+35=98</think>",                                   "★ 没有 <answer> 标签"),
        ("The answer is 44 + 19 + 35",                                   "★ 完全没格式"),
        ("<think>...</think><answer>__import__('os')</answer>",          "★ 注入尝试 → 应被拒"),
        ("<think>...</think><answer>44 + 19 + 35 = 98</answer>",         "★★ 带等号 —— 实测最常见的写法"),
        ("<think>...</think><answer>\\(44 + 19 + 35\\)</answer>",          "★ LaTeX 包裹"),
        ("<think>...</think><answer>`44 + 19 + 35`</answer>",            "★ markdown 代码块"),
        ("<think>...</think><answer>44 + 19 + 35 = 98.</answer>",        "★ 等号 + 句号"),
    ]
    print(f"   nums {NUMS}   target {TGT}\n")
    print(f"   {'模型输出':<58}{'总分':>6}  明细")
    print("   " + "─" * 84)
    for t, note in CASES:
        r, d = reward(t, NUMS, TGT)
        flags = (("F" if d["fmt"] else "·") + ("P" if d["parse"] else "·")
                 + f" 数{d['nums_frac']*100:>3.0f}% "
                 + ("C" if d["correct"] else "·"))
        print(f"   {t[:54]:<56}{r:>6.2f}  {flags}   {note}")
    print("\n   明细：F=有格式  P=能解析  N=★数字用对的【比例】  C=答案正确")
    print("   ★ 阶梯  格式 0.05 + 解析 0.05 + ★数字比例 0.15 + 正确 0.75")
    print("   ★★ 数字那一级是【连续】的：用对 2/3 拿 0.10，用对 3/3 拿 0.15")
    print("      → 模型能学到「往对的方向靠」，不是只有全对才给分")
