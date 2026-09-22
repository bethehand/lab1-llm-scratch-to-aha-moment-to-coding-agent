#!/usr/bin/env python3
"""
guess_env.py —— E 猜数字环境：秘密数 s ∈ [1, N]，模型写 <guess>k</guess>，环境回 <obs>higher|lower|correct|invalid</obs>，最多 T 轮，猜中后写 <answer>k</answer>。
奖励（v3 精神，只奖不罚）：answer == 秘密数 → 1 − step_cost × (轮数 − 1)；否则 0。假观测由训练器罚。
提示词：8 个训练模板 + 2 个留出（英 / 中 / 清单 / 问答 / 规格 / 聊天 …），全部以 <think> 收尾。
"""
import re, random

GUESS_RE = re.compile(r"<guess>(.*?)</guess>", re.S)
OBS_RE = re.compile(r"<obs>(.*?)</obs>", re.S)
ANS_RE = re.compile(r"<answer>(.*?)</answer>", re.S)
PAIR_RE = re.compile(r"<guess>(.*?)</guess>\s*<obs>(.*?)</obs>", re.S)


class GuessEnv:
    stops = ["</guess>"]
    fake_tags = ["<obs>"]

    def __init__(self, N=100, T=10, step_cost=0.05):
        self.N, self.T, self.step_cost = N, T, step_cost

    def reset(self, secret):
        return dict(secret=int(secret), turns=0, guesses=[], obs=[], found=False, invalid=0, repeat=0)

    def step(self, st, full):
        if st["found"]:                                                # ★ 猜中之后还在猜（E1 评测里见过 26 轮的）：直接结束，没写 <answer> 就 0 分 → RL 学「中了就停」
            return None, True
        raw = GUESS_RE.findall(full)[-1].strip() if GUESS_RE.findall(full) else ""
        st["turns"] += 1
        try:
            k = int(raw)
        except ValueError:
            k = None
        if k is None or not (1 <= k <= self.N):
            o = "invalid"; st["invalid"] += 1
        elif k in st["guesses"]:
            o = "invalid"; st["repeat"] += 1
        elif k == st["secret"]:
            o = "correct"; st["found"] = True
        else:
            o = "higher" if st["secret"] > k else "lower"
        st["guesses"].append(k); st["obs"].append(o)
        done = (not st["found"]) and st["turns"] >= self.T           # 轮数用完还没中：结束；猜中了让它接着写 <answer>
        return f"<obs>{o}</obs>", done

    def score(self, st, txt):
        m = ANS_RE.findall(txt)
        ans = m[-1].strip() if m else ""
        try:
            correct = st["found"] and int(ans) == st["secret"]        # ★ 环境说过 correct 才算：瞎答撞中（1/N）不给分，否则 RL 会放大「不猜直接答」
        except ValueError:
            correct = False
        r = (1.0 - self.step_cost * (max(1, st["turns"]) - 1)) if correct else 0.0
        return max(0.0, r), dict(correct=float(correct), found=st["found"], turns=st["turns"], invalid=st["invalid"], repeat=st["repeat"],
                                fmt=float(len(st["guesses"]) > 0), has_answer=float(bool(m)))


# ── 行为读数（从轨迹的 (guess, obs) 序列算）──
def behavior(txt, N=100):
    """→ dict(dir_ok, dir_n, in_range_ok, in_range_n, bisect_ok, bisect_n)
    dir：上一观测说 higher 下一猜要更大 / lower 要更小；in_range：下一猜落在所有观测圈出的 [lo, hi] 里；bisect：下一猜离 [lo, hi] 中点 ≤ max(1, 5% 区间)"""
    pairs = [(g, o) for g, o in PAIR_RE.findall(txt)]
    lo, hi = 1, N
    d = dict(dir_ok=0, dir_n=0, in_range_ok=0, in_range_n=0, bisect_ok=0, bisect_n=0)
    prev = None
    for g, o in pairs:
        try:
            k = int(g.strip())
        except ValueError:
            prev = None; continue
        if prev is not None:
            pk, po = prev
            if po in ("higher", "lower"):
                d["dir_n"] += 1; d["dir_ok"] += (k > pk) if po == "higher" else (k < pk)
        if lo <= hi:
            d["in_range_n"] += 1; d["in_range_ok"] += (lo <= k <= hi)
            d["bisect_n"] += 1; d["bisect_ok"] += abs(k - (lo + hi) / 2) <= max(1, 0.05 * (hi - lo))
        o = o.strip()
        if o == "higher": lo = max(lo, k + 1)
        elif o == "lower": hi = min(hi, k - 1)
        prev = (k, o)
    return d


# ── 提示词 ──
def t0(N, T): return (f"Let's play a guessing game. I am thinking of a whole number between 1 and {N}. Make a guess by writing <guess>NUMBER</guess>; "
                      f"I will reply with <obs>higher</obs> if my number is higher than your guess, <obs>lower</obs> if it is lower, or <obs>correct</obs> when you get it. "
                      f"You have at most {T} guesses. When you get it, write the number in <answer> </answer> tags. Think inside <think> </think>.\n<think>")
def t1(N, T): return (f"Number guessing: secret integer in [1, {N}]. Guess with <guess>N</guess>; feedback comes back as <obs>higher</obs>, <obs>lower</obs> or <obs>correct</obs>. "
                      f"Max {T} guesses. Reason in <think></think>, final number in <answer></answer>.\n<think>")
def t2(N, T): return (f"You are playing higher-or-lower. The hidden number is between 1 and {N} inclusive. Each turn, write <guess>k</guess> and read the reply: "
                      f"'higher' means the hidden number is larger than k, 'lower' means smaller, 'correct' means found. Use at most {T} turns, then put the found number in <answer> </answer>.\n<think>")
def t3(N, T): return (f"我心里想了一个 1 到 {N} 之间的整数。你写 <guess>数字</guess> 来猜，我会回 <obs>higher</obs>（比你猜的大）、<obs>lower</obs>（比你猜的小）或 <obs>correct</obs>（猜中）。"
                      f"最多猜 {T} 次。思考写在 <think> </think> 里，猜中后把数字写在 <answer> </answer> 里。\n<think>")
def t4(N, T): return (f"Guessing game\n- range: 1 to {N}\n- action: <guess>k</guess>\n- feedback: <obs>higher</obs> / <obs>lower</obs> / <obs>correct</obs>\n"
                      f"- limit: {T} guesses\n- output: reasoning in <think> </think>, the number in <answer> </answer>\n<think>")
def t5(N, T): return (f"Q: A secret number is hidden between 1 and {N}. You can ask by guessing: write <guess>k</guess> and you will be told <obs>higher</obs>, "
                      f"<obs>lower</obs> or <obs>correct</obs>. Find it in at most {T} guesses and report it in <answer> </answer>.\nA: <think>")
def t6(N, T): return (f"TASK: find hidden integer x, 1 <= x <= {N}\nACTION: <guess>k</guess>\nOBS: <obs>higher</obs> (x > k) | <obs>lower</obs> (x < k) | <obs>correct</obs>\n"
                      f"LIMIT: {T} guesses\nOUTPUT: <think>...</think><answer>x</answer>\n<think>")
def t7(N, T): return (f"User: I'm thinking of a number from 1 to {N}. Try to guess it! Write <guess>k</guess> and I'll answer <obs>higher</obs>, <obs>lower</obs> or <obs>correct</obs>. "
                      f"You get {T} tries. When you find it, put it in <answer> </answer>.\nAssistant: Sure. <think>")
def h8(N, T): return (f"Here is a small game. I have secretly chosen an integer somewhere from 1 up to {N}. On each turn you may propose one integer by writing <guess>k</guess>; "
                      f"in return you will read <obs>higher</obs> when my number exceeds your proposal, <obs>lower</obs> when it falls short, and <obs>correct</obs> the moment you hit it. "
                      f"Please find my number within {T} proposals, thinking inside <think> </think>, and finally write it inside <answer> </answer>.\n<think>")
def h9(N, T): return (f"猜数游戏：秘密数字在 1 到 {N} 之间。每次写 <guess>k</guess>，会得到 <obs>higher</obs>、<obs>lower</obs> 或 <obs>correct</obs>。"
                      f"最多 {T} 次，找到后把数字放进 <answer> </answer>。\n<think>")

TEMPLATES = [t0, t1, t2, t3, t4, t5, t6, t7, h8, h9]
SETS = {"train": list(range(8)), "heldout": [8, 9], "all": list(range(10)), "t0": [0]}


def render(pid, N=100, T=10):
    return TEMPLATES[pid](N, T)


def make_problems(n, N=100, seed=0, lo=1):
    """n 个秘密数（可重复），seed 定死；idx 用来固定模板"""
    rng = random.Random(seed)
    return [dict(secret=rng.randint(lo, N), idx=i) for i in range(n)]
