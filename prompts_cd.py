#!/usr/bin/env python3
"""
prompts_cd.py —— Countdown 提示词的一组改写（臂 A 第二轮：让做法不绑死在一个模板上，提示词写得不好也要对）。

    TRAIN   pid 0~9    训练用（SFT 教材换提示词 + RL rollout 随机抽）
    HELDOUT pid 10~12  评测用，训练时不见：10 啰嗦英文、11 另一种中文、12 ★ 完全不提工具（看「没说明也会用」）
每个模板：(nums, target, th, ah) → 提示词，结尾都停在 <think>（教材的 response 从这里接，判分器只认 <answer>）。
工具说明 th / 求助说明 ah 各 3 种英文措辞（中文模板一种），也可以不给。pid 0 + 措辞 0 = 原来的 R1 模板，跟 calc_tool_vllm.tool_prompt / ask_prompt 一字不差。

    render(pid, nums, target, tool=True, ask=False, max_asks=2, tool_style=0, ask_style=0)
    pick(rng, nums, target, set="train", p_tool=0.7, p_ask=0.7, max_asks=3) → (prompt, meta)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import countdown as CD

WORD = {1: "once", 2: "twice", 3: "three times", 4: "four times"}
TOOL = [
    "You have a calculator: write <calc>EXPRESSION</calc> and the exact value will be returned as <result>VALUE</result>. Use it for every calculation. ",
    "A calculator is available: put an expression inside <calc></calc> and its value comes back inside <result></result>. Use it instead of computing in your head. ",
    "Tool: <calc>EXPR</calc> returns <result>VALUE</result>. Do all arithmetic with it. ",
]
TOOL_ZH = "你有一个计算器：写 <calc>算式</calc> 会返回 <result>值</result>，所有计算都用它。"


def ASK(n):
    return [
        f"You may also ask an expert at most {WORD[n]}: write <ask>QUESTION</ask> and the expert's answer will be returned as <reply>ANSWER</reply>. Always verify the expert's equation with the calculator before answering. ",
        f"If you get stuck you can consult an expert (at most {n} times) by writing <ask>QUESTION</ask>; the answer comes back as <reply>ANSWER</reply>. Check any equation the expert gives with the calculator before you answer. ",
        f"Expert help: <ask>...</ask> (max {n} uses) returns <reply>...</reply>. Verify replies with <calc> before answering. ",
    ]


def ASK_ZH(n):
    return f"还可以向专家求助最多 {n} 次：写 <ask>问题</ask> 会返回 <reply>回答</reply>，专家给的算式要先用计算器验证再作答。"


def t0(nums, target, th, ah):                                     # 原 R1 模板（countdown.make_prompt），说明插在 Show your work 之前
    return CD.make_prompt(nums, target).replace("Show your work in", th + ah + "Show your work in")


def t1(nums, target, th, ah):
    return (f"Numbers: {nums}\nTarget: {target}\nUsing each number exactly once with +, -, *, /, write an equation that equals the target. "
            f"{th}{ah}Think inside <think> </think>, then put the final equation inside <answer> </answer>.\n<think>")


def t2(nums, target, th, ah):
    return (f"You are a careful solver of Countdown number puzzles.\nPuzzle: reach {target} from the numbers {nums}. Every number must be used exactly once; "
            f"allowed operations are +, -, *, /. {th}{ah}\nWrite your reasoning between <think> and </think>. Give only the equation between <answer> and </answer>.\n<think>")


def t3(nums, target, th, ah):
    return f"Make {target} from {nums} (each number once, + - * /). {th}{ah}Reason in <think></think>, answer in <answer></answer>.\n<think>"


def t4(nums, target, th, ah):
    return (f"Q: With the numbers {nums}, each used exactly once and combined with +, -, *, /, which equation gives {target}? {th}{ah}"
            f"Show your work inside <think> </think> and put the final equation inside <answer> </answer>.\nA: <think>")


def t5(nums, target, th, ah):
    return (f"{th}{ah}Task: combine the numbers {nums} with +, -, *, / so that the result is {target}. Use all of them, each exactly once. "
            f"Think step by step inside <think> </think> tags and return the final equation inside <answer> </answer> tags.\n<think>")


def t6(nums, target, th, ah):
    lines = ["Countdown puzzle", f"- numbers: {nums}", f"- target: {target}", "- rules: use every number exactly once; operations + - * / and parentheses"]
    if th: lines.append("- calculator: " + th.strip())
    if ah: lines.append("- expert: " + ah.strip())
    lines.append("- output: reasoning in <think> </think>, final equation in <answer> </answer>")
    return "\n".join(lines) + "\n<think>"


def t7(nums, target, th, ah):                                     # 中文
    return (f"用 {nums} 这几个数各用一次，通过加减乘除（可以加括号）得到 {target}。{th}{ah}"
            f"思考过程写在 <think> </think> 里，最后的算式写在 <answer> </answer> 里。\n<think>")


def t8(nums, target, th, ah):
    return (f"User: I have the numbers {nums}. Can you find an equation that equals {target}? Use each number exactly once, only + - * /. {th}{ah}"
            f"Please think in <think> </think> and then give the equation in <answer> </answer>.\nAssistant: Sure. <think>")


def t9(nums, target, th, ah):
    tools = ("TOOLS: " + (th + ah).strip() + "\n") if (th or ah) else ""
    return (f"INPUT: nums={nums} target={target}\nRULES: each number exactly once; ops + - * /; parentheses ok\n{tools}"
            f"OUTPUT: <think>reasoning</think><answer>equation</answer>\n<think>")


def h10(nums, target, th, ah):
    return (f"Here is a small arithmetic challenge. You are given the numbers {nums}. By adding, subtracting, multiplying or dividing them, using each one exactly one time, "
            f"produce the value {target}. {th}{ah}Work through it inside <think> </think> tags and write the final expression inside <answer> </answer> tags.\n<think>")


def h11(nums, target, th, ah):                                    # 中文，另一种写法
    return (f"题目：给定数字 {nums}，目标 {target}。每个数字恰好用一次，运算只能用 + - * /。{th}{ah}"
            f"请把推理放在 <think> </think> 中，把最终算式放在 <answer> </answer> 中。\n<think>")


def h12(nums, target, th, ah):                                    # ★ 完全不提工具
    return (f"Numbers {nums}, target {target}. Each number once, + - * / allowed. Put your thinking in <think> </think> and the equation in <answer> </answer>.\n<think>")


TEMPLATES = [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, h10, h11, h12]
ZH = {7, 11}
NO_HINT = {12}
SETS = {"train": list(range(10)), "heldout": [10, 11, 12], "all": list(range(13)), "r1": [0]}


def render(pid, nums, target, tool=True, ask=False, max_asks=2, tool_style=0, ask_style=0):
    if pid in ZH:
        th = TOOL_ZH if tool else ""; ah = ASK_ZH(max_asks) if ask else ""
    elif pid in NO_HINT:
        th = ah = ""
    else:
        th = TOOL[tool_style % 3] if tool else ""; ah = ASK(max_asks)[ask_style % 3] if ask else ""
    return TEMPLATES[pid](nums, target, th, ah)


def pick(rng, nums, target, set="train", p_tool=0.7, p_ask=0.7, max_asks=3, force_ask=None):
    """随机抽模板 + 随机决定给不给说明、用哪种措辞 → (prompt, meta)"""
    pid = rng.choice(SETS[set])
    tool = rng.random() < p_tool
    ask = (rng.random() < p_ask) if force_ask is None else force_ask
    ts, as_ = rng.randrange(3), rng.randrange(3)
    return render(pid, nums, target, tool=tool, ask=ask, max_asks=max_asks, tool_style=ts, ask_style=as_), \
        dict(pid=pid, tool_hint=tool, ask_hint=ask, tool_style=ts, ask_style=as_)


if __name__ == "__main__":
    for pid in SETS["all"]:
        print(f"═══ pid {pid} ═══"); print(render(pid, [5, 7, 14, 33], 43, tool=True, ask=True, max_asks=3, tool_style=pid % 3, ask_style=(pid // 3) % 3)); print()
