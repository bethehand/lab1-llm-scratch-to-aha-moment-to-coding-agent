#!/usr/bin/env python3
"""
chat_countdown.py —— 部署推理：给模型出 Countdown 题，肉眼看它有没有「顿悟」
                     （试错 → 标 too high / too low → 往对的方向改 → 收敛）

    python3 chat_countdown.py --ckpt out_mix4/ckpt_step300.pt              # 顿悟模型，内置 8 题
    python3 chat_countdown.py                                               # Base，同 8 题
    python3 chat_countdown.py --both --ckpt out_mix4/ckpt_step300.pt        # ★ 两个模型并排，同一题
    python3 chat_countdown.py --ckpt ... --nums 6 12 18 42 --target 30      # 单题
    python3 chat_countdown.py --ckpt ... -i                                  # 交互：输入 "6 12 18 42 = 30"
    python3 chat_countdown.py --ckpt ... --temp 1 -s 4                      # 温度 1 抽 4 条，看多样性
    python3 chat_countdown.py --list                                        # 只看内置题和参考解，不载模型
    python3 chat_countdown.py --both --ckpt ... --template raw -i           # ★ 不加任何提示词，你打什么模型收什么
    python3 chat_countdown.py --both --ckpt ... --template plain            # 中性模板：只有题目 + <answer> 要求

三种模板（--template）：
    r1     训练时用的那段（系统说明 + User/Assistant + 结尾 <think>），默认。RL 学到的东西绑在这上面
    plain  中性模板：Numbers/Target 两行 + 一句要求。跟 crosstask 的中性模板同用途，测离了训练模板还在不在
    raw    什么都不加。交互模式下你打的那行原样送进模型；内置题模式下只送一句「Using the numbers …」

每条输出后面一行判读：对/错、得分、搜索词几个、标注几条算对几条、长度，
以及三个红旗：「说了 perfect 但答案错」（幻觉验证）、「幻觉续写 User:」、「撞顶」。
"""
import os, re, sys, argparse
from fractions import Fraction

# ══════════════════════════════════════════════════════════════
#  内置测试题：从「Base 也能碰到」到「mix4 也解不了」
# ══════════════════════════════════════════════════════════════
DEMO = [
    ([1, 13, 42],    56, "热身：三数纯加。Base 64 抽里也能碰到；顿悟模型一次该对"),
    ([3, 7, 25],     46, "三数需 ×：3×7+25。看它加减不通后换不换运算符"),
    ([4, 8, 20],     22, "三数需 ÷：8/4+20。除法更少见"),
    ([8, 9, 10],     71, "三数需 ×，且第一反应（全加=27）差很远。看方向标注"),
    ([6, 12, 18, 42], 30, "四数纯加减：42+12−6−18。800 条里它没搜到过 —— 8 种符号形态它试不全"),
    ([2, 3, 5, 7],   31, "四数混合形态：3×7+5×2。要两个乘积相加"),
    ([5, 7, 14, 33], 43, "四数嵌套：33+14×5/7。硬尾巴那 26 道的样子，预计解不了"),
    ([9, 11, 4, 25], 50, "四数嵌套除：25×4/(11−9)。同上"),
]

# ══════════════════════════════════════════════════════════════
#  穷举求解器：给参考解，判题型（+− 可解 / 需 ×÷ / 无解）
# ══════════════════════════════════════════════════════════════
def solve(nums, target, ops="+-*/"):
    tgt = Fraction(target)

    def rec(items):
        if len(items) == 1:
            return items[0][1] if items[0][0] == tgt else None
        n = len(items)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                (a, ea), (b, eb) = items[i], items[j]
                rest = [items[k] for k in range(n) if k not in (i, j)]
                cands = []
                if i < j and "+" in ops: cands.append((a + b, f"({ea} + {eb})"))
                if i < j and "*" in ops: cands.append((a * b, f"({ea} * {eb})"))
                if "-" in ops:           cands.append((a - b, f"({ea} - {eb})"))
                if "/" in ops and b != 0: cands.append((a / b, f"({ea} / {eb})"))
                for v, e in cands:
                    r = rec(rest + [(v, e)])
                    if r is not None:
                        return r
        return None

    r = rec([(Fraction(x), str(x)) for x in nums])
    return r[1:-1] if r and r.startswith("(") and r.endswith(")") else r


def classify(nums, target):
    pm = solve(nums, target, "+-")
    if pm:
        return "+− 可解", pm
    full = solve(nums, target)
    return ("需 ×÷", full) if full else ("无解", None)


# ══════════════════════════════════════════════════════════════
#  探针（跟 grpo_mix_mp.py / analyze_search.py 同一套正则）
# ══════════════════════════════════════════════════════════════
SEARCH = re.compile(
    r"\b(too (?:high|low|big|small|large)|not (?:equal|right|correct)|doesn'?t (?:work|equal)"
    r"|does not (?:work|equal)|perfect|close|try (?:another|again|different|a different|the next)"
    r"|let'?s try|next,? try|another (?:combination|approach|way|try)|nope|wrong|incorrect)\b", re.I)
ANNOT = re.compile(r"((?:\(|\d)[\d\s\+\-\*/\(\)]*?)\s*=\s*(-?\d+(?:\.\d+)?)\s*\(?\s*(too (?:high|low|big|small|large))", re.I)   # ★ 算式必须以数字或 ( 开头：否则上一行标注的 ")\n" 会被吞进算式，eval 失败被跳过 → 连续标注只数到第一条
CLAIM = re.compile(r"\b(perfect|works|correct|matches)\b", re.I)
CONT = "\nUser:"


def annot_stats(CD, txt, target):
    n = ok = 0
    for m in ANNOT.finditer(txt):
        try:
            real = CD.safe_eval(CD.clean_expr(m.group(1)))
        except Exception:
            continue
        n += 1
        said_high = any(w in m.group(3).lower() for w in ("high", "big", "large"))
        if abs(real - float(m.group(2))) < 1e-6 and said_high == (real > target):
            ok += 1
    return n, ok


def judge(CD, tok, txt, nums, target, max_new):
    cut = txt.find(CONT)
    cont = cut >= 0
    eff = txt[:cut] if cont else txt
    score, d = CD.reward(eff, nums, target)
    n_an, ok_an = annot_stats(CD, eff, target)
    ntok = len(tok(eff).input_ids)
    calcs = re.findall(r"<calc>(.*?)</calc>", eff, re.S)
    calc_ok = 0
    for e in calcs:
        try:
            CD.safe_eval(CD.clean_expr(e)); calc_ok += 1
        except Exception:
            pass
    return dict(eff=eff, score=score, correct=d["correct"] > 0, nums_ok=d["nums_ok"] > 0,
                nsrch=len(SEARCH.findall(eff)), n_an=n_an, ok_an=ok_an,
                claim=bool(CLAIM.search(eff)), cont=cont, ntok=ntok, cut=ntok >= max_new - 2,
                n_calc=len(calcs), calc_ok=calc_ok,
                fake_result=("⚠ harness: 模型自己写 <result>" in eff) if USE_TOOL else ("<result>" in eff))   # 工具模式下 <result> 是 harness 注入的，只认 harness 的假结果标记


def verdict(j):
    head = "✓ 对" if j["correct"] else ("✗ 错" if j["nums_ok"] else "✗ 错（数字没用对）")
    prec = f"{j['ok_an']}/{j['n_an']}" if j["n_an"] else "—"
    flags = []
    if j["claim"] and not j["correct"]:
        flags.append("⚠ 说了 perfect/works 但答案错 → 幻觉验证")
    if j["cont"]:
        flags.append("⚠ 幻觉续写 User:")
    if j["cut"]:
        flags.append("⚠ 撞顶")
    line = (f"  {head}   得分 {j['score']:.2f} │ 搜索词 {j['nsrch']} │ 标注 {j['n_an']} 条，算对且方向对 {prec}"
            f" │ 长度 {j['ntok']} tok")
    if j.get("n_calc") or j.get("fake_result"):
        line += f" │ calc {j['n_calc']} 次（式子合法 {j['calc_ok']}）" + ("  ⚠ 自己写了 <result>" if j["fake_result"] else "")
    return line + ("\n  " + "  ".join(flags) if flags else "")


# ══════════════════════════════════════════════════════════════
#  模型
# ══════════════════════════════════════════════════════════════
ENGINE, HF_DIR, HF_BASE_DIR, GPU_MEM = "hf", None, None, 0.6

def load_model(model_name, ckpt, dev):
    if ENGINE == "vllm":                                   # ★ vLLM：权重从 HF 目录装（--hf-dir / --hf-base-dir），不载 HF 模型
        from transformers import AutoTokenizer
        import calc_tool_vllm as calc_tool
        d = HF_DIR if ckpt else (HF_BASE_DIR or HF_DIR)
        assert d, "--engine vllm 要 --hf-dir（--both 时 Base 还要 --hf-base-dir）"
        tok = AutoTokenizer.from_pretrained(model_name); tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        print(f"  ★ vLLM ← {d}", flush=True)
        return tok, calc_tool.make_backend("vllm", tok=tok, hf_dir=d, gpu_mem=GPU_MEM)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    if ckpt:
        c = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(c["model"])
        print(f"  ★ {ckpt}（step {c.get('step', '?')}）", flush=True)
        del c
    model.to(dev).eval()
    return tok, model


USE_TOOL = False

def generate(tok, model, prompts, temp, max_new):
    import calc_tool_vllm as calc_tool
    if USE_TOOL:
        res = calc_tool.generate_with_tools(model, tok, prompts, max_new=max_new, temp=temp, gen_bs=len(prompts))
        return [o["txt"] + ("\n⚠ harness: 模型自己写 <result>，已停" if o["fake"] else "") for o in res]
    be = model if not hasattr(model, "parameters") else calc_tool.HFBackend(model, tok, gen_bs=len(prompts))
    return [o["txt"] + (CONT if o["cont"] else "") for o in calc_tool.plain_generate(be, tok, prompts, max_new=max_new, temp=temp)]


# ══════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════
def show(name, CD, tok, texts, nums, target, max_new, tally, prefix="<think>"):
    print(f"\n  ┌── {name} ──")
    for si, txt in enumerate(texts):
        if len(texts) > 1:
            print(f"  │ 第 {si+1} 条")
        if nums is None:
            cut = txt.find(CONT)
            for line in (txt[:cut] if cut >= 0 else txt).rstrip().split("\n"):
                print("  │ " + line)
            continue
        j = judge(CD, tok, txt, nums, target, max_new)
        body = prefix + j["eff"]
        for line in body.rstrip().split("\n"):
            print("  │ " + line)
        print("  │")
        print(verdict(j).replace("\n  ", "\n  │ ", 1).replace("  ✓", "  │ ✓", 1).replace("  ✗", "  │ ✗", 1))
        t = tally.setdefault(name, dict(n=0, ok=0, srch=0, an=0, an_ok=0, halluc=0, tok=0, calc_any=0, calc_n=0, calc_ok=0, fake=0))
        t["n"] += 1; t["ok"] += j["correct"]; t["srch"] += j["nsrch"]
        t["calc_any"] += (j["n_calc"] > 0); t["calc_n"] += j["n_calc"]; t["calc_ok"] += j["calc_ok"]; t["fake"] += j["fake_result"]
        t["an"] += j["n_an"]; t["an_ok"] += j["ok_an"]; t["tok"] += j["ntok"]
        t["halluc"] += (j["claim"] and not j["correct"])
    print("  └" + "─" * 60)


def run_problem(models, CD, nums, target, note, temp, samples, max_new, tally,
                template="r1", raw_text=None, show_prompt=False):
    print("\n" + "=" * 74)
    if nums is not None:
        kind, ref = classify(nums, target)
        tag = "（从你粘贴的文字里抓的，核对一下）" if raw_text is not None else ""
        print(f"  题  nums {nums}   target {target}      题型：{kind}    参考解：{ref}  {tag}")
    else:
        print("  题  （没解析出数字和目标，只看输出，不判分）")
    if note:
        print(f"  {note}")
    print("=" * 74)
    prompt, prefix = build_prompt(CD, template, nums, target, raw_text)
    if show_prompt or template != "r1":
        print(f"  ── 送进模型的全文（模板 {template}）──")
        for line in prompt.split("\n"):
            print("  ┆ " + line)
    for name, tok, model in models:
        texts = generate(tok, model, [prompt] * samples, temp, max_new)
        show(name, CD, tok, texts, nums, target, max_new, tally, prefix)


def plain_prompt(nums, target):
    return (f"Numbers: {nums}\nTarget: {target}\n"
            f"Using each number exactly once with +, -, *, /, write an equation that equals the target. "
            f"Put the final equation in <answer> </answer> tags.\n")


TOOL_HINT = ("You have a calculator: write <calc>EXPRESSION</calc> and the exact value will be returned as "
             "<result>VALUE</result>. For example: <calc>7 + 5 - 3</calc><result>9</result> (too low). "
             "Use it for every calculation. ")


def build_prompt(CD, template, nums, target, raw_text=None):
    if template == "r1":
        return CD.make_prompt(nums, target), "<think>"
    if template == "tooldemo":                      # ★ 第 0 步：不 SFT，只靠提示词，看它写不写得出 <calc>
        pr = CD.make_prompt(nums, target)
        return pr.replace("Show your work in", TOOL_HINT + "Show your work in"), "<think>"
    if template == "tool":                          # ★ 工具版模型：跟 SFT 教材同一段提示词，生成走 harness
        import calc_tool_vllm as calc_tool
        return calc_tool.tool_prompt(nums, target), "<think>"
    if template == "plain":
        return plain_prompt(nums, target), ""
    if raw_text is not None:
        return raw_text, ""
    return f"Using the numbers {nums}, create an equation that equals {target}.", ""


def read_block(prompt):
    """
    读一段输入，可多行。规则：
      · 粘贴进来的多行一次收齐（粘贴时后面的行会立刻到，用 select 探测）
      · 在【空行】上按回车 = 发送。粘贴内容中间夹的空行不会触发（因为后面还有数据在等）
      · 第一行是 q / quit / exit = 退出；Ctrl-D 退出；Ctrl-C 只清空当前这段，不退出
    """
    import select
    print(prompt, end="", flush=True)
    lines = []

    def pending():
        try:
            return bool(select.select([sys.stdin], [], [], 0.15)[0])
        except (OSError, ValueError):          # Windows 的 stdin 不支持 select → 当没有
            return False

    while True:
        try:
            line = sys.stdin.readline()
        except KeyboardInterrupt:
            print("\n  （已清空这段，重新输入；q 退出）")
            lines = []
            print(prompt, end="", flush=True)
            continue
        if line == "":                          # Ctrl-D
            return "\n".join(lines) if lines else None
        line = line.rstrip("\n").rstrip("\r")
        if not lines and line.strip().lower() in ("q", "quit", "exit"):
            return None
        more = pending()
        if line.strip() == "":
            if not lines:                       # 开头的空行：忽略，不退出
                continue
            if more:                            # 粘贴中间的空行：保留
                lines.append("")
                continue
            return "\n".join(lines)            # 空行且后面没数据 → 发送
        lines.append(line)


def parse_line(s):
    """
    '6 12 18 42 = 30' 或 '6,12,18,42 30' → (nums, target)
    ★ 粘贴的是整段题面时，优先认 'numbers [3, 7, 25]' 和 'equals 46' 这两个位置 ——
      否则模板里的例子 '(1 + 2) / 3' 也会被当成题目里的数，判分全错
    """
    m_n = re.search(r"numbers?\s*[\[\(]\s*([\d,\s]+?)\s*[\]\)]", s, re.I)
    m_t = re.search(r"equals?\s*(?:to\s*)?(-?\d+)", s, re.I)
    if m_n and m_t:
        nums = [int(x) for x in re.findall(r"\d+", m_n.group(1))]
        if len(nums) >= 2:
            return nums, int(m_t.group(1))
    xs = [int(x) for x in re.findall(r"-?\d+", s)]
    if len(xs) < 3:
        return None
    return xs[:-1], xs[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--ckpt", default=None, help="训好的 checkpoint；不给 = Base")
    ap.add_argument("--both", action="store_true", help="★ Base 和 --ckpt 一起载入，每题并排出")
    ap.add_argument("--nums", type=int, nargs="+")
    ap.add_argument("--target", type=int)
    ap.add_argument("-i", "--interactive", action="store_true")
    ap.add_argument("--temp", type=float, default=0.0, help="0 = 贪心（默认，可复现）")
    ap.add_argument("-s", "--samples", type=int, default=1, help="温度 > 0 时每题抽几条")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--list", action="store_true", help="只打印内置题 + 参考解")
    ap.add_argument("--template", choices=["r1", "plain", "raw", "tooldemo", "tool"], default="r1",
                    help="r1 = 训练模板（默认）；plain = 中性模板；raw = 什么都不加")
    ap.add_argument("--show-prompt", action="store_true", help="打印送进模型的全文（plain/raw 模式默认打印）")
    ap.add_argument("--engine", choices=["hf", "vllm"], default="hf", help="★ vllm = vLLM 引擎（要 --hf-dir；--both 再加 --hf-base-dir）")
    ap.add_argument("--hf-dir", default=None); ap.add_argument("--hf-base-dir", default=None)
    ap.add_argument("--gpu-mem", type=float, default=0.6)
    args = ap.parse_args()

    if args.list:
        for nums, target, note in DEMO:
            kind, ref = classify(nums, target)
            print(f"  {str(nums):22} → {target:4}   {kind:8}  {ref}\n      {note}")
        return

    if args.both and not args.ckpt:
        raise SystemExit("--both 需要 --ckpt")
    global USE_TOOL, ENGINE, HF_DIR, HF_BASE_DIR, GPU_MEM
    USE_TOOL = args.template == "tool"
    ENGINE, HF_DIR, HF_BASE_DIR, GPU_MEM = args.engine, args.hf_dir, args.hf_base_dir, args.gpu_mem
    samples = args.samples if args.temp > 0 else 1

    import countdown as CD
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    models = []
    if args.both or not args.ckpt:
        print(f"载入 Base {args.model} …", flush=True)
        tok, m = load_model(args.model, None, dev)
        models.append(("Base " + args.model.split("/")[-1], tok, m))
    if args.ckpt:
        print(f"载入 {args.ckpt} …", flush=True)
        tok, m = load_model(args.model, args.ckpt, dev)
        models.append((os.path.basename(os.path.dirname(args.ckpt)) + "/" + os.path.basename(args.ckpt), tok, m))
    print(f"  显存 {torch.cuda.memory_allocated()/1024**3:.1f} GB   温度 {args.temp}   每题 {samples} 条\n")

    tally = {}
    if args.nums and args.target is not None:
        run_problem(models, CD, args.nums, args.target, "", args.temp, samples, args.max_new, tally,
                    args.template, None, args.show_prompt)
    elif args.interactive:
        if args.template == "raw":
            print("★ raw 模式：你打的这行原样送进模型，不加任何东西。判分靠从这行里抓数字（最后一个当目标），抓不到就只看输出")
        print("输入或粘贴一道题（可多行）。★ 在空行上按回车发送。例：  6 12 18 42 = 30   最后一个数是目标")
        print("q 回车退出；Ctrl-C 清空重输，不退出")
        while True:
            s = read_block("\n题 > ")
            if s is None:
                break
            s = s.strip()
            if not s:
                continue
            p = parse_line(s)
            if not p and args.template != "raw":
                print("  至少两个数 + 一个目标，例如  3 7 25 = 46")
                continue
            nums, target = p if p else (None, None)
            run_problem(models, CD, nums, target, "", args.temp, samples, args.max_new, tally,
                        args.template, s if args.template == "raw" else None, args.show_prompt)
    else:
        for nums, target, note in DEMO:
            run_problem(models, CD, nums, target, note, args.temp, samples, args.max_new, tally,
                        args.template, None, args.show_prompt)

    if tally:
        print("\n" + "=" * 74)
        print("  汇总")
        print(f"  {'模型':34} {'解对':>6} {'搜索词/条':>9} {'标注精度':>9} {'幻觉验证':>8} {'长度':>6}")
        for name, t in tally.items():
            prec = f"{t['an_ok']}/{t['an']}" if t["an"] else "—"
            print(f"  {name:34} {t['ok']:>3}/{t['n']:<3} {t['srch']/t['n']:>9.1f} {prec:>9} "
                  f"{t['halluc']:>8} {t['tok']/t['n']:>6.0f}")
        for name, t in tally.items():
            if t["calc_n"] or t["fake"]:
                print(f"  ★ 第 0 步读数 {name}：写出 <calc> 的轨迹 {t['calc_any']}/{t['n']}   共 {t['calc_n']} 次，式子合法 {t['calc_ok']}"
                      f"   自己编 <result> 的轨迹 {t['fake']}/{t['n']}")
        print("=" * 74)


if __name__ == "__main__":
    main()
