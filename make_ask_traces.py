#!/usr/bin/env python3
"""
make_ask_traces.py —— 臂 A 的教材：教模型「自己先搜（第 0、1 层，跟老师 v1 一字不差）、搜完没中才限量求助、拿回来先验证再交卷」。要先 export DEEPSEEK_API_KEY。

    export DEEPSEEK_API_KEY=sk-...
    python3 make_ask_traces.py --counts data/countdown4_mixed_counts.json --data data/countdown4.json \\
        --n-ask 800 --n-plain 800 --out sft_cd4_ask.jsonl
    python3 make_ask_traces.py --probs "5,7,14,33:43" "1,2,3:6" --out /tmp/x.jsonl        # 指定几道题（本地冒烟）

选题：硬题 = 筛池 counts 里 c ≤ --hard-c（默认 2）的题抽 --n-ask 道；简单题 = c ≥ --easy-c（默认 6）的题抽 --n-plain 道
每道题都先走老师 v1 的第 0 层（符号形态 8 行）和第 1 层（两数乘积/商 + 其余做 ±，CAP 40 行），格式跟 sft_cd4_tool.jsonl 相同：
    第 0/1 层命中 → 普通轨迹（kind plain_l0 / plain_l1），不问 —— 教「自己能搜到的不问」
    没命中 →「None of these work; this one is hard, let me ask the expert.」→ <ask>问题</ask><reply>EQUATION: …</reply>（真调 reasoner，缓存、并发）
            → Let me verify: <calc>式子</calc><result>V</result>
            专家对 → (perfect match, verified) → answer；错 → (not T, the expert is wrong) → 再问一次（告诉它上次的式子错了）→ 再验
            专家没答（error）→ (the expert did not answer) → 再问一次；两次都不行 → 丢弃（记进统计）。最终答案过判分器才落盘
规则的落点：<ask> 只出现在「自己的第 0、1 层都搜完没中」这一处 —— 决策点无歧义，简单题永远不问；第 2 层被求助取代（臂 T/S 的 RL 本来就把第 2 层剪没了）
两种轨迹的 prompt 都是 calc_tool_vllm.ask_prompt（TOOL_HINT + ASK_HINT）。SFT 时 <result> 和 <reply> 段 label −100（sft_qwen.RES_RE）。
"""
import os, sys, json, random, argparse, time
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import countdown as CD
import calc_tool_vllm as T
import enum_traces as ET
import deepseek_tool as DS

DS.get_key()                                               # 没密钥就在这儿停
ET.TOOL = True                                             # 值写成 <calc>…</calc><result>…</result>

ap = argparse.ArgumentParser()
ap.add_argument("--counts", default=None, help="screen_pool 的 *_counts.json（每题 8 次里对的条数 c）")
ap.add_argument("--hard-c", type=int, default=2, help="c ≤ 这个数算硬题")
ap.add_argument("--easy-c", type=int, default=6, help="c ≥ 这个数算简单题")
ap.add_argument("--data", default="data/countdown4.json")
ap.add_argument("--probs", nargs="*", default=None, help='直接给题："5,7,14,33:43"')
ap.add_argument("--n-ask", type=int, default=800, help="硬题抽几道")
ap.add_argument("--n-plain", type=int, default=800, help="简单题抽几道")
ap.add_argument("--model", default="deepseek-reasoner")
ap.add_argument("--max-tokens", type=int, default=8000, help="专家的推理+正文上限；4000 偶尔截断（正文空 → error）")
ap.add_argument("--wrong-frac", type=float, default=0.0,
                help="★ 抽这个比例的求助题先问弱专家（--wrong-model），它答错就把错答案当第一次回复写进教材 → 验证抓错 → 再问强专家。"
                     "强专家答了就几乎不错（GS01 第一轮 646/646），不造就没有「the expert is wrong」的样本")
ap.add_argument("--wrong-model", default="deepseek-chat", help="弱专家（非思考模式，硬题 1/10）")
ap.add_argument("--workers", type=int, default=8, help="并发问专家的线程数（一次 3~20 s，串行 800 道要两小时）")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
args = ap.parse_args()
rng = random.Random(args.seed)

# ── 选题 ──
if args.probs:
    probs = [(list(map(int, p.split(":")[0].split(","))), int(p.split(":")[1]), -1, "?", rng.random() < args.wrong_frac) for p in args.probs]
else:
    D = json.load(open(args.data)); cnt = json.load(open(args.counts))["counts"]
    hard = sorted(int(i) for i, c in cnt.items() if c <= args.hard_c); rng.shuffle(hard)
    easy = sorted(int(i) for i, c in cnt.items() if c >= args.easy_c); rng.shuffle(easy)
    probs = ([(D[i]["nums"], D[i]["target"], i, "hard", rng.random() < args.wrong_frac) for i in hard[: args.n_ask]]
             + [(D[i]["nums"], D[i]["target"], i, "easy", rng.random() < args.wrong_frac) for i in easy[: args.n_plain]])
    print(f"池里 c ≤ {args.hard_c} 的硬题 {len(hard)} 道、c ≥ {args.easy_c} 的简单题 {len(easy)} 道；"
          f"抽硬 {min(len(hard), args.n_ask)} + 简单 {min(len(easy), args.n_plain)}")
print(f"共 {len(probs)} 道  专家 {args.model}  max_tokens {args.max_tokens}  并发 {args.workers}"
      + (f"  ★ {args.wrong_frac:.0%} 的求助题先问弱专家 {args.wrong_model}" if args.wrong_frac > 0 else ""), flush=True)


def level01(nums, target):
    """老师 v1 的第 0 层 + 第 1 层（enum_traces.trace 的前半段，一字不差）→ (行列表, 命中式子或 None, 命中层)"""
    Tg = Fraction(target); L = []; n = 0

    def emit(v, line):
        nonlocal n
        L.append(f"{ET.calc(line, ET.fmt_v(v))} {ET.tag(v, target)}"); n += 1
        return v == Tg

    L.append("First, only + and -:")
    lv0 = ET.sign_lines([(Fraction(x), str(x)) for x in nums])
    for v, line in lv0:
        if emit(v, line):
            return L, line, 0
    vals = [v for v, _ in lv0]
    L.append(f"None of the {len(lv0)} sign patterns gives {target} (they range from "
             f"{ET.fmt_v(min(vals))} to {ET.fmt_v(max(vals))}), so we need * or /.")
    b1 = sorted(ET.blocks1(nums), key=lambda t: (abs(t[0] - target), t[0]))
    L.append("Pair products and quotients: " + ", ".join(ET.calc(e, v) for v, e, _ in b1) + ".")
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
        for vv, line in ET.sign_lines([(Fraction(v), e)] + [(Fraction(x), str(x)) for x in rest]):
            if emit(vv, line):
                return L, line, 1
        if n >= ET.CAP:
            break
    return L, None, None


def parse(r):
    """专家回复 → (回复第一行 ≤120 字, 式子)；正文空（截断/网络错）→ error"""
    a = r["content"].split("\n")[0][:120] if r["content"] else "error"
    eq = a.split("EQUATION:")[-1].strip().split("=")[0].strip() if "EQUATION:" in a else a
    return a, eq


def verify_line(eq, target):
    v = T.evaluate(eq)
    ok = v != "error" and abs(float(v) - target) < 1e-9
    return (f"Let me verify: <calc>{eq}</calc><result>{v}</result> "
            + ("(perfect match, verified)" if ok else f"(not {target}, the expert is wrong)")), ok


KEYS = ("plain_l0", "plain_l1", "ok1", "ok2", "drop", "api_err", "strong_wrong", "weak_wrong_used", "bad_final")


def build(item):
    """一道题 → (row 或 None, 统计增量)"""
    nums, target, idx, bucket, wrong = item
    inc = dict.fromkeys(KEYS, 0)
    head = (f" We need to use the numbers {nums} exactly once, combined with the operations "
            f"+, -, *, / to get a result of {target}.\nLet's try different combinations:\n")
    L, hit, lv = level01(nums, target)
    if hit:                                                   # 自己搜到了：不问
        inc[f"plain_l{lv}"] = 1
        resp = head + "\n".join(L) + f"\n</think>\n<answer> {hit} </answer>"
        return dict(prompt=T.ask_prompt(nums, target), response=resp, nums=nums, target=target, idx=idx,
                    bucket=bucket, kind=f"plain_l{lv}", n_asks=0), inc
    q = f"Using the numbers {nums} each exactly once with + - * /, write one equation that equals {target}."
    L.append("None of these work; this one is hard, let me ask the expert.")

    def ask_once(qq, model, max_tokens):
        """→ (回复行, 式子, 验证通过?, 验证行, 没答?)"""
        a, eq = parse(DS.ask(qq, model=model, system=DS.CD_SYS, max_tokens=max_tokens))
        if a == "error":
            return a, eq, False, "(the expert did not answer)", True
        vline, ok = verify_line(eq, target)
        return a, eq, ok, vline, False

    # 第一次：抽中的题先问弱专家，它答错才用（教「验证抓错 → 再问」）；碰巧答对或没答就当没问过，照常问强专家
    a, eq, ok, vline, err = ask_once(q, args.wrong_model, 200) if wrong else (None,) * 5
    if not wrong or ok or err:
        wrong = False
        a, eq, ok, vline, err = ask_once(q, args.model, args.max_tokens)
    n_asks, final = 1, None
    L.append(f"<ask>{q}</ask><reply>{a}</reply>"); L.append(vline)
    if err: inc["api_err"] += 1
    elif wrong: inc["weak_wrong_used"] += 1
    elif not ok: inc["strong_wrong"] += 1
    if ok:
        final = eq; inc["ok1"] = 1
    else:                                                     # 第二次：告诉它上次的式子错了 / 没答就请它答一行
        L.append("Let me ask once more.")
        qq = q + (" Please answer in one line." if err else f" Your previous answer {eq} does not work. Give a different equation.")
        a, eq, ok, vline, err = ask_once(qq, args.model, args.max_tokens)
        n_asks = 2
        L.append(f"<ask>{qq}</ask><reply>{a}</reply>"); L.append(vline)
        if err: inc["api_err"] += 1
        elif not ok: inc["strong_wrong"] += 1
        if ok:
            final = eq; inc["ok2"] = 1
    if final is None:
        inc["drop"] = 1; return None, inc
    resp = head + "\n".join(L) + f"\n</think>\n<answer> {final} </answer>"
    sc, d = CD.reward(resp, nums, target)
    if d["correct"] <= 0:                                     # 值对但数字没各用一次之类：判分器不认
        inc["bad_final"] = 1; return None, inc
    return dict(prompt=T.ask_prompt(nums, target), response=resp, nums=nums, target=target, idx=idx,
                bucket=bucket, kind="ask", n_asks=n_asks), inc


rows, stat, t0 = [], dict.fromkeys(KEYS, 0), time.time()
with ThreadPoolExecutor(max_workers=args.workers) as ex:
    for k, (row, inc) in enumerate(ex.map(build, probs)):
        for key, v in inc.items():
            stat[key] += v
        if row:
            rows.append(row)
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(probs)}  {stat}  {time.time()-t0:.0f} s", flush=True)

with open(args.out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
n_ask = sum(r["kind"] == "ask" for r in rows)
by = {}
for r in rows:
    by.setdefault(r["bucket"], {}).setdefault(r["kind"], 0); by[r["bucket"]][r["kind"]] += 1
print(f"\n落盘 {len(rows)} 条 → {args.out}：求助 {n_ask}（一次就对 {stat['ok1']}，第二次才对 {stat['ok2']}）  "
      f"自己搜到 第 0 层 {stat['plain_l0']} / 第 1 层 {stat['plain_l1']}  丢弃 {stat['drop']}（两次都不行）  "
      f"判分不认 {stat['bad_final']}  {time.time()-t0:.0f} s")
print(f"专家回复：没答（正文空）{stat['api_err']}  强专家答错 {stat['strong_wrong']}  弱专家错答用作第一次回复 {stat['weak_wrong_used']}"
      f"（教材里「the expert is wrong → 再问」的样本 = {stat['strong_wrong'] + stat['weak_wrong_used']} 条里最终收下的）")
print(f"按桶：{by}   最长响应 {max((len(r['response']) for r in rows), default=0)} 字符")
asked = stat['ok1'] + stat['ok2'] + stat['drop'] + stat['bad_final']
print("专家一次答对率", f"{stat['ok1']}/{asked}" if asked else "—")
