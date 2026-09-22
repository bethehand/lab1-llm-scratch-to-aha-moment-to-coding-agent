#!/usr/bin/env python3
"""
make_code_demos.py —— 阶段 ①「修 bug」的教材（撒种）：程序写调试轨迹，注入段（<result>）是环境真跑出来的，SFT 时 label −100。
    python3 make_code_demos.py --bugs data/code_bugs.json --out sft_code.jsonl [--n 1200 --per-task 2 --p-tb 0.3 --p-retry 0.2]
一条轨迹（不带报错开局，70%）：
    <test></test> → ⟦passed 0/2 + traceback⟧ → 「The failing test is …; the traceback points at line N: `bug 行`. <kind 对应的一句解释>. It should be `ref 行`.」
    → <write>整个文件（ref_code）</write> → ⟦written⟧ → <test></test> → ⟦passed 2/2⟧ → </think> <answer>done</answer>
带报错开局（30%）：题面里已有输出，跳过第一次 test。
带一次失败的重试（--p-retry）：先 write 同一题的另一个变异体（另一处错）→ test 挂（真输出）→ 「Still failing: …」→ write ref → test 过。教「改完必须测，测挂了再改」
只用训练 split、训练 kind；提示词从 8 个训练模板随机抽，工具措辞随机。每条记 kind / bug_type / with_tb / retry。
"""
import json, random, argparse, collections, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import code_env as C

ap = argparse.ArgumentParser()
ap.add_argument("--bugs", default="data/code_bugs.json")
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=1200)
ap.add_argument("--per-task", type=int, default=2)
ap.add_argument("--p-tb", type=float, default=0.3)
ap.add_argument("--p-retry", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
rng = random.Random(args.seed)

EXPLAIN = {
    "cmp_flip": "the comparison goes the wrong way",
    "off_by_one": "the number is off by one",
    "arith_swap": "the arithmetic operator is wrong",
    "bool_neg": "the condition is negated",
    "ret_wrong": "it returns the wrong variable",
    "del_stmt": "a statement is missing",
    "swap_args": "the arguments are in the wrong order",
}


def bug_line(b):
    plus = [l[1:].strip() for l in b["diff"] if l.startswith("+")]
    minus = [l[1:].strip() for l in b["diff"] if l.startswith("-")]
    cur = plus[0] if plus else None
    fixed = minus[0] if minus else None
    lines = b["bug_code"].splitlines()
    n = next((i + 1 for i, l in enumerate(lines) if cur and l.strip() == cur), None)
    return n, cur, fixed


def reason(b, obs):
    """看着观测写的那一句：哪条测试挂、指到哪行、错在哪、该是什么"""
    failing = next((l for l in obs.splitlines() if l.startswith("assert")), "").strip()
    n, cur, fixed = bug_line(b)
    s = f"The failing test is `{failing}`. " if failing else ""
    if cur and fixed:
        s += f"Looking at the function, the bug is on line {n}: `{cur}` — {EXPLAIN[b['kind']]}. It should be `{fixed}`. "
    elif fixed:
        s += f"Looking at the function, a line is missing — {EXPLAIN[b['kind']]}. It should have `{fixed}`. "
    s += "I will rewrite the whole file with the fix and run the tests again."
    return s


B = json.load(open(args.bugs))
train = [b for b in B if b["split"] == "train"]
by_task = collections.defaultdict(list)
for b in train:
    by_task[b["task_id"]].append(b)
tasks = sorted(by_task); rng.shuffle(tasks)
env = C.CodeEnv(max_calls=8)
rows, stat, t0 = [], collections.Counter(), time.time()
for tid in tasks:
    variants = by_task[tid]; rng.shuffle(variants)
    for b in variants[: args.per_task]:
        if len(rows) >= args.n: break
        with_tb = rng.random() < args.p_tb
        retry = (not with_tb or True) and rng.random() < args.p_retry and len(variants) >= 2
        pid = rng.choice(C.SETS["train"]); ts = rng.randrange(3)
        prompt = C.render(pid, b, with_tb=with_tb, tool_style=ts, env=env)
        st = env.reset(b); parts = []
        def call(tag, arg=""):
            """模型写一个工具调用，环境真跑，把注入段接上；返回注入的观测文本"""
            piece = f"<{tag}>{arg}</{tag}>"
            parts.append(piece)
            inj, done = env.step(st, "".join(parts))
            parts.append(inj)
            return inj[len(C.RESULT_OPEN):-len(C.RESULT_CLOSE)]
        if with_tb:
            obs = b["initial_obs"] if "initial_obs" in b else env.initial_obs(b)
            parts.append("The test output is already given above. ")
        else:
            parts.append("Let me run the tests first to see what fails.\n")
            obs = call("test")
            parts.append("\n")
        if retry:                                                     # 先交一版还是错的（同题另一个变异体），测挂，再改对
            other = next(v for v in variants if v is not b)
            parts.append(reason(b, obs).replace("I will rewrite the whole file with the fix and run the tests again.", "Let me try a fix and run the tests.") + "\n")
            call("write", "\n" + other["bug_code"] + "\n"); parts.append("\n")
            obs2 = call("test"); parts.append("\n")
            failing2 = next((l for l in obs2.splitlines() if l.startswith("assert")), "")
            n2, cur2, fixed2 = bug_line(other)
            parts.append(f"Still failing: `{failing2}`. My fix introduced another mistake on line {n2}: `{cur2}` — {EXPLAIN[other['kind']]}. "
                         f"The correct line is `{fixed2}`. Let me write the whole file correctly and test again.\n")
        else:
            parts.append(reason(b, obs) + "\n")
        call("write", "\n" + b["ref_code"] + "\n"); parts.append("\n")
        obs3 = call("test")
        ok = obs3.startswith("passed 2/2")
        parts.append("\nAll visible tests pass.\n</think>\n<answer>done</answer>")
        resp = "".join(parts)
        r, d = env.score(st, resp)
        if not (ok and d["correct"] > 0):
            stat["bad"] += 1; continue
        rows.append(dict(prompt=prompt, response=resp, task="code", task_id=tid, kind=b["kind"], bug_type=b["bug_type"], pid=pid, with_tb=with_tb, retry=retry,
                         n_calls=st["calls"]))
        stat["ok"] += 1; stat[f"kind_{b['kind']}"] += 1; stat["tb"] += with_tb; stat["retry"] += retry
    if len(rows) >= args.n: break
    if stat["ok"] and stat["ok"] % 100 == 0:
        print(f"  {stat['ok']} 条  {time.time()-t0:.0f} s", flush=True)
with open(args.out, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
L = sorted(len(r["prompt"]) + len(r["response"]) for r in rows)
print(f"\n落盘 {len(rows)} 条 → {args.out}   带报错开局 {stat['tb']}   带重试 {stat['retry']}   丢 {stat['bad']}   字符中位 {L[len(L)//2]} P90 {L[int(.9*len(L))]} 最长 {L[-1]}   {time.time()-t0:.0f} s")
print("  kind：", {k[5:]: v for k, v in stat.items() if k.startswith("kind_")})
