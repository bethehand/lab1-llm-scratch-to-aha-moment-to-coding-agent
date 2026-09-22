#!/usr/bin/env python3
"""
make_pkg_demos.py —— 阶段 ②-A 的教材：程序老师按文件挨个修，注入段（<result>）是 PkgEnv 真跑出来的（SFT 时 label −100）。
    python3 make_pkg_demos.py --out-a sft_pkg_A.jsonl --out-b sft_pkg_B.jsonl [--n 1200 --k-weights 0.2,0.3,0.5 --p-tb 0.3 --p-retry 0.15 --p-revert 0.10 --p-badwrite 0.10]
A 版：每次 <test> 的观测之后写一行状态「Status: fixed a.py; remaining b.py, c.py.」；B 版 = A 版去掉这些行，其余一字不差（E2 的唯一变量）。
一条轨迹：（不带报错开局）<test> → ⟦按文件的观测⟧ → [Status] → 对每个还挂的文件：「In a.py, the failing test is …; the bug is on line N: `…` — 解释. It should be `…`.
  I will rewrite the whole file a.py with the fix and run the tests again.」→ <write file="a.py">整个文件（ref_code，含 import）</write> → ⟦written a.py⟧ → <test> → ⟦观测⟧ → [Status]
  → … → 全过 → 「All tests pass.」</think><answer>done</answer>
恢复段（每个文件独立抽，互斥）：
  retry    先交同题另一个变异体（还是错）→ 测挂 → 「Still failing in a.py … My fix introduced another mistake …」→ 写对
  revert   先交同题的一个崩溃型变异体 → 测出新异常 → 「That made it worse … restore the original a.py」→ 写回原文件 → 测（回到原来的错）→ 写对   ← ① 欠的债
  badwrite 第一次 write 忘了 file= → ⟦not written: say which file…⟧ → 「I forgot the file name.」→ 带文件名重写（只在 K ≥ 2）
K 混 1/2/3（--k-weights），模板 8 套训练 × 3 种措辞 × 开局带/不带报错。每条都用秤验过（含严秤隐藏测试）才落盘。
"""
import json, random, argparse, collections, os, sys, time, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import code_env as C
import pkg_env as P

ap = argparse.ArgumentParser()
ap.add_argument("--bugs", default="data/code_bugs.json")
ap.add_argument("--out-a", required=True)
ap.add_argument("--out-b", required=True)
ap.add_argument("--n", type=int, default=1200)
ap.add_argument("--k-weights", default="0.2,0.3,0.5", help="K=1,2,3 的比例")
ap.add_argument("--p-tb", type=float, default=0.3)
ap.add_argument("--p-retry", type=float, default=0.15)
ap.add_argument("--p-revert", type=float, default=0.15, help="只有同题有崩溃型变异体（237/863 道）时才可能触发，实际比例约一半")
ap.add_argument("--p-badwrite", type=float, default=0.10)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--show", type=int, default=0, help="打前几条 A 版看看")
args = ap.parse_args()
rng = random.Random(args.seed)
KW = [float(x) for x in args.k_weights.split(",")]

EXPLAIN = {"cmp_flip": "the comparison goes the wrong way", "off_by_one": "the number is off by one", "arith_swap": "the arithmetic operator is wrong",
           "bool_neg": "the condition is negated", "ret_wrong": "it returns the wrong variable", "del_stmt": "a statement is missing", "swap_args": "the arguments are in the wrong order"}


def bug_line(b):
    plus = [l[1:].strip() for l in b["diff"] if l.startswith("+")]; minus = [l[1:].strip() for l in b["diff"] if l.startswith("-")]
    cur = plus[0] if plus else None; fixed = minus[0] if minus else None
    n = next((i + 1 for i, l in enumerate(b["bug_code"].splitlines()) if cur and l.strip() == cur), None)
    return n, cur, fixed


def section(obs, name):
    """观测里这个文件那一段 → (k, n, 第一条挂的 assert, 异常名)"""
    lines = obs.splitlines(); out = []; on = False
    for l in lines:
        if re.match(r"^\S+\.py: \d+/\d+$", l):
            on = l.startswith(name + ":")
            if on: out.append(l)
            continue
        if on: out.append(l)
    if not out: return None
    k, n = map(int, re.match(r"^\S+\.py: (\d+)/(\d+)$", out[0]).groups())
    failing = next((l.strip() for l in out[1:] if l.strip().startswith("assert")), "")
    exc = next((l.strip().split(":")[0] for l in out[1:] if re.match(r"^\s*[A-Z][A-Za-z]*Error\b|^\s*Timeout", l)), "")
    return k, n, failing, exc


def status_line(obs, names):
    fixed = [nm for nm in names if (sec := section(obs, nm)) and sec[0] == sec[1]]
    rem = [nm for nm in names if nm not in fixed]
    return f"Status: fixed {', '.join(fixed) if fixed else 'none'}; remaining {', '.join(rem) if rem else 'none'}."


def reason(f, failing, first=True):
    n, cur, fixed = bug_line(f)
    s = f"{'In' if first else 'Next,'} {f['name']}, the failing test is `{failing}`. " if failing else f"{'In' if first else 'Next,'} {f['name']}: "
    if cur and fixed:
        s += f"Looking at the code, the bug is on line {n}: `{cur}` — {EXPLAIN[f['kind']]}. It should be `{fixed}`. "
    elif fixed:
        s += f"Looking at the code, a line is missing — {EXPLAIN[f['kind']]}. It should have `{fixed}`. "
    return s + f"I will rewrite the whole file {f['name']} with the fix and run the tests again."


B = json.load(open(args.bugs)); train = [b for b in B if b["split"] == "train"]
by_task = collections.defaultdict(list)
for b in train: by_task[b["task_id"]].append(b)
ext = P.load_ext()
env = P.PkgEnv(max_calls=16, call_cost=0.01)
rows_a, rows_b, stat, t0 = [], [], collections.Counter(), time.time()

while len(rows_a) < args.n:
    K = rng.choices([1, 2, 3], weights=KW)[0]
    for _ in range(50):
        pick = rng.sample(train, K)
        if len({p["task_id"] for p in pick}) < K: continue
        pkg = P.make_pkg(pick, ext, pkg_id=f"t{len(rows_a)}")
        if pkg: break
    else:
        continue
    with_tb = rng.random() < args.p_tb
    pid = rng.choice(P.SETS["train"]); ts = rng.randrange(3)
    prompt = P.render(pid, pkg, with_tb=with_tb, tool_style=ts, env=env)
    names = [f["name"] for f in pkg["files"]]
    st = env.reset(pkg); parts = []                                  # parts: (text, is_status)
    def add(t, status=False): parts.append((t, status))
    def call(tag, arg="", fname=None):
        piece = f"<{tag}{(' file=%s' % json.dumps(fname)) if fname else ''}>{arg}</{tag}>"
        add(piece)
        inj, done = env.step(st, "".join(t for t, _ in parts))
        add(inj)
        return inj[len(C.RESULT_OPEN):-len(C.RESULT_CLOSE)]
    if with_tb:
        obs = pkg["initial_obs"]; add("The test output is already given above.\n")
    else:
        add("Let me run the tests first to see what fails.\n"); obs = call("test"); add("\n")
    add(status_line(obs, names) + "\n", status=True)
    branches = []
    for idx, f in enumerate(pkg["files"]):
        sec = section(obs, f["name"]); failing = sec[2] if sec else ""
        others = [v for v in by_task[f["task_id"]] if v is not f and v["bug_code"] != f["bug_code"]]
        crash = [v for v in others if v["bug_type"] == "crash"]
        r = rng.random()
        branch = ("badwrite" if (K >= 2 and idx == 0 and r < args.p_badwrite) else
                  "revert" if (crash and r < args.p_badwrite + args.p_revert) else
                  "retry" if (others and r < args.p_badwrite + args.p_revert + args.p_retry) else "plain")
        if branch == "badwrite":
            add(reason(f, failing, idx == 0) + "\n")
            bad = call("write", "\n" + f["ref_code"].strip("\n") + "\n")            # 忘了 file=
            add("\nI forgot the file name. Writing " + f["name"] + ":\n")
            call("write", "\n" + f["ref_code"].strip("\n") + "\n", f["name"]); add("\n")
        elif branch == "revert":
            v = rng.choice(crash)
            n2, cur2, fixed2 = bug_line(f)
            add(reason(f, failing, idx == 0).replace("I will rewrite the whole file", "Let me try rewriting the whole file").replace("and run the tests again.", "and run the tests.") + "\n")
            call("write", "\n" + v["bug_code"].strip("\n") + "\n", f["name"]); add("\n")
            obs2 = call("test"); add("\n")
            s2 = section(obs2, f["name"]); exc = (s2[3] if s2 else "") or "an error"
            add(f"That made it worse: {f['name']} now fails with {exc}. Let me restore the original {f['name']} and look at the traceback again.\n")
            call("write", "\n" + f["bug_code"].strip("\n") + "\n", f["name"]); add("\n")
            obs3 = call("test"); add("\n")
            s3 = section(obs3, f["name"]); failing3 = s3[2] if s3 else failing
            add(f"Back to the original failure in {f['name']}: `{failing3}`. The bug is on line {n2}: `{cur2}` — {EXPLAIN[f['kind']]}. It should be `{fixed2}`. "
                f"I will rewrite the whole file {f['name']} with only that change and run the tests again.\n")
            call("write", "\n" + f["ref_code"].strip("\n") + "\n", f["name"]); add("\n")
        elif branch == "retry":
            v = rng.choice(others)
            add(reason(f, failing, idx == 0).replace("I will rewrite the whole file", "Let me try rewriting the whole file").replace("and run the tests again.", "and run the tests.") + "\n")
            call("write", "\n" + v["bug_code"].strip("\n") + "\n", f["name"]); add("\n")
            obs2 = call("test"); add("\n")
            s2 = section(obs2, f["name"]); failing2 = s2[2] if s2 else failing
            n2, cur2, fixed2 = bug_line(v)
            add(f"Still failing in {f['name']}: `{failing2}`. My fix introduced another mistake on line {n2}: `{cur2}` — {EXPLAIN[v['kind']]}. "
                f"The correct line is `{fixed2}`. Let me write the whole file {f['name']} correctly and test again.\n")
            call("write", "\n" + f["ref_code"].strip("\n") + "\n", f["name"]); add("\n")
        else:
            add(reason(f, failing, idx == 0) + "\n")
            call("write", "\n" + f["ref_code"].strip("\n") + "\n", f["name"]); add("\n")
        obs = call("test"); add("\n")
        add(status_line(obs, names) + "\n", status=True)
        branches.append(branch)
    add("All tests pass.\n</think>\n<answer>done</answer>")
    resp_a = "".join(t for t, _ in parts); resp_b = "".join(t for t, s_ in parts if not s_)
    r_, d = env.score(st, resp_a)
    if d["correct"] < 1 or not obs.startswith(f"passed {2*K}/{2*K}") or st["calls"] > 16:
        stat["bad"] += 1; continue
    meta = dict(task="pkg", K=K, task_ids=pkg["task_ids"], kinds=pkg["kinds"], pid=pid, with_tb=with_tb, branches=branches, n_calls=st["calls"])
    rows_a.append(dict(prompt=prompt, response=resp_a, version="A", **meta)); rows_b.append(dict(prompt=prompt, response=resp_b, version="B", **meta))
    stat["ok"] += 1; stat[f"K{K}"] += 1; stat["tb"] += with_tb
    for br in branches: stat[br] += 1
    if stat["ok"] % 100 == 0:
        print(f"  {stat['ok']} 条  {time.time()-t0:.0f} s", flush=True)

for path, rows in ((args.out_a, rows_a), (args.out_b, rows_b)):
    with open(path, "w") as fo:
        for r in rows: fo.write(json.dumps(r, ensure_ascii=False) + "\n")
L = sorted(len(r["prompt"]) + len(r["response"]) for r in rows_a)
print(f"\n落盘 {len(rows_a)} 条 × 2 版 → {args.out_a} / {args.out_b}   K1/K2/K3 {stat['K1']}/{stat['K2']}/{stat['K3']}   带报错开局 {stat['tb']}   丢 {stat['bad']}"
      f"   文件级分支 plain/retry/revert/badwrite {stat['plain']}/{stat['retry']}/{stat['revert']}/{stat['badwrite']}   字符中位 {L[len(L)//2]} P90 {L[int(.9*len(L))]} 最长 {L[-1]}   {time.time()-t0:.0f} s")
print(f"  A 版每条状态行均值 {sum(len(P.STATE_RE.findall(r['response'])) for r in rows_a)/max(1,len(rows_a)):.2f}   B 版 {sum(len(P.STATE_RE.findall(r['response'])) for r in rows_b)/max(1,len(rows_b)):.2f}")
for r in rows_a[: args.show]:
    print("█" * 100); print(r["prompt"]); print("── response（A 版）──"); print(r["response"])
