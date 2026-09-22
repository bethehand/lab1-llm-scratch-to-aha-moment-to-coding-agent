#!/usr/bin/env python3
"""严秤：用 ref_code 当 oracle 随机造 N 条隐藏测试，再拿它给已经跑完的评测答卷重新打分（不用 GPU，只重新改卷）。

    # ① 造测试（只给这些 raw 里出现过的题造），存下来复用
    python3 strict_score.py --build --raws dyn_notb_raw.json rl_notb_raw.json --n 20 --out data/tests_ext.json
    # ② 重新打分
    python3 strict_score.py --raws sftc_notb_raw.json rl_notb_raw.json dyn_notb_raw.json --tests data/tests_ext.json

原则：观测不变（模型仍只看 2 条可见测试），只把「模型看不见的那部分」从 1 条加到 N 条。
验收：ref_code 必须全过新测试（造错了就丢），bug_code 必须至少挂一条（挂不住说明没覆盖到 bug，这道题的新测试不算数）。
"""
import os, sys, ast, json, random, argparse, collections, string
from concurrent.futures import ThreadPoolExecutor
import code_env as C

ap = argparse.ArgumentParser()
ap.add_argument("--raws", nargs="*", default=[], help="评测的 *_raw.json")
ap.add_argument("--data", default="data/code_bugs.json")
ap.add_argument("--tests", default="data/tests_ext.json", help="造好的新测试（--build 时是输出，否则是输入）")
ap.add_argument("--build", action="store_true", help="造测试而不是打分")
ap.add_argument("-n", type=int, default=20, help="每题造几条")
ap.add_argument("--tries", type=int, default=60, help="每题最多试几组输入")
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--show", type=int, default=2, help="造完打几道题的例子")
args = ap.parse_args()

BUGS = {b["task_id"]: b for b in json.load(open(args.data))}


# ══════════════════ ① 从可见测试里读出调用形状 ══════════════════
def parse_call(src):
    """'assert f(a, b) == c' → (函数名, [参数字面量], 期望值, 比较方式)；读不懂返回 None"""
    try:
        node = ast.parse(src.strip()).body[0]
        if not isinstance(node, ast.Assert):
            return None
        t, neg = node.test, False
        if isinstance(t, ast.UnaryOp) and isinstance(t.op, ast.Not):
            t, neg = t.operand, True
        exp, kind = (False if neg else True), "bool"
        if isinstance(t, ast.Compare) and len(t.ops) == 1 and isinstance(t.ops[0], ast.Eq):
            exp, kind, t = ast.literal_eval(t.comparators[0]), "eq", t.left
        if not isinstance(t, ast.Call) or not isinstance(t.func, ast.Name) or t.keywords:
            return None
        return t.func.id, [ast.literal_eval(a) for a in t.args], exp, kind
    except Exception:
        return None


# ══════════════════ ② 按样例的形状造随机输入 ══════════════════
def spec_of(vals):
    """一列同位置的参数样例 → 形状描述"""
    ts = {type(v).__name__ for v in vals}
    if ts == {"bool"}: return dict(t="bool")
    if ts <= {"int", "bool"}:
        lo, hi = min(vals), max(vals)
        return dict(t="int", lo=min(lo, 0) if lo < 0 else max(0, lo - 3), hi=max(hi + 3, 6))
    if ts <= {"int", "float", "bool"}:
        return dict(t="float", lo=float(min(vals)) - 3, hi=float(max(vals)) + 3)
    if ts == {"str"}:
        pool = set()
        for v in vals:
            for ch in v:
                pool.add("d" if ch.isdigit() else "a" if ch.isalpha() else "s" if ch.isspace() else "p")
        if pool & {"a", "d"}: pool |= {"a", "d"}                  # ★ 见过字母或数字 → 两类都造，否则「可见样例全是纯字母」会造出 20 条答案全一样的废测试
        return dict(t="str", pool=sorted(pool) or ["a"], lo=max(0, min(len(v) for v in vals) - 1), hi=max(len(v) for v in vals) + 2)
    if ts <= {"list", "tuple", "set"}:
        kind = type(vals[0]).__name__
        inner = [x for v in vals for x in v]
        if not inner: return None
        e = spec_of(inner)
        if e is None: return None
        lens = {len(v) for v in vals}
        if len(lens) == 1:                                            # ★ 可见样例里这个位置的长度都一样（3×3 的方阵、定长元组）→ 新造的保持同样长度，别造出超出题意的形状
            L = lens.pop(); return dict(t=kind, e=e, lo=L, hi=L)
        return dict(t=kind, e=e, lo=max(0, min(lens) - 1), hi=max(lens) + 2)
    if ts == {"dict"}:
        ks = [k for v in vals for k in v]; vs = [x for v in vals for x in v.values()]
        if not ks: return None
        sk, sv = spec_of(ks), spec_of(vs)
        if sk is None or sv is None: return None
        return dict(t="dict", k=sk, v=sv, lo=1, hi=max(len(v) for v in vals) + 1)
    return None


CH = dict(a=string.ascii_lowercase, d=string.digits, s=" ", p="_-.,!")


def gen(sp, rng, depth=0):
    if depth > 3: return 0
    t = sp["t"]
    if t == "bool": return rng.random() < 0.5
    if t == "int": return rng.randint(int(sp["lo"]), int(sp["hi"]))
    if t == "float": return round(rng.uniform(sp["lo"], sp["hi"]), 3)
    if t == "str":
        pool = "".join(CH[c] for c in sp["pool"])
        return "".join(rng.choice(pool) for _ in range(rng.randint(sp["lo"], sp["hi"])))
    if t in ("list", "tuple", "set"):
        v = [gen(sp["e"], rng, depth + 1) for _ in range(rng.randint(sp["lo"], sp["hi"]))]
        return v if t == "list" else tuple(v) if t == "tuple" else set(v)
    if t == "dict":
        return {gen(sp["k"], rng, depth + 1): gen(sp["v"], rng, depth + 1) for _ in range(rng.randint(sp["lo"], sp["hi"]))}
    return 0


def build_args(calls, rng):
    """calls = 可见测试解析出的若干组参数 → 一组新参数。保留「某个 int 参数 == 某个序列参数的长度」这种关系"""
    nargs = len(calls[0])
    if any(len(c) != nargs for c in calls): return None
    specs = []
    for i in range(nargs):
        sp = spec_of([c[i] for c in calls])
        if sp is None: return None
        specs.append(sp)
    lenrel = {}                                                      # i -> j：第 i 个参数在所有样例里都等于第 j 个参数的长度
    for i in range(nargs):
        if specs[i]["t"] != "int": continue
        for j in range(nargs):
            if i != j and all(hasattr(c[j], "__len__") and c[i] == len(c[j]) for c in calls):
                lenrel[i] = j; break
    out = [gen(sp, rng) for sp in specs]
    for i, j in lenrel.items():
        out[i] = len(out[j])
    return out


# ══════════════════ ③ 用 ref_code 当 oracle 求期望值 ══════════════════
PROBE = '''
_cases = @CASES@
_out = []
for _a in _cases:
    try:
        _out.append(("ok", repr(@FN@(*_a))))
    except Exception as _e:
        _out.append(("err", type(_e).__name__))
print("__PROBE__" + repr(_out))
'''


def oracle(prob, cases):
    """在沙盒里跑一次 ref_code，拿到每组输入的返回值 repr"""
    fn = prob["_fn"]
    src = (prob.get("setup") or "") + "\n" + prob["ref_code"] + "\n" + PROBE.replace("@CASES@", repr(cases)).replace("@FN@", fn)
    d = C.tempfile.mkdtemp()
    try:
        rc, out, err = C.run_in_sandbox(src, d, timeout=10, raw=True)
    finally:
        C.shutil.rmtree(d, ignore_errors=True)
    i = (out or "").find("__PROBE__")
    if i < 0: return None
    try:
        return ast.literal_eval(out[i + len("__PROBE__"):].strip())
    except Exception:
        return None


def make_tests(prob, n, tries, seed):
    """→ (新测试列表, 说明)。造不出来就返回 ([], 原因)"""
    parsed = [parse_call(t) for t in prob["visible_tests"]]
    parsed = [p for p in parsed if p]
    if not parsed: return [], "可见测试读不懂"
    fn = parsed[0][0]
    if any(p[0] != fn for p in parsed): return [], "可见测试调的不是同一个函数"
    prob["_fn"] = fn
    rng = random.Random(seed + prob["task_id"])
    cases, seen = [], {repr(list(p[1])) for p in parsed}          # 可见测试用过的输入不再造
    for _ in range(tries):
        a = build_args([p[1] for p in parsed], rng)
        if a is None: return [], "参数形状推不出来"
        k = repr(a)
        if k in seen: continue
        seen.add(k); cases.append(a)
        if len(cases) >= tries: break
    if not cases: return [], "造不出新输入"
    r1 = oracle(prob, cases)
    if r1 is None: return [], "ref 跑不起来"
    r2 = oracle(prob, cases)                                        # ★ 跑两次：结果不一样的（用了集合顺序 / 随机 / 时间）整题丢掉
    if r2 is None or r1 != r2: return [], "ref 输出不稳定"
    byval = collections.OrderedDict()                               # ★ 按期望值分桶再轮转着取：整组答案都一样的测试集抓不住「碰巧过」的错答案
    for a, (st, v) in zip(cases, r1):
        if st != "ok": continue                                     # ref 自己崩的输入丢掉
        try:
            val = ast.literal_eval(v)
        except Exception:
            continue                                                # 返回值不是字面量（对象、生成器）→ 不用
        call = f"{fn}({', '.join(repr(x) for x in a)})"
        byval.setdefault(v, []).append(f"assert abs({call} - {v}) < 1e-6" if isinstance(val, float) else f"assert {call} == {v}")
    tests = []
    while len(tests) < n and any(byval.values()):
        for k in list(byval):
            if byval[k]:
                tests.append(byval[k].pop(0))
                if len(tests) >= n: break
    if len(tests) < 3: return [], "可用的输入太少"
    return tests, "ok"


def check(prob, tests):
    """验收：ref 必须全过；bug 必须至少挂一条"""
    d1 = C.tempfile.mkdtemp(); d2 = C.tempfile.mkdtemp()
    try:
        k1, ok1, _ = C.run_tests(prob["ref_code"], tests, prob.get("setup", ""), d1, 10)
        k2, ok2, _ = C.run_tests(prob["bug_code"], tests, prob.get("setup", ""), d2, 10)
    finally:
        C.shutil.rmtree(d1, ignore_errors=True); C.shutil.rmtree(d2, ignore_errors=True)
    return all(ok1), not all(ok2)


# ══════════════════ ④ 重新打分 ══════════════════
def rescore(raw, EXT):
    R = raw["records"]
    def one(r):
        p = BUGS[r["task_id"]]; ext = EXT.get(str(r["task_id"]))
        if not ext: return None
        d = C.tempfile.mkdtemp()
        try:
            k, oks, _ = C.run_tests(r["final_code"] or p["bug_code"], list(p["visible_tests"]) + [p["hidden_test"]] + ext,
                                    p.get("setup", ""), d, 10)
        finally:
            C.shutil.rmtree(d, ignore_errors=True)
        n_old = len(p["visible_tests"]) + 1
        return dict(task_id=r["task_id"], kind=r["kind"], old=all(oks[:n_old]), new=all(oks), n_ext_fail=sum(1 for o in oks[n_old:] if not o))
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        return [x for x in ex.map(one, R) if x]


if args.build:
    ids = set()
    for f in args.raws:
        try:
            ids |= {r["task_id"] for r in json.load(open(f))["records"]}
        except Exception as e:
            print(f"  ⚠️ 读不了 {f}（{e.__class__.__name__}），跳过（评测输出名带 _raw 后缀，例如 sftc_coder_notb_raw.json）")
    ids = sorted(ids) or sorted(BUGS)                                  # 不给 --raws（或都读不了）= 整个题库
    print(f"给 {len(ids)} 道题造新隐藏测试（每题 ≤ {args.n} 条，最多试 {args.tries} 组输入）…", flush=True)
    EXT, why = {}, collections.Counter()
    def work(tid):
        p = BUGS[tid]
        tests, msg = make_tests(dict(p), args.n, args.tries, args.seed)
        if not tests: return tid, [], msg
        ok_ref, ok_bug = check(p, tests)
        if not ok_ref: return tid, [], "ref 没全过（造错了）"
        if not ok_bug: return tid, [], "bug 版也全过（没覆盖到 bug）"
        return tid, tests, "ok"
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for tid, tests, msg in ex.map(work, ids):
            why[msg] += 1
            if tests: EXT[str(tid)] = tests
    os.makedirs(os.path.dirname(args.tests) or ".", exist_ok=True)
    json.dump(EXT, open(args.tests, "w"))
    print(f"  成功 {len(EXT)}/{len(ids)} 道（{len(EXT)/len(ids):.0%}），平均 {sum(len(v) for v in EXT.values())/max(1,len(EXT)):.1f} 条/题  →  {args.tests}")
    for k, v in why.most_common():
        print(f"    {k:<24} {v}")
    for tid in list(EXT)[: args.show]:
        p = BUGS[int(tid)]
        print(f"\n  ── 例：task {tid}  {p['kind']}")
        for t in p["visible_tests"]: print(f"    可见（模型能跑）  {t}")
        print(f"    原隐藏            {p['hidden_test']}")
        for t in EXT[tid][:5]: print(f"    新隐藏            {t}")
        vals = {t.split("==")[-1].strip() for t in EXT[tid] if "==" in t}
        print(f"    …共 {len(EXT[tid])} 条，期望值有 {len(vals)} 种")
    sys.exit()

EXT = json.load(open(args.tests))
print(f"新隐藏测试：{len(EXT)} 道题，平均 {sum(len(v) for v in EXT.values())/max(1,len(EXT)):.1f} 条/题\n")
print(f"  {'评测':<28}{'覆盖':>7}{'旧秤':>8}{'严秤':>8}{'掉了':>8}   过旧秤但被新测试抓住的")
for f in args.raws:
    if not os.path.exists(f):
        print(f"  ⚠️ 找不到 {f}，跳过"); continue
    raw = json.load(open(f)); rs = rescore(raw, EXT)
    if not rs: print(f"  {os.path.basename(f):<28} 没有一道题有新测试"); continue
    old = sum(r["old"] for r in rs) / len(rs); new = sum(r["new"] for r in rs) / len(rs)
    caught = [r for r in rs if r["old"] and not r["new"]]
    by = collections.Counter(r["kind"] for r in caught)
    print(f"  {os.path.basename(f):<28}{len(rs)/len(raw['records']):>6.0%}{old:>8.3f}{new:>8.3f}{new-old:>+8.3f}   {len(caught)} 条  {dict(by.most_common(4))}")
