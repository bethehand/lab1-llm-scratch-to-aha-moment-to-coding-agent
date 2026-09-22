#!/usr/bin/env python3
"""pass@k 曲线 + 两个模型的「谁能做到」对照表。回答一个问题：RL 是把已有的路走稳了，还是开了新路（或者掐死了细路）。

    python3 pass_k.py sftc_s32_raw.json rl_s32_raw.json dyn_s32_raw.json
    python3 pass_k.py *_s32_raw.json --strict data/tests_ext.json      # 顺便用严秤再算一遍
pass@k 用无偏估计：某题 n 次里对了 c 次 → pass@k = 1 − C(n−c, k)/C(n, k)，不是「前 k 次里有没有对」。
"""
import os, json, argparse, itertools, collections
from math import comb
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("raws", nargs="+")
ap.add_argument("--strict", default=None, help="data/tests_ext.json：给的话用严秤再算一遍")
ap.add_argument("--data", default="data/code_bugs.json")
ap.add_argument("--workers", type=int, default=32)
ap.add_argument("--ks", default="1,2,4,8,16,32")
ap.add_argument("--unit", choices=["task", "file"], default="task",
                help="task = 一条轨迹算一个单位（K 个文件全修好才算对）；file = 多文件包里每个文件各算一个单位（★ K>1 时量支撑集要用这个："
                     "包级 pass@k 问的是「32 次里有没有一次同时修好 K 个」，那是可靠性不是覆盖面）")
args = ap.parse_args()
KS = [int(x) for x in args.ks.split(",")]


def pass_at_k(n, c, k):
    if k > n: return None
    if n - c < k: return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def curve(byprob, k):
    vals = [pass_at_k(n, c, k) for n, c in byprob.values()]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def load(f, strict_ext=None, BUGS=None):
    raw = json.load(open(f)); R = raw["records"]
    ok = {r["pi"]: r["d"]["correct"] > 0 for r in R} if False else None
    if strict_ext is None and args.unit == "file" and R and "per_file" in R[0]["d"]:
        flags = [(f"{r['pi']}#{i}", r.get("task_ids", [r.get("task_id")] * 9)[i] if isinstance(r.get("task_ids"), list) else r.get("task_id"), p["correct"] > 0)
                 for r in R for i, p in enumerate(r["d"]["per_file"])]
    elif strict_ext is None:
        flags = [(r["pi"], r.get("task_id") or r.get("pkg_id") or r.get("slug"), r["d"]["correct"] > 0) for r in R]   # ex 的题号是 slug
    else:
        import code_env as C
        def one(r):
            p = BUGS[r["task_id"]]; ext = strict_ext.get(str(r["task_id"]))
            if ext is None: return None
            d = C.tempfile.mkdtemp()
            try:
                k, oks, _ = C.run_tests(r["final_code"] or p["bug_code"], list(p["visible_tests"]) + [p["hidden_test"]] + ext,
                                        p.get("setup", ""), d, 10)
            finally:
                C.shutil.rmtree(d, ignore_errors=True)
            return (r["pi"], r["task_id"], all(oks))
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            flags = [x for x in ex.map(one, R) if x]
    by = {}
    for pi, tid, good in flags:
        n, c = by.get(pi, (0, 0))
        by[pi] = (n + 1, c + int(good))
    tid_of = {pi: tid for pi, tid, _ in flags}
    return raw, by, tid_of


BUGS = {b["task_id"]: b for b in json.load(open(args.data))} if args.strict else None
EXT = json.load(open(args.strict)) if args.strict else None

for tag, ext in ([("旧秤", None)] + ([("严秤", EXT)] if EXT else [])):
    print(f"\n{'='*92}\n  pass@k（{tag}，单位 = {'每个文件' if args.unit == 'file' else '整条轨迹'}）  —— 无偏估计，每题 n 次采样\n{'='*92}")
    print(f"  {'模型':<26}" + "".join(f"{'@'+str(k):>9}" for k in KS) + f"{'题数':>7}{'n':>5}")
    SOLVED = {}
    for f in args.raws:
        if not os.path.exists(f):
            print(f"  ⚠️ 找不到 {f}"); continue
        raw, by, tid_of = load(f, ext, BUGS)
        name = raw.get("name", os.path.basename(f))
        n_s = min(n for n, _ in by.values())
        row = "".join((f"{curve(by, k):>9.3f}" if curve(by, k) is not None else f"{'—':>9}") for k in KS)
        print(f"  {name[:25]:<26}{row}{len(by):>7}{n_s:>5}")
        SOLVED[name] = {tid_of[pi]: c > 0 for pi, (n, c) in by.items()}
    names = list(SOLVED)
    if len(names) >= 2:
        print(f"\n  ── 「{n_s} 次里至少对一次」的题目集合对照（这才是有没有路的直接证据）")
        for a, b in itertools.combinations(names, 2):
            A, B = SOLVED[a], SOLVED[b]
            ids = set(A) & set(B)
            only_a = [t for t in ids if A[t] and not B[t]]
            only_b = [t for t in ids if B[t] and not A[t]]
            both = sum(1 for t in ids if A[t] and B[t]); none = sum(1 for t in ids if not A[t] and not B[t])
            print(f"    {a[:22]:<24} vs {b[:22]:<24} 都能 {both:>3}   都不能 {none:>3}   只有前者能 {len(only_a):>3}   只有后者能 {len(only_b):>3}")
            if only_a[:6] or only_b[:6]:
                print(f"      {'':<24}    只有前者能的题号 {sorted(only_a)[:8]}   只有后者能的 {sorted(only_b)[:8]}")
