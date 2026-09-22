#!/usr/bin/env python3
"""
collect_mbpp_weak.py —— 把阶段 ① 的 MBPP 题库（data/code_bugs.json 里的 task：题面 / 参考实现 / 断言）做成 ex_env 能吃的「从头写」题（阶段 ③ 扩池）。
    python3 collect_mbpp_weak.py --out data/mbpp_weak.json [--ext data/tests_ext_all.json] [--limit 0]
每道题：
    text     一句描述 + 一条示例断言（题面留一条例子，和 exercism 题面自带例子同一标准；弱模式下模型看不到任何测试）
    stub     参考实现里每个顶层函数的签名 + pass（多函数的全列）
    test_src unittest 文件：示例断言 + 其余原断言 + 隐藏断言 + --ext 里 ref 造的 oracle 断言，各一个 test 方法；秤跑全部
    module   mbpp_<task_id>；slug mbpp-<task_id>；split 沿用题库的 train / heldout
只收参考实现在本机沙盒里全过 test_src 的题（和 collect_exercism 同一条规矩）
"""
import os, re, sys, ast, json, argparse, tempfile, shutil, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ex_env as E

ap = argparse.ArgumentParser()
ap.add_argument("--bugs", default="data/code_bugs.json")
ap.add_argument("--ext", default=None, help="strict_score.py --build 造的 oracle 断言（{task_id: [...]}）；没有就只用原来的 3 条")
ap.add_argument("--out", default="data/mbpp_weak.json")
ap.add_argument("--limit", type=int, default=0, help="只做前几道（本机验流水线用）")
ap.add_argument("--max-ext", type=int, default=20)
args = ap.parse_args()


def load_ext(path):
    if not path or not os.path.exists(path):
        return {}
    raw = json.load(open(path))
    out = {}
    items = raw.items() if isinstance(raw, dict) else ((r.get("task_id"), r) for r in raw)
    for k, v in items:
        if isinstance(v, dict):
            v = v.get("tests") or v.get("hidden") or v.get("asserts") or []
        tests = []
        for t in v:
            if isinstance(t, dict):
                t = t.get("test") or t.get("src") or t.get("code") or ""
            t = str(t).strip()
            if t.startswith("assert"):
                tests.append(t)
        out[str(k)] = tests
    return out


def signatures(ref_code):
    tree = ast.parse(ref_code)
    sigs = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            a = node.args
            names = [x.arg for x in a.posonlyargs + a.args]
            defaults = [None] * (len(names) - len(a.defaults)) + [ast.unparse(d) for d in a.defaults]
            parts = [n if d is None else f"{n}={d}" for n, d in zip(names, defaults)]
            if a.vararg: parts.append("*" + a.vararg.arg)
            for kw, d in zip(a.kwonlyargs, a.kw_defaults):
                parts.append(kw.arg if d is None else f"{kw.arg}={ast.unparse(d)}")
            if a.kwarg: parts.append("**" + a.kwarg.arg)
            sigs.append(f"def {node.name}({', '.join(parts)}):\n    pass")
    return sigs


def make_test_src(module, tests):
    lines = ["import unittest", "import math", "import re", "import collections", "import itertools", "import heapq", f"from {module} import *", "", "", "class MbppTest(unittest.TestCase):"]
    for i, t in enumerate(tests):
        lines.append(f"    def test_{i:02d}(self):")
        lines.append(f"        {t}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main():
    bugs = json.load(open(args.bugs))
    tasks = {}
    for r in bugs:
        tasks.setdefault(r["task_id"], r)
    ext = load_ext(args.ext)
    ids = sorted(tasks)
    if args.limit:
        ids = ids[: args.limit]
    out, stat = [], collections.Counter()
    tmp = tempfile.mkdtemp()
    for tid in ids:
        r = tasks[tid]
        vis = [t.strip() for t in r["visible_tests"] if t.strip().startswith("assert")]
        hid = [r["hidden_test"].strip()] if str(r.get("hidden_test", "")).strip().startswith("assert") else []
        extra = [t for t in ext.get(str(tid), []) if t not in vis and t not in hid][: args.max_ext]
        tests = vis + hid + extra
        if not vis or len(tests) < 3:
            stat["断言不够"] += 1; continue
        ref = ((r.get("setup") or "").strip() + "\n\n" + r["ref_code"].strip()).strip() + "\n"
        try:
            sigs = signatures(ref)
        except SyntaxError:
            stat["参考实现语法错"] += 1; continue
        if not sigs:
            stat["没有函数"] += 1; continue
        module = f"mbpp_{tid}"
        test_src = make_test_src(module, tests)
        res = E.run_unittests(module, ref, test_src, None, tmp, per_test=2.0, max_timeouts=2)
        if res.get("timeout") or res.get("load_error") or any(x["status"] != "ok" for x in res["rows"]) or len(res["rows"]) != len(tests):
            stat["参考实现没全过"] += 1; continue
        text = r["text"].strip() + "\n\nExample:\n```python\n" + vis[0] + "\n```"
        stub = "\n\n\n".join(sigs) + "\n"
        out.append(dict(slug=f"mbpp-{tid}", module=module, split=r.get("split", "train"), text=text, stub=stub, ref_code=ref, test_src=test_src,
                        test_names=[f"MbppTest.test_{i:02d}" for i in range(len(tests))], n_tests=len(tests), ref_lines=len(r["ref_code"].strip().splitlines()),
                        n_classes=0, n_funcs=len(sigs), instr_chars=len(text), source="mbpp", n_ext=len(extra)))
        stat["收"] += 1
    shutil.rmtree(tmp, ignore_errors=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False)
    sp = collections.Counter(p["split"] for p in out)
    nt = sorted(p["n_tests"] for p in out) or [0]
    print(f"题库 {len(tasks)} 道 → 收 {len(out)} 道（{dict(sp)}）→ {args.out}   丢：{dict((k, v) for k, v in stat.items() if k != '收')}")
    print(f"每题测试条数 中位 {nt[len(nt) // 2]} 最少 {nt[0]} 最多 {nt[-1]}   有 oracle 断言的 {sum(1 for p in out if p['n_ext'] > 0)} 道   参考实现行数中位 {sorted(p['ref_lines'] for p in out)[len(out) // 2] if out else 0}")
    if out:
        p = E.make_prob(out[0])
        print("── 样例（弱模式题面）──"); print(E.render(1, p, env=E.ExEnv(obs="weak"))[:1200])


if __name__ == "__main__":
    main()
