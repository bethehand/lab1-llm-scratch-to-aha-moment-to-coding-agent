#!/usr/bin/env python3
"""
collect_exercism.py —— 把 exercism/python 的 practice 练习整理成阶段 ①′ 的题库（从头写、给测试）。
    python3 collect_exercism.py --repo /path/to/exercism_py --out data/exercism.json [--heldout 30 --seed 0]
每道题：slug / 题面（instructions.md + append）/ 空壳 stub / 参考实现 example.py / 测试文件全文 / 测试函数名列表 / 模块名。
验收：参考实现在沙盒里跑测试文件必须全过（unittest），过不了的丢；只收「一个模块 + 一个测试文件」的题（有额外数据文件、多模块的丢）。
切分：按 seed 抽 --heldout 道留出（SFT / RL 都不碰），其余 train。
"""
import os, re, sys, json, ast, random, argparse, tempfile, shutil, subprocess, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--out", default="data/exercism.json")
ap.add_argument("--heldout", type=int, default=30)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--timeout", type=int, default=20)
args = ap.parse_args()

ROOT = os.path.join(args.repo, "exercises", "practice")


def read(p):
    return open(p, encoding="utf-8").read() if os.path.exists(p) else ""


def test_names(src):
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name.startswith("test"):
                    out.append(f.name)
    return out


def run_unittest(module_name, code, test_src, timeout):
    """在临时目录里放模块 + 测试文件，python -m unittest 跑一遍 → (全过?, 通过数, 总数, stderr 尾巴)"""
    d = tempfile.mkdtemp()
    try:
        open(os.path.join(d, module_name + ".py"), "w").write(code)
        open(os.path.join(d, module_name + "_test.py"), "w").write(test_src)
        try:
            p = subprocess.run([sys.executable, "-S", "-m", "unittest", "-q", module_name + "_test"], cwd=d, capture_output=True, text=True, timeout=timeout,
                               env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0", "PYTHONPATH": d})   # -I 含 -E 会忽略 PYTHONPATH 且不放 cwd；这里只是验参考实现，用 -S
        except subprocess.TimeoutExpired:
            return False, 0, 0, "timeout"
        err = p.stderr
        m = re.search(r"Ran (\d+) tests?", err)
        n = int(m.group(1)) if m else 0
        fails = len(re.findall(r"^(FAIL|ERROR):", err, re.M))
        return (p.returncode == 0 and n > 0), n - fails, n, err[-600:]
    finally:
        shutil.rmtree(d, ignore_errors=True)


rows, why = [], collections.Counter()
for slug in sorted(os.listdir(ROOT)):
    d = os.path.join(ROOT, slug)
    if not os.path.isdir(d):
        continue
    mod = slug.replace("-", "_")
    stub = read(os.path.join(d, mod + ".py"))
    test_src = read(os.path.join(d, mod + "_test.py"))
    ref = read(os.path.join(d, ".meta", "example.py")) or read(os.path.join(d, ".meta", "exemplar.py"))
    instr = read(os.path.join(d, ".docs", "instructions.md")) + "\n" + read(os.path.join(d, ".docs", "instructions.append.md"))
    intro = read(os.path.join(d, ".docs", "introduction.md"))
    if not (stub and test_src and ref):
        why["缺文件"] += 1; continue
    extra = [f for f in os.listdir(d) if f.endswith(".py") and f not in (mod + ".py", mod + "_test.py")]
    if extra:
        why["多模块"] += 1; continue
    data_files = [f for f in os.listdir(d) if not f.startswith(".") and not f.endswith((".py", ".md"))]
    if data_files:
        why["带数据文件"] += 1; continue
    imports = re.findall(r"^from (\w+) import", test_src, re.M) + re.findall(r"^import (\w+)", test_src, re.M)
    if any(i not in (mod, "unittest", "pytest", "math", "re", "itertools", "collections", "functools", "random", "string", "datetime", "fractions", "decimal") for i in imports):
        why["测试 import 了别的"] += 1; continue
    names = test_names(test_src)
    if len(names) < 3:
        why["测试太少"] += 1; continue
    if len(stub) > 1500:                                              # ledger / markdown / tree-building：空壳就是一份完整实现（重构题），不是从头写
        why["重构题（空壳带实现）"] += 1; continue
    ok, k, n, err = run_unittest(mod, ref, test_src, args.timeout)
    if not ok:
        why["参考实现没全过"] += 1; continue
    try:
        tree = ast.parse(ref)
        n_cls = sum(isinstance(x, ast.ClassDef) for x in tree.body)
        n_fn = sum(isinstance(x, ast.FunctionDef) for x in tree.body) + sum(isinstance(y, ast.FunctionDef) for x in tree.body if isinstance(x, ast.ClassDef) for y in x.body)
    except SyntaxError:
        n_cls, n_fn = 0, 0
    rows.append(dict(slug=slug, module=mod, text=(intro + "\n" + instr).strip(), stub=stub.strip("\n"), ref_code=ref.strip("\n"), test_src=test_src,
                     test_names=names, n_tests=n, ref_lines=len(ref.strip().splitlines()), n_classes=n_cls, n_funcs=n_fn, instr_chars=len(instr)))
    why["ok"] += 1

rng = random.Random(args.seed); rng.shuffle(rows)
for i, r in enumerate(rows):
    r["split"] = "heldout" if i < args.heldout else "train"
rows.sort(key=lambda r: r["slug"])
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
json.dump(rows, open(args.out, "w"), ensure_ascii=False)

import statistics as st
print(f"收下 {len(rows)} 道（train {sum(r['split']=='train' for r in rows)} / heldout {sum(r['split']=='heldout' for r in rows)}）→ {args.out}")
print("  丢弃：", {k: v for k, v in why.items() if k != "ok"})
T = [r["n_tests"] for r in rows]; L = [r["ref_lines"] for r in rows]; C = [r["instr_chars"] for r in rows]
print(f"  每题测试数 中位 {st.median(T):.0f}  P10 {sorted(T)[len(T)//10]}  P90 {sorted(T)[9*len(T)//10]}  最多 {max(T)}")
print(f"  参考实现行数 中位 {st.median(L):.0f}  P90 {sorted(L)[9*len(L)//10]}  最长 {max(L)}   带类的 {sum(r['n_classes']>0 for r in rows)} 道   函数/方法数中位 {st.median([r['n_funcs'] for r in rows]):.0f}")
print(f"  题面字符 中位 {st.median(C):.0f}  P90 {sorted(C)[9*len(C)//10]}  最长 {max(C)}（≈ token /3.5）")
print("  最长的三道（参考行数）：", [(r["slug"], r["ref_lines"], r["n_tests"]) for r in sorted(rows, key=lambda r: -r["ref_lines"])[:3]])
print("  测试最多的三道：", [(r["slug"], r["n_tests"]) for r in sorted(rows, key=lambda r: -r["n_tests"])[:3]])
