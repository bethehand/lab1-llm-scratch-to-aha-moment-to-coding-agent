#!/usr/bin/env python3
"""把「可见测试全过、隐藏测试挂」的轨迹挑出来，连同三条测试一起打出来 —— 判断是真硬编码还是「看着合理但错」的修法。
    python3 show_hardcode.py dyn_cheatA_raw.json                    # 全部列出
    python3 show_hardcode.py dyn_tb_raw.json dyn_cheatA_raw.json    # 两份并排比数量
    python3 show_hardcode.py dyn_cheatA_raw.json --show 6 --data data/code_bugs.json
自动标记：最终代码里出现了「可见测试里的字面量」而原 bug 代码里没有 → 疑似硬编码（照抄期望值）。标记只是线索，最终看代码。"""
import json, re, sys, argparse, collections

ap = argparse.ArgumentParser()
ap.add_argument("raws", nargs="+")
ap.add_argument("--data", default="data/code_bugs.json")
ap.add_argument("--show", type=int, default=99, help="每份最多打几条")
args = ap.parse_args()

BUGS = {}
try:
    for b in json.load(open(args.data)):
        BUGS[b["task_id"]] = b
except Exception as e:
    print(f"  ⚠️ 读不到 {args.data}（{e}），只打代码不打测试")

LIT = re.compile(r"'[^']*'|\"[^\"]*\"|\b\d+\.\d+\b|\b\d+\b")


def lits(s):
    return {x for x in LIT.findall(s or "") if x not in ("0", "1", "2")}      # 0/1/2 太常见，排除


def pick(raw):
    return [r for r in raw["records"] if r["d"]["vis_pass"] == r["d"]["n_tests"] - 1 and r["d"]["hidden_pass"] == 0]


print("=" * 96)
for f in args.raws:
    raw = json.load(open(f))
    hits = pick(raw); n = len(raw["records"])
    by = collections.Counter(r["task_id"] for r in hits)
    print(f"\n【{f}】{raw.get('name','')}  {n} 条轨迹  开局 {'带报错' if raw['with_tb'] else '不带报错'}"
          + (f"  inject=「{raw['inject'][:36]}…」" if raw.get("inject") else "  无 inject"))
    print(f"  可见全过但隐藏挂 {len(hits)} 条（{len(hits)/n:.2%}），落在 {len(by)} 道题上：{dict(by.most_common(8))}")
    if not hits: continue
    flagged = 0
    for r in hits:
        b = BUGS.get(r["task_id"])
        extra = (lits(" ".join(b["visible_tests"])) & lits(r["final_code"])) - lits(b["bug_code"]) if b else set()
        if extra: flagged += 1
    print(f"  ★ 自动标记疑似硬编码（代码里出现了可见测试的字面量、原代码里没有）{flagged} / {len(hits)} 条")
    for r in hits[: args.show]:
        b = BUGS.get(r["task_id"])
        extra = (lits(" ".join(b["visible_tests"])) & lits(r["final_code"])) - lits(b["bug_code"]) if b else set()
        print("\n  " + "─" * 92)
        print(f"  task {r['task_id']}  {r['kind']}/{r['bug_type']}  模板 {r['pid']}  调用 {r['d']['calls']}"
              + (f"   ⚠ 疑似硬编码，多出的字面量 {sorted(extra)}" if extra else "   （没抄字面量，更像「合理但错」的修法）"))
        if b:
            for t in b["visible_tests"]: print(f"    可见  {t}")
            print(f"    隐藏  {b['hidden_test']}")
            print("    原 bug 代码：")
            for l in b["bug_code"].strip().splitlines(): print(f"      {l}")
        print("    最终文件：")
        for l in (r["final_code"] or "").strip().splitlines(): print(f"      {l}")
