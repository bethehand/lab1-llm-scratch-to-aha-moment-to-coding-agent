#!/usr/bin/env python3
"""
★ 从 baseline_countdown.py 的 *_raw.json 里，把「搜索措辞有没有用」拆开看。

  池化 ★差 有一个混淆：难题更容易触发多轮试错、也更容易错，两者叠一起可能把正相关抵消成零。
  这里加三个视角：
    ① 组内 ★差   —— 同一道题的 8 条里，带的 vs 不带的，控制了难度
    ② 剂量响应   —— 试错 0 / 1~2 / 3~5 / 6+ 次，正确率各多少
    ③ 验证精度   —— 模型写了「perfect / correct / works」的那些，真的对了几成？
                    写了「too high / too low」的那些，标注方向跟实际算出来的对不对？

    python3 analyze_search.py cd_base_s8_off0_raw.json cd_out_mix_mp_ckpt_best_s8_off0_raw.json
"""
import sys, json, re
from collections import defaultdict
import numpy as np
import countdown as CD

SEARCH = re.compile(
    r"\b(too (?:high|low|big|small|large)|not (?:equal|right|correct)|doesn'?t (?:work|equal)"
    r"|does not (?:work|equal)|perfect|close|try (?:another|again|different|a different|the next)"
    r"|let'?s try|next,? try|another (?:combination|approach|way|try)|nope|wrong|incorrect)\b", re.I)
CLAIM_OK = re.compile(r"\b(perfect(?: match)?|correct|works|that'?s it|equals? the target|matches)\b", re.I)
# ★ 「X = V (too high)」这种带方向的标注 —— 抓表达式、数值、方向三样
ANNOT = re.compile(r"((?:\(|\d)[\d\s\+\-\*/\(\)]*?)\s*=\s*(-?\d+(?:\.\d+)?)\s*\(?\s*(too (?:high|low|big|small|large))", re.I)   # ★ 算式必须以数字或 ( 开头：否则上一行标注的 ")\n" 会被吞进算式，eval 失败被跳过 → 连续标注只数到第一条

TOOLCALL = re.compile(r"<calc>(.*?)</calc>\s*<result>(.*?)</result>", re.S)


def load(path):
    d = json.load(open(path))
    R = d["records"]
    if not R or "txt" not in R[0]:
        sys.exit(f"{path} 没有文本字段 —— 是旧脚本跑的，重跑一次")
    for r in R:                                   # ★ 工具版轨迹：<calc>E</calc><result>V</result> → 「E = V」，所有探针照用
        if "<calc>" in r["txt"]:
            r["txt"] = TOOLCALL.sub(r"\1 = \2", r["txt"])
    # ★ raw 里没存目标值，按 offset/n 从数据文件对回去
    P = json.load(open(d.get("data", "data/countdown.json")))[d["offset"]: d["offset"] + d["n"]]
    d["targets"] = {str(i): p["target"] for i, p in enumerate(P)}
    d["nums"] = {str(i): p["nums"] for i, p in enumerate(P)}
    return d, R


def gap(rs, key):
    y = [r["r"] for r in rs if r[key] > 0]; n = [r["r"] for r in rs if r[key] == 0]
    return (np.mean(y) - np.mean(n)) if (y and n) else float("nan"), len(y), len(n)


def report(path):
    d, R = load(path)
    tag = d.get("ckpt") or d["model"]
    print("=" * 78 + f"\n  {tag}   {len(R)} 条   温度 {d['temp']}\n" + "=" * 78)

    # ── ① 池化 vs 组内 ──
    g_pool, ny, nn = gap(R, "search")
    G = defaultdict(list)
    for r in R:
        G[r["pi"]].append(r)
    within = [gap(rs, "search")[0] for rs in G.values()]
    within = [x for x in within if x == x]                     # 去 nan（组里全带或全不带）
    print(f"\n  ── ① 搜索措辞 ★差 ──")
    print(f"    池化（所有题混在一起）     {g_pool:+.4f}   带 {ny} / 不带 {nn}")
    print(f"    ★ 组内（同一题里比，控难度） {np.mean(within):+.4f}   可比的题 {len(within)}/{len(G)}"
          f"   标准误 {np.std(within)/np.sqrt(len(within)):.4f}")

    # ── ② 剂量响应 ──
    print(f"\n  ── ② 剂量响应：试错次数 → 正确率 ──")
    buckets = [(0, 0, "0 次"), (1, 2, "1~2 次"), (3, 5, "3~5 次"), (6, 999, "6+ 次")]
    for lo, hi, name in buckets:
        rs = [r for r in R if lo <= r["search"] <= hi]
        if rs:
            c = np.mean([r["d"]["correct"] for r in rs]); L = np.mean([r["L_eff"] for r in rs])
            print(f"    {name:8s}  {len(rs):4d} 条   正确率 {c:.3f}   有效长度 {L:.0f}")

    # ── ③ 验证精度 ──
    print(f"\n  ── ③ 验证精度：模型说「对了」的时候，真的对了吗 ──")
    claim = [r for r in R if CLAIM_OK.search(r["txt"])]
    if claim:
        prec = np.mean([r["d"]["correct"] for r in claim])
        base = np.mean([r["d"]["correct"] for r in R])
        print(f"    写了 perfect/correct/works 的   {len(claim):4d} 条   其中真对 {prec:.3f}   （全体正确率 {base:.3f}）")
        print(f"    ★ 若 {prec:.3f} ≈ {base:.3f} → 「说对了」不携带信息，是表演；若明显更高 → 真在验证")

    # ── ⑤ ★★ 窄行为的 ★差：ANNOT（算式 = 值 (方向)）有没有 ──
    #    前面 ①② 用的是宽泛的 SEARCH 词表。真正被放大的是这个窄形态，要单独算它的 ★差。
    for r in R:
        r["_na"] = sum(1 for m in ANNOT.finditer(r["txt"]))
    g_pool, ny, nn = gap(R, "_na")
    within = [gap(rs, "_na")[0] for rs in G.values()]
    within = [x for x in within if x == x]
    yc = np.mean([r["d"]["correct"] for r in R if r["_na"] > 0]) if ny else float("nan")
    nc = np.mean([r["d"]["correct"] for r in R if r["_na"] == 0]) if nn else float("nan")
    print(f"\n  ── ⑤ ★★ 窄行为 ★差：带「算式 = 值 (方向)」标注的 vs 不带的 ──")
    print(f"    池化 ★差   {g_pool:+.4f}   带 {ny} 条 / 不带 {nn} 条")
    if within:
        print(f"    组内 ★差   {np.mean(within):+.4f}   可比的题 {len(within)}   标准误 {np.std(within)/np.sqrt(len(within)):.4f}")
    print(f"    正确率     带 {yc:.3f}   不带 {nc:.3f}")
    print(f"    ★ 这个才是「条件 ②」的直接读数：正 → 这个行为跟奖励挂钩，RL 会放大它")

    # ── ③b 方向标注对不对 ── ★ 分层：撞顶/没撞顶 × 答对/答错，错再分「算错」和「方向标反」
    max_new = d.get("max_new") or max(r.get("L", 0) for r in R)   # 旧 raw 没存 max_new → 用文件里的最长当上限
    C = {}   # key → [ok, arith_bad, dir_bad]
    def acc(key, kind):
        C.setdefault(key, [0, 0, 0])[kind] += 1
    for r in R:
        tgt = None
        cut = r.get("L", 0) >= max_new - 2
        corr = (r.get("d") or {}).get("correct", 0) > 0
        for m in ANNOT.finditer(r["txt"]):
            expr, val, dirn = m.group(1), float(m.group(2)), m.group(3).lower()
            try:
                real = CD.safe_eval(CD.clean_expr(expr))
            except Exception:
                continue
            if tgt is None:
                tgt = d["targets"][str(r["pi"])]
            said_high = "high" in dirn or "big" in dirn or "large" in dirn
            kind = 1 if abs(real - val) > 1e-3 * max(1.0, abs(real)) else (0 if said_high == (real > tgt) else 2)   # ★ 相对容差：工具只印 4 位有效数字
            for key in ("全部", "撞顶" if cut else "没撞顶", "答对的轨迹" if corr else "答错的轨迹"):
                acc(key, kind)
    if C:
        print(f"\n  ── ③b 「X = V (too high/low)」标注：算得对且方向对的比例 ──")
        print(f"    {'':12} {'条数':>7} {'精度':>7} {'算错':>7} {'方向反':>7}")
        for key in ("全部", "没撞顶", "撞顶", "答对的轨迹", "答错的轨迹"):
            if key not in C: continue
            o, a, dr = C[key]; n = o + a + dr
            print(f"    {key:12} {n:>7} {o/max(n,1):>7.3f} {a/max(n,1):>7.3f} {dr/max(n,1):>7.3f}")
        print(f"    ★ 看「没撞顶」那行才是程序本身的精度；「撞顶」是循环里的垃圾标注。低于 0.7 → 试错是编的")
    # ── ③c 任意标签的尝试行（judge_v2.terms）：「(not 30)」「(no)」也算，改标签躲不掉；方向只在写了 high/low 时验 ──
    try:
        import judge_v2 as J2
        agg = dict(n=0, ok=0, dup=0, viol=0, fc=0, claim=0, tr=0, tr_lt3_wrong=0)
        for r in R:
            t = J2.terms(r["txt"], d["nums"][str(r["pi"])], d["targets"][str(r["pi"])])
            agg["n"] += t["n_annot"]; agg["ok"] += t["n_ok"]; agg["dup"] += t["dup"]; agg["viol"] += t["viol"]
            agg["fc"] += t["false_claim"]; agg["claim"] += (t["n_claim"] > 0); agg["tr"] += 1
            agg["tr_lt3_wrong"] += (t["n_annot"] < 3 and (r.get("d") or {}).get("correct", 0) <= 0)
        n, tr = agg["n"], max(agg["tr"], 1)
        print(f"\n  ── ③c 任意标签的尝试行（v2.1 口径，改标签躲不掉）──")
        print(f"    尝试行/条 {n/tr:.2f}   算对率 {agg['ok']/max(n,1):.3f}   重复行/条 {agg['dup']/tr:.2f}   违规行/条 {agg['viol']/tr:.2f}")
        print(f"    假宣告轨迹 {agg['fc']/tr:.3f}   写了宣告的 {agg['claim']/tr:.3f}   答错且尝试 < 3 行 {agg['tr_lt3_wrong']/tr:.3f}")
    except ImportError:
        pass


for p in sys.argv[1:]:
    report(p)
print()
