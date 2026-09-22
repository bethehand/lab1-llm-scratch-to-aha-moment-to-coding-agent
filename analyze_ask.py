#!/usr/bin/env python3
"""
analyze_ask.py —— 臂 A 评测的读数：把 baseline_countdown_vllm.py --ask 落盘的 *_raw.json 按「题的难度」拆开，看模型什么时候问、问完验不验、专家错了抓不抓。

    python3 analyze_ask.py cd_out_sft_ask_ckpt_tool_ask_countdown4_raw.json [更多 raw …]

难度标签不看模型，看老师程序：enum_traces.trace 在第 0/1 层就解出 = 易题（模型自己该搜到，不该问）；要到第 2 层或解不出 = 硬题（该问）。
每条样本量：对没对、问了几次、问的位置（第 0 层前 / 第 0 层后第 1 层前 / 第 1 层后）、问完有没有紧跟 <calc> 验证、
专家给的式子对不对、专家错时模型是抓住了（没把它写进 answer）还是照抄。
"""
import sys, os, json, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import countdown as CD
import enum_traces as ET
import calc_tool_vllm as T

REPLY_RE = re.compile(r"<ask>(.*?)</ask>\s*<reply>(.*?)</reply>", re.S)
ANS_RE = re.compile(r"<answer>(.*?)</answer>", re.S)


def hard_label(nums, target):
    r = ET.trace(nums, target)
    return r is None or r[2] >= 2


def eq_of(reply):
    a = reply.strip().split("\n")[0]
    return a.split("EQUATION:")[-1].strip().split("=")[0].strip() if "EQUATION:" in a else a.strip()


def value_ok(eq, target):
    v = T.evaluate(eq)
    return v != "error" and abs(float(v) - target) < 1e-9


def ask_position(txt):
    i = txt.find("<ask>")
    if i < 0: return None
    head = txt[:i]
    if "Pair products" in head: return "l1后"
    if "sign patterns" in head: return "l0后"
    return "更早"


def analyze(path):
    raw = json.load(open(path))
    D = json.load(open(raw["data"])); P = D[raw["offset"]: raw["offset"] + raw["n"]]
    hard = {pi: hard_label(p["nums"], p["target"]) for pi, p in enumerate(P)}
    rows = []
    for r in raw["records"]:
        p = P[r["pi"]]; txt = r["txt"]; tgt = p["target"]
        pairs = REPLY_RE.findall(txt)
        asks = max(r.get("asks", 0), len(pairs))                  # 没开求助工具时模型自己写的 <ask>…<reply> 也数（自问自答 + 验证）
        ans = ANS_RE.findall(txt); ans = ans[-1].strip() if ans else ""
        verified = sum(1 for m in REPLY_RE.finditer(txt) if "<calc>" in txt[m.end(): m.end() + 200])
        exp_wrong = [q for q, a in pairs if a.strip() != "error" and not value_ok(eq_of(a), tgt)]
        exp_err = sum(1 for q, a in pairs if a.strip() == "error")
        copied_wrong = sum(1 for q, a in pairs if a.strip() != "error" and not value_ok(eq_of(a), tgt)
                           and ans.replace(" ", "") == eq_of(a).replace(" ", ""))
        rows.append(dict(pi=r["pi"], hard=hard[r["pi"]], ok=r["d"]["correct"] > 0, asks=asks, asked=asks > 0,
                         pos=ask_position(txt), verified=verified, n_reply=len(pairs), exp_wrong=len(exp_wrong), exp_err=exp_err,
                         copied_wrong=copied_wrong, fake=r.get("fake", 0), cut=r["L"] >= raw["max_new"] - 2, calls=r.get("calls", 0), L=r["L"]))
    pset = raw.get("prompt_set")
    if pset:                                                  # 改写模板：题 i 用集合里第 i % len 个，按模板拆一行
        import prompts_cd as PC
        ids = PC.SETS[pset]
        for x in rows:
            x["pid"] = ids[x["pi"] % len(ids)]
    n_hard = sum(hard.values())
    print(f"\n{os.path.basename(path)}   {raw['n']} 题 × {raw['n_samples']}   老师标签：硬题 {n_hard} 道 / 易题 {raw['n'] - n_hard} 道   T={raw['temp']}  max_new {raw['max_new']}")
    print(f"{'':6} {'条':>5} {'pass@1':>7} {'求助率':>7} {'次/条':>6} {'问后验证':>8} {'位置 l1后/l0后/更早':>20} {'专家错':>6} {'抓住':>5} {'照抄':>5} {'没答':>5} {'撞顶':>6} {'假':>4} {'调用/条':>7} {'问了的对':>8} {'没问的对':>8}")
    groups = [("硬题", [x for x in rows if x["hard"]]), ("易题", [x for x in rows if not x["hard"]]), ("全部", rows)]
    if pset:
        groups += [(f"模板{pid}", [x for x in rows if x.get("pid") == pid]) for pid in sorted({x.get("pid") for x in rows})]
    for name, sel in groups:
        if not sel: continue
        n = len(sel); asked = [x for x in sel if x["asked"]]; not_asked = [x for x in sel if not x["asked"]]
        pos = {k: sum(1 for x in asked if x["pos"] == k) for k in ("l1后", "l0后", "更早")}
        n_reply = sum(x["n_reply"] for x in sel); ver = sum(x["verified"] for x in sel)
        ew = sum(x["exp_wrong"] for x in sel); cw = sum(x["copied_wrong"] for x in sel)
        print(f"{name:6} {n:>5} {np.mean([x['ok'] for x in sel]):>7.3f} {len(asked)/n:>7.3f} {np.mean([x['asks'] for x in sel]):>6.2f} "
              f"{(ver / n_reply if n_reply else 0):>8.2f} {pos['l1后']:>8}/{pos['l0后']}/{pos['更早']:<8} {ew:>6} {ew - cw:>5} {cw:>5} {sum(x['exp_err'] for x in sel):>5} "
              f"{np.mean([x['cut'] for x in sel]):>6.3f} {sum(x['fake'] for x in sel):>4} {np.mean([x['calls'] for x in sel]):>7.1f} "
              f"{(np.mean([x['ok'] for x in asked]) if asked else float('nan')):>8.3f} {(np.mean([x['ok'] for x in not_asked]) if not_asked else float('nan')):>8.3f}")
    print("读法：求助率 = 有 ≥1 次 <ask> 的样本占比；问后验证 = <reply> 之后 200 字符内出现 <calc> 的比例；位置 = 第一次 <ask> 出现在第 1 层之后 / 只做完第 0 层 / 更早；"
          "专家错 = 回复的式子值 ≠ 目标；抓住 = 专家错但 answer 没照抄；照抄 = 把错式子写进了 answer；撞顶 = 长度到 max_new")


SHOW = 0
argv = sys.argv[1:]
if "--show" in argv:                                          # --show N：每份 raw 打 N 条有求助的完整轨迹（先挑专家答错的，再挑普通的）
    i = argv.index("--show"); SHOW = int(argv[i + 1]); argv = argv[:i] + argv[i + 2:]


def show(path, n):
    raw = json.load(open(path))
    D = json.load(open(raw["data"])); P = D[raw["offset"]: raw["offset"] + raw["n"]]
    recs = [r for r in raw["records"] if r.get("asks", 0) > 0 or REPLY_RE.search(r["txt"])]
    def wrong(r):
        return any(a.strip() != "error" and not value_ok(eq_of(a), P[r["pi"]]["target"]) for _, a in REPLY_RE.findall(r["txt"]))
    picked = [r for r in recs if wrong(r)][: max(1, n // 2)]
    picked += [r for r in recs if r not in picked][: n - len(picked)]
    for r in picked:
        p = P[r["pi"]]
        print("─" * 100); print(f"【题 {r['pi']}】nums {p['nums']}  target {p['target']}   得分 {r['r']:.2f}   求助 {r.get('asks', 0)} 次   {r['L']} token"
                              + ("   ★ 专家答错过" if wrong(r) else ""))
        print(r["txt"])


for path in argv:
    analyze(path)
    if SHOW:
        show(path, SHOW)
