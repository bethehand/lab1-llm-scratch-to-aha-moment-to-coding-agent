"""从训练日志的 vLLM 唤醒/睡下时间戳反推四张卡各自的 rollout 时长（不用改训练器、不用重启）。
每一步：四卡在合梯度后同时醒来 → 各自 rollout → 各自睡下；「睡下 − 第一次醒来」= 该卡的 rollout 时长；最晚睡下的就是大家在合梯度处等的那张卡。
用法：python3 rank_spread.py codeRL_dyn.txt [--since 续训]   （--since：只看最后一次含该字样的行之后的步；默认取最后一次「续训」）"""
import re, sys, statistics as st
path = sys.argv[1]; since = "续训"
if "--since" in sys.argv: since = sys.argv[sys.argv.index("--since") + 1]
lines = open(path, errors="replace").read().splitlines()
start = max([i for i, l in enumerate(lines) if since in l], default=-1) + 1
TS = re.compile(r"^INFO (\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)")
def ts(l):
    m = TS.match(l)
    if not m: return None
    mo, d, h, mi, s = map(int, m.groups()); return ((mo * 31 + d) * 24 + h) * 3600 + mi * 60 + s
STEP = re.compile(r"^\s*(\d+)/(\d+) \|")
blocks, cur = [], dict(wake=[], sleep=[], r0=None)
for l in lines[start:]:
    t = ts(l)
    if t is not None and "wake up tags ['weights']" in l: cur["wake"].append(t)
    elif t is not None and "fall asleep" in l: cur["sleep"].append(t)
    elif "[rollout 计时]" in l:
        m = re.search(r"harness ([\d.]+) s", l); cur["r0"] = float(m.group(1)) if m else None
    m = STEP.match(l)
    if m:
        cur["step"] = int(m.group(1)); m2 = re.search(r"\| *(\d+)\+(\d+)s", l); cur["gt"] = (int(m2.group(1)), int(m2.group(2))) if m2 else None
        blocks.append(cur); cur = dict(wake=[], sleep=[], r0=None)
rows = []
for b, nb in zip(blocks, blocks[1:] + [None]):
    if not b["wake"] or not b["sleep"]: continue
    w0 = min(b["wake"]); sl = sorted(b["sleep"])[:4]                       # 评测步会多出一轮唤醒/睡下：只取前 4 次睡下（训练 rollout 的）
    per = [s - w0 for s in sl]
    wall = (min(nb["wake"]) - w0) if nb and nb["wake"] else None
    rows.append((b["step"], wall, b["r0"], per, b["gt"]))
print(f"{'步':>4} {'整步墙钟':>8} {'r0 harness':>10} {'四卡 rollout（醒→睡，升序）':>28} {'最慢−最快':>9}  rank0 行")
for step, wall, r0, per, gt in rows[-20:]:
    print(f"{step:>4} {('%.0f s' % wall) if wall else '   -':>8} {('%.1f s' % r0) if r0 else '   -':>10} {' '.join('%3.0f' % x for x in per):>28} {max(per)-min(per):>7.0f} s  {gt[0]}+{gt[1]}s" if gt else "")
W = [r[1] for r in rows if r[1]]; S = [max(r[3]) for r in rows]; F = [min(r[3]) for r in rows]; D = [max(r[3]) - min(r[3]) for r in rows]
if rows:
    print(f"\n{len(rows)} 步：整步墙钟 中位 {st.median(W):.0f} s（均 {st.mean(W):.0f}）| 最慢卡 rollout 中位 {st.median(S):.0f} s | 最快卡 中位 {st.median(F):.0f} s | 最慢−最快 中位 {st.median(D):.0f} s，> 15 s 的步占 {sum(d > 15 for d in D) / len(D):.0%}")
    print("注意：vLLM 的日志行不带 rank 号，这里看不出慢的是不是同一张卡；训练器新加的 [四卡 rollout] 行才带 rank 号和生成/环境/超时的分解")
