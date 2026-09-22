#!/usr/bin/env python3
"""
一条命令看全部：loss 曲线 + 四张 GPU 状态 + 健康检查。

用法:
    python3 monitor.py                     # 看一次
    watch -n 30 'python3 monitor.py'       # 每 30 秒刷新
    python3 monitor.py --session train     # 指定 tmux 会话名
"""
import re, sys, subprocess, argparse, os

ap = argparse.ArgumentParser()
ap.add_argument("--session", default="train", help="tmux 会话名")
ap.add_argument("--log", default=None, help="直接读日志文件（不从 tmux 抓）")
ap.add_argument("--plot", action="store_true", help="额外画 ASCII loss 曲线")
args = ap.parse_args()

# ══ 拿训练输出：优先读日志文件，否则从 tmux 抓 scrollback ══
text = ""
if args.log and os.path.exists(args.log):
    text = open(args.log, errors="ignore").read()
else:
    for target in (f"{args.session}:0.0", args.session):
        try:
            text = subprocess.run(["tmux", "capture-pane", "-p", "-S", "-6000",
                                   "-t", target],
                                  capture_output=True, text=True, timeout=10).stdout
            if "step" in text:
                break
        except Exception:
            pass

pat = re.compile(r"step\s+([\d,]+)/([\d,]+)\s+\|\s+loss\s+([\d.naN]+)\s+\|\s+lr\s+([\d.e+-]+)"
                 r"\s+\|\s+norm\s+([\d.naN]+)\s+\|\s+(\d+)ms\s+\|\s+MFU\s+([\d.]+)%\s+\|\s+([\d.]+)GB")
rows = []
for m in pat.finditer(text):
    rows.append((int(m.group(1).replace(",", "")), int(m.group(2).replace(",", "")),
                 float(m.group(3)), float(m.group(4)), float(m.group(5)),
                 int(m.group(6)), float(m.group(7)), float(m.group(8))))

# 日志里可能混着之前几次运行（5b/5c 的 scrollback）。
# 找最后一次「step 回退」的位置，只保留它之后的记录。
# 不然那几个两三万毫秒的首步（含 torch.compile）会把平均步时拉高好几倍。
start = 0
for i in range(1, len(rows)):
    if rows[i][0] <= rows[i-1][0]:      # step 号没往前走 = 新的一次运行开始了
        start = i
n_dropped = start
rows = rows[start:]

print()
if not rows:
    print("  找不到训练输出。可能还在 torch.compile 编译中，")
    print(f"  或者 tmux 会话名不是 '{args.session}' —— 用 --session 指定，或 --log 直接读日志。")
else:
    step, tot, loss, lr, norm, ms, mfu, gb = rows[-1]
    log_every = rows[1][0] - rows[0][0] if len(rows) > 1 else 1
    recent = [r[5] for r in rows[-30:]] if len(rows) > 30 else [r[5] for r in rows[1:]] or [ms]
    avg_ms = sum(recent) / len(recent)
    eta = (tot - step) * avg_ms / 1000
    done = sum(r[5] for r in rows[1:]) * log_every / 1000

    bar_n = int(38 * step / tot)
    print(f"  ┌── 训练 ───────────────────────────────────────────┐")
    print(f"    step {step:,} / {tot:,}   [{'█'*bar_n}{'░'*(38-bar_n)}] {100*step/tot:.1f}%")
    print(f"    loss {loss:.4f}    lr {lr:.2e}    norm {norm:.2f}")
    print(f"    {avg_ms:.0f} ms/步    MFU {mfu:.1f}%    显存 {gb:.1f} GB")
    print(f"    已用 {done/60:.0f} 分    剩余 {eta/60:.0f} 分")
    if n_dropped:
        print(f"    （日志里有 {n_dropped} 行属于之前的运行，已忽略）")

    # eval 记录
    ev = re.findall(r"eval @ ([\d,]+): train ([\d.]+)\s+val ([\d.]+)", text)
    if ev:
        print(f"\n    ── eval ──")
        for st, tr, va in ev[-4:]:
            gap = float(va) - float(tr)
            f = "  ⚠️ val 远低，查污染" if gap < -0.3 else ("  ⚠️ 过拟合迹象" if gap > 0.5 else "")
            print(f"    step {st:>7}   train {tr}   val {va}   ({gap:+.4f}){f}")

    # 健康检查
    warn = []
    if loss != loss:
        warn.append("❌ loss 是 NaN，立刻 Ctrl-C")
    if len(rows) > 25 and loss > max(r[2] for r in rows[-25:-12]):
        warn.append("⚠️ loss 近 25 步在上升")
    if norm > 5:
        warn.append(f"⚠️ grad norm {norm:.1f} 偏高（正常 0.3~2.0）")
    if mfu < 45:
        warn.append(f"⚠️ MFU {mfu:.0f}%（正常约 55%）")
    # 泄漏的信号是「持续增长」，不是「数值高」。
    # 拿最近的显存跟前 20% 的记录比，涨超过 15% 才告警。
    if len(rows) > 20:
        early = max(r[7] for r in rows[2:max(3, len(rows)//5)])
        if gb > early * 1.15:
            warn.append(f"⚠️ 显存从 {early:.1f} 涨到 {gb:.1f} GB，可能泄漏")
    print(f"\n    {'   '.join(warn) if warn else '✅ 训练指标全部正常'}")

    if args.plot and len(rows) > 3:
        H, W = 12, 48
        ls = [r[2] for r in rows]
        lo, hi = min(ls), max(ls)
        pad = (hi - lo) * .05 or .1
        lo, hi = lo - pad, hi + pad
        g = [[" "] * W for _ in range(H)]
        for r in rows:
            x = min(W-1, int((W-1) * r[0] / tot))
            y = min(H-1, max(0, int((H-1) * (hi - r[2]) / (hi - lo))))
            g[y][x] = "●"
        print()
        for i in range(H):
            v = hi - (hi-lo)*i/(H-1)
            print(f"    {v:6.2f} │" + "".join(g[i]) if i % 2 == 0 else "           │" + "".join(g[i]))
        print("           └" + "─"*W)

# ══ GPU ══
print(f"\n  ┌── GPU ────────────────────────────────────────────┐")
try:
    q = ("index,utilization.gpu,temperature.gpu,power.draw,power.limit,"
         "memory.used,memory.total,clocks.sm")
    out = subprocess.run(["nvidia-smi", f"--query-gpu={q}",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=10).stdout.strip()
    tot_pwr = 0
    for line in out.splitlines():
        i, util, temp, pwr, plim, mused, mtot, clk = [x.strip() for x in line.split(",")]
        util, temp, pwr, mused = float(util), float(temp), float(pwr), float(mused)
        tot_pwr += pwr
        fu = "✅" if util >= 90 else ("⚠️" if util >= 70 else "❌")
        ft = "✅" if temp < 80 else ("⚠️" if temp < 86 else "❌降频")
        bar = "█" * int(util/10) + "░" * (10 - int(util/10))
        print(f"    GPU{i}  [{bar}] {util:3.0f}%  {temp:2.0f}°C{ft}  "
              f"{pwr:5.1f}/{plim}W  {mused/1024:.1f}/{float(mtot)/1024:.0f}GB  {clk}MHz")
    print(f"    合计功耗 {tot_pwr:.0f} W")
except FileNotFoundError:
    print("    （本机没有 nvidia-smi）")
except Exception as e:
    print(f"    读取失败: {e}")
print()
