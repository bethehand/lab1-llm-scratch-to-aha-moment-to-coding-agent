#!/usr/bin/env python3
"""生成报告用的 SVG 图（文字转路径，任何机器上显示一致）。数字全部来自 PLAN.md。
    python3 docs/tools/make_charts.py            # 中文 → docs/assets/
    LANG_EN=1 python3 docs/tools/make_charts.py  # 英文 → docs/assets/en/
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

EN = os.environ.get("LANG_EN") == "1"
rcParams["font.family"] = ["Hiragino Sans GB", "PingFang HK", "Heiti TC", "DejaVu Sans"] if not EN else ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
rcParams["svg.fonttype"] = "path"
rcParams["axes.unicode_minus"] = False
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False
rcParams["font.size"] = 11
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "assets", "en" if EN else "")
os.makedirs(OUT, exist_ok=True)
INK, MUTED, GRID = "#1B1F26", "#5F6773", "#E1E4E8"
STRONG, WEAK, TINT, SFT_C, RL_C = "#1F5F8B", "#B45A1C", "#BFD6E8", "#8FA9C2", "#1F5F8B"

# 中英对照：键是中文原文，值是英文
T = {
    "Qwen2.5-1.5B Base，零训练": "Qwen2.5-1.5B Base, untrained",
    "混训 600 步，CD-4 零训练": "Mixed training 600 steps, no CD-4 training",
    "mix4-300，三代 RL 共 900 步": "mix4-300, three RL rounds, 900 steps",
    "臂 0 只 RL 再 300 步": "Arm 0: RL only, +300 steps",
    "臂 2 自蒸馏 SFT + RL": "Arm 2: self-distillation SFT + RL",
    "臂 1 程序老师撒种 SFT": "Arm 1: program-teacher seeding SFT",
    "臂 1 撒种 SFT + RL": "Arm 1: seeding SFT + RL",
    "臂 T 计算器 SFT": "Arm T: calculator tool SFT",
    "臂 T 计算器 SFT + RL，v1 秤": "Arm T: calculator SFT + RL, v1 scale",
    "臂 S 秤做减法 + 筛池 + 锚": "Arm S: subtractive scale + pool screening + anchor",
    "臂 A 求助教材 SFT，不开求助": "Arm A: ask-expert SFT, expert unavailable",
    "臂 A RL 后，开求助": "Arm A after RL, expert available",
    "Countdown 4 数：pass@1 一级级怎么抬上来的": "Countdown with 4 numbers: how pass@1 was lifted step by step",
    "pass@1（池外 100 题 × 8，温度 1）": "pass@1 (100 held-out problems × 8 samples, temperature 1)",
    "浅色 = SFT 后，深色 = RL 后。最后一根开了专家出口，其余都是模型自己解题；开出口的 .98 里硬题 .946。": "Light = after SFT, dark = after RL. Only the last bar has the expert available; hard problems alone score .946 there.",
    "SFT-②-B\n强房间": "SFT-②-B\nstrong room", "RL-②-B\n强房间": "RL-②-B\nstrong room", "SFT-②-B\n弱房间": "SFT-②-B\nweak room",
    "RL-②-B\n弱房间": "RL-②-B\nweak room", "SFT-③\n弱房间": "SFT-③\nweak room", "RL-③\n弱房间": "RL-③\nweak room",
    "首版": "first version",
    "pass@1（100 题 × 8，温度 1）": "pass@1 (100 problems × 8, temperature 1)",
    "观测的价格：拿走测试，价全落在首版之后的改循环里": "The price of observation: remove the tests and the whole price lands in the fix loop after the first version",
    "实心 = 首版就对且保住；浅色 = 改循环赚到；斜线 = 改循环丢掉；虚线框到首版高度。强房间首版按可见测试记，为上界。": "Solid = first version correct and kept; light = gained in the fix loop; hatched = lost in the fix loop; dashed box = first-version height. Strong-room first versions are scored on visible tests (upper bound).",
    "单进程 HF，估算": "Single-process HF, estimated", "torchrun 四进程": "torchrun, four processes", "工具版 HF，臂 T": "Tool-use HF, Arm T",
    "vLLM 同卡共存": "vLLM co-located on the same GPUs", "代码题 ①，vfork 前": "Coding stage ①, before vfork", "① 动态采样 + vfork": "① dynamic sampling + vfork",
    "长程 ②-A，异步 + 缓存": "Long-horizon ②-A, async harness + cache", "弱观测 ③，序列长 5 倍": "Weak observation ③, 5× longer sequences",
    "一步 RL 的墙钟时间（秒，四卡）": "Wall-clock time per RL step (seconds, four GPUs)",
    "基建：每一格都是量到瓶颈再修": "Infrastructure: every rung was measured before it was fixed",
    "红色两格是任务变难后的回升（工具调用、代码沙盒），随后各被一项基建压回去。": "The two red rungs are regressions when the task got harder (tool calls, code sandboxes); each was pushed back down by one piece of infrastructure.",
    "S 形拟合 A = .330, C_mid ≈ 第 40 步, B = 2.6": "Sigmoid fit A = .330, C_mid ≈ step 40, B = 2.6",
    "仪表盘实测（20 未见 + 20 见过，温度 1 × 4）": "Dashboard measurements (20 unseen + 20 seen, temperature 1 × 4)",
    "渐近线 A = .330": "asymptote A = .330", "实际停在 150 步": "stopped at step 150",
    "RL 步数": "RL steps", "仪表盘通过率": "dashboard pass rate",
    "阶段 ③：第 75 步到顶，续跑到 300 步只多 .003": "Stage ③: saturated by step 75; running to step 300 would add .003",
    "原始 Qwen2.5-1.5B-Instruct": "Original Qwen2.5-1.5B-Instruct", "GRPO 第 100 步": "GRPO step 100", "GRPO 第 300 步": "GRPO step 300",
    "k（每题采样条数，温度 1）": "k (samples per problem, temperature 1)",
    "k = 64 三条线重合在 .980：RL 只重排，不创造": "At k = 64 all three meet at .980: RL reorders, it does not create",
    "GSM8K GRPO：pass@1 +.21，pass@64 一分没动": "GSM8K GRPO: pass@1 +.21, pass@64 unchanged",
    "① 修 bug": "① fix a bug", "②-A 拼包 K=3\n（包级）": "②-A bundle K=3\n(bundle level)", "②-B exercism\n从头写": "②-B exercism\nfrom scratch", "③ 弱观测\n自己写测试": "③ weak obs.\nself-written tests",
    "SFT 后 pass@1": "pass@1 after SFT", "RL 后 pass@1": "pass@1 after RL", "pass@32 天花板": "pass@32 ceiling",
    "通过率（100 × 8 温度 1；横线 = 100 × 32）": "pass rate (100 × 8, temperature 1; bars = 100 × 32)",
    "编程线四级：RL 抬 pass@1，天花板几乎不动": "The four coding stages: RL lifts pass@1, the ceiling barely moves",
    "SFT-B 若各文件独立：.546^K": "SFT-B if files were independent: .546^K", "SFT-B 实测": "SFT-B measured",
    "RL-bin 若各文件独立：.703^K": "RL-bin if files were independent: .703^K", "RL-bin 实测": "RL-bin measured",
    "K，一个包里的文件数": "K, files per bundle", "整包全修好的比例，对数轴": "fraction of bundles fully fixed, log axis",
    "②-A 独立拼包：实测低于乘积基线，RL 把长度的代价收回一半": "②-A independent bundles: measured below the product baseline; RL recovers half of the length penalty",
    "① 修 bug，100 × 32": "① fix a bug, 100 × 32", "②-B 训练题，100 × 32": "②-B training problems, 100 × 32", "②-B 留出题，30 × 32": "②-B held-out problems, 30 × 32",
    "RL（动态采样趟）": "RL (dynamic-sampling run)",
    "k ≥ 16 时 RL 反而更低：削尾巴": "for k ≥ 16 RL is lower: the tail is cut",
    "编程线的 pass@k：RL 抬 pass@1，大 k 处不动或反降": "pass@k on the coding line: RL lifts pass@1; at large k it is flat or lower",
    "起点 hf_armA": "start: hf_armA", "E1 纯 RL": "E1 pure RL", "E3 课程 20→100": "E3 curriculum 20→100", "E2 撒种 SFT": "E2 seeding SFT", "E2 撒种 SFT + RL": "E2 seeding SFT + RL",
    "训练模板，N=100": "training prompts, N=100", "留出模板，N=100": "held-out prompts, N=100", "N=1000，越界，10 轮刚够二分": "N=1000, out of range, 10 rounds just enough for bisection",
    "猜中率（100 个秘密数 × 8，温度 1）": "success rate (100 secret numbers × 8, temperature 1)",
    "猜数字：撒种 200 条示范 2 分钟，超过两条 RL 臂；N=1000 的差距来自「写下区间」": "Number guessing: 200 seeded demos in 2 minutes beat both RL arms; the N=1000 gap comes from writing the interval down",
}
def _(s):
    return T.get(s, s) if EN else s

def save(fig, name):
    p = os.path.join(OUT, name); fig.savefig(p, format="svg", bbox_inches="tight"); plt.close(fig); print("wrote", p)

def hbar_ladder(rows, title, xlabel, name, xmax=1.0, colors=None, note=None):
    fig, ax = plt.subplots(figsize=(8.4, 0.46 * len(rows) + 1.4))
    labels = [_(r[0]) for r in rows]; vals = [r[1] for r in rows]
    cols = colors or [STRONG] * len(rows)
    y = list(range(len(rows)))[::-1]
    ax.barh(y, vals, color=cols, height=0.62)
    for yi, v in zip(y, vals):
        ax.text(v + 0.008 * xmax, yi, f"{v:.3f}" if xmax <= 1 else f"{v:g}", va="center", fontsize=10.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_xlim(0, xmax * 1.12)
    ax.set_xlabel(_(xlabel), color=MUTED); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.set_title(_(title), loc="left", fontsize=13, color=INK, pad=10)
    if note: fig.text(0.01, -0.02, _(note), fontsize=9, color=MUTED)
    save(fig, name)

# 1 Countdown-4 pass@1 阶梯（池外 [95000:95100] 100 题 × 8，温度 1；同口径「不开求助」）
hbar_ladder([
    ("Qwen2.5-1.5B Base，零训练", 0.0175), ("混训 600 步，CD-4 零训练", 0.208), ("mix4-300，三代 RL 共 900 步", 0.393),
    ("臂 0 只 RL 再 300 步", 0.471), ("臂 2 自蒸馏 SFT + RL", 0.473), ("臂 1 程序老师撒种 SFT", 0.506), ("臂 1 撒种 SFT + RL", 0.563),
    ("臂 T 计算器 SFT", 0.710), ("臂 T 计算器 SFT + RL，v1 秤", 0.711), ("臂 S 秤做减法 + 筛池 + 锚", 0.748),
    ("臂 A 求助教材 SFT，不开求助", 0.729), ("臂 A RL 后，开求助", 0.979),
], "Countdown 4 数：pass@1 一级级怎么抬上来的", "pass@1（池外 100 题 × 8，温度 1）", "countdown_ladder.svg",
   colors=[MUTED, STRONG, STRONG, STRONG, STRONG, SFT_C, STRONG, SFT_C, STRONG, STRONG, SFT_C, STRONG],
   note="浅色 = SFT 后，深色 = RL 后。最后一根开了专家出口，其余都是模型自己解题；开出口的 .98 里硬题 .946。")

# 3 观测的价格
def obs_price():
    bars = [("SFT-②-B\n强房间", .251, .329, "s"), ("RL-②-B\n强房间", .400, .446, "s"), ("SFT-②-B\n弱房间", .248, .240, "w"),
            ("RL-②-B\n弱房间", .327, .295, "w"), ("SFT-③\n弱房间", .235, .186, "w"), ("RL-③\n弱房间", .302, .282, "w")]
    fig, ax = plt.subplots(figsize=(8.6, 4.6)); xs = [0, 1, 2.4, 3.4, 4.8, 5.8]
    for x, (lab, first, final, room) in zip(xs, bars):
        c = STRONG if room == "s" else WEAK
        ax.bar(x, min(first, final), width=0.7, color=c)
        if final > first: ax.bar(x, final - first, bottom=first, width=0.7, color=TINT if room == "s" else "#F0D2B8")
        else: ax.bar(x, first - final, bottom=final, width=0.7, color="none", edgecolor=c, hatch="////", linewidth=0.8)
        ax.bar(x, first, width=0.7, color="none", edgecolor=INK, linestyle=(0, (3, 3)), linewidth=1)
        ax.text(x, max(first, final) + 0.012, f"{final:.3f}", ha="center", fontsize=10.5, color=INK, fontweight="bold")
        ax.text(x, -0.03, _(lab), ha="center", va="top", fontsize=9.5, color=INK)
        ax.text(x, min(first, final) - 0.02, f"{_('首版')} {first:.3f}", ha="center", va="top", fontsize=8.5, color="white")
    ax.set_xticks([]); ax.set_ylim(0, 0.5); ax.set_xlim(-0.6, 6.4)
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.spines["bottom"].set_visible(False)
    ax.set_ylabel(_("pass@1（100 题 × 8，温度 1）"), color=MUTED)
    ax.set_title(_("观测的价格：拿走测试，价全落在首版之后的改循环里"), loc="left", fontsize=13, pad=10)
    fig.text(0.01, -0.06, _("实心 = 首版就对且保住；浅色 = 改循环赚到；斜线 = 改循环丢掉；虚线框到首版高度。强房间首版按可见测试记，为上界。"), fontsize=9, color=MUTED)
    save(fig, "observation_price.svg")
obs_price()

# 4 基建阶梯
def infra_ladder():
    rows = [("单进程 HF，估算", 93), ("torchrun 四进程", 19.7), ("工具版 HF，臂 T", 190), ("vLLM 同卡共存", 25),
            ("代码题 ①，vfork 前", 76), ("① 动态采样 + vfork", 40), ("长程 ②-A，异步 + 缓存", 59), ("弱观测 ③，序列长 5 倍", 78)]
    fig, ax = plt.subplots(figsize=(8.4, 4.2)); y = list(range(len(rows)))[::-1]
    ax.barh(y, [r[1] for r in rows], color=[MUTED, STRONG, "#C0392B", STRONG, "#C0392B", STRONG, STRONG, STRONG], height=0.62)
    for yi, (lab, v) in zip(y, rows): ax.text(v + 2, yi, f"{v:g} s", va="center", fontsize=10.5)
    ax.set_yticks(y); ax.set_yticklabels([_(r[0]) for r in rows]); ax.set_xlim(0, 215)
    ax.set_xlabel(_("一步 RL 的墙钟时间（秒，四卡）"), color=MUTED); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.set_title(_("基建：每一格都是量到瓶颈再修"), loc="left", fontsize=13, pad=10)
    fig.text(0.01, -0.04, _("红色两格是任务变难后的回升（工具调用、代码沙盒），随后各被一项基建压回去。"), fontsize=9, color=MUTED)
    save(fig, "infra_ladder.svg")
infra_ladder()

# 8 阶段 ③ S 形拟合
def sigmoid_fit():
    steps = [0, 25, 50, 75, 100, 125, 150]; vals = [.2145, .2440, .2844, .3191, .3159, .3294, .3228]
    R0, A, Cmid, B = .2145, .330, 40, 2.6
    xs = list(range(1, 301)); fit = [R0 + (A - R0) / (1 + (Cmid / x) ** B) for x in xs]
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.plot(xs, fit, color=STRONG, lw=1.6, label=_("S 形拟合 A = .330, C_mid ≈ 第 40 步, B = 2.6"))
    ax.scatter(steps, vals, color=INK, zorder=3, s=28, label=_("仪表盘实测（20 未见 + 20 见过，温度 1 × 4）"))
    ax.axhline(A, color=MUTED, ls=(0, (3, 3)), lw=1); ax.text(298, A + 0.004, _("渐近线 A = .330"), ha="right", fontsize=9, color=MUTED)
    ax.axvline(150, color=GRID, lw=1); ax.text(152, .215, _("实际停在 150 步"), fontsize=9, color=MUTED)
    ax.set_xlim(0, 300); ax.set_ylim(.20, .35); ax.set_xlabel(_("RL 步数"), color=MUTED); ax.set_ylabel(_("仪表盘通过率"), color=MUTED)
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.set_title(_("阶段 ③：第 75 步到顶，续跑到 300 步只多 .003"), loc="left", fontsize=13, pad=10)
    save(fig, "sigmoid_fit_stage3.svg")
sigmoid_fit()

# 9 GSM8K pass@k
def gsm8k_passk():
    ks = [1, 2, 4, 8, 16, 32, 64]
    orig = [.5262, .6829, .7978, .8767, .9247, .9536, .9800]; s100 = [.6994, .8122, .8809, .9269, .9568, .9697, .9800]; s300 = [.7369, .8454, .9077, .9445, .9619, .9699, .9800]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for ys, lab, c, m in [(orig, "原始 Qwen2.5-1.5B-Instruct", MUTED, "o"), (s100, "GRPO 第 100 步", SFT_C, "s"), (s300, "GRPO 第 300 步", STRONG, "D")]:
        ax.plot(ks, ys, marker=m, color=c, lw=1.6, ms=5, label=_(lab))
    ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
    ax.set_ylim(0.5, 1.0); ax.set_xlabel(_("k（每题采样条数，温度 1）"), color=MUTED); ax.set_ylabel("pass@k", color=MUTED)
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.annotate(_("k = 64 三条线重合在 .980：RL 只重排，不创造"), xy=(64, .98), xytext=(9, .60), fontsize=9.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_title(_("GSM8K GRPO：pass@1 +.21，pass@64 一分没动"), loc="left", fontsize=13, pad=10)
    save(fig, "gsm8k_passk.svg")
gsm8k_passk()

# 10 编程线四级
def coding_ladder():
    names = ["① 修 bug", "②-A 拼包 K=3\n（包级）", "②-B exercism\n从头写", "③ 弱观测\n自己写测试"]
    sft1 = [.606, .069, .329, .186]; rl1 = [.749, .245, .446, .282]; sft32 = [.960, .55, .620, .560]; rl32 = [.980, .79, .680, .560]
    fig, ax = plt.subplots(figsize=(8.6, 4.4)); w = 0.34; xs = list(range(4))
    ax.bar([x - w/2 for x in xs], sft1, w, color=SFT_C, label=_("SFT 后 pass@1")); ax.bar([x + w/2 for x in xs], rl1, w, color=RL_C, label=_("RL 后 pass@1"))
    ax.scatter([x - w/2 for x in xs], sft32, marker="_", s=420, color=INK, linewidths=2, label=_("pass@32 天花板"), zorder=3)
    ax.scatter([x + w/2 for x in xs], rl32, marker="_", s=420, color=INK, linewidths=2, zorder=3)
    for x, a, b, c, d in zip(xs, sft1, rl1, sft32, rl32):
        ax.text(x - w/2, a + 0.015, f"{a:.2f}", ha="center", fontsize=9.5); ax.text(x + w/2, b + 0.015, f"{b:.2f}", ha="center", fontsize=9.5)
        ax.text(x - w/2, c + 0.02, f"{c:.2f}", ha="center", fontsize=8.5, color=MUTED); ax.text(x + w/2, d + 0.02, f"{d:.2f}", ha="center", fontsize=8.5, color=MUTED)
    ax.set_xticks(xs); ax.set_xticklabels([_(n) for n in names]); ax.set_ylim(0, 1.08); ax.set_ylabel(_("通过率（100 × 8 温度 1；横线 = 100 × 32）"), color=MUTED)
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.set_title(_("编程线四级：RL 抬 pass@1，天花板几乎不动"), loc="left", fontsize=13, pad=10)
    save(fig, "coding_ladder.svg")
coding_ladder()

# 11 p^K
def pk_multiplication():
    K = [1, 2, 3, 5]; sft = [.546, .256, .069, .007]; rl = [.703, .454, .245, .055]
    sft_b = [.546 ** k for k in K]; rl_b = [.703 ** k for k in K]
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.plot(K, sft_b, ls=(0, (3, 3)), color=SFT_C, lw=1.4, label=_("SFT-B 若各文件独立：.546^K")); ax.plot(K, sft, marker="o", color=SFT_C, lw=1.8, label=_("SFT-B 实测"))
    ax.plot(K, rl_b, ls=(0, (3, 3)), color=RL_C, lw=1.4, label=_("RL-bin 若各文件独立：.703^K")); ax.plot(K, rl, marker="D", color=RL_C, lw=1.8, label=_("RL-bin 实测"))
    ax.set_yscale("log"); ax.set_ylim(0.004, 1.0); ax.set_xticks(K); ax.set_xlabel(_("K，一个包里的文件数"), color=MUTED); ax.set_ylabel(_("整包全修好的比例，对数轴"), color=MUTED)
    ax.yaxis.grid(True, color=GRID, which="both"); ax.set_axisbelow(True); ax.legend(fontsize=9, frameon=False, loc="lower left")
    for k, a, b in zip(K, sft, rl):
        ax.text(k + 0.06, a, f"{a:.3f}", fontsize=8.5, color=SFT_C, va="center"); ax.text(k + 0.06, b, f"{b:.3f}", fontsize=8.5, color=RL_C, va="center")
    ax.set_title(_("②-A 独立拼包：实测低于乘积基线，RL 把长度的代价收回一半"), loc="left", fontsize=13, pad=10)
    save(fig, "pk_multiplication.svg")
pk_multiplication()

# 12 编程线 pass@k 三面板
def coding_passk():
    ks = [1, 2, 4, 8, 16, 32]
    panels = [("① 修 bug，100 × 32", [.612, .734, .823, .885, .930, .960], [.733, .818, .874, .917, .953, .980], "SFT-Coder", "RL（动态采样趟）"),
              ("②-B 训练题，100 × 32", [.328, .414, .475, .526, .575, .620], [.451, .531, .587, .626, .656, .680], "SFT-ex", "RL-bin"),
              ("②-B 留出题，30 × 32", [.244, .323, .402, .472, .536, .600], [.299, .375, .446, .496, .523, .533], "SFT-ex", "RL-bin")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, (title, s, r, ls, lr) in zip(axes, panels):
        ax.plot(ks, s, marker="o", color=SFT_C, lw=1.7, label=_(ls)); ax.plot(ks, r, marker="D", color=RL_C, lw=1.7, label=_(lr))
        ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
        ax.set_title(_(title), loc="left", fontsize=11.5); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.legend(fontsize=8.5, frameon=False, loc="lower right"); ax.set_xlabel("k", color=MUTED)
    axes[0].set_ylabel("pass@k", color=MUTED); axes[0].set_ylim(.55, 1.0); axes[1].set_ylim(.25, .75); axes[2].set_ylim(.2, .65)
    axes[2].annotate(_("k ≥ 16 时 RL 反而更低：削尾巴"), xy=(32, .533), xytext=(3.2, .30), fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    fig.suptitle(_("编程线的 pass@k：RL 抬 pass@1，大 k 处不动或反降"), x=0.01, ha="left", fontsize=13); fig.tight_layout()
    save(fig, "coding_passk.svg")
coding_passk()

# 13 猜数字三臂
def guess_number():
    arms = ["起点 hf_armA", "E1 纯 RL", "E3 课程 20→100", "E2 撒种 SFT", "E2 撒种 SFT + RL"]
    train = [.043, .632, .838, .976, .996]; held = [None, .359, .512, .873, .926]; big = [None, .142, .171, .621, .713]
    fig, ax = plt.subplots(figsize=(8.6, 4.2)); w = 0.26; xs = list(range(5))
    ax.bar([x - w for x in xs], train, w, color=STRONG, label=_("训练模板，N=100")); ax.bar(xs, [h or 0 for h in held], w, color=SFT_C, label=_("留出模板，N=100"))
    ax.bar([x + w for x in xs], [b or 0 for b in big], w, color=WEAK, label=_("N=1000，越界，10 轮刚够二分"))
    for x, a, h, b in zip(xs, train, held, big):
        ax.text(x - w, a + .015, f"{a:.2f}", ha="center", fontsize=9)
        if h: ax.text(x, h + .015, f"{h:.2f}", ha="center", fontsize=9)
        if b: ax.text(x + w, b + .015, f"{b:.2f}", ha="center", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels([_(a) for a in arms], fontsize=9.5); ax.set_ylim(0, 1.1); ax.set_ylabel(_("猜中率（100 个秘密数 × 8，温度 1）"), color=MUTED)
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.set_title(_("猜数字：撒种 200 条示范 2 分钟，超过两条 RL 臂；N=1000 的差距来自「写下区间」"), loc="left", fontsize=12.5, pad=10)
    save(fig, "guess_number.svg")
guess_number()
print("done", "en" if EN else "zh")
