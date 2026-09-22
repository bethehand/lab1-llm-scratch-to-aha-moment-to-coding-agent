#!/usr/bin/env python3
"""
grpo_tool_mp_vllm.py = grpo_tool_mp.py + --engine vllm：rollout 交给同卡共存的 vLLM（external_launcher + 睡眠模式），每步 opt.step 后本地灌权重。
原 grpo_tool_mp.py 不动。要在 vllm_env 里跑（还需 pip install bitsandbytes datasets）。--engine hf 时与原版行为一致（只多了 EOS 保留）。

★ 阶段一：混训 Countdown + GSM8K —— 验证「条件性策略绑定」

    torchrun --nproc_per_node=4 grpo_mix_mp.py --smoke
    torchrun --nproc_per_node=4 grpo_mix_mp.py --probs 8 -k 16 --gen-bs 32 --steps 300

★★ 要验证的机制 ──────────────────────────────────────────────
  单任务 RL 后，模型学到的是 P(短答案 │ R1 模板)，不是 P(短答案)
  → 模板成了「开关」，看到 <think> 就闭嘴（长度 183→71，GSM8K 正确率 0.52→0.16）

  ★ 声称：退化策略学得成，是因为【所有训练轨迹共享同一个上下文】
  ★★ 推论：同一个模板下同时训两种题（一种要短答案、一种要长推理），
            「看到模板就闭嘴」不再最优 → 模型被迫【读题】

★★★ 主判据：【长度分化】 ─────────────────────────────────────
  日志里【分任务】打长度：
     step 0     CD ~230   GSM ~250   → 差 ~20    （没分化）
     step 300   CD ~70    GSM ~250   → 差 ~180   （★ 分化 = 在读题，不是在看模板）
  ★ 平均长度不崩 ≠ 绑定被打破（可能只是 GSM 把均值拉高）—— 所以必须分任务看

★ 跟 grpo_countdown_mp.py 只差【混不混】这一个变量：
  优势公式、lr、KL β、k、max_new 全部不变。
  ★ 注意 GSM8K 是 0/1 奖励，组内方差比 Countdown 大 2~7 倍 → 可能主导梯度
    → 日志打每个任务的平均 |A|，让不平衡【可见】；--adv-std 备用（默认关）
"""
import os, re, json, time, random, argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import countdown as CD
import _reward as RW                              # ★ GSM8K 判分器，唯一定义处
import judge_v2 as J2                             # ★ 判分器 v2 的过程项（--judge v2）
import calc_tool_vllm as calc_tool                # ★ 计算器 harness 的后端层版本：--engine hf 走 HFBackend（修了 pad==eos 丢 EOS），--engine vllm 走 vLLM

ap = argparse.ArgumentParser()
ap.add_argument("--model",   default="Qwen/Qwen2.5-1.5B")
ap.add_argument("--data",    default="data/countdown.json")
ap.add_argument("--out",     default="out_mix_mp")
ap.add_argument("--steps",   type=int,   default=300)
ap.add_argument("--probs",   type=int,   default=8,  help="★ 每步几道题（两任务合计，能被卡数整除）")
ap.add_argument("--gsm-frac", type=float, default=0.5, help="★ GSM8K 占比（默认一半一半）")
ap.add_argument("-k", "--k", type=int,   default=16)
ap.add_argument("--micro",   type=int,   default=2)
ap.add_argument("--gen-bs",  type=int,   default=32)
ap.add_argument("--max-new", type=int,   default=512)
ap.add_argument("--temp",    type=float, default=1.0)
ap.add_argument("--lr",      type=float, default=5e-6)
ap.add_argument("--beta",    type=float, default=0.0005)
ap.add_argument("--warmup",  type=int,   default=10)
ap.add_argument("--clip-grad", type=float, default=1.0)
ap.add_argument("--len-soft", type=int, default=0,
                help="★ 长度软惩罚起点（token）。0 = 关。DAPO 的 overlong shaping：soft 以内不罚，soft→max_new 线性罚到 --len-pen。"
                     "池外实测 43% 的 CD 轨迹试 6 次以上还不停（均值 375 token，正确率 0.18），这是在打它")
ap.add_argument("--len-pen",  type=float, default=0.5, help="撞到 max_new 时扣多少分（阶梯满分 1.0）")
ap.add_argument("--adv-std", action="store_true",
                help="★ 优势除以组内 std（标准 GRPO）。默认关，跟上一轮保持同公式")
ap.add_argument("--tool", action="store_true", help="★ 计算器：CD 题提示词加工具说明，rollout 走 calc_tool，注入段 mask，假结果罚")
ap.add_argument("--fake-pen", type=float, default=0.2, help="模型自己写 <result>（--ask 时也含 <reply>）扣多少")
ap.add_argument("--max-calls", type=int, default=48)
ap.add_argument("--ask", action="store_true",
                help="★ 臂 A：CD 题提示词每步从 prompts_cd 的 --prompt-set 随机抽（工具/求助说明各以 --p-hint 概率出现），harness 开求助"
                     "（<ask> 停 → 问专家 → 注入 <reply>，注入段 mask，自己编 <reply> 按 --fake-pen 罚），每次求助扣 --ask-pen。隐含 --tool")
ap.add_argument("--ask-pen", type=float, default=0.05, help="每次求助扣多少（价格：硬题问 0→0.965 划算，易题问 0.983→1.0 亏本）")
ap.add_argument("--max-asks", type=int, default=3)
ap.add_argument("--ask-weak-frac", type=float, default=0.2, help="每次求助落到弱专家 deepseek-chat 的概率：专家偶尔错，「验证抓错再问」才有回报")
ap.add_argument("--ask-workers", type=int, default=16)
ap.add_argument("--p-hint", type=float, default=0.7, help="--ask 时工具说明 / 求助说明各自出现的概率（不出现 = 学「没说明也会用」）")
ap.add_argument("--prompt-set", default="train", choices=["train", "all", "r1"], help="--ask 时抽哪组模板（heldout 留给评测，训练不碰）")
ap.add_argument("--task", choices=["mix", "guess", "code", "pkg", "ex"], default="mix", help="★ mix = Countdown + GSM（默认，一字不变）；guess = E 猜数字；code = 修 bug（code_env + harness）；pkg = 多文件包（pkg_env，阶段 ②-A）；ex = 从头写 exercism（ex_env，阶段 ①′）")
ap.add_argument("--ex-data", nargs="+", default=["data/exercism.json"], help="ex：第一份是主池（仪表盘从它里面取），后面的是扩池（如 data/mbpp_weak.json）")
ap.add_argument("--ex-mix", type=float, default=0.5, help="ex：有扩池时每步从主池抽的比例（其余从扩池抽）")
ap.add_argument("--ex-obs", choices=["strong", "weak"], default="strong", help="★ 阶段 ③：weak = 题面不给测试、<test> 不可用，模型用 <run> 自己验证（ExEnv(obs=weak)）")
ap.add_argument("--ex-max-timeouts", type=int, default=2, help="ex：一份代码连着几条测试超时就不再跑剩下的（只改速度，二进制奖励不变；0 = 关）")
ap.add_argument("--ex-dash", type=int, default=20, help="ex：从训练 split 里固定留几道当仪表盘（不进 RL 采样；--eval-n 取这个数）")
ap.add_argument("--ex-dash-k", type=int, default=0, help="★ 仪表盘 v2：>0 = 固定 --ex-dash 道没见过 + 同数见过的（seed 12345），温度 1 各采这么多条，报见过/没见过；0 = 老的贪心仪表盘")
ap.add_argument("--logp-diff", action="store_true", help="★ 诊断：rollout 时向 vLLM 要每个 token 的 logp，训练时和 HF 算的比，日志打 Δlp（采样器和训练器的概率错位有多大）")
ap.add_argument("--pkg-K", type=int, default=3, help="pkg：每个包几个文件（训练和验证同一个 K）")
ap.add_argument("--pkg-reward", choices=["bin", "frac"], default="bin", help="pkg：bin = 全过才给分；frac = 按通过测试比例给分（E3 的变量）")
ap.add_argument("--code-bugs", default="data/code_bugs.json")
ap.add_argument("--code-p-tb", type=float, default=0.3, help="修 bug：题面带报错开局的比例")
ap.add_argument("--code-max-calls", type=int, default=None, help="默认 code 8 / pkg 16")
ap.add_argument("--code-call-cost", type=float, default=None, help="默认 code 0.02 / pkg 0.01")
ap.add_argument("--code-timeout", type=float, default=None, help="沙盒超时秒数，默认 code 5 / pkg 2（死循环 bug 的每次 <test> 都等满它，是异步下拖慢整卡的主因）")
ap.add_argument("--code-ask", action="store_true", help="修 bug：开专家 <ask>（要 DEEPSEEK_API_KEY）；不开时 <ask> 回 error")
ap.add_argument("--code-step-workers", type=int, default=8, help="沙盒并发")
ap.add_argument("--score-workers", type=int, default=0, help="★ 终评（秤跑 3 条测试）并发数；0 = 同 --code-step-workers；1 = 串行（老行为，量对照用）")
ap.add_argument("--dyn-sample", type=float, default=0.0,
                help="★ DAPO 动态采样：每步多抽这个比例的题一起 rollout（如 0.5 = 8 道抽 12 道），全组同分的丢掉，取前 probs/WORLD 道有差异的进梯度；"
                     "不够就用同分的补。日志多打「有效组」。只对 guess / code 生效（题无限）")
ap.add_argument("--guess-N", type=int, default=100)
ap.add_argument("--guess-T", type=int, default=10)
ap.add_argument("--guess-cost", type=float, default=0.05, help="猜数字每多一轮扣多少")
ap.add_argument("--val-data", default=None, help="★ 训练中 CD 验证题从这份文件取（[--eval-offset : +eval_n]）；--data 是筛过的小池时必须给原始文件")
ap.add_argument("--judge", choices=["v1", "v2", "v3"], default="v1",
                help="★ v2：阶梯缩到 .10、对 .9；假宣告 −.2、错步率 −.1、重复行/违规行各 −.05（封顶 4）；对的只扣 .05·len/max_new。见 judge_v2.py")
ap.add_argument("--eval-every", type=int, default=25)
ap.add_argument("--eval-n",  type=int,   default=100, help="★ 每个任务各评几题")
ap.add_argument("--eval-offset", type=int, default=90000, help="Countdown 评估集起点")
ap.add_argument("--gsm-eval-offset", type=int, default=200,
                help="★ GSM8K test 评估起点。test[0:200] 留给 crosstask_eval 做对照，不碰")
ap.add_argument("--save-every", type=int, default=25)
ap.add_argument("--patience", type=int,  default=5)
ap.add_argument("--resume",  action="store_true")
ap.add_argument("--init",    default=None,
                help="★ 从别的目录的 ckpt 起（只读权重和优化器），写到新 --out，step 从 0 数。换任务用这个，别用 --resume")
ap.add_argument("--ref-init", default=None,
                help="★ KL 参照用这个 ckpt（如 SFT 模型）而不是原始 Base：护种，RL 只能在种的邻域里选。配 --beta 0.01~0.05")
ap.add_argument("--engine",  choices=["hf", "vllm"], default="hf",
                help="★ rollout 引擎：vllm = 每个 rank 自己一份 vLLM 与训练共存（external_launcher + 睡眠模式），每步训练后本地灌权重。要在装了 vLLM 的 venv 里跑")
ap.add_argument("--vllm-mem", type=float, default=0.6,
                help="vLLM 占整卡显存的比例上限。★ 它算 KV 预算时把卡上已占的（训练模型 3 + 参照 3 + 杂项 1 GB）也算在内：0.65×24=15.6 − 7 − 权重 3 − 激活 1 ≈ 4.6 GB KV。0.3 会报 No available memory for the cache blocks")
ap.add_argument("--smoke",   action="store_true")
ap.add_argument("--seed",    type=int,   default=1337)
args = ap.parse_args()
if args.code_max_calls is None: args.code_max_calls = 16 if args.task == "pkg" else 12 if args.task == "ex" else 8
if args.code_call_cost is None: args.code_call_cost = 0.01 if args.task in ("pkg", "ex") else 0.02
if args.code_timeout is None: args.code_timeout = 2 if args.task in ("pkg", "ex") else 5
if args.task == "ex": args.eval_n = args.ex_dash
import shutil as _sh
_free = _sh.disk_usage(".").free / 1024 ** 3
assert args.smoke or _free >= 15, f"盘只剩 {_free:.1f} GB，跑不完（每次存盘 5.8 GB + best 2.9 GB）。先清盘：du -sh out_* | sort -h；slim_ckpt.py 剥优化器"
if args.ask:
    args.tool = True                                          # 求助版一定带计算器

# ══════════════════════════════════════════════════════════════
#  分布式
# ══════════════════════════════════════════════════════════════
RANK = int(os.environ.get("RANK", 0))
WORLD = int(os.environ.get("WORLD_SIZE", 1))
LOCAL = int(os.environ.get("LOCAL_RANK", 0))
if WORLD > 1:
    dist.init_process_group("nccl")
torch.cuda.set_device(LOCAL)
DEV = f"cuda:{LOCAL}"
M0 = RANK == 0


def p0(*a, **kw):
    if M0:
        print(*a, **kw)


if args.smoke:
    args.steps, args.probs, args.k = 3, 2 * WORLD, 4      # ★ 每 rank 2 题，两任务都能碰到
    args.out = args.out.rstrip("/") + "_smoke"            # ★ 别把正式目录里的 ckpt_best 覆盖了
    args.max_new, args.eval_every, args.eval_n, args.save_every = (args.max_new if args.task in ("pkg", "ex") else 1400 if args.ask else 256), 2, 2 * WORLD, 999   # ★ --ask 的冒烟要走到求助那步，256 不够；pkg 的冒烟就是量显存，max_new 照用户给的

assert args.probs % WORLD == 0, f"--probs {args.probs} 必须能被卡数 {WORLD} 整除"
assert args.eval_n % WORLD == 0, f"--eval-n {args.eval_n} 必须能被卡数 {WORLD} 整除"
PER = args.probs // WORLD
N_GSM = int(round(args.probs * args.gsm_frac)); N_CD = args.probs - N_GSM
assert 0 <= N_GSM <= args.probs

torch.manual_seed(args.seed + RANK); np.random.seed(args.seed + RANK)
if M0:
    os.makedirs(args.out, exist_ok=True)

REFLECT = re.compile(
    r"\b(wait|hold on|hmm|actually|alternatively|instead|let me (?:try|check|recheck|reconsider)"
    r"|that (?:doesn'?t|does not) work|not (?:right|correct)|recheck|reconsider)\b", re.I)

# ★★ 搜索措辞 —— 上一轮训完抽样发现模型会写「(too high)」「(too low)」「(perfect match)」，
#    这是带反馈的试错，REFLECT 抓不到。单独计一个 ★差，看它跟奖励是正相关还是负相关。
SEARCH = re.compile(
    r"\b(too (?:high|low|big|small|large)|not (?:equal|right|correct)|doesn'?t (?:work|equal)"
    r"|does not (?:work|equal)|perfect|close|try (?:another|again|different|a different|the next)"
    r"|let'?s try|next,? try|another (?:combination|approach|way|try)|nope|wrong|incorrect)\b", re.I)

# ★ 上一轮的坑：Base 写完 </answer> 不发 EOS，接着幻觉 "User: …" 撞到 max_new。
#   85% 的贪心输出被撑到 512，长度指标全废；训练时这几百个幻觉 token 也在吃梯度。
#   → 撞到就停；截掉后【补一个 EOS】，让模型学会「答完就停」。
CONT = "\nUser:"

# ★★★ 顿悟探针（analyze_search.py 验证过的那套，替换掉钝的二值 ★差）
#   二值 SEARCH ★差 一直报零，是因为 let's try / close 这类装饰词把信号淹了。
#   真正跟奖励挂钩的是「算式 = 值 (too high/low)」这种【可验证的中间步骤】：
#     混训 ckpt 上 87 处标注、精度 0.82；试错 3~5 次的正确率 0.342 vs 不试 0.210。
#   所以训练时改记三样：标注数、标注精度、剂量响应（3~5 次 − 0 次）。
ANNOT = re.compile(r"((?:\(|\d)[\d\s\+\-\*/\(\)]*?)\s*=\s*(-?\d+(?:\.\d+)?)\s*\(?\s*(too (?:high|low|big|small|large))", re.I)   # ★ 算式必须以数字或 ( 开头：否则上一行标注的 ")\n" 会被吞进算式，eval 失败被跳过 → 连续标注只数到第一条


def annot_stats(txt, target):
    """→ (标注数, 其中【算式算对 且 方向标对】的数)"""
    n = ok = 0
    for m in ANNOT.finditer(txt):
        try:
            real = CD.safe_eval(CD.clean_expr(m.group(1)))
        except Exception:
            continue
        n += 1
        said_high = any(w in m.group(3).lower() for w in ("high", "big", "large"))
        if abs(real - float(m.group(2))) < 1e-6 and said_high == (real > target):
            ok += 1
    return n, ok


TOOL_ANNOT = re.compile(r"<calc>(.*?)</calc><result>(-?\d+(?:\.\d+)?|error)</result>\s*\(?\s*(too (?:high|low|big|small|large))", re.I)

def annot_stats_tool(txt, target):
    """工具版：值是 harness 算的，只验方向标没标对 → (标注数, 方向对的数)"""
    n = ok = 0
    for m in TOOL_ANNOT.finditer(txt):
        if m.group(2) == "error":
            n += 1; continue
        n += 1
        said_high = any(w in m.group(3).lower() for w in ("high", "big", "large"))
        if said_high == (float(m.group(2)) > target):
            ok += 1
    return n, ok


def bucket(nsrch):
    return 0 if nsrch == 0 else 12 if nsrch <= 2 else 35 if nsrch <= 5 else 6

# ══════════════════════════════════════════════════════════════
#  ★ 两个任务：数据 / 提示 / 判分  —— 统一成 {"task": "cd"|"gsm", ...}
# ══════════════════════════════════════════════════════════════
TASKS = ("guess",) if args.task == "guess" else ("code",) if args.task in ("code", "pkg", "ex") else ("cd", "gsm")
if args.task == "ex":                                                      # ★ 阶段 ①′ 从头写 exercism：训练 split 100 道，固定 --ex-dash 道当仪表盘（贪心，不进采样）
    import code_env as CE, harness as HN, ex_env as EX
    _all = sorted(EX.load_problems(args.ex_data[0], "train"), key=lambda q: q["slug"])
    _dash = _all[::max(1, len(_all) // args.ex_dash)][: args.ex_dash]      # 仪表盘的 20 道固定不进采样（和 ②-B 同一批，留出/见过没见过的对照才连得上）
    _dash_slugs = {q["slug"] for q in _dash}
    CODE_TRAIN_MAIN = [q for q in _all if q["slug"] not in _dash_slugs]
    CODE_TRAIN_EXTRA = [q for f in args.ex_data[1:] for q in EX.load_problems(f, "train")]          # ★ 扩池（MBPP 那类），不进仪表盘
    CODE_TRAIN = CODE_TRAIN_MAIN + CODE_TRAIN_EXTRA
    if args.ex_dash_k > 0:                                                 # ★ 仪表盘 v2：没见过的 20 道 + 见过的 20 道（主池里 seed 12345 抽），温度 1 各 k 条
        _seen = random.Random(12345).sample(CODE_TRAIN_MAIN, min(args.ex_dash, len(CODE_TRAIN_MAIN)))
        CODE_EVAL = ([dict(task="code", ex=q, idx=i, grp="unseen") for i, q in enumerate(_dash)]
                     + [dict(task="code", ex=q, idx=len(_dash) + i, grp="seen") for i, q in enumerate(_seen)])
        if not args.smoke:
            args.eval_n = len(CODE_EVAL)
        CODE_EVAL = CODE_EVAL[: args.eval_n]
        assert args.eval_n % WORLD == 0, f"--eval-n {args.eval_n} 必须能被卡数 {WORLD} 整除"
    else:
        CODE_EVAL = [dict(task="code", ex=q, idx=i, grp="unseen") for i, q in enumerate(_dash[: args.eval_n])]
    assert len(CODE_EVAL) == args.eval_n, (len(CODE_EVAL), args.eval_n)
    CENV = EX.ExEnv(max_calls=args.code_max_calls, call_cost=args.code_call_cost, per_test=args.code_timeout, obs=args.ex_obs, max_timeouts=args.ex_max_timeouts,
                    ask_fn=(lambda q, pr, code: EX.ask_expert(q, pr, code, weak=(args.ex_obs == "weak"))) if args.code_ask else None)
    CD_TRAIN = GSM_TRAIN = CD_EVAL = GSM_EVAL = []
    p0(f"★ 任务 ex（观测 {args.ex_obs}）：主池 {len(CODE_TRAIN_MAIN)} 道 + 扩池 {len(CODE_TRAIN_EXTRA)} 道（每步抽 {args.probs} 道 × {args.k}，主池占 {args.ex_mix if CODE_TRAIN_EXTRA else 1.0:.2f}）"
       f"/ 仪表盘 {len(CODE_EVAL)} 道（{'没见过 ' + str(len(_dash)) + ' + 见过 ' + str(len(CODE_EVAL) - len(_dash)) + '，温度 1 × ' + str(args.ex_dash_k) if args.ex_dash_k > 0 else '固定，贪心'}）；"
       f"{'题面不给测试，<run> 自验，秤 = 全部测试' if args.ex_obs == 'weak' else '可见测试七成、隐藏三成只进秤'}；"
       f"最多 {args.code_max_calls} 次调用、每次扣 {args.code_call_cost}；单条测试限时 {args.code_timeout} s，连续超时 {args.ex_max_timeouts} 条早停；专家 {'开' if args.code_ask else '关'}；沙盒并发 {args.code_step_workers}")
elif args.task == "pkg":                                                   # ★ 阶段 ②-A 多文件包：每步现抽 K 个 bug 拼包，验证集固定 80 包（task_id 末位 0 的题）
    import code_env as CE, harness as HN, pkg_env as PE
    _bugs = [b for b in json.load(open(args.code_bugs)) if b["split"] == "train"]
    PKG_EXT = PE.load_ext()
    CODE_TRAIN = [b for b in _bugs if b["task_id"] % 10 != 0]
    _val = [b for b in _bugs if b["task_id"] % 10 == 0]
    CODE_EVAL = [dict(task="code", pkg=pk, idx=i) for i, pk in enumerate(PE.make_pkgs(_val, args.pkg_K, args.eval_n, seed=12345, ext=PKG_EXT))]
    CENV = PE.PkgEnv(max_calls=args.code_max_calls, call_cost=args.code_call_cost, reward=args.pkg_reward, timeout=args.code_timeout,
                     ask_fn=(lambda q, pkg, files: PE.ask_expert(q, pkg, files)) if args.code_ask else None)
    CD_TRAIN = GSM_TRAIN = CD_EVAL = GSM_EVAL = []
    p0(f"★ 任务 pkg：K={args.pkg_K}，训练 bug 池 {len(CODE_TRAIN)} 个（每步现拼 {args.probs} 包 × {args.k}）/ 验证 {len(CODE_EVAL)} 包（task_id 末位 0 的题拼的，seed 12345）；"
       f"严秤隐藏测试 {len(PKG_EXT)} 道题有；带报错开局 {args.code_p_tb:.0%}；最多 {args.code_max_calls} 次调用、每次扣 {args.code_call_cost}；沙盒超时 {args.code_timeout} s；奖励 {args.pkg_reward}；"
       f"专家 {'开' if args.code_ask else '关'}；沙盒并发 {args.code_step_workers}")
elif args.task == "code":                                                  # ★ 修 bug：不读 Countdown / GSM
    import code_env as CE, harness as HN
    _bugs = [b for b in json.load(open(args.code_bugs)) if b["split"] == "train"]
    CODE_TRAIN = [dict(task="code", bug=b, idx=i) for i, b in enumerate(_bugs) if b["task_id"] % 10 != 0]      # task_id 末位 0 的题留作训练中验证
    _val, _seen = [], set()
    for b in _bugs:
        if b["task_id"] % 10 == 0 and b["task_id"] not in _seen:
            _seen.add(b["task_id"]); _val.append(b)
    CODE_EVAL = [dict(task="code", bug=b, idx=i) for i, b in enumerate(_val[: args.eval_n])]
    assert len(CODE_EVAL) == args.eval_n, f"验证题不够 {len(CODE_EVAL)} / {args.eval_n}"
    CENV = CE.CodeEnv(max_calls=args.code_max_calls, call_cost=args.code_call_cost, timeout=args.code_timeout,
                      ask_fn=(lambda q, p, code: CE.ask_expert(q, p, code)) if args.code_ask else None)
    CD_TRAIN = GSM_TRAIN = CD_EVAL = GSM_EVAL = []
    p0(f"★ 任务 code：训练 bug {len(CODE_TRAIN)} 个（{len(set(x['bug']['task_id'] for x in CODE_TRAIN))} 道题）/ 验证 {len(CODE_EVAL)}（task_id 末位 0）；"
       f"每步 {args.probs} 个 bug × {args.k}；带报错开局 {args.code_p_tb:.0%}；最多 {args.code_max_calls} 次调用、每次扣 {args.code_call_cost}；"
       f"专家 {'开' if args.code_ask else '关'}；沙盒并发 {args.code_step_workers}")
elif args.task == "guess":                                                 # ★ E 猜数字：不读 Countdown / GSM
    import guess_env as GE, harness as HN
    GENV = GE.GuessEnv(N=args.guess_N, T=args.guess_T, step_cost=args.guess_cost)
    GUESS_EVAL = [dict(task="guess", secret=p["secret"], idx=p["idx"]) for p in GE.make_problems(args.eval_n, N=args.guess_N, seed=12345)]
    CD_TRAIN = GSM_TRAIN = CD_EVAL = GSM_EVAL = []
    p0(f"★ 任务 guess：秘密数 [1, {args.guess_N}]，最多 {args.guess_T} 轮，每多一轮扣 {args.guess_cost}；每步 {args.probs} 个新秘密数 × {args.k}；"
       f"eval {args.eval_n} 个固定秘密数（seed 12345，模板按题号固定）；提示词每步从 guess_env 训练模板抽")
else:
    D = json.load(open(args.data))
    CD_TRAIN = [dict(task="cd", **p) for p in D[: args.eval_offset]]
    DV = json.load(open(args.val_data)) if args.val_data else D                 # ★ 验证题可以来自另一份文件
    CD_EVAL = [dict(task="cd", **p) for p in DV[args.eval_offset: args.eval_offset + args.eval_n]]
    assert len(CD_EVAL) == args.eval_n, f"验证题不够：{len(CD_EVAL)}/{args.eval_n}，--data 是小池时请给 --val-data data/countdown4.json"

    from datasets import load_dataset
    _g = load_dataset("openai/gsm8k", "main")
    def _gsm(split, lo=0, hi=None):
        rows = _g[split]
        hi = len(rows) if hi is None else hi
        return [dict(task="gsm", q=r["question"],
                     gold=r["answer"].split("####")[-1].strip().replace(",", ""))
                for r in rows.select(range(lo, hi))]
    GSM_TRAIN = _gsm("train")
    GSM_EVAL = _gsm("test", args.gsm_eval_offset, args.gsm_eval_offset + args.eval_n)
    p0(f"Countdown train {len(CD_TRAIN):,} / eval {len(CD_EVAL)}   "
       f"GSM8K train {len(GSM_TRAIN):,} / eval test[{args.gsm_eval_offset}:{args.gsm_eval_offset+len(GSM_EVAL)}]")
    p0(f"★ 每步 {N_CD} 道 Countdown + {N_GSM} 道 GSM8K，{WORLD} 进程，每进程 {PER} 题")


def gsm_prompt(q):
    """★ 跟 crosstask_eval.py 的 R1 模板【逐字相同】—— 同一个模板，两种题"""
    return (f"{CD.SYS}\nUser: {q}\nShow your work in <think> </think> tags. "
            f"Return the final numeric answer in <answer> </answer> tags, "
            f"for example <answer> 42 </answer>.\n"
            f"Assistant: Let me solve this step by step.\n<think>")


PRNG = random.Random(args.seed * 100 + RANK)                     # ★ --ask：每步抽提示词用，各 rank 不同


def make_prompt(p):
    if p["task"] in ("guess", "code"):
        return p["_prompt"]
    if p["task"] == "cd":
        if args.ask:
            return p["_prompt"]                                     # rollout 开头抽好、存在题的 dict 里；训练编码时取同一份
        return calc_tool.tool_prompt(p["nums"], p["target"]) if args.tool else CD.make_prompt(p["nums"], p["target"])
    return gsm_prompt(p["q"])


def pick_prompts(probs, eval_mode=False):
    """★ --ask：给这批 CD 题各抽一个提示词（模板、说明有无、措辞）。评测时按题号定种子，每次评测同一份"""
    import prompts_cd as PC
    for i, p in enumerate(probs):
        if p["task"] == "code":                                     # 修 bug / 多文件包：训练时随机模板 / 措辞 / 开局，评测按题号固定
            rng = random.Random(7919 + p["idx"]) if eval_mode else PRNG
            M = EX if "ex" in p else PE if "pkg" in p else CE
            pid = rng.choice(M.SETS["train"]); ts = rng.randrange(3); tb = rng.random() < args.code_p_tb
            p["_prompt"], p["_pmeta"] = M.render(pid, p.get("ex") or p.get("pkg") or p["bug"], with_tb=tb, tool_style=ts, env=CENV), dict(pid=pid, with_tb=tb)
            continue
        if p["task"] == "guess":                                    # 猜数字：训练时随机模板，评测按题号固定
            pid = GE.SETS["train"][p.get("idx", i) % len(GE.SETS["train"])] if eval_mode else PRNG.choice(GE.SETS["train"])
            p["_prompt"], p["_pmeta"] = GE.render(pid, args.guess_N, args.guess_T), dict(pid=pid)
            continue
        if p["task"] != "cd":
            continue
        rng = random.Random(7919 + i) if eval_mode else PRNG
        p["_prompt"], p["_pmeta"] = PC.pick(rng, p["nums"], p["target"], set=args.prompt_set, p_tool=args.p_hint, p_ask=args.p_hint, max_asks=args.max_asks)


def ask_mixed(q):
    """★ 环境里的专家：--ask-weak-frac 的概率落到弱专家 deepseek-chat，其余 reasoner。没答由 harness 重试一次（calc_tool_vllm._ask_retry）"""
    m = "deepseek-chat" if random.random() < args.ask_weak_frac else "deepseek-reasoner"
    return calc_tool.ask_expert(q, model=m)


ASK_KW = dict(ask=True, max_asks=args.max_asks, ask_fn=ask_mixed, ask_workers=args.ask_workers) if args.ask else {}


def gsm_reward(txt, gold):
    """
    ★ 跟 Countdown 一样满分 1.0（尺度归一）：
        有 <answer> 标签   0.1
        ★ 标签里的数字对   0.9   ← 跟 Countdown 一致：没标签不给正确分
    """
    d = {"fmt": 0.0, "correct": 0.0}
    m = CD.ANS.search(txt)
    if m:
        d["fmt"] = 1.0
        a, _ = RW.extract(m.group(1))
        try:
            if a is not None and abs(float(a) - float(gold)) < 1e-4:
                d["correct"] = 1.0
        except ValueError:
            pass
    return 0.1 * d["fmt"] + 0.9 * d["correct"], d


def score(p, txt):
    if p["task"] == "cd":
        if args.judge == "v2":
            return J2.score(txt, p["nums"], p["target"])          # ★ v2：阶梯缩小 + 四个过程项
        if args.judge == "v3":                                    # ★ v3：只奖不罚。对 1.0，错 / 没答案 / 循环 0，格式阶梯全去掉（「任何错 > 不答」教猜）
            r, d = CD.reward(txt, p["nums"], p["target"])
            return (1.0 if d["correct"] > 0 else 0.0), d
        return CD.reward(txt, p["nums"], p["target"])
    return gsm_reward(txt, p["gold"])


# ══════════════════════════════════════════════════════════════
#  模型（跟单任务版完全一样）
# ══════════════════════════════════════════════════════════════
tok = AutoTokenizer.from_pretrained(args.model)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
PAD, EOS = tok.pad_token_id, tok.eos_token_id
STOP_IDS = tok.encode(CONT, add_special_tokens=False)   # ★ 用来从 ids 尾部剥掉停止串


def load():
    try:
        m = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    return m.to(DEV)


p0("载入模型（每个 rank 一份）…", flush=True)
raw = load()
raw.gradient_checkpointing_enable()
raw.config.use_cache = False
ref = None
if args.beta > 0:
    ref = load().eval()
    for p in ref.parameters():
        p.requires_grad_(False)
model = raw


def allreduce_grads():
    if WORLD == 1:
        return 0.0
    t0 = time.time()
    for p_ in raw.parameters():
        if p_.grad is not None:
            dist.all_reduce(p_.grad, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    return time.time() - t0


n_par = sum(p.numel() for p in raw.parameters())
try:
    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(raw.parameters(), lr=args.lr, betas=(0.9, 0.95),
                              weight_decay=0.0, eps=1e-8)
    OPT = "AdamW8bit"
except ImportError:
    opt = torch.optim.AdamW(raw.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    OPT = "AdamW(fp32)"
    p0("⚠️ 没装 bitsandbytes，可能 OOM")


def set_lr(step):
    lr = args.lr * min(1.0, (step + 1) / max(args.warmup, 1))
    for g in opt.param_groups:
        g["lr"] = lr
    return lr


# ══════════════════════════════════════════════════════════════
#  采样 —— 每条轨迹一个 dict（★ 带 task 字段，日志按任务分）
# ══════════════════════════════════════════════════════════════
@torch.no_grad()
def _finish(pi, ids, txt, cont, spans, n_calls, n_err, fake, probs, max_new, n_asks=0):
    r, d = score(probs[pi], txt)
    if args.ask and probs[pi]["task"] == "cd":
        r -= args.ask_pen * n_asks                                  # ★ 求助的价格，按次扣
    if args.judge in ("v2", "v3") and probs[pi]["task"] == "cd" and d["correct"] > 0:
        r -= J2.PEN["len_ok"] * len(ids) / max_new                  # 对的之间：短者略高
    elif args.judge == "v3" and probs[pi]["task"] == "cd":
        pass                                                        # ★ v3：错的不罚长度，循环也是 0，不罚探索
    elif args.len_soft > 0 and len(ids) > args.len_soft:
        frac = min(1.0, (len(ids) - args.len_soft) / max(1, max_new - args.len_soft))
        r -= args.len_pen * frac
    if fake:
        r -= args.fake_pen                                          # ★ 自己写 <result>
    tgt = probs[pi]["target"] if probs[pi]["task"] == "cd" else float(probs[pi]["gold"])
    na, nok = (annot_stats_tool(txt, tgt) if (args.tool and probs[pi]["task"] == "cd") else annot_stats(txt, tgt))
    return dict(pi=pi, task=probs[pi]["task"], ids=ids, txt=txt, r=r, d=d,
                nref=len(REFLECT.findall(txt)), nsrch=len(SEARCH.findall(txt)),
                nannot=na, annot_ok=nok, cont=cont, spans=spans, n_calls=n_calls, n_err=n_err, fake=fake, n_asks=n_asks)


def _finish_guess(pi, o, probs):
    """猜数字一条轨迹 → 和 _finish 同形的记录：calls = 轮数，errs = invalid + 重复"""
    r, d = GENV.score(o["state"], o["txt"])
    if o["fake"]:
        r -= args.fake_pen                                          # ★ 自己编 <obs>
    st = o["state"]
    return dict(pi=pi, task="guess", ids=o["ids"], txt=o["txt"], r=r, d=d, nref=len(REFLECT.findall(o["txt"])), nsrch=0, nannot=0, annot_ok=0,
                cont=o["cont"], spans=o["spans"], n_calls=st["turns"], n_err=st["invalid"] + st["repeat"], fake=o["fake"], n_asks=0)


def _finish_code(pi, o, probs):
    """修 bug 一条轨迹 → 和 _finish 同形：calls = 工具调用数，errs = 语法错次数"""
    r, d = CENV.score(o["state"], o["txt"])
    if o["fake"]:
        r -= args.fake_pen                                          # ★ 自己编 <result> / <reply>
    st = o["state"]
    return dict(pi=pi, task="code", ids=o["ids"], txt=o["txt"], r=r, d=d, nref=len(REFLECT.findall(o["txt"])), nsrch=0, nannot=0, annot_ok=0,
                cont=o["cont"], spans=o["spans"], n_calls=st["calls"], n_err=st["syntax_err"], fake=o["fake"], n_asks=st["n_ask"], lp=o.get("lp"))


def rollout(probs, k, temp, max_new, eval_mode=False):
    raw.gradient_checkpointing_disable()
    raw.eval(); raw.config.use_cache = True
    if args.ask or args.task in ("guess", "code", "pkg", "ex"):                      # ★ ex 的题也是 task="code", 同样要先渲染 _prompt
        pick_prompts(probs, eval_mode)
    flat = [(i, make_prompt(p)) for i, p in enumerate(probs) for _ in range(k)]
    out = []
    if args.task in ("code", "pkg", "ex"):                                         # ★ 修 bug：通用 harness + code_env，沙盒并发
        T = {"t0": time.time(), "tout0": CE.N_TIMEOUT, "run0": getattr(CENV, "n_run", 0), "hit0": getattr(CENV, "n_hit", 0)}   # ★ 分段计时 + 沙盒缓存命中
        if VL is not None and (VL_DIRTY or VL.asleep):
            vl_sync()
        T["sync"] = time.time()
        states = [CENV.reset(probs[i].get("ex") or probs[i].get("pkg") or probs[i]["bug"]) for i, _ in flat]
        res = HN.generate_with_env(VL if VL is not None else raw, tok, [pr for _, pr in flat], CENV, states,
                                   max_new=max_new, temp=temp, gen_bs=args.gen_bs, dev=DEV, step_workers=args.code_step_workers,
                                   max_total=VL_MAX_LEN if VL is not None else None, want_lp=(args.logp_diff and not eval_mode))
        T["harness"] = time.time()
        sw = args.score_workers or args.code_step_workers               # ★ 终评 = 每条一次沙盒子进程；和 env.step 一样放进线程池，别让 GPU 等一个串行的秤
        if sw > 1 and len(res) > 1:
            with ThreadPoolExecutor(max_workers=sw) as ex:
                out = list(ex.map(lambda t: _finish_code(t[0][0], t[1], probs), zip(flat, res)))
        else:
            out = [_finish_code(pi, o, probs) for (pi, _), o in zip(flat, res)]
        T["score"] = time.time()
        if VL is not None:
            VL.sleep(); import gc; gc.collect(); torch.cuda.empty_cache()
        T["sleep"] = time.time()
        tm = getattr(HN.generate_with_env, "last_timing", None)
        global ROLL_T                                                   # ★ 本卡这次 rollout 的分解，步末 all_gather 到 rank 0 打「四卡 rollout」
        ROLL_T = dict(total=T["sleep"] - T["t0"], gen=tm["gen"] if tm else 0.0, env=tm["env"] if tm else 0.0,
                      score=T["score"] - T["harness"], tout=float(CE.N_TIMEOUT - T["tout0"]), mode=(tm or {}).get("mode"))
        if tm and M0 and not eval_mode:
            hn = T["harness"] - T["sync"]
            if tm.get("mode") == "async":                              # 异步：引擎和沙盒重叠，沙盒是各线程累计，不能相加
                print(f"    [rollout 计时·异步 窗口{tm.get('inflight')}] 唤醒+灌权重 {T['sync']-T['t0']:.1f} s | harness 墙钟 {hn:.1f} s（引擎 {tm['gen']:.1f} s ∥ 沙盒累计 {tm['env']:.1f} s）"
                      f" | 终评 {T['score']-T['harness']:.1f} s（{len(res)} 条，{sw} 并发）| 睡下 {T['sleep']-T['score']:.1f} s"
                      f" | {tm['env_calls']} 次调用，最多一条 {tm['rounds']} 次，最多 {tm['max_pending']} 条同时在跑沙盒"
                      + (f" | 沙盒真跑 {getattr(CENV, 'n_run', 0) - T['run0']} 次 / 缓存命中 {getattr(CENV, 'n_hit', 0) - T['hit0']} 次" if hasattr(CENV, "n_run") else ""), flush=True)
            else:
              print(f"    [rollout 计时] 唤醒+灌权重 {T['sync']-T['t0']:.1f} s | harness {hn:.1f} s = 生成 {tm['gen']:.1f} + 环境 {tm['env']:.1f} + 其余 {hn-tm['gen']-tm['env']:.1f}"
                  f" | 终评 {T['score']-T['harness']:.1f} s（{len(res)} 条，{sw} 并发）| 睡下 {T['sleep']-T['score']:.1f} s"
                  f" | {tm['env_calls']} 次调用 / {tm['rounds']} 轮  一轮最多 {tm['max_pending']} 条同时等", flush=True)
        raw.config.use_cache = False; raw.gradient_checkpointing_enable(); raw.train()
        return out
    if args.task == "guess":                                        # ★ E：全部走通用 harness + 猜数字环境
        if VL is not None and (VL_DIRTY or VL.asleep):
            vl_sync()
        states = [GENV.reset(probs[i]["secret"]) for i, _ in flat]
        res = HN.generate_with_env(VL if VL is not None else raw, tok, [pr for _, pr in flat], GENV, states,
                                   max_new=max_new, temp=temp, gen_bs=args.gen_bs, dev=DEV)
        out = [_finish_guess(pi, o, probs) for (pi, _), o in zip(flat, res)]
        if VL is not None:
            VL.sleep(); import gc; gc.collect(); torch.cuda.empty_cache()
        raw.config.use_cache = False; raw.gradient_checkpointing_enable(); raw.train()
        return out
    if VL is not None:                                              # ★★ vLLM 采样端：醒来、灌最新权重、CD 和 GSM 都一次全交给它
        if VL_DIRTY or VL.asleep:
            vl_sync()
        gen = VL
        flat_cd = [(i, pr) for i, pr in flat if probs[i]["task"] == "cd"] if args.tool else []
        flat = [(i, pr) for i, pr in flat if not (args.tool and probs[i]["task"] == "cd")]
        if flat_cd:
            res = calc_tool.generate_with_tools(VL, tok, [pr for _, pr in flat_cd], max_new=max_new, temp=temp, max_calls=args.max_calls, **ASK_KW)
            for (pi, _), o in zip(flat_cd, res):
                out.append(_finish(pi, o["ids"], o["txt"], o["cont"], o["spans"], o["n_calls"], o["n_err"], o["fake"], probs, max_new, o.get("n_asks", 0)))
        if flat:
            res = calc_tool.plain_generate(VL, tok, [pr for _, pr in flat], max_new=max_new, temp=temp)
            for (pi, _), o in zip(flat, res):
                ids, txt, cont = o["ids"], o["txt"], o["cont"]
                if cont:                                     # ★ 幻觉出下一轮：截掉，补 EOS（同 HF 路径）
                    if len(ids) >= len(STOP_IDS) and ids[-len(STOP_IDS):] == STOP_IDS:
                        ids = ids[:-len(STOP_IDS)]
                    else:
                        ids = tok.encode(txt, add_special_tokens=False)
                    ids.append(EOS)
                out.append(_finish(pi, ids, txt, cont, [], 0, 0, False, probs, max_new))
        VL.sleep()                                               # ★ 采完就让出显存，反向传播要用
        import gc; gc.collect(); torch.cuda.empty_cache()        # ★ 睡下后把 PyTorch 缓存的空闲块还掉再进反向，治碎片（expandable_segments 跟睡眠分配器不兼容，不能用）
        raw.config.use_cache = False
        raw.gradient_checkpointing_enable()
        raw.train()
        return out
    if args.tool:                                                   # ★ CD 题走 harness，GSM 照旧
        flat_cd = [(i, pr) for i, pr in flat if probs[i]["task"] == "cd"]
        flat = [(i, pr) for i, pr in flat if probs[i]["task"] != "cd"]
        for s in range(0, len(flat_cd), args.gen_bs):
            chunk = flat_cd[s: s + args.gen_bs]
            res = calc_tool.generate_with_tools(raw, tok, [pr for _, pr in chunk], max_new=max_new, temp=temp,
                                                max_calls=args.max_calls, gen_bs=args.gen_bs, dev=DEV, **ASK_KW)
            for (pi, _), o in zip(chunk, res):
                out.append(_finish(pi, o["ids"], o["txt"], o["cont"], o["spans"], o["n_calls"], o["n_err"], o["fake"], probs, max_new, o.get("n_asks", 0)))
    for s in range(0, len(flat), args.gen_bs):
        chunk = flat[s: s + args.gen_bs]
        enc = tok([p for _, p in chunk], return_tensors="pt", padding=True).to(DEV)
        gen_kw = dict(max_new_tokens=max_new, do_sample=temp > 0,
                      temperature=max(temp, 1e-5), top_p=1.0, pad_token_id=PAD)
        try:
            g = raw.generate(**enc, stop_strings=[CONT], tokenizer=tok, **gen_kw)
        except (TypeError, ValueError):
            g = raw.generate(**enc, **gen_kw)
        for (pi, _), row in zip(chunk, g[:, enc.input_ids.size(1):]):
            ids = row.tolist()
            if EOS in ids:
                ids = ids[: ids.index(EOS) + 1]
            else:
                while ids and ids[-1] == PAD:
                    ids.pop()
            txt = tok.decode(ids, skip_special_tokens=True)
            cont = CONT in txt
            if cont:                                     # ★ 幻觉出下一轮：截掉，补 EOS
                txt = txt.split(CONT)[0]
                if len(ids) >= len(STOP_IDS) and ids[-len(STOP_IDS):] == STOP_IDS:
                    ids = ids[:-len(STOP_IDS)]
                else:                                    # 停止串没对齐上 token 边界 → 按文本重编码
                    ids = tok.encode(txt, add_special_tokens=False)
                ids.append(EOS)                          # ★ 教它「答完就停」
            out.append(_finish(pi, ids, txt, cont, [], 0, 0, False, probs, max_new))
    raw.config.use_cache = False
    raw.gradient_checkpointing_enable()
    raw.train()
    return out


def token_logp_full(m, ids, attn, tgt_mask):
    """老路径：整段 [B, L, V] 的 logits 一次算出来再 .float()。L=5000 时光 logits 就 1.5 GB bf16 + 3 GB fp32 + 反向再存 3 GB → ②-A 冒烟 OOM"""
    logits = m(input_ids=ids, attention_mask=attn).logits[:, :-1]
    tgt = ids[:, 1:]
    lp = -F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                          tgt.reshape(-1), reduction="none").view_as(tgt)
    return lp, tgt_mask[:, 1:]


def token_logp(m, ids, attn, tgt_mask, chunk=1024):
    """★ 分块算 logits + 交叉熵，每块套 checkpoint：任何时刻显存里只有一块 [B, chunk, V]（fp32 0.6 GB），反向时重算。
    数值和老路径一样（本机小模型对过：值和梯度都一致）。模型没有 .model/.lm_head 就退回老路径"""
    base, head = getattr(m, "model", None), getattr(m, "lm_head", None)
    if base is None or head is None:
        return token_logp_full(m, ids, attn, tgt_mask)
    from torch.utils.checkpoint import checkpoint
    h = base(input_ids=ids, attention_mask=attn).last_hidden_state[:, :-1]
    tgt = ids[:, 1:]

    def f(hc, tc):
        lg = head(hc).float()
        return -F.cross_entropy(lg.reshape(-1, lg.size(-1)), tc.reshape(-1), reduction="none").view_as(tc)
    outs = [checkpoint(f, h[:, s: s + chunk], tgt[:, s: s + chunk], use_reentrant=False) for s in range(0, h.size(1), chunk)]
    return torch.cat(outs, 1), tgt_mask[:, 1:]


def pad_batch(items):
    """items: (full_ids, prompt_len, resp_mask)。resp_mask 里 0 = harness 注入的 token，不进 loss / KL"""
    L = max(len(x[0]) for x in items)
    ids = torch.full((len(items), L), PAD, dtype=torch.long)
    attn = torch.zeros((len(items), L), dtype=torch.long)
    rm = torch.zeros((len(items), L), dtype=torch.float)
    for i, (full, np_, m) in enumerate(items):
        n = len(full)
        ids[i, :n] = torch.tensor(full); attn[i, :n] = 1
        rm[i, np_:n] = torch.tensor(m, dtype=torch.float)
    return ids.to(DEV), attn.to(DEV), rm.to(DEV)


# ══════════════════════════════════════════════════════════════
#  ★ 按任务分开的统计 —— 用【求和 + 计数】跨 rank 归约，而不是「rank 均值的均值」
#    （某个 rank 可能一道 GSM 都没抽到，简单平均会偏）
# ══════════════════════════════════════════════════════════════
KEYS = ["n", "r", "correct", "length", "ref_any", "ref_n",
        "ry_n", "ry_r", "rn_n", "rn_r", "adv",
        "srch_any", "srch_n", "sy_n", "sy_r", "sn_n", "sn_r",       # 搜索措辞那套（保留做对照）
        "annot_n", "annot_ok", "cont_n",                            # ★★ 顿悟探针
        "b0_n", "b0_c", "b12_n", "b12_c", "b35_n", "b35_c", "b6_n", "b6_c",   # ★★ 剂量响应桶
        "fc_n", "se_sum", "dup_n", "viol_n", "n_c", "len_c", "n_w", "len_w",   # ★ 判分器 v2 各项 + 对/错各自的长度
        "calls", "errs", "fake_n",                                             # ★ 工具
        "asks", "asked_n", "ask_c", "noask_n", "noask_c",                      # ★ 求助：次数、有求助的条数、求助过的对、没求助的条数、没求助的对
        "ffix", "capn",                                                        # ★ pkg：文件修好比例之和、撞调用上限的条数
        "selft", "verif", "fp_n", "st_n", "fv"]                                # ★ ex 弱观测：自测过的条数、验过最终版的条数、假阳次数、断言 run 总次数、首版（全秤）全过条数


def task_sums(items, adv=None):
    v = dict.fromkeys(KEYS, 0.0)
    for j, x in enumerate(items):
        v["n"] += 1; v["r"] += x["r"]; v["correct"] += x["d"]["correct"]
        v["length"] += len(x["ids"]); v["ref_any"] += (x["nref"] > 0); v["ref_n"] += x["nref"]
        if x["nref"] > 0: v["ry_n"] += 1; v["ry_r"] += x["r"]
        else:             v["rn_n"] += 1; v["rn_r"] += x["r"]
        v["srch_any"] += (x["nsrch"] > 0); v["srch_n"] += x["nsrch"]
        if x["nsrch"] > 0: v["sy_n"] += 1; v["sy_r"] += x["r"]
        else:              v["sn_n"] += 1; v["sn_r"] += x["r"]
        v["annot_n"] += x["nannot"]; v["annot_ok"] += x["annot_ok"]; v["cont_n"] += x["cont"]
        b = bucket(x["nsrch"])
        v[f"b{b}_n"] += 1; v[f"b{b}_c"] += x["d"]["correct"]
        dd = x["d"]
        v["fc_n"] += dd.get("false_claim", 0); v["se_sum"] += dd.get("step_err", 0.0)
        v["dup_n"] += dd.get("dup", 0); v["viol_n"] += dd.get("viol", 0)
        if dd["correct"] > 0: v["n_c"] += 1; v["len_c"] += len(x["ids"])
        else:                 v["n_w"] += 1; v["len_w"] += len(x["ids"])
        v["calls"] += x.get("n_calls", 0); v["errs"] += x.get("n_err", 0); v["fake_n"] += x.get("fake", False)
        na_ = x.get("n_asks", 0); v["asks"] += na_
        if na_ > 0: v["asked_n"] += 1; v["ask_c"] += x["d"]["correct"]
        else:       v["noask_n"] += 1; v["noask_c"] += x["d"]["correct"]
        v["ffix"] += dd.get("frac_files", dd.get("frac_tests", dd["correct"])); v["capn"] += (x.get("n_calls", 0) >= args.code_max_calls) if args.task in ("pkg", "ex") else 0
        v["selft"] += (dd.get("n_selftest", 0) > 0); v["verif"] += dd.get("verified_final", 0); v["fp_n"] += dd.get("n_fp", 0)
        v["st_n"] += dd.get("n_tp", 0) + dd.get("n_fn", 0) + dd.get("n_fp", 0) + dd.get("n_tn", 0); v["fv"] += dd.get("first_ver_all", 0)
        if adv is not None: v["adv"] += abs(adv[j])
    return [v[k] for k in KEYS]


def reduce_tasks(roll, adv=None):
    """→ {"cd": {...均值...}, "gsm": {...}}，全 rank 汇总"""
    vec = []
    for t in TASKS:
        idx = [j for j, x in enumerate(roll) if x["task"] == t]
        vec += task_sums([roll[j] for j in idx], None if adv is None else [adv[j] for j in idx])
    tt = torch.tensor(vec, device=DEV, dtype=torch.float64)
    if WORLD > 1:
        dist.all_reduce(tt)
    vec = tt.tolist(); out = {}
    for ti, t in enumerate(TASKS):
        s = dict(zip(KEYS, vec[ti * len(KEYS): (ti + 1) * len(KEYS)]))
        n = s["n"]
        div = lambda a, b: (a / b) if b > 0 else float("nan")
        out[t] = dict(n=int(n), score=div(s["r"], n), correct=div(s["correct"], n),
                      length=div(s["length"], n), reflect=div(s["ref_any"], n),
                      reflect_n=div(s["ref_n"], n), adv=div(s["adv"], n),
                      reflect_gap=(div(s["ry_r"], s["ry_n"]) - div(s["rn_r"], s["rn_n"]))
                                  if (s["ry_n"] > 0 and s["rn_n"] > 0) else float("nan"),
                      search=div(s["srch_any"], n), search_n=div(s["srch_n"], n),
                      search_gap=(div(s["sy_r"], s["sy_n"]) - div(s["sn_r"], s["sn_n"]))
                                 if (s["sy_n"] > 0 and s["sn_n"] > 0) else float("nan"),
                      # ★★ 顿悟探针
                      annot=div(s["annot_n"], n),                        # 标注数/条
                      annot_prec=div(s["annot_ok"], s["annot_n"]),       # 标注精度（算对且方向对）
                      fc=div(s["fc_n"], n), se=div(s["se_sum"], n),        # ★ v2：假宣告率、错步率
                      dup=div(s["dup_n"], n), viol=div(s["viol_n"], n),    # ★ v2：重复行/条、违规行/条
                      len_ok=div(s["len_c"], s["n_c"]), len_bad=div(s["len_w"], s["n_w"]),   # 对/错各自均长
                      calls=div(s["calls"], n), errs=div(s["errs"], n), fake=div(s["fake_n"], n),   # ★ 工具：调用/条、报错/条、假结果率
                      asks=div(s["asks"], n), asked=div(s["asked_n"], n),                          # ★ 求助/条、有求助的轨迹比例
                      acc_ask=div(s["ask_c"], s["asked_n"]), acc_noask=div(s["noask_c"], s["noask_n"]),   # 求助过的对 / 没求助的对
                      cont=div(s["cont_n"], n),                          # 写完不停的比例
                      ffix=div(s["ffix"], n), capn=div(s["capn"], n),      # ★ pkg：文件修好比例、撞调用上限比例
                      selft=div(s["selft"], n), verif=div(s["verif"], n), fpr=div(s["fp_n"], s["st_n"]), first_all=div(s["fv"], n),   # ★ ex 弱观测：自测率、验过最终版、假阳率、首版全过
                      acc_b0=div(s["b0_c"], s["b0_n"]), acc_b12=div(s["b12_c"], s["b12_n"]),
                      acc_b35=div(s["b35_c"], s["b35_n"]), acc_b6=div(s["b6_c"], s["b6_n"]),
                      dose=(div(s["b35_c"], s["b35_n"]) - div(s["b0_c"], s["b0_n"]))
                           if (s["b35_n"] > 0 and s["b0_n"] > 0) else float("nan"))
    return out


def allmean(d):
    if WORLD == 1:
        return d
    keys = sorted(d)
    t = torch.tensor([0.0 if (d[k] != d[k]) else d[k] for k in keys], device=DEV, dtype=torch.float64)
    n = torch.tensor([0.0 if (d[k] != d[k]) else 1.0 for k in keys], device=DEV, dtype=torch.float64)
    dist.all_reduce(t); dist.all_reduce(n)
    return {k: (float(t[i] / n[i]) if n[i] > 0 else float("nan")) for i, k in enumerate(keys)}


def flat_ev(T):
    """把 {"cd":{...},"gsm":{...}} 压平成 {"cd_len":..}, 附带总分"""
    f = {f"{t}_{k}": v for t in T for k, v in T[t].items()}
    f["score"] = sum(T[t]["score"] for t in T) / len(T)                # ★ 各任务平均（guess 只有一个）
    f["gap"] = (T["gsm"]["length"] - T["cd"]["length"]) if ("cd" in T and "gsm" in T) else float("nan")   # ★★ 分化指标
    return f


@torch.no_grad()
def evaluate():
    """★ 贪心。两个评估集各按 rank 切，各评各的，按任务汇总"""
    per = args.eval_n // WORLD
    mine = [dict(x) for x in CODE_EVAL[RANK * per: (RANK + 1) * per]] if args.task in ("code", "pkg", "ex") else \
        GUESS_EVAL[RANK * per: (RANK + 1) * per] if args.task == "guess" else \
        (CD_EVAL[RANK * per: (RANK + 1) * per] + GSM_EVAL[RANK * per: (RANK + 1) * per])
    if args.task == "ex" and args.ex_dash_k > 0:                            # ★ 仪表盘 v2：温度 1 × k，见过 / 没见过分开报（求和 + 计数跨 rank 归约）
        r = rollout(mine, args.ex_dash_k, 1.0, args.max_new, eval_mode=True)
        ev = flat_ev(reduce_tasks(r))
        acc = torch.zeros(2, 4, device=DEV, dtype=torch.float64)          # [组][n, 对, 自测过, 首版全过]
        for x in r:
            g = 0 if mine[x["pi"]].get("grp") == "unseen" else 1; dd = x["d"]
            acc[g, 0] += 1; acc[g, 1] += dd["correct"]; acc[g, 2] += (dd.get("n_selftest", 0) > 0); acc[g, 3] += dd.get("first_ver_all", 0)
        if WORLD > 1:
            dist.all_reduce(acc)
        for g, name in ((0, "unseen"), (1, "seen")):
            n = float(acc[g, 0]); ev[f"code_{name}_n"] = n
            ev[f"code_correct_{name}"] = float(acc[g, 1] / n) if n else float("nan")
            ev[f"code_selft_{name}"] = float(acc[g, 2] / n) if n else float("nan")
            ev[f"code_first_{name}"] = float(acc[g, 3] / n) if n else float("nan")
        return ev
    r = rollout(mine, 1, 0.0, args.max_new, eval_mode=True)
    return flat_ev(reduce_tasks(r))


def fmt_ev(e):
    if args.task in ("code", "pkg", "ex"):
        base = (f"总 {e['score']:.4f} | 修好 {e['code_correct']:.3f}" + (f" {'测试比例' if args.task == 'ex' else '文件'} {e['code_ffix']:.3f} 撞调用上限 {e['code_capn']:.2f}" if args.task in ("pkg", "ex") else "")
                + f" 调用 {e['code_calls']:.2f} 语法错 {e['code_errs']:.2f} 假观测 {e['code_fake']:.3f} 长 {e['code_length']:.0f} 不停 {e['code_cont']:.2f}")
        if "code_correct_unseen" in e:                                      # ★ 仪表盘 v2
            base += (f" ‖ 没见过 {e['code_correct_unseen']:.3f}（首版 {e['code_first_unseen']:.2f} 自测 {e['code_selft_unseen']:.2f}）"
                     f" 见过 {e['code_correct_seen']:.3f}（首版 {e['code_first_seen']:.2f} 自测 {e['code_selft_seen']:.2f}）")
        if args.task == "ex" and args.ex_obs == "weak":
            base += f" ‖ 自测率 {e['code_selft']:.2f} 验过 {e['code_verif']:.2f} 假阳 {e['code_fpr']:.2f} 首版 {e['code_first_all']:.2f}"
        return base
    if args.task == "guess":
        return (f"总 {e['score']:.4f} | 猜中 {e['guess_correct']:.3f} 轮 {e['guess_calls']:.2f} 无效 {e['guess_errs']:.2f} 假观测 {e['guess_fake']:.3f}"
                f" 长 {e['guess_length']:.0f} 不停 {e['guess_cont']:.2f}")
    return (f"总 {e['score']:.4f} | CD 正{e['cd_correct']:.3f} 长{e['cd_length']:.0f}"
            f" 标{e['cd_annot']:.1f}@{e['cd_annot_prec']:.2f} 不停{e['cd_cont']:.2f}"
            f" | GSM 正{e['gsm_correct']:.3f} 长{e['gsm_length']:.0f}")


# ══════════════════════════════════════════════════════════════
#  训练循环
# ══════════════════════════════════════════════════════════════
step0, HIST = 0, []
BEST = {"score": -1.0, "step": -1}; no_improve = 0
ck_latest = os.path.join(args.out, "ckpt_latest.pt")
if args.resume and os.path.exists(ck_latest):
    ck = torch.load(ck_latest, map_location="cpu", weights_only=False)
    raw.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
    step0, HIST = ck["step"], ck.get("hist", [])
    prev = [h for h in HIST if "ev" in h]
    if prev:
        b = max(prev, key=lambda h: h["ev"]["score"])
        BEST = {"score": b["ev"]["score"], "step": b["step"] + 1}
    p0(f"★ 从 step {step0} 续训（峰值 {BEST['score']:.4f}）")
    del ck; import gc; gc.collect(); torch.cuda.empty_cache()   # ★ 6 GB 的 ckpt 留在 CPU 内存里，每次沙盒 fork 都要多复制它的页表（续跑第 26 步环境 8 s → 51 s 的元凶之一）
elif args.init:
    ck = torch.load(args.init, map_location="cpu", weights_only=False)
    raw.load_state_dict(ck["model"])
    if "opt" in ck:
        opt.load_state_dict(ck["opt"])
    p0(f"★ 从 {args.init}（step {ck.get('step', '?')}）初始化权重，本次 step 从 0 数，写到 {args.out}/")
    del ck; torch.cuda.empty_cache()
if args.ref_init and ref is not None:                      # ★ KL 锚到 SFT 模型而不是 Base
    ck = torch.load(args.ref_init, map_location="cpu", weights_only=False)
    ref.load_state_dict(ck["model"])
    p0(f"★ KL 参照 = {args.ref_init}（step {ck.get('step', '?')}），β={args.beta}")
    del ck; torch.cuda.empty_cache()

# ★ vLLM 采样端：权重从 HF 仓库装一次（结构和 tokenizer），随即用训练端当前权重覆盖；之后每步 opt.step 后再灌
VL, VL_DIRTY = None, False
ROLL_T = None
def vl_sync():
    """训练端 → 采样端。同卡本地拷贝，1.5B 约 1~2 s"""
    global VL_DIRTY
    if VL is not None:                                          # ★ 分段唤醒：先醒权重 → 灌 → 再醒 KV（官方 RLHF 推荐顺序，峰值更低）
        if VL.asleep:
            VL.wake_up(["weights"]); VL.load_weights(raw.state_dict()); VL.wake_up(["kv_cache"])
        else:
            VL.load_weights(raw.state_dict())
        VL_DIRTY = False
        if args.smoke:
            p0(f"    [vl_sync] 唤醒 + 灌权重后 显存 allocated {torch.cuda.memory_allocated()/1024**3:.1f} / reserved {torch.cuda.memory_reserved()/1024**3:.1f} GB")
if args.engine == "vllm":
    p0(f"★ 起 vLLM 采样端（每 rank 一份，显存比例 {args.vllm_mem}，external_launcher + 睡眠）…", flush=True)
    VL_MAX_LEN = args.max_new + (2400 if args.task == 'pkg' else 1600 if args.task in ('code', 'pkg', 'ex') else 400)       # 修 bug 的题面带代码 + traceback，见过 1300+；超出的条 harness 自己截，不让引擎报错
    VL = calc_tool.VLLMBackend(args.model, tok, gpu_mem=args.vllm_mem, max_model_len=VL_MAX_LEN,
                               seed=args.seed + RANK, external=True, sleep=True)
    vl_sync()
    p0(f"  vLLM 就绪，已灌入当前权重   显存 {torch.cuda.memory_allocated()/1024**3:.1f} GB")
if WORLD > 1:
    dist.barrier()


def free_gb(path="."):
    import shutil
    return shutil.disk_usage(path).free / 1024 ** 3


def save(name, step, with_opt=True, ev=None):
    if not M0:
        return
    path = os.path.join(args.out, name); tmp = path + ".tmp"
    need = 6.5 if with_opt else 3.5                                  # ★ 带优化器约 5.8 GB，只权重约 2.9 GB；臂 T / 臂 A 都在这儿盘满崩过
    if free_gb(args.out) < need:
        p0(f"  ★★★ 盘只剩 {free_gb(args.out):.1f} GB，不够存 {name}（要 {need} GB）→ " + ("改成只存权重" if with_opt else "跳过这次存盘") + "，赶紧清盘")
        if not with_opt:
            return
        with_opt = False
    d = {"model": raw.state_dict(), "step": step, "hist": HIST, "args": vars(args)}
    if with_opt: d["opt"] = opt.state_dict()
    if ev: d["ev"] = ev
    try:
        torch.save(d, tmp); os.replace(tmp, path)
    except Exception as e:                                           # 写失败别把进程带崩（之前 segfault 就是这么来的），留着继续训
        p0(f"  ★★★ 存 {name} 失败：{e}  盘剩 {free_gb(args.out):.1f} GB。继续训，先清盘")
        if os.path.exists(tmp): os.remove(tmp)


def dump_hist():
    if M0:
        json.dump(HIST, open(os.path.join(args.out, "hist.json"), "w"), indent=1)


p0("=" * 110)
_BN = {"mix": "混训 Countdown + GSM8K", "guess": "E 猜数字", "code": "修 bug", "pkg": "多文件包", "ex": "从头写 exercism"}[args.task]
p0(f"  ★ {_BN}（{WORLD} 卡）  {args.model}  {n_par/1e9:.2f}B  {OPT}")
p0((f"  每步 {N_CD} CD + {N_GSM} GSM" if args.task == "mix" else f"  每步 {args.probs} 题") + f"，k={args.k} → {args.probs*args.k} 轨迹"
   f"   lr {args.lr:.1e}  KL β {args.beta}  max_new {args.max_new}"
   f"   优势 {'(r−μ)/σ' if args.adv_std else 'r−μ（同上一轮）'}")
if args.task == "mix":
    p0("  ★★★ 主判据：分化 = GSM 长度 − CD 长度。起点 ≈ 20；若涨到 >100 说明模型在读题，不在看模板")
if args.len_soft > 0:
    p0(f"  ★ 长度软惩罚：{args.len_soft} token 内不罚，到 {args.max_new} 扣 {args.len_pen}"
       f"（375 token 约扣 {args.len_pen * min(1, (375 - args.len_soft) / max(1, args.max_new - args.len_soft)):.2f}）")
if args.tool:
    p0(f"  ★★ 计算器 harness：CD 题 <calc>…</calc> 停 → safe_eval → 注入 <result>…</result> 续；注入段不进 loss/KL；"
       f"自己写 <result> 扣 {args.fake_pen}；每条最多 {args.max_calls} 次")
if args.ask:
    p0(f"  ★★★ 臂 A 求助环境：提示词每步从 prompts_cd[{args.prompt_set}] 抽（说明各 {args.p_hint:.0%} 出现）；<ask> 停 → 专家（{1-args.ask_weak_frac:.0%} reasoner / "
       f"{args.ask_weak_frac:.0%} chat，没答重试一次）→ 注入 <reply>（mask）；每条最多 {args.max_asks} 次，每次扣 {args.ask_pen}；自己编 <reply> 扣 {args.fake_pen}")
if args.judge == "v2":
    p0("  ★★ 判分器 v2.1：阶梯 .02/.02/.06 + 对 .9 − 假宣告 .2 − 错步率 .1 − 重复 .05/行 − 违规 .05/行（各封顶 4）+ 算对行 .01（封顶 .05）；"
       "尝试行认任意短标签；答错且尝试 < 3 行错步率按缺行数算；对的只扣 .05·len/max_new → 任何对 > 任何错")
p0("=" * 110, flush=True)

if not args.smoke and step0 == 0:
    p0(f"训练前评估（每任务 {args.eval_n} 题，贪心）…", flush=True)
    e0 = evaluate()
    p0(f"  ★ 起点 {fmt_ev(e0)}", flush=True)
    HIST.append(dict(step=-1, ev=e0))

t_start = time.time()
torch.cuda.reset_peak_memory_stats()
STOP = torch.zeros(1, device=DEV)
for step in range(step0, args.steps):
    lr = set_lr(step)
    opt.zero_grad(set_to_none=True)

    # ★ 所有 rank 同一个 RNG：抽 N_CD 道 + N_GSM 道，打乱，再各取自己那份
    rng = random.Random(args.seed * 1000003 + step)
    n_draw = args.probs if (args.dyn_sample <= 0 or args.task == "mix") else int(round(args.probs * (1 + args.dyn_sample) / WORLD)) * WORLD   # 多抽的也要能被卡数整除
    if args.task == "ex":                                          # 每步抽 n_draw 道题：有扩池时主池占 --ex-mix，其余从扩池抽（所有 rank 同 seed → 同一批，再各切自己那份）
        if CODE_TRAIN_EXTRA:
            n_main = max(0, min(n_draw, int(round(n_draw * args.ex_mix))))
            qs = rng.sample(CODE_TRAIN_MAIN, min(n_main, len(CODE_TRAIN_MAIN))) + rng.sample(CODE_TRAIN_EXTRA, min(n_draw - n_main, len(CODE_TRAIN_EXTRA)))
            rng.shuffle(qs)
        else:
            qs = rng.sample(CODE_TRAIN, min(n_draw, len(CODE_TRAIN)))
        allp = [dict(task="code", ex=q, idx=step * n_draw + i) for i, q in enumerate(qs)]
    elif args.task == "pkg":                                       # 每步现拼 n_draw 个 K 文件的包（所有 rank 同一个 seed → 同一批，再各切自己那份）
        allp = [dict(task="code", pkg=pk, idx=step * n_draw + i) for i, pk in enumerate(PE.make_pkgs(CODE_TRAIN, args.pkg_K, n_draw, seed=rng.randrange(1 << 30), ext=PKG_EXT))]
    elif args.task == "code":                                      # 每步抽 probs 个 bug（题无限，不筛池）；--dyn-sample 时多抽
        allp = [dict(x) for x in rng.sample(CODE_TRAIN, n_draw)]
    elif args.task == "guess":                                     # 每步一批新秘密数（不筛池：每个秘密数都是新组）
        allp = [dict(task="guess", secret=rng.randint(1, args.guess_N), idx=step * args.probs + i) for i in range(n_draw)]
    else:
        allp = rng.sample(CD_TRAIN, N_CD) + rng.sample(GSM_TRAIN, N_GSM)
        rng.shuffle(allp)
    PER_DRAW = n_draw // WORLD
    myp = allp[RANK * PER_DRAW: (RANK + 1) * PER_DRAW]

    ts = time.time()
    roll = rollout(myp, args.k, args.temp, args.max_new)
    t_gen = time.time() - ts
    if args.task in ("code", "pkg", "ex") and WORLD > 1 and ROLL_T is not None:      # ★ 四卡各自的 rollout 时长：rank 0 的计时行只说自己，合梯度前要等最慢的卡
        v = torch.tensor([ROLL_T[k] for k in ("total", "gen", "env", "score", "tout")], device=DEV)
        allv = [torch.zeros_like(v) for _ in range(WORLD)]
        dist.all_gather(allv, v)                                       # 四卡都走这条（集合通信要对齐）
        if M0:
            rows = [x.tolist() for x in allv]
            lab = "沙盒累计" if (ROLL_T or {}).get("mode") == "async" else "环境"
            print("    [四卡 rollout] " + "  ".join(f"r{i} {a[0]:>3.0f}s(生成{a[1]:.0f} {lab}{a[2]:.0f} 终评{a[3]:.0f} 超时{int(a[4])})" for i, a in enumerate(rows))
                  + f"  | 最慢−最快 {max(a[0] for a in rows) - min(a[0] for a in rows):.0f} s", flush=True)
    n_varied = None
    if PER_DRAW > PER:                                             # ★ 动态采样：按组看奖励有没有差异，有差异的优先留下 PER 道，其余丢掉（不进梯度）
        by = {}
        for x in roll:
            by.setdefault(x["pi"], []).append(x["r"])
        varied = [pi for pi, rs in by.items() if max(rs) - min(rs) > 1e-6]
        flat_ = [pi for pi in by if pi not in varied]
        keep = (varied + flat_)[:PER]; n_varied = min(len(varied), PER)
        remap = {pi: i for i, pi in enumerate(keep)}
        roll = [dict(x, pi=remap[x["pi"]]) for x in roll if x["pi"] in remap]
        myp = [myp[pi] for pi in keep]

    # ── 优势（组完整落在本 rank）──
    R = np.zeros((PER, args.k), dtype=np.float32); cnt = [0] * PER; order = []
    for x in roll:
        R[x["pi"], cnt[x["pi"]]] = x["r"]; order.append((x["pi"], cnt[x["pi"]])); cnt[x["pi"]] += 1
    adv_m = R - R.mean(1, keepdims=True)
    if args.adv_std:
        adv_m = adv_m / (R.std(1, keepdims=True) + 1e-4)
    adv = [float(adv_m[p, j]) for p, j in order]

    pc = [tok.encode(make_prompt(p)) for p in myp]
    recs = [(pc[x["pi"]] + x["ids"], len(pc[x["pi"]]), a, calc_tool.response_mask(len(x["ids"]), x.get("spans", [])), x.get("lp"))
            for x, a in zip(roll, adv) if len(x["ids"]) > 0]
    my_tok = sum(sum(m) for _, _, _, m, _ in recs)
    tt_ = torch.tensor([float(my_tok)], device=DEV)
    if WORLD > 1:
        dist.all_reduce(tt_)
    tot_tok = float(tt_.item())
    if not recs or tot_tok == 0:
        p0(f"  {step:>4} ⚠️ 无有效轨迹，跳过"); continue

    # ── 微批 + 梯度累积 ──
    tt = time.time(); pg_s, kl_s = 0.0, 0.0
    dlp = torch.zeros(3, device=DEV, dtype=torch.float64)           # ★ Δlogp：[和, 个数, 最大]
    for s in range(0, len(recs), args.micro):
        mb = recs[s: s + args.micro]
        ids, attn, rm = pad_batch([(f, n, m) for f, n, _, m, _ in mb])
        A = torch.tensor([a for _, _, a, _, _ in mb], device=DEV).unsqueeze(1)
        rlp = None
        if ref is not None:
            with torch.no_grad():
                rlp, _ = token_logp(ref, ids, attn, rm)
        lp, m_ = token_logp(model, ids, attn, rm)
        if args.logp_diff and any(x[4] for x in mb):                  # ★ vLLM 采样时的 logp vs 训练器算的：只看模型自己写的 token（注入段 nan）
            vl = torch.full(ids.shape, float("nan"), device=DEV)
            for i, (f, n, _, _, lpv) in enumerate(mb):
                if lpv:
                    vl[i, n: n + len(lpv)] = torch.tensor(lpv[: ids.shape[1] - n], device=DEV)
            with torch.no_grad():
                dif = (lp.detach() - vl[:, 1:]).abs(); ok = torch.isfinite(dif) & (m_ > 0)
                if ok.any():
                    dlp[0] += dif[ok].sum(); dlp[1] += ok.sum(); dlp[2] = torch.maximum(dlp[2], dif[ok].max().double())
        pg = -(A * lp * m_).sum() / tot_tok
        kl_t = torch.zeros((), device=DEV)
        if rlp is not None:
            lr_ = rlp - lp
            kl_t = ((torch.exp(lr_) - lr_ - 1) * m_).sum() / tot_tok
        (pg + args.beta * kl_t).backward()
        pg_s += pg.item(); kl_s += kl_t.item()
    t_comm = allreduce_grads()
    if args.logp_diff and WORLD > 1:
        dist.all_reduce(dlp[:2]); dist.all_reduce(dlp[2:], op=dist.ReduceOp.MAX)
    gn = torch.nn.utils.clip_grad_norm_(raw.parameters(), args.clip_grad)
    opt.step()
    VL_DIRTY = True                                             # ★ 权重变了，下次 rollout 前灌给 vLLM
    t_tr = time.time() - tt
    peak = torch.cuda.max_memory_allocated() / 1024**3
    torch.cuda.reset_peak_memory_stats()

    # ── 日志：总体 + ★ 按任务 ──
    T = reduce_tasks(roll, adv)
    g = allmean(dict(sd=float(R.std(1).mean()), kl=kl_s, gn=float(gn), nvar=float(n_varied if n_varied is not None else sum(1 for i in range(PER) if R[i].max() - R[i].min() > 1e-6))))
    if M0:
        if args.task in ("guess", "code", "pkg", "ex"):
            c = T[TASKS[0]]; m = {k: float("nan") for k in c}
        else:
            c, m = T["cd"], T["gsm"]
        gap = m["length"] - c["length"]
        rec = dict(step=step, lr=lr, t_gen=t_gen, t_train=t_tr, t_comm=t_comm, peak=peak,
                   gap=gap, **g, **{f"{TASKS[0] if args.task != 'mix' else 'cd'}_{k}": v for k, v in c.items()},
                   **{f"gsm_{k}": v for k, v in m.items()})
        HIST.append(rec)
        el = time.time() - t_start; done = step - step0 + 1
        v2col = (f" 假{c['fc']:.2f} 错步{c['se']:.2f} 重{c['dup']:.1f} 违{c['viol']:.1f} 对长{c['len_ok']:.0f}/错长{c['len_bad']:.0f} |"
                 if args.judge == "v2" else "")
        v2col += (f" 调{c['calls']:.1f} 错{c['errs']:.2f} 假果{c['fake']:.2f} |" if args.tool else "")
        v2col += (f" 问{c['asks']:.2f}/条 问过{c['asked']:.2f} 问对{c['acc_ask']:.2f} 没问对{c['acc_noask']:.2f} |" if args.ask else "")
        if args.task in ("code", "pkg", "ex"):
            wk = (f" 自测{c['selft']:.2f} 验过{c['verif']:.2f} 假阳{c['fpr']:.2f} 首版{c['first_all']:.2f}" if (args.task == "ex" and args.ex_obs == "weak") else "")
            dl = (f" Δlp{float(dlp[0] / dlp[1]):.3f}/{float(dlp[2]):.2f}" if (args.logp_diff and float(dlp[1]) > 0) else "")
            print(f"  {step:>4}/{args.steps} | 修好{c['correct']:.2f}" + (f" {'测比' if args.task == 'ex' else '文件'}{c['ffix']:.2f} 撞限{c['capn']:.2f}" if args.task in ("pkg", "ex") else "") + wk + f" 分{c['score']:.2f} 调用{c['calls']:.2f} 语法错{c['errs']:.2f} 假观测{c['fake']:.2f}"
                  f" 长{c['length']:>4.0f} 不停{c['cont']:.2f} |A|{c['adv']:.2f} 有效组{g['nvar']:.1f}/{PER} | KL{g['kl']:.4f} |g|{g['gn']:.2f}{dl} | {peak:.0f}GB |"
                  f" {t_gen:.0f}+{t_tr:.0f}s | 剩{el/done*(args.steps-step-1)/60:>4.0f}分", flush=True)
        elif args.task == "guess":
            print(f"  {step:>4}/{args.steps} | 猜中{c['correct']:.2f} 分{c['score']:.2f} 轮{c['calls']:.2f} 无效{c['errs']:.2f} 假观测{c['fake']:.2f}"
                  f" 长{c['length']:>4.0f} 不停{c['cont']:.2f} |A|{c['adv']:.2f} | KL{g['kl']:.4f} |g|{g['gn']:.2f} | {peak:.0f}GB |"
                  f" {t_gen:.0f}+{t_tr:.0f}s | 剩{el/done*(args.steps-step-1)/60:>4.0f}分", flush=True)
        else:
          print(f"  {step:>4}/{args.steps} |"
              f" CD 分{c['score']:.2f} 正{c['correct']:.2f} 长{c['length']:>4.0f}"
              f" 标{c['annot']:.1f}@{c['annot_prec']:.2f} 剂{c['dose']:+.2f} 不停{c['cont']:.2f} |A|{c['adv']:.2f} |" + v2col +
              f" GSM 分{m['score']:.2f} 正{m['correct']:.2f} 长{m['length']:>4.0f} |A|{m['adv']:.2f} |"
              f" KL{g['kl']:.4f} |g|{g['gn']:.2f} | {peak:.0f}GB |"
              f" {t_gen:.0f}+{t_tr:.0f}s | 剩{el/done*(args.steps-step-1)/60:>4.0f}分", flush=True)

    if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
        ev = evaluate()
        if M0:
            HIST[-1]["ev"] = ev
            best = ev["score"] > BEST["score"]
            if best:
                BEST = {"score": ev["score"], "step": step + 1}; no_improve = 0
            else:
                no_improve += 1
            tag = ("★ 新高，存 ckpt_best" if best else
                   f"峰值 {BEST['score']:.4f} @ {BEST['step']}，{no_improve} 次没新高")
            print(f"    ── ★ eval {fmt_ev(ev)} | " + tag, flush=True)
            if best:
                save("ckpt_best.pt", step + 1, with_opt=False, ev=ev)
            if args.patience and no_improve >= args.patience:
                print(f"    ★ 连续 {no_improve} 次没新高 → 早停"); STOP[0] = 1
            dump_hist()

    if (step + 1) % args.save_every == 0 or step == args.steps - 1:
        save("ckpt_latest.pt", step + 1); dump_hist()

    if WORLD > 1:
        dist.broadcast(STOP, src=0)
    if STOP[0] > 0:
        save("ckpt_latest.pt", step + 1)
        p0("已存盘退出。--resume 可继续。"); break

# ══════════════════════════════════════════════════════════════
#  总结 —— ★★★ 主判据：分化
# ══════════════════════════════════════════════════════════════
if M0:
    dump_hist()
    E = [h["ev"] for h in HIST if "ev" in h]
    TR = [h for h in HIST if "gap" in h and "cd_length" in h]
    print("=" * 110)
    print(f"  完成   {(time.time()-t_start)/60:.1f} 分钟")
    if len(E) >= 2 and len(TR) >= 2:
        a, b = E[0], E[-1]
        f = lambda key, sl: float(np.nanmean([x.get(key, float("nan")) for x in sl])) if sl else float("nan")
        SEG = [x for x in TR if x["step"] >= step0] or TR      # ★ 续训时只看本段
        h, tl = SEG[:20], SEG[-20:]
        print(f"""
  ── ★★★ 主判据：长度分化（GSM 长度 − CD 长度）──
       eval（贪心）      {a['gap']:+.0f} → {b['gap']:+.0f}
     ★ 训练（温度 1）    {f('gap', h):+.0f} → {f('gap', tl):+.0f}
       ★ 单任务那一轮：CD 长度 249→71 且 GSM 上也只写 71 —— 分化 ≈ 0
       ★★ 若这里 >100：模型按【题目内容】决定长度 → 绑定被打破
  ── ★ 分任务长度 ──
       CD    eval {a['cd_length']:.0f} → {b['cd_length']:.0f}    训练 {f('cd_length', h):.0f} → {f('cd_length', tl):.0f}
       GSM   eval {a['gsm_length']:.0f} → {b['gsm_length']:.0f}    训练 {f('gsm_length', h):.0f} → {f('gsm_length', tl):.0f}
  ── ★ 分任务正确率（eval 贪心）──
       CD    {a['cd_correct']:.4f} → {b['cd_correct']:.4f}    （单任务那轮 0.035 → 0.405）
       GSM   {a['gsm_correct']:.4f} → {b['gsm_correct']:.4f}    （单任务那轮训完后在 test[0:200] 只有 0.16）
  ── ★ 梯度平衡（平均 |A|，前 20 步 → 后 20 步）──
       CD  {f('cd_adv', h):.3f} → {f('cd_adv', tl):.3f}     GSM {f('gsm_adv', h):.3f} → {f('gsm_adv', tl):.3f}
       ★ 若 GSM 一直是 CD 的 3 倍以上 → GSM 主导梯度，下一轮考虑 --adv-std
  ── ★ ★差（wait/hmm 类纠错措辞 vs 奖励）──
       CD  {f('cd_reflect_gap', h):+.3f} → {f('cd_reflect_gap', tl):+.3f}     GSM {f('gsm_reflect_gap', h):+.3f} → {f('gsm_reflect_gap', tl):+.3f}
  ── ★★ 搜索措辞（too high / too low / perfect …）──
       CD  出现率 {f('cd_search', h):.3f} → {f('cd_search', tl):.3f}   ★差 {f('cd_search_gap', h):+.3f} → {f('cd_search_gap', tl):+.3f}
       GSM 出现率 {f('gsm_search', h):.3f} → {f('gsm_search', tl):.3f}   ★差 {f('gsm_search_gap', h):+.3f} → {f('gsm_search_gap', tl):+.3f}
       （二值 ★差 是钝探针，装饰词会把信号淹掉；留着只为跟上一轮对照）
  ── ★★★ 顿悟探针（本段前 20 步 → 后 20 步，只看 CD）──
       标注数/条        {f('cd_annot', h):.2f} → {f('cd_annot', tl):.2f}
       标注精度         {f('cd_annot_prec', h):.3f} → {f('cd_annot_prec', tl):.3f}   ← 稳在 0.8 以上才是真验算
       剂量响应         {f('cd_dose', h):+.3f} → {f('cd_dose', tl):+.3f}   （试错 3~5 次 − 0 次 的正确率差）
       3~5 次桶正确率   {f('cd_acc_b35', h):.3f} → {f('cd_acc_b35', tl):.3f}     0 次桶 {f('cd_acc_b0', h):.3f} → {f('cd_acc_b0', tl):.3f}
       6+ 次桶正确率    {f('cd_acc_b6', h):.3f} → {f('cd_acc_b6', tl):.3f}   ← 跑飞的那桶
       写完不停         {f('cd_cont', h):.3f} → {f('cd_cont', tl):.3f}   ← 停止串+补 EOS 应把它压到接近 0
       ★ 数量涨 + 精度稳 + 剂量为正 → 顿悟在被放大
       ★ 数量涨 + 精度掉            → 变成表演了（奖励在选「多写」不是「算对」）
       ★ 都不动                     → 功能形态到此为止
  ★★ 峰值 {BEST['score']:.4f} @ step {BEST['step']}  →  {os.path.join(args.out, 'ckpt_best.pt')}
  ★ 训完跑：python3 crosstask_eval.py --ckpt {os.path.join(args.out, 'ckpt_best.pt')} --out crosstask_mix.json
     跟单任务那轮（R1 模板 0.16 / 71 token）在同一批 test[0:200] 上对照""")
        if args.tool:
            print(f"""  ── ★★ 工具（本段前 20 步 → 后 20 步，只看 CD，温度 1）──
       调用/条          {f('cd_calls', h):.2f} → {f('cd_calls', tl):.2f}
       报错/条          {f('cd_errs', h):.3f} → {f('cd_errs', tl):.3f}
       假结果率         {f('cd_fake', h):.3f} → {f('cd_fake', tl):.3f}   ← 模型自己写 <result> 的轨迹占比
       标注精度（方向）  {f('cd_annot_prec', h):.3f} → {f('cd_annot_prec', tl):.3f}   ← 工具版：值是算的，只看方向标没标对""")
        if args.ask:
            print(f"""       求助/条 · 问过 · 问对 · 没问对   {f('cd_asks', h):.2f} · {f('cd_asked', h):.2f} · {f('cd_acc_ask', h):.2f} · {f('cd_acc_noask', h):.2f}  →  """
                  f"""{f('cd_asks', tl):.2f} · {f('cd_asked', tl):.2f} · {f('cd_acc_ask', tl):.2f} · {f('cd_acc_noask', tl):.2f}   ← 臂 A：RL 前后问的多少、问得准不准""")
        if args.judge == "v2":
            print(f"""  ── ★★ 判分器 v2 各项（本段前 20 步 → 后 20 步，只看 CD，温度 1）──
       假宣告率         {f('cd_fc', h):.3f} → {f('cd_fc', tl):.3f}   ← 说 perfect 而式子 ≠ 目标的轨迹占比
       错步率           {f('cd_se', h):.3f} → {f('cd_se', tl):.3f}   ← 标注行里算错/标反的比例
       重复行/条        {f('cd_dup', h):.2f} → {f('cd_dup', tl):.2f}
       违规行/条        {f('cd_viol', h):.2f} → {f('cd_viol', tl):.2f}   ← 一次尝试里某个数用超了
       对的均长/错的均长 {f('cd_len_ok', h):.0f}/{f('cd_len_bad', h):.0f} → {f('cd_len_ok', tl):.0f}/{f('cd_len_bad', tl):.0f}
       ★ 标注/条跌到 3 以下 = 在躲错步罚（不搜直接猜）""")
    print("=" * 110)

# ★ 收尾：vLLM 睡眠模式的分配器在解释器退出时会报「Trying to free a pointer not allocated here」并段错误（结果早已落盘，纯善后问题）。
#   所有 rank 先同步、刷输出，再用 os._exit 跳过析构，torchrun 就不会把一次正常结束记成 FAILED
if VL is not None:
    import sys
    if WORLD > 1:
        dist.barrier()
        dist.destroy_process_group()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
