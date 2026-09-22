#!/usr/bin/env python3
"""
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
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import countdown as CD
import _reward as RW                              # ★ GSM8K 判分器，唯一定义处
import judge_v2 as J2                             # ★ 判分器 v2 的过程项（--judge v2）

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
ap.add_argument("--judge", choices=["v1", "v2"], default="v1",
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
ap.add_argument("--smoke",   action="store_true")
ap.add_argument("--seed",    type=int,   default=1337)
args = ap.parse_args()

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
    args.max_new, args.eval_every, args.eval_n, args.save_every = 256, 2, 2 * WORLD, 999

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


def bucket(nsrch):
    return 0 if nsrch == 0 else 12 if nsrch <= 2 else 35 if nsrch <= 5 else 6

# ══════════════════════════════════════════════════════════════
#  ★ 两个任务：数据 / 提示 / 判分  —— 统一成 {"task": "cd"|"gsm", ...}
# ══════════════════════════════════════════════════════════════
D = json.load(open(args.data))
CD_TRAIN = [dict(task="cd", **p) for p in D[: args.eval_offset]]
CD_EVAL = [dict(task="cd", **p) for p in D[args.eval_offset: args.eval_offset + args.eval_n]]

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


def make_prompt(p):
    return CD.make_prompt(p["nums"], p["target"]) if p["task"] == "cd" else gsm_prompt(p["q"])


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
def rollout(probs, k, temp, max_new):
    raw.gradient_checkpointing_disable()
    raw.eval(); raw.config.use_cache = True
    flat = [(i, make_prompt(p)) for i, p in enumerate(probs) for _ in range(k)]
    out = []
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
            r, d = score(probs[pi], txt)
            if args.judge == "v2" and probs[pi]["task"] == "cd" and d["correct"] > 0:
                r -= J2.PEN["len_ok"] * len(ids) / max_new                 # ★ v2：对的只轻扣，任何对 > 任何错
            elif args.len_soft > 0 and len(ids) > args.len_soft:       # ★ 长度软惩罚（进优势，不进 d）
                frac = min(1.0, (len(ids) - args.len_soft) / max(1, max_new - args.len_soft))
                r -= args.len_pen * frac
            tgt = probs[pi]["target"] if probs[pi]["task"] == "cd" else float(probs[pi]["gold"])
            na, nok = annot_stats(txt, tgt)
            out.append(dict(pi=pi, task=probs[pi]["task"], ids=ids, txt=txt, r=r, d=d,
                            nref=len(REFLECT.findall(txt)), nsrch=len(SEARCH.findall(txt)),
                            nannot=na, annot_ok=nok, cont=cont))
    raw.config.use_cache = False
    raw.gradient_checkpointing_enable()
    raw.train()
    return out


def token_logp(m, ids, attn, tgt_mask):
    logits = m(input_ids=ids, attention_mask=attn).logits[:, :-1]
    tgt = ids[:, 1:]
    lp = -F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                          tgt.reshape(-1), reduction="none").view_as(tgt)
    return lp, tgt_mask[:, 1:]


def pad_batch(items):
    L = max(len(x[0]) for x in items)
    ids = torch.full((len(items), L), PAD, dtype=torch.long)
    attn = torch.zeros((len(items), L), dtype=torch.long)
    rm = torch.zeros((len(items), L), dtype=torch.float)
    for i, (full, np_) in enumerate(items):
        n = len(full)
        ids[i, :n] = torch.tensor(full); attn[i, :n] = 1; rm[i, np_:n] = 1.0
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
        "fc_n", "se_sum", "dup_n", "viol_n", "n_c", "len_c", "n_w", "len_w"]   # ★ 判分器 v2 各项 + 对/错各自的长度


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
        if adv is not None: v["adv"] += abs(adv[j])
    return [v[k] for k in KEYS]


def reduce_tasks(roll, adv=None):
    """→ {"cd": {...均值...}, "gsm": {...}}，全 rank 汇总"""
    vec = []
    for t in ("cd", "gsm"):
        idx = [j for j, x in enumerate(roll) if x["task"] == t]
        vec += task_sums([roll[j] for j in idx], None if adv is None else [adv[j] for j in idx])
    tt = torch.tensor(vec, device=DEV, dtype=torch.float64)
    if WORLD > 1:
        dist.all_reduce(tt)
    vec = tt.tolist(); out = {}
    for ti, t in enumerate(("cd", "gsm")):
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
                      cont=div(s["cont_n"], n),                          # 写完不停的比例
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
    f["score"] = (T["cd"]["score"] + T["gsm"]["score"]) / 2           # ★ 两任务平均
    f["gap"] = T["gsm"]["length"] - T["cd"]["length"]                 # ★★ 分化指标
    return f


@torch.no_grad()
def evaluate():
    """★ 贪心。两个评估集各按 rank 切，各评各的，按任务汇总"""
    per = args.eval_n // WORLD
    mine = (CD_EVAL[RANK * per: (RANK + 1) * per] + GSM_EVAL[RANK * per: (RANK + 1) * per])
    r = rollout(mine, 1, 0.0, args.max_new)
    return flat_ev(reduce_tasks(r))


def fmt_ev(e):
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
elif args.init:
    ck = torch.load(args.init, map_location="cpu", weights_only=False)
    raw.load_state_dict(ck["model"])
    if "opt" in ck:
        opt.load_state_dict(ck["opt"])
    p0(f"★ 从 {args.init}（step {ck.get('step', '?')}）初始化权重，本次 step 从 0 数，写到 {args.out}/")
    del ck; torch.cuda.empty_cache()
if WORLD > 1:
    dist.barrier()


def save(name, step, with_opt=True, ev=None):
    if not M0:
        return
    path = os.path.join(args.out, name); tmp = path + ".tmp"
    d = {"model": raw.state_dict(), "step": step, "hist": HIST, "args": vars(args)}
    if with_opt: d["opt"] = opt.state_dict()
    if ev: d["ev"] = ev
    torch.save(d, tmp); os.replace(tmp, path)


def dump_hist():
    if M0:
        json.dump(HIST, open(os.path.join(args.out, "hist.json"), "w"), indent=1)


p0("=" * 110)
p0(f"  ★ 混训 Countdown + GSM8K（{WORLD} 卡）  {args.model}  {n_par/1e9:.2f}B  {OPT}")
p0(f"  每步 {N_CD} CD + {N_GSM} GSM，k={args.k} → {args.probs*args.k} 轨迹"
   f"   lr {args.lr:.1e}  KL β {args.beta}  max_new {args.max_new}"
   f"   优势 {'(r−μ)/σ' if args.adv_std else 'r−μ（同上一轮）'}")
p0("  ★★★ 主判据：分化 = GSM 长度 − CD 长度。起点 ≈ 20；若涨到 >100 说明模型在读题，不在看模板")
if args.len_soft > 0:
    p0(f"  ★ 长度软惩罚：{args.len_soft} token 内不罚，到 {args.max_new} 扣 {args.len_pen}"
       f"（375 token 约扣 {args.len_pen * min(1, (375 - args.len_soft) / max(1, args.max_new - args.len_soft)):.2f}）")
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
    allp = rng.sample(CD_TRAIN, N_CD) + rng.sample(GSM_TRAIN, N_GSM)
    rng.shuffle(allp)
    myp = allp[RANK * PER: (RANK + 1) * PER]

    ts = time.time()
    roll = rollout(myp, args.k, args.temp, args.max_new)
    t_gen = time.time() - ts

    # ── 优势（组完整落在本 rank）──
    R = np.zeros((PER, args.k), dtype=np.float32); cnt = [0] * PER; order = []
    for x in roll:
        R[x["pi"], cnt[x["pi"]]] = x["r"]; order.append((x["pi"], cnt[x["pi"]])); cnt[x["pi"]] += 1
    adv_m = R - R.mean(1, keepdims=True)
    if args.adv_std:
        adv_m = adv_m / (R.std(1, keepdims=True) + 1e-4)
    adv = [float(adv_m[p, j]) for p, j in order]

    pc = [tok.encode(make_prompt(p)) for p in myp]
    recs = [(pc[x["pi"]] + x["ids"], len(pc[x["pi"]]), a)
            for x, a in zip(roll, adv) if len(x["ids"]) > 0]
    my_tok = sum(len(f) - n for f, n, _ in recs)
    tt_ = torch.tensor([float(my_tok)], device=DEV)
    if WORLD > 1:
        dist.all_reduce(tt_)
    tot_tok = float(tt_.item())
    if not recs or tot_tok == 0:
        p0(f"  {step:>4} ⚠️ 无有效轨迹，跳过"); continue

    # ── 微批 + 梯度累积 ──
    tt = time.time(); pg_s, kl_s = 0.0, 0.0
    for s in range(0, len(recs), args.micro):
        mb = recs[s: s + args.micro]
        ids, attn, rm = pad_batch([(f, n) for f, n, _ in mb])
        A = torch.tensor([a for *_, a in mb], device=DEV).unsqueeze(1)
        rlp = None
        if ref is not None:
            with torch.no_grad():
                rlp, _ = token_logp(ref, ids, attn, rm)
        lp, m_ = token_logp(model, ids, attn, rm)
        pg = -(A * lp * m_).sum() / tot_tok
        kl_t = torch.zeros((), device=DEV)
        if rlp is not None:
            lr_ = rlp - lp
            kl_t = ((torch.exp(lr_) - lr_ - 1) * m_).sum() / tot_tok
        (pg + args.beta * kl_t).backward()
        pg_s += pg.item(); kl_s += kl_t.item()
    t_comm = allreduce_grads()
    gn = torch.nn.utils.clip_grad_norm_(raw.parameters(), args.clip_grad)
    opt.step()
    t_tr = time.time() - tt
    peak = torch.cuda.max_memory_allocated() / 1024**3
    torch.cuda.reset_peak_memory_stats()

    # ── 日志：总体 + ★ 按任务 ──
    T = reduce_tasks(roll, adv)
    g = allmean(dict(sd=float(R.std(1).mean()), kl=kl_s, gn=float(gn)))
    if M0:
        c, m = T["cd"], T["gsm"]
        gap = m["length"] - c["length"]
        rec = dict(step=step, lr=lr, t_gen=t_gen, t_train=t_tr, t_comm=t_comm, peak=peak,
                   gap=gap, **g, **{f"cd_{k}": v for k, v in c.items()},
                   **{f"gsm_{k}": v for k, v in m.items()})
        HIST.append(rec)
        el = time.time() - t_start; done = step - step0 + 1
        v2col = (f" 假{c['fc']:.2f} 错步{c['se']:.2f} 重{c['dup']:.1f} 违{c['viol']:.1f} 对长{c['len_ok']:.0f}/错长{c['len_bad']:.0f} |"
                 if args.judge == "v2" else "")
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
        if args.judge == "v2":
            print(f"""  ── ★★ 判分器 v2 各项（本段前 20 步 → 后 20 步，只看 CD，温度 1）──
       假宣告率         {f('cd_fc', h):.3f} → {f('cd_fc', tl):.3f}   ← 说 perfect 而式子 ≠ 目标的轨迹占比
       错步率           {f('cd_se', h):.3f} → {f('cd_se', tl):.3f}   ← 标注行里算错/标反的比例
       重复行/条        {f('cd_dup', h):.2f} → {f('cd_dup', tl):.2f}
       违规行/条        {f('cd_viol', h):.2f} → {f('cd_viol', tl):.2f}   ← 一次尝试里某个数用超了
       对的均长/错的均长 {f('cd_len_ok', h):.0f}/{f('cd_len_bad', h):.0f} → {f('cd_len_ok', tl):.0f}/{f('cd_len_bad', tl):.0f}
       ★ 标注/条跌到 3 以下 = 在躲错步罚（不搜直接猜）""")
    print("=" * 110)
