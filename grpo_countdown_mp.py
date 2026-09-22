#!/usr/bin/env python3
"""
R1-Zero 复现（★ 多进程 4 卡版）—— 待办 B 的实现。

    torchrun --nproc_per_node=4 grpo_countdown_mp.py --smoke
    torchrun --nproc_per_node=4 grpo_countdown_mp.py --probs 8 -k 16 --steps 300

★★ 跟单进程版（grpo_countdown.py）的核心区别：

  单进程 + 4 线程   ★ 共享一个 GIL → HF generate 的 Python 部分互相抢锁
                    ★ 实测：比单卡【慢 4.5 倍】

  ★ 多进程（本文件） torchrun 起 4 个独立进程 → ★ 各有各的 GIL → 真并行

★★ 而且【不需要】手动同步权重：
   各 rank 的梯度 all_reduce(SUM) 后完全相同 → 应用相同更新 → ★ 权重天然保持一致
   → ★ 采样直接用自己那份训练模型，零同步开销
   （对比单进程线程版：那个方案要每步把 3.08 GB 权重复制给 3 张卡）

★ 关键设计：按【题目】切分，不是按轨迹切分
   8 题 / 4 rank = 每 rank 负责 2 题的【全部】k 条
   → ★ 组内平均自己就能算，不需要跨进程通信
"""
import os, re, json, time, random, argparse
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import countdown as CD

ap = argparse.ArgumentParser()
ap.add_argument("--model",   default="Qwen/Qwen2.5-1.5B")
ap.add_argument("--data",    default="data/countdown.json")
ap.add_argument("--out",     default="out_cd_mp")
ap.add_argument("--steps",   type=int,   default=300)
ap.add_argument("--probs",   type=int,   default=8,  help="★ 每步几道题（必须能被卡数整除）")
ap.add_argument("-k", "--k", type=int,   default=16, help="组大小。采样半径≈0.7/k")
ap.add_argument("--micro",   type=int,   default=2)
ap.add_argument("--gen-bs",  type=int,   default=32)
ap.add_argument("--max-new", type=int,   default=512)
ap.add_argument("--temp",    type=float, default=1.0)
ap.add_argument("--lr",      type=float, default=5e-6)
ap.add_argument("--beta",    type=float, default=0.0005)
ap.add_argument("--warmup",  type=int,   default=10)
ap.add_argument("--clip-grad", type=float, default=1.0)
ap.add_argument("--eval-every", type=int, default=25)
ap.add_argument("--eval-n",  type=int,   default=200)
ap.add_argument("--eval-offset", type=int, default=90000)
ap.add_argument("--save-every", type=int, default=25)
ap.add_argument("--patience", type=int,  default=5)
ap.add_argument("--resume",  action="store_true")
ap.add_argument("--smoke",   action="store_true")
ap.add_argument("--seed",    type=int,   default=1337)
args = ap.parse_args()

# ══════════════════════════════════════════════════════════════
#  ★ 分布式初始化（torchrun 会设好这些环境变量）
# ══════════════════════════════════════════════════════════════
RANK = int(os.environ.get("RANK", 0))
WORLD = int(os.environ.get("WORLD_SIZE", 1))
LOCAL = int(os.environ.get("LOCAL_RANK", 0))
if WORLD > 1:
    dist.init_process_group("nccl")
torch.cuda.set_device(LOCAL)
DEV = f"cuda:{LOCAL}"
M0 = RANK == 0                                   # ★ 只有 rank 0 打日志/存盘


def p0(*a, **kw):
    if M0:
        print(*a, **kw)


if args.smoke:
    args.steps, args.probs, args.k = 3, WORLD, 4
    args.max_new, args.eval_every, args.eval_n, args.save_every = 256, 2, 4 * WORLD, 999

assert args.probs % WORLD == 0, f"--probs {args.probs} 必须能被卡数 {WORLD} 整除"
assert args.eval_n % WORLD == 0, f"--eval-n {args.eval_n} 必须能被卡数 {WORLD} 整除"
PER = args.probs // WORLD                        # 每个 rank 负责几道题

torch.manual_seed(args.seed + RANK); np.random.seed(args.seed + RANK)
if M0:
    os.makedirs(args.out, exist_ok=True)

REFLECT = re.compile(
    r"\b(wait|hold on|hmm|actually|alternatively|instead|let me (?:try|check|recheck|reconsider)"
    r"|that (?:doesn'?t|does not) work|not (?:right|correct)|recheck|reconsider)\b", re.I)

D = json.load(open(args.data))
TRAIN = D[: args.eval_offset]
EVAL = D[args.eval_offset: args.eval_offset + args.eval_n]
p0(f"Countdown  train {len(TRAIN):,}   eval {len(EVAL)}   ★ {WORLD} 进程，每进程 {PER} 题/步")

# ══════════════════════════════════════════════════════════════
#  模型
# ══════════════════════════════════════════════════════════════
tok = AutoTokenizer.from_pretrained(args.model)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
PAD, EOS = tok.pad_token_id, tok.eos_token_id


def load():
    try:
        m = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    return m.to(DEV)


p0(f"载入模型（每个 rank 一份）…", flush=True)
raw = load()
raw.gradient_checkpointing_enable()
raw.config.use_cache = False

ref = None
if args.beta > 0:
    ref = load().eval()
    for p in ref.parameters():
        p.requires_grad_(False)

# ★★ 不用 DDP 包装，改成【手动 all_reduce 梯度】。三个原因：
#   ① DDP 的 static_graph 和 no_sync() 不兼容，而梯度检查点又要求 static_graph
#   ② 我们有微批梯度累积，DDP 会在每次 backward 都尝试同步 —— 要一堆 no_sync 补丁
#   ③ ★ 手动版逻辑一眼看得懂：全部微批 backward 完 → all_reduce(SUM) → step
# 代价：通信不能跟 backward 重叠，约 1.2 秒/步（占 5%），换来的是没有隐藏坑
model = raw


def allreduce_grads():
    """★ 各 rank 的梯度求和。因为每个 rank 都已按【全局】token 数归一，
       求和后就是全局梯度 → 各 rank 拿到相同梯度 → ★ 权重天然保持一致"""
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
#  采样（★ 每个 rank 只采自己那几道题，进程独立 → 真并行）
# ══════════════════════════════════════════════════════════════
@torch.no_grad()
def rollout(probs, k, temp, max_new):
    raw.gradient_checkpointing_disable()
    raw.eval(); raw.config.use_cache = True
    flat = [(i, CD.make_prompt(p["nums"], p["target"]))
            for i, p in enumerate(probs) for _ in range(k)]
    out = []
    for s in range(0, len(flat), args.gen_bs):
        chunk = flat[s: s + args.gen_bs]
        enc = tok([p for _, p in chunk], return_tensors="pt", padding=True).to(DEV)
        g = raw.generate(**enc, max_new_tokens=max_new, do_sample=temp > 0,
                         temperature=max(temp, 1e-5), top_p=1.0, pad_token_id=PAD)
        for (pi, _), row in zip(chunk, g[:, enc.input_ids.size(1):]):
            ids = row.tolist()
            if EOS in ids:
                ids = ids[: ids.index(EOS) + 1]
            else:
                while ids and ids[-1] == PAD:
                    ids.pop()
            txt = tok.decode(ids, skip_special_tokens=True)
            r, d = CD.reward(txt, probs[pi]["nums"], probs[pi]["target"])
            out.append((pi, ids, txt, r, d, len(REFLECT.findall(txt))))
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


def allmean(d):
    """★ 把各 rank 的标量指标求平均（nan 安全）"""
    if WORLD == 1:
        return d
    keys = sorted(d)
    t = torch.tensor([0.0 if (d[k] != d[k]) else d[k] for k in keys],
                     device=DEV, dtype=torch.float64)
    n = torch.tensor([0.0 if (d[k] != d[k]) else 1.0 for k in keys],
                     device=DEV, dtype=torch.float64)
    dist.all_reduce(t); dist.all_reduce(n)
    return {k: (float(t[i] / n[i]) if n[i] > 0 else float("nan")) for i, k in enumerate(keys)}


@torch.no_grad()
def evaluate():
    """★ 贪心。评估集也按 rank 切分，各评各的再汇总"""
    mine = EVAL[RANK * (len(EVAL) // WORLD): (RANK + 1) * (len(EVAL) // WORLD)]
    r = rollout(mine, 1, 0.0, args.max_new)
    d = dict(score=float(np.mean([x[3] for x in r])),
             length=float(np.mean([len(x[1]) for x in r])),
             reflect=float(np.mean([x[5] > 0 for x in r])),
             reflect_n=float(np.mean([x[5] for x in r])),
             **{k: float(np.mean([x[4][k] for x in r]))
                for k in ("fmt", "parse", "nums_frac", "correct")})
    return allmean(d)


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


p0("=" * 100)
p0(f"  ★ R1-Zero 复现（多进程 {WORLD} 卡）  {args.model}  {n_par/1e9:.2f}B  {OPT}")
p0(f"  每步 {args.probs} 题 × k={args.k} = {args.probs*args.k} 轨迹"
   f"   ★ 每 rank {PER} 题 × {args.k} = {PER*args.k} 条")
p0(f"  ★ 采样半径 ≈ 0.7/{args.k} = {0.7/args.k:.3f}   lr {args.lr:.1e}"
   f"   KL β {args.beta}   max_new {args.max_new}   micro {args.micro}")
p0("  ★★ 各 rank 梯度 all_reduce 后相同 → 权重天然一致 → 采样零同步开销")
p0("=" * 100, flush=True)

if not args.smoke and step0 == 0:
    p0(f"训练前评估（{len(EVAL)} 题，贪心）…", flush=True)
    e0 = evaluate()
    p0(f"  ★ 起点 总分 {e0['score']:.4f} | F{e0['fmt']:.3f} P{e0['parse']:.3f}"
       f" N{e0['nums_frac']:.3f} ★C{e0['correct']:.4f} | 长{e0['length']:.0f}"
       f" | ★纠错{e0['reflect']:.3f}/{e0['reflect_n']:.2f}次", flush=True)
    HIST.append(dict(step=-1, ev=e0))

t_start = time.time()
torch.cuda.reset_peak_memory_stats()
STOP = torch.zeros(1, device=DEV)
for step in range(step0, args.steps):
    lr = set_lr(step)
    opt.zero_grad(set_to_none=True)

    # ★ 所有 rank 用同一个 RNG 抽同一批题，再各取自己那份 → 无需通信
    rng = random.Random(args.seed * 1000003 + step)
    allp = rng.sample(TRAIN, args.probs)
    myp = allp[RANK * PER: (RANK + 1) * PER]

    ts = time.time()
    roll = rollout(myp, args.k, args.temp, args.max_new)
    t_gen = time.time() - ts

    # ── 优势（★ 组完整落在本 rank 上，本地就能算） ──
    R = np.zeros((PER, args.k), dtype=np.float32); cnt = [0] * PER; order = []
    for pi, _ids, _txt, r, _d, _rf in roll:
        R[pi, cnt[pi]] = r; order.append((pi, cnt[pi])); cnt[pi] += 1
    adv_m = R - R.mean(1, keepdims=True)
    adv = [float(adv_m[p, j]) for p, j in order]

    pc = [tok.encode(CD.make_prompt(p["nums"], p["target"])) for p in myp]
    recs = [(pc[pi] + ids, len(pc[pi]), a)
            for (pi, ids, *_), a in zip(roll, adv) if len(ids) > 0]
    my_tok = sum(len(f) - n for f, n, _ in recs)

    # ★ 全局 token 总数（loss 要按全局归一，否则各 rank 权重不等）
    tt_ = torch.tensor([float(my_tok)], device=DEV)
    if WORLD > 1:
        dist.all_reduce(tt_)
    tot_tok = float(tt_.item())
    if not recs or tot_tok == 0:
        p0(f"  {step:>4} ⚠️ 无有效轨迹，跳过"); continue

    # ── 微批 + 梯度累积（★ 纯本地累积，最后统一 all_reduce）──
    tt = time.time(); pg_s, kl_s = 0.0, 0.0
    for s in range(0, len(recs), args.micro):
        mb = recs[s: s + args.micro]
        ids, attn, rm = pad_batch([(f, n) for f, n, _ in mb])
        A = torch.tensor([a for *_, a in mb], device=DEV).unsqueeze(1)
        rlp = None
        if ref is not None:                      # ★ 先算参考模型，logits 立刻释放
            with torch.no_grad():
                rlp, _ = token_logp(ref, ids, attn, rm)
        lp, m_ = token_logp(model, ids, attn, rm)
        pg = -(A * lp * m_).sum() / tot_tok      # ★ 按【全局】token 数归一
        kl_t = torch.zeros((), device=DEV)
        if rlp is not None:
            lr_ = rlp - lp
            kl_t = ((torch.exp(lr_) - lr_ - 1) * m_).sum() / tot_tok
        (pg + args.beta * kl_t).backward()
        pg_s += pg.item(); kl_s += kl_t.item()
    t_comm = allreduce_grads()                   # ★★ 所有微批做完，一次性同步梯度
    gn = torch.nn.utils.clip_grad_norm_(raw.parameters(), args.clip_grad)
    opt.step()
    t_tr = time.time() - tt
    peak = torch.cuda.max_memory_allocated() / 1024**3
    torch.cuda.reset_peak_memory_stats()

    # ── 日志（各 rank 指标求平均） ──
    rs_y = [x[3] for x in roll if x[5] > 0]; rs_n = [x[3] for x in roll if x[5] == 0]
    loc = dict(score=float(R.mean()), sd=float(R.std(1).mean()),
               **{k: float(np.mean([x[4][k] for x in roll]))
                  for k in ("fmt", "parse", "nums_frac", "correct")},
               length=float(np.mean([len(x[1]) for x in roll])),
               reflect=float(np.mean([x[5] > 0 for x in roll])),
               reflect_n=float(np.mean([x[5] for x in roll])),
               reflect_gap=(float(np.mean(rs_y)) - float(np.mean(rs_n)))
                           if (rs_y and rs_n) else float("nan"),
               kl=kl_s, gn=float(gn))
    g = allmean(loc)
    if M0:
        rec = dict(step=step, lr=lr, t_gen=t_gen, t_train=t_tr,
                   t_comm=t_comm, peak=peak, **g)
        HIST.append(rec)
        el = time.time() - t_start; done = step - step0 + 1
        print(f"  {step:>4}/{args.steps} | 分 {g['score']:.3f} sd{g['sd']:.3f} |"
              f" F{g['fmt']:.2f} P{g['parse']:.2f} N{g['nums_frac']:.2f} ★C{g['correct']:.3f} |"
              f" 长{g['length']:>4.0f} | ★纠错{g['reflect']:.2f}/{g['reflect_n']:.1f}次"
              f" ★差{g['reflect_gap']:+.3f} | KL{g['kl']:.4f} |g|{g['gn']:.2f} |"
              f" {peak:.0f}GB | {t_gen:.0f}+{t_tr:.0f}s(通信{t_comm:.1f})"
              f" | 剩{el/done*(args.steps-step-1)/60:>4.0f}分", flush=True)

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
            print(f"    ── ★ eval 总分 {ev['score']:.4f} | F{ev['fmt']:.3f} P{ev['parse']:.3f}"
                  f" N{ev['nums_frac']:.3f} ★C{ev['correct']:.4f} | 长{ev['length']:.0f}"
                  f" | ★纠错{ev['reflect']:.3f} | " + tag, flush=True)
            if best:
                save("ckpt_best.pt", step + 1, with_opt=False, ev=ev)
            if args.patience and no_improve >= args.patience:
                print(f"    ★ 连续 {no_improve} 次没新高 → 早停"); STOP[0] = 1
            json.dump(HIST, open(os.path.join(args.out, "hist.json"), "w"), indent=1)

    if (step + 1) % args.save_every == 0 or step == args.steps - 1:
        save("ckpt_latest.pt", step + 1)
        if M0:
            json.dump(HIST, open(os.path.join(args.out, "hist.json"), "w"), indent=1)

    if WORLD > 1:                               # ★ 早停要所有 rank 一起退，否则死锁
        dist.broadcast(STOP, src=0)
    if STOP[0] > 0:
        save("ckpt_latest.pt", step + 1)
        p0("已存盘退出。--resume 可继续。"); break

if M0:
    json.dump(HIST, open(os.path.join(args.out, "hist.json"), "w"), indent=1)
    E = [h["ev"] for h in HIST if "ev" in h]
    TR = [h for h in HIST if "reflect_n" in h]
    print("=" * 100)
    print(f"  完成   {(time.time()-t_start)/60:.1f} 分钟")
    if len(E) >= 2 and TR:
        a, b = E[0], E[-1]
        f = lambda key, sl: float(np.mean([h[key] for h in sl]))
        print(f"""
  ── ★ H1 长度 ──
       eval（贪心）   {a['length']:.0f} → {b['length']:.0f}   ({b['length']/max(a['length'],1):.2f}×)
     ★ 训练（温度1）  {f('length', TR[:20]):.0f} → {f('length', TR[-20:]):.0f}
  ── ★★ H2 纠错措辞（★ 取训练时的数，贪心几乎不出现）──
       宽松（有没有）  {f('reflect', TR[:20]):.3f} → {f('reflect', TR[-20:]):.3f}
     ★ 严格（几次/条） {f('reflect_n', TR[:20]):.2f} → {f('reflect_n', TR[-20:]):.2f}   ← >1.0 才算真在搜索
     ★★ 因果诊断      {f('reflect_gap', TR[:20]):+.3f} → {f('reflect_gap', TR[-20:]):+.3f}
        （有纠错措辞的轨迹 − 没有的，★ 正数才说明它跟奖励挂钩）
  ── ★ H4 四级阶梯 ──
       格式  {a['fmt']:.3f} → {b['fmt']:.3f}
       解析  {a['parse']:.3f} → {b['parse']:.3f}
       数字  {a['nums_frac']:.3f} → {b['nums_frac']:.3f}
     ★ 正确  {a['correct']:.4f} → {b['correct']:.4f}
  ★★ 峰值 {BEST['score']:.4f} @ step {BEST['step']}  →  {args.out}/ckpt_best.pt""")
    print("=" * 100)

if WORLD > 1:
    dist.barrier(); dist.destroy_process_group()
