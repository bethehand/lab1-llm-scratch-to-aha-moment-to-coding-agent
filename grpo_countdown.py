#!/usr/bin/env python3
"""
R1-Zero 复现：Base 模型 + Countdown + GRPO，★ 4 卡并行采样。

    python3 grpo_countdown.py --smoke        冒烟测试，3 步
    python3 grpo_countdown.py                正式训练 300 步
    python3 grpo_countdown.py --resume

★ 跟 grpo.py（GSM8K 那次）的四点不同：
  ① Base 模型，不是 Instruct —— 没做过任何后训练
  ② ★ 四级阶梯奖励（格式/解析/数字比例/正确），不是 0/1
  ③ ★★ 4 卡并行采样：4 个模型副本 + 线程（GPU 算子释放 GIL → 真并行）
  ④ ★★ H2 实时诊断：「有纠错措辞的轨迹」vs「没有的」平均奖励差
     → ★ 前 100 步就能判断 1.5B 值不值得继续训
"""
import os, re, json, time, random, signal, argparse, copy
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import countdown as CD

ap = argparse.ArgumentParser()
ap.add_argument("--model",   default="Qwen/Qwen2.5-1.5B", help="★ Base 版")
ap.add_argument("--data",    default="data/countdown.json")
ap.add_argument("--out",     default="out_cd")
ap.add_argument("--steps",   type=int,   default=300)
ap.add_argument("--probs",   type=int,   default=16,  help="每步几道题")
ap.add_argument("-k", "--k", type=int,   default=16,  help="★ 组大小。采样半径≈0.7/k")
ap.add_argument("--micro",   type=int,   default=2,   help="★ 算 loss 时一次几条（logits 显存）")
ap.add_argument("--gen-gpus", type=int,  default=4,   help="★ 采样用几张卡")
ap.add_argument("--gen-bs",  type=int,   default=16,  help="每卡一次并行几条")
ap.add_argument("--max-new", type=int,   default=768)
ap.add_argument("--temp",    type=float, default=1.0)
ap.add_argument("--lr",      type=float, default=5e-6)
ap.add_argument("--beta",    type=float, default=0.0005, help="KL 系数；★ 比 GSM8K 那次小，让它跑得更远")
ap.add_argument("--warmup",  type=int,   default=10)
ap.add_argument("--clip-grad", type=float, default=1.0)
ap.add_argument("--eval-every", type=int, default=25)
ap.add_argument("--eval-n",  type=int,   default=200)
ap.add_argument("--eval-offset", type=int, default=90000, help="★ 评估用训练没碰过的题")
ap.add_argument("--save-every", type=int, default=25)
ap.add_argument("--patience", type=int,  default=5, help="★ 比 GSM8K 那次放宽（200 题的噪声大）")
ap.add_argument("--resume",  action="store_true")
ap.add_argument("--smoke",   action="store_true")
ap.add_argument("--seed",    type=int,   default=1337)
args = ap.parse_args()

if args.smoke:
    args.steps, args.probs, args.k = 3, 4, 4
    args.max_new, args.eval_every, args.eval_n = 256, 2, 16
    args.save_every, args.gen_gpus = 999, min(args.gen_gpus, torch.cuda.device_count())

assert torch.cuda.is_available()
NG = min(args.gen_gpus, torch.cuda.device_count())
DEV = "cuda:0"
torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
os.makedirs(args.out, exist_ok=True)

# ★ H2 的检测器 —— 跟 baseline_countdown.py 保持一致
REFLECT = re.compile(
    r"\b(wait|hold on|hmm|actually|alternatively|instead|let me (?:try|check|recheck|reconsider)"
    r"|that (?:doesn'?t|does not) work|not (?:right|correct)|recheck|reconsider)\b", re.I)

# ══════════════════════════════════════════════════════════════
#  数据
# ══════════════════════════════════════════════════════════════
D = json.load(open(args.data))
TRAIN = D[: args.eval_offset]
EVAL = D[args.eval_offset: args.eval_offset + args.eval_n]
print(f"Countdown  train {len(TRAIN):,}   eval {len(EVAL)}（★ TEST[{args.eval_offset}:] 训练不碰）")

# ══════════════════════════════════════════════════════════════
#  模型：训练模型在 cuda:0，另外 NG-1 张卡各放一个只读副本
# ══════════════════════════════════════════════════════════════
tok = AutoTokenizer.from_pretrained(args.model)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
PAD, EOS = tok.pad_token_id, tok.eos_token_id


def load(dev):
    try:
        m = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    return m.to(dev)


print(f"载入训练模型 → cuda:0 …", flush=True)
model = load(DEV)
model.gradient_checkpointing_enable()
model.config.use_cache = False

ref = None
if args.beta > 0:
    print("载入参考模型（冻结）…", flush=True)
    ref = load(DEV).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

# ★★ 采样副本：cuda:1..NG-1。cuda:0 直接用训练模型本身，不额外占显存
print(f"★ 建 {NG-1} 个采样副本 → cuda:1..{NG-1} …", flush=True)
REPS = [model] + [load(f"cuda:{i}").eval() for i in range(1, NG)]
for r in REPS[1:]:
    for p in r.parameters():
        p.requires_grad_(False)
    r.config.use_cache = True


@torch.no_grad()
def sync_replicas():
    """★ 每步把训练模型的权重复制给采样副本 —— 保证 on-policy"""
    if NG == 1:
        return 0.0
    t0 = time.time()
    sd = model.state_dict()
    for r in REPS[1:]:
        rd = r.state_dict()
        for k, v in sd.items():
            rd[k].copy_(v, non_blocking=True)
    torch.cuda.synchronize()
    return time.time() - t0


n_par = sum(p.numel() for p in model.parameters())
try:
    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                              weight_decay=0.0, eps=1e-8)
    OPT = "AdamW8bit"
except ImportError:
    print("⚠️ 没装 bitsandbytes，退回 fp32 AdamW（可能 OOM）\n   pip install bitsandbytes --no-deps")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    OPT = "AdamW(fp32)"


def set_lr(step):
    lr = args.lr * min(1.0, (step + 1) / max(args.warmup, 1))
    for g in opt.param_groups:
        g["lr"] = lr
    return lr


# ══════════════════════════════════════════════════════════════
#  ★★ 4 卡并行采样
# ══════════════════════════════════════════════════════════════
@torch.no_grad()
def _gen_one_device(idx, prompts, temp, max_new):
    """在第 idx 张卡上生成一批。★ generate 内部释放 GIL → 多线程真并行"""
    if not prompts:
        return []
    m, dev = REPS[idx], f"cuda:{idx}"
    was = m.config.use_cache
    m.config.use_cache = True
    out = []
    for s in range(0, len(prompts), args.gen_bs):
        enc = tok(prompts[s: s + args.gen_bs], return_tensors="pt", padding=True).to(dev)
        g = m.generate(**enc, max_new_tokens=max_new, do_sample=temp > 0,
                       temperature=max(temp, 1e-5), top_p=1.0, pad_token_id=PAD)
        for row in g[:, enc.input_ids.size(1):]:
            r = row.tolist()
            if EOS in r:
                r = r[: r.index(EOS) + 1]        # ★ 保留 EOS
            else:
                while r and r[-1] == PAD:
                    r.pop()
            out.append(r)
    m.config.use_cache = was
    return out


def rollout(probs, k, temp, max_new):
    """→ [(题号, resp_ids, 文本, 总分, 明细dict, 有无纠错措辞)]"""
    model.gradient_checkpointing_disable()
    model.eval(); model.config.use_cache = True
    flat = [(i, CD.make_prompt(p["nums"], p["target"]))
            for i, p in enumerate(probs) for _ in range(k)]
    # ★ 连续切：同一题的 k 条尽量落在同一卡的同一批里
    #   → 该批 prompt 完全相同 → ★ 零 padding（stride 切法会打散，浪费大量算力）
    per = (len(flat) + NG - 1) // NG
    chunks = [flat[i * per: (i + 1) * per] for i in range(NG)]
    with ThreadPoolExecutor(NG) as ex:
        res = list(ex.map(lambda t: _gen_one_device(t[0], [p for _, p in t[1]],
                                                    temp, max_new),
                          enumerate(chunks)))
    out = []
    for ci, ids_list in enumerate(res):
        for (pi, _), ids in zip(chunks[ci], ids_list):
            txt = tok.decode(ids, skip_special_tokens=True)
            r, d = CD.reward(txt, probs[pi]["nums"], probs[pi]["target"])
            out.append((pi, ids, txt, r, d, len(REFLECT.findall(txt))))
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()
    return out


# ══════════════════════════════════════════════════════════════
#  loss
# ══════════════════════════════════════════════════════════════
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


@torch.no_grad()
def evaluate():
    """
    ★ 贪心解码（温度 0），零采样噪声
    ★★ 必须先同步副本 —— sync 在每步【开头】，evaluate 在每步【末尾】，
       不同步的话副本还是 opt.step() 之前的权重，评的是旧模型
    """
    sync_replicas()
    r = rollout(EVAL, 1, 0.0, args.max_new)
    keys = ("fmt", "parse", "nums_frac", "correct")
    return (dict(score=float(np.mean([x[3] for x in r])),
                 **{k: float(np.mean([x[4][k] for x in r])) for k in keys},
                 length=float(np.mean([len(x[1]) for x in r])),
                 reflect=float(np.mean([x[5] > 0 for x in r])),
                 reflect_n=float(np.mean([x[5] for x in r]))))


# ══════════════════════════════════════════════════════════════
#  训练循环
# ══════════════════════════════════════════════════════════════
step0, HIST = 0, []
BEST = {"score": -1.0, "step": -1}
no_improve = 0
ck_latest = os.path.join(args.out, "ckpt_latest.pt")
if args.resume and os.path.exists(ck_latest):
    ck = torch.load(ck_latest, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
    step0, HIST = ck["step"], ck.get("hist", [])
    prev = [h for h in HIST if "ev" in h]
    if prev:
        b = max(prev, key=lambda h: h["ev"]["score"])
        BEST = {"score": b["ev"]["score"], "step": b["step"] + 1}
    print(f"★ 从 step {step0} 续训（历史峰值 {BEST['score']:.4f}）")

STOP = False
def _sig(*_):
    global STOP; STOP = True
    print("\n⚠️ Ctrl-C，本步跑完存盘退出…", flush=True)
signal.signal(signal.SIGINT, _sig)


def save(name, step, with_opt=True, ev=None):
    path = os.path.join(args.out, name); tmp = path + ".tmp"
    d = {"model": model.state_dict(), "step": step, "hist": HIST, "args": vars(args)}
    if with_opt: d["opt"] = opt.state_dict()
    if ev: d["ev"] = ev
    torch.save(d, tmp); os.replace(tmp, path)


print("=" * 96)
print(f"  ★ R1-Zero 复现   {args.model}   {n_par/1e9:.2f}B   {OPT}")
print(f"  每步 {args.probs} 题 × k={args.k} = {args.probs*args.k} 轨迹"
      f"   微批 {args.micro}   ★ 采样 {NG} 卡并行")
print(f"  ★ 采样半径 ≈ 0.7/{args.k} = {0.7/args.k:.3f}"
      f"   lr {args.lr:.1e}   KL β {args.beta}   max_new {args.max_new}")
print(f"  {args.steps} 步   {'★ 冒烟' if args.smoke else ''}")
print("=" * 96, flush=True)

if not args.smoke and step0 == 0:
    print(f"训练前评估（{len(EVAL)} 题，贪心）…", flush=True)
    e0 = evaluate()
    print(f"  ★ 起点  总分 {e0['score']:.4f} | 格式 {e0['fmt']:.3f} | 解析 {e0['parse']:.3f}"
          f" | 数字 {e0['nums_frac']:.3f} | ★正确 {e0['correct']:.4f}"
          f" | 长度 {e0['length']:.0f} | ★纠错 {e0['reflect']:.3f}", flush=True)
    HIST.append(dict(step=-1, ev=e0))

t_start = time.time()
torch.cuda.reset_peak_memory_stats()
for step in range(step0, args.steps):
    lr = set_lr(step)
    opt.zero_grad(set_to_none=True)

    t_sync = sync_replicas()                        # ★ 权重同步，保证 on-policy
    probs = random.sample(TRAIN, args.probs)

    ts = time.time()
    roll = rollout(probs, args.k, args.temp, args.max_new)
    t_gen = time.time() - ts

    # ── 优势 ──
    R = np.zeros((args.probs, args.k), dtype=np.float32)
    cnt = [0] * args.probs
    order = []
    for pi, ids, txt, r, d, rf in roll:
        R[pi, cnt[pi]] = r; order.append((pi, cnt[pi])); cnt[pi] += 1
    adv_m = R - R.mean(1, keepdims=True)
    adv = [float(adv_m[p, j]) for p, j in order]

    pc = [tok.encode(CD.make_prompt(p["nums"], p["target"])) for p in probs]
    recs = [(pc[pi] + ids, len(pc[pi]), a)
            for (pi, ids, *_), a in zip(roll, adv) if len(ids) > 0]
    tot_tok = sum(len(f) - n for f, n, _ in recs)
    if not recs or tot_tok == 0:
        print(f"  {step:>4} ⚠️ 无有效轨迹，跳过"); continue

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
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    opt.step()
    t_tr = time.time() - tt
    peak = torch.cuda.max_memory_allocated() / 1024**3
    torch.cuda.reset_peak_memory_stats()

    # ── 日志 ──
    D4 = {k: float(np.mean([x[4][k] for x in roll]))
          for k in ("fmt", "parse", "nums_frac", "correct")}
    ln = float(np.mean([len(x[1]) for x in roll]))
    refl = float(np.mean([x[5] > 0 for x in roll]))
    refl_n = float(np.mean([x[5] for x in roll]))     # ★ 严格口径：平均几次/条
    # ★★ H2 诊断：有纠错措辞的轨迹 vs 没有的，平均奖励差
    rs_y = [x[3] for x in roll if x[5] > 0]
    rs_n = [x[3] for x in roll if x[5] == 0]
    gap = (float(np.mean(rs_y)) - float(np.mean(rs_n))) if rs_y and rs_n else float("nan")
    sd = float(np.mean(R.std(1)))
    rec = dict(step=step, score=float(R.mean()), sd=sd, **D4, len=ln,
               reflect=refl, reflect_n=refl_n,
               reflect_gap=gap, kl=kl_s, gn=float(gn), lr=lr,
               t_gen=t_gen, t_train=t_tr, t_sync=t_sync, peak=peak)
    HIST.append(rec)
    el = time.time() - t_start; done = step - step0 + 1
    print(f"  {step:>4}/{args.steps} | 分 {R.mean():.3f} sd{sd:.3f} |"
          f" F{D4['fmt']:.2f} P{D4['parse']:.2f} N{D4['nums_frac']:.2f} ★C{D4['correct']:.3f} |"
          f" 长{ln:>4.0f} | ★纠错{refl:.2f}/{refl_n:.1f}次 ★差{gap:+.3f} |"
          f" KL{kl_s:.4f} |g|{gn:.2f} | {peak:.0f}GB |"
          f" {t_sync:.0f}+{t_gen:.0f}+{t_tr:.0f}s | 剩{el/done*(args.steps-step-1)/60:>4.0f}分",
          flush=True)

    if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
        ev = evaluate(); HIST[-1]["ev"] = ev
        best = ev["score"] > BEST["score"]
        if best:
            BEST = {"score": ev["score"], "step": step + 1}; no_improve = 0
            save("ckpt_best.pt", step + 1, with_opt=False, ev=ev)
        else:
            no_improve += 1
        tag = ("★ 新高，存 ckpt_best" if best else
               f"峰值 {BEST['score']:.4f} @ step {BEST['step']}，{no_improve} 次没新高")
        print(f"    ── ★ eval  总分 {ev['score']:.4f} | F{ev['fmt']:.3f} P{ev['parse']:.3f}"
              f" N{ev['nums_frac']:.3f} ★C{ev['correct']:.4f} | 长{ev['length']:.0f}"
              f" | ★纠错{ev['reflect']:.3f}/{ev['reflect_n']:.2f}次 | "
              + tag, flush=True)
        if args.patience and no_improve >= args.patience:
            print(f"    ★ 连续 {no_improve} 次没新高 → 早停。最佳在 {args.out}/ckpt_best.pt"); STOP = True
        json.dump(HIST, open(os.path.join(args.out, "hist.json"), "w"), indent=1)

    if (step + 1) % args.save_every == 0 or step == args.steps - 1 or STOP:
        save("ckpt_latest.pt", step + 1)
        json.dump(HIST, open(os.path.join(args.out, "hist.json"), "w"), indent=1)
    if STOP:
        print("已存盘退出。--resume 可继续。"); break

json.dump(HIST, open(os.path.join(args.out, "hist.json"), "w"), indent=1)
E = [h["ev"] for h in HIST if "ev" in h]
print("=" * 96)
print(f"  完成   {(time.time()-t_start)/60:.1f} 分钟")
TR = [h for h in HIST if "reflect_n" in h]
r0 = float(np.mean([h["reflect"] for h in TR[:20]])) if TR else float("nan")
r1 = float(np.mean([h["reflect"] for h in TR[-20:]])) if TR else float("nan")
rn0 = float(np.mean([h["reflect_n"] for h in TR[:20]])) if TR else float("nan")
rn1 = float(np.mean([h["reflect_n"] for h in TR[-20:]])) if TR else float("nan")
if len(E) >= 2:
    a, b = E[0], E[-1]
    print(f"""
  ── ★ H1 长度 ──
       eval（贪心）   {a['length']:.0f} → {b['length']:.0f}   ({b['length']/max(a['length'],1):.2f}×)
     ★ 训练（温度1）  {float(np.mean([h['len'] for h in TR[:20]])):.0f} → {float(np.mean([h['len'] for h in TR[-20:]])):.0f}
  ── ★★ H2 纠错措辞（★ 取【训练时】的数，温度 1.0）──
     ⚠️ 不能用 eval 的数 —— 贪心解码几乎不出现纠错措辞，恒为 0
       宽松（有没有）  {r0:.3f} → {r1:.3f}
     ★ 严格（几次/条） {rn0:.2f} → {rn1:.2f}   ← ★ >1.0 才算真在搜索
  ── ★ H4 四级阶梯 ──
       格式  {a['fmt']:.3f} → {b['fmt']:.3f}
       解析  {a['parse']:.3f} → {b['parse']:.3f}
       数字  {a['nums_frac']:.3f} → {b['nums_frac']:.3f}
     ★ 正确  {a['correct']:.4f} → {b['correct']:.4f}
  ★★ 峰值 {BEST['score']:.4f} @ step {BEST['step']}  →  {args.out}/ckpt_best.pt""")
print("=" * 96)
