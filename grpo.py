#!/usr/bin/env python3
"""
GRPO on GSM8K —— 全参微调 + 8-bit Adam。

    python3 grpo.py --smoke              冒烟测试，3 步，约 3 分钟
    python3 grpo.py                      正式训练 300 步，约 4~5 小时
    python3 grpo.py --resume             断点续训

一步做七件事（跟讲的顺序一一对应）：
  ① 抽 8 道题
  ② 每题采样 8 条          ← 占 80% 的时间
  ③ 程序判分（_reward.py） ← 全部的『环境』
  ④ 优势 = 得分 − 组内平均  ← ★ GRPO 唯一的算法创新
  ⑤ loss = −A × logP       ← 微批 8 条，梯度累积 8 次（躲开 logits 的 5.8 GB）
  ⑥ 加 KL 惩罚（防跑飞）
  ⑦ backward + 更新

★ 没有 clip / 重要性采样比：每批 rollout 只更新一次，采样时和算 loss 时
  是同一份权重，ratio 恒等于 1，clip 是空操作。只有一批 rollout 要更新
  多次（PPO 的多 epoch）时才需要它。
"""
import os, re, json, time, math, random, signal, argparse
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from _reward import reward, extract          # ★ 判分器唯一定义处

ap = argparse.ArgumentParser()
ap.add_argument("--model",    default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--out",      default="out_grpo")
ap.add_argument("--steps",    type=int,   default=300)
ap.add_argument("--probs",    type=int,   default=8,    help="每步几道题")
ap.add_argument("-k", "--k",  type=int,   default=8,    help="每题采样几次（组大小）")
ap.add_argument("--micro",    type=int,   default=4,
                help="★ 算 loss 时一次几条。logits 会有个 fp32 副本："
                     "4条×450位置×151936×4字节≈1.1GB，8 条就翻倍。OOM 就调到 2")
ap.add_argument("--gen-bs",   type=int,   default=32,   help="采样时一次几条")
ap.add_argument("--max-new",  type=int,   default=1024)
ap.add_argument("--temp",     type=float, default=1.0,  help="★ 训练温度，必须高，要多样性")
ap.add_argument("--lr",       type=float, default=1e-6)
ap.add_argument("--beta",     type=float, default=0.001, help="KL 系数；0 = 不要参考模型")
ap.add_argument("--warmup",   type=int,   default=10)
ap.add_argument("--clip-grad", type=float, default=1.0)
ap.add_argument("--eval-every", type=int, default=25,   help="每多少步在 test 上评一次")
ap.add_argument("--eval-temp", type=float, default=0.0,
                help="★ 评估温度。0=贪心，完全确定，消除采样噪声，"
                     "步与步之间可以干净比较。注意它跟基线的 1.0 不可比")
ap.add_argument("--eval-n",   type=int,   default=200,
                help="★ 评估题数。60 题的标准误 ±6.4%%，看不出 5%% 的提升；"
                     "200 题降到 ±3.5%%。而且每次评的是同样的前 N 道，"
                     "跨步比较是配对的，噪声比这个还小")
ap.add_argument("--save-every", type=int, default=25)
ap.add_argument("--patience", type=int, default=3,
                help="★ 早停：连续几次评估没创新高就停。0 = 关闭")
ap.add_argument("--resume",   action="store_true")
ap.add_argument("--smoke",    action="store_true")
ap.add_argument("--seed",     type=int,   default=1337)
args = ap.parse_args()

if args.smoke:
    args.steps, args.probs, args.k = 3, 2, 4
    args.max_new, args.eval_every, args.eval_n, args.save_every = 256, 2, 8, 999

dev = "cuda" if torch.cuda.is_available() else "cpu"
assert dev == "cuda", "需要 GPU"
torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
os.makedirs(args.out, exist_ok=True)

INSTR = "\n\nSolve this step by step. End your answer with '#### <number>'."


# ══════════════════════════════════════════════════════════════════
#  数据
# ══════════════════════════════════════════════════════════════════
from datasets import load_dataset


def prep(split):
    d = load_dataset("openai/gsm8k", "main", split=split)
    return [(x["question"], x["answer"].split("####")[-1].strip().replace(",", "")) for x in d]


TRAIN, TEST = prep("train"), prep("test")     # ★ 训练只碰 train，评估只碰 test
print(f"GSM8K   train {len(TRAIN):,}   test {len(TEST):,}")


# ══════════════════════════════════════════════════════════════════
#  模型
# ══════════════════════════════════════════════════════════════════
def load(path, dtype=torch.bfloat16):
    try:
        m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
    return m.to(dev)


print(f"载入策略模型 {args.model} …", flush=True)
tok = AutoTokenizer.from_pretrained(args.model)
tok.padding_side = "left"                      # 生成要左填充
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
PAD, EOS = tok.pad_token_id, tok.eos_token_id

model = load(args.model)
model.gradient_checkpointing_enable()          # ★ 省激活值显存
model.config.use_cache = False                 # 和梯度检查点冲突，训练时必须关

ref = None
if args.beta > 0:
    print("载入参考模型（冻结，算 KL 用）…", flush=True)
    ref = load(args.model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

n_par = sum(p.numel() for p in model.parameters())

# ── 8-bit Adam ──
try:
    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                              weight_decay=0.0, eps=1e-8)
    OPT = "AdamW8bit"
except ImportError:
    print("⚠️ 没装 bitsandbytes，退回普通 AdamW（显存会多 9 GB，很可能 OOM）")
    print("   pip install bitsandbytes --no-deps")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    OPT = "AdamW(fp32)"


def set_lr(step):
    lr = args.lr * min(1.0, (step + 1) / max(args.warmup, 1))     # 只 warmup，不衰减
    for g in opt.param_groups:
        g["lr"] = lr
    return lr


# ══════════════════════════════════════════════════════════════════
#  ② 采样
# ══════════════════════════════════════════════════════════════════
def prompt_of(q):
    return tok.apply_chat_template([{"role": "user", "content": q + INSTR}],
                                   tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def rollout(problems, k, temp, max_new):
    """
    → [(prompt_ids, resp_ids, reward), …]，长度 = len(problems) * k
    同一道题的 k 条连续排列，方便后面按组算平均。
    """
    model.eval(); model.config.use_cache = True
    flat = [(i, prompt_of(q)) for i, (q, _) in enumerate(problems) for _ in range(k)]
    out = []
    for s in range(0, len(flat), args.gen_bs):
        chunk = flat[s: s + args.gen_bs]
        enc = tok([p for _, p in chunk], return_tensors="pt", padding=True).to(dev)
        gen = model.generate(**enc, max_new_tokens=max_new,
                             do_sample=temp > 0,                 # ★ temp=0 → 贪心
                             temperature=max(temp, 1e-5), top_p=1.0,
                             pad_token_id=PAD)
        new = gen[:, enc.input_ids.size(1):]
        for (pi, _), row in zip(chunk, new):
            resp = row.tolist()
            if EOS in resp:
                resp = resp[: resp.index(EOS) + 1]      # ★ 保留 EOS —— 模型要学会"答完就停"
            else:
                while resp and resp[-1] == PAD:         # 没停下来的，剥掉右侧 padding
                    resp.pop()
            txt = tok.decode(resp, skip_special_tokens=True)
            out.append((pi, resp, reward(txt, problems[pi][1]), txt))
    model.config.use_cache = False; model.train()
    return out


# ══════════════════════════════════════════════════════════════════
#  ⑤ 每个 token 的 log P
# ══════════════════════════════════════════════════════════════════
def token_logp(m, ids, attn, tgt_mask):
    """
    → (每个位置的 logP, 对应的 mask)   两者形状都是 (B, T-1)

    用 cross_entropy(reduction='none') 而不是 log_softmax + gather：
    ★ 前者是融合 kernel，不用把 (B,T,151936) 的 log_softmax 结果materialize 出来
    """
    logits = m(input_ids=ids, attention_mask=attn).logits[:, :-1]      # (B,T-1,V)
    tgt = ids[:, 1:]
    lp = -F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                          tgt.reshape(-1), reduction="none").view_as(tgt)
    return lp, tgt_mask[:, 1:]


def pad_batch(items):
    """items: [(full_ids, n_prompt)] → (ids, attn, resp_mask)  右填充"""
    L = max(len(x[0]) for x in items)
    ids = torch.full((len(items), L), PAD, dtype=torch.long)
    attn = torch.zeros((len(items), L), dtype=torch.long)
    rm = torch.zeros((len(items), L), dtype=torch.float)
    for i, (full, np_) in enumerate(items):
        n = len(full)
        ids[i, :n] = torch.tensor(full)
        attn[i, :n] = 1
        rm[i, np_:n] = 1.0                    # ★ 只有回答部分算 loss
    return ids.to(dev), attn.to(dev), rm.to(dev)


# ══════════════════════════════════════════════════════════════════
#  评估
# ══════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(n):
    probs = TEST[:n]
    r = rollout(probs, 1, args.eval_temp, args.max_new)
    acc = sum(x[2] for x in r) / len(r)
    ln = np.mean([len(x[1]) for x in r])
    return acc, ln


# ══════════════════════════════════════════════════════════════════
#  训练循环
# ══════════════════════════════════════════════════════════════════
step0, HIST = 0, []
BEST = {"acc": -1.0, "step": -1}          # ★ 峰值 checkpoint 的记录
no_improve = 0
ck_latest = os.path.join(args.out, "ckpt_latest.pt")
if args.resume and os.path.exists(ck_latest):
    ck = torch.load(ck_latest, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
    step0, HIST = ck["step"], ck.get("hist", [])
    prev = [h for h in HIST if "test_acc" in h]
    if prev:
        b = max(prev, key=lambda h: h["test_acc"])
        BEST = {"acc": b["test_acc"], "step": b["step"] + 1}
    print(f"★ 从 step {step0} 续训（历史峰值 {BEST['acc']:.4f} @ step {BEST['step']}）")

STOP = False
def _sig(*_):
    global STOP
    STOP = True
    print("\n⚠️ 收到 Ctrl-C，本步跑完后存盘退出…", flush=True)
signal.signal(signal.SIGINT, _sig)


def save(name, step, with_opt=True, acc=None):
    path = os.path.join(args.out, name); tmp = path + ".tmp"
    d = {"model": model.state_dict(), "step": step, "hist": HIST, "args": vars(args)}
    if with_opt:
        d["opt"] = opt.state_dict()            # 续训要，但体积翻倍
    if acc is not None:
        d["test_acc"] = acc
    torch.save(d, tmp)
    os.replace(tmp, path)                      # ★ 原子写


print("=" * 78)
print(f"  GRPO   {args.model}")
print(f"  {n_par/1e9:.2f}B 参数   全参微调   {OPT}   梯度检查点 ✅")
print(f"  每步 {args.probs} 题 × {args.k} 条 = {args.probs*args.k} 轨迹"
      f"   微批 {args.micro}（梯度累积 {args.probs*args.k//args.micro} 次）")
print(f"  lr {args.lr:.1e}   KL β {args.beta}   T {args.temp}   max_new {args.max_new}")
print(f"  {args.steps} 步   {'★ 冒烟测试' if args.smoke else ''}")
print("=" * 78, flush=True)

def gb(x=None):
    return (x if x is not None else torch.cuda.max_memory_allocated()) / 1024**3


# ★ 训练前先评一次，这才是同温度下真正的起点
BASE_ACC = None
if not args.smoke and step0 == 0:
    print(f"训练前评估（{args.eval_n} 题，温度 {args.eval_temp}）…", flush=True)
    BASE_ACC, bl = evaluate(args.eval_n)
    print(f"  ★ 起点 test 正确率 {BASE_ACC:.4f}   平均长度 {bl:.0f} token", flush=True)
    HIST.append(dict(step=-1, test_acc=BASE_ACC, test_len=bl))

t_start = time.time()
torch.cuda.reset_peak_memory_stats()
for step in range(step0, args.steps):
    t0 = time.time()
    lr = set_lr(step)
    opt.zero_grad(set_to_none=True)            # ★ 采样前释放梯度显存

    # ── ① 抽题 ──
    probs = random.sample(TRAIN, args.probs)

    # ── ② 采样 ──
    ts = time.time()
    roll = rollout(probs, args.k, args.temp, args.max_new)
    t_gen = time.time() - ts

    # ── ③④ 判分 + 优势 ──
    rs = np.array([x[2] for x in roll], dtype=np.float32).reshape(args.probs, args.k)
    # ★ GRPO 的全部算法：减组内平均。
    #   原论文还除以组内标准差，但 std 很小时会放大噪声
    #   （8 条里只对 1 条时 std≈0.33，优势被放大 3 倍），
    #   后续工作（DAPO 等）建议去掉。这里不除。
    adv = rs - rs.mean(1, keepdims=True)
    adv = adv.reshape(-1)
    n_useful = int(((rs.sum(1) > 0) & (rs.sum(1) < args.k)).sum())

    # 组装成训练样本
    pcache = [tok.encode(prompt_of(q)) for q, _ in probs]   # 每题只编码一次
    recs = []
    for (pi, resp, r, txt), a in zip(roll, adv):
        if len(resp) == 0:
            continue
        recs.append((pcache[pi] + resp, len(pcache[pi]), float(a)))
    tot_tok = sum(len(f) - np_ for f, np_, _ in recs)       # ★ 全步的回答 token 总数
    if not recs or tot_tok == 0:
        print(f"  {step:>4} ⚠️ 本步没有有效轨迹，跳过"); continue

    # ── ⑤⑥⑦ 微批 + 梯度累积 ──
    tt = time.time()
    pg_sum, kl_sum, lp_sum = 0.0, 0.0, 0.0
    for s in range(0, len(recs), args.micro):
        mb = recs[s: s + args.micro]
        ids, attn, rm = pad_batch([(f, n) for f, n, _ in mb])
        A = torch.tensor([a for *_, a in mb], device=dev).unsqueeze(1)   # (B,1)

        # ★ 顺序很重要：先算参考模型（no_grad，logits 立刻释放），
        #   再算策略模型（logits 要留给 backward）。反过来会两份 logits 同时占显存。
        rlp = None
        if ref is not None:
            with torch.no_grad():
                rlp, _ = token_logp(ref, ids, attn, rm)

        lp, m = token_logp(model, ids, attn, rm)
        pg = -(A * lp * m).sum() / tot_tok                  # ★ loss = −A × logP

        kl_t = torch.zeros((), device=dev)
        if rlp is not None:
            lr_ratio = rlp - lp                              # log(ref/pol)
            kl = torch.exp(lr_ratio) - lr_ratio - 1          # ★ k3 估计量，恒 ≥0
            kl_t = (kl * m).sum() / tot_tok

        (pg + args.beta * kl_t).backward()
        pg_sum += pg.item(); kl_sum += kl_t.item()
        lp_sum += (lp.detach() * m).sum().item()

    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    opt.step()
    t_train = time.time() - tt
    peak = gb()                                   # ★ 本步显存峰值
    torch.cuda.reset_peak_memory_stats()

    # ── 日志 ──
    acc = float(rs.mean())
    ln = float(np.mean([len(x[1]) for x in roll]))
    maxlen = max(len(x[1]) for x in roll)
    rec = dict(step=step, acc=acc, useful=n_useful / args.probs, len=ln, maxlen=maxlen,
               kl=kl_sum, pg=pg_sum, logp=lp_sum / max(tot_tok, 1),
               gn=float(gn), lr=lr, t_gen=t_gen, t_train=t_train, peak_gb=peak)
    HIST.append(rec)
    el = time.time() - t_start
    done = step - step0 + 1
    print(f"  {step:>4}/{args.steps} | 正确率 {acc:.3f} | 有信号 {n_useful}/{args.probs}"
          f" | 长度 {ln:>5.0f}/{maxlen:<5} | KL {kl_sum:.4f} | logP {rec['logp']:>7.3f}"
          f" | |g| {gn:>5.2f} | ★{peak:>4.1f}GB | {t_gen:>3.0f}+{t_train:>2.0f}s"
          f" | 剩 {el/done*(args.steps-step-1)/60:>5.1f}分", flush=True)
    if peak > 21.5:
        print(f"    ⚠️ 显存 {peak:.1f} GB 逼近 24 GB —— 建议 --micro {max(args.micro//2,1)}",
              flush=True)

    if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
        a, l = evaluate(args.eval_n)
        HIST[-1]["test_acc"] = a; HIST[-1]["test_len"] = l

        if a > BEST["acc"]:
            BEST = {"acc": a, "step": step + 1}
            no_improve = 0
            save("ckpt_best.pt", step + 1, with_opt=False, acc=a)   # ★ 只存模型
            tag = "★ 新高，已存 ckpt_best.pt"
        else:
            no_improve += 1
            tag = (f"（峰值 {BEST['acc']:.4f} @ step {BEST['step']}，"
                   f"已 {no_improve} 次没创新高）")
        print(f"    ── ★ test 正确率 {a:.4f}   平均长度 {l:.0f} token   {tag}", flush=True)

        # ★ 熵坍缩 / 多样性预警
        if len(HIST) > 40:
            lp0 = np.mean([h["logp"] for h in HIST[:20] if "logp" in h])
            lp1 = np.mean([h["logp"] for h in HIST[-20:] if "logp" in h])
            u1 = np.mean([h["useful"] for h in HIST[-20:] if "useful" in h])
            if lp1 - lp0 > 0.40:
                print(f"    ⚠️ ★ logP {lp0:.3f}→{lp1:.3f}（{np.exp(lp1):.0%} 把握）"
                      f"，有信号率 {u1:.2f} —— 熵坍缩，多样性在收窄", flush=True)

        if args.patience and no_improve >= args.patience:
            print(f"    ★ 连续 {no_improve} 次评估没创新高 → 早停。"
                  f"最佳权重在 {args.out}/ckpt_best.pt（step {BEST['step']}，"
                  f"{BEST['acc']:.4f}）", flush=True)
            STOP = True

        json.dump(HIST, open(os.path.join(args.out, "hist.json"), "w"), indent=1)

    if (step + 1) % args.save_every == 0 or step == args.steps - 1 or STOP:
        save("ckpt_latest.pt", step + 1)
        json.dump(HIST, open(os.path.join(args.out, "hist.json"), "w"), indent=1)
    if STOP:
        print("已存盘，退出。--resume 可继续。"); break

json.dump(HIST, open(os.path.join(args.out, "hist.json"), "w"), indent=1)
print("=" * 78)
print(f"  完成   用时 {(time.time()-t_start)/60:.1f} 分钟")
if HIST:
    fa = [h for h in HIST if "test_acc" in h]
    if fa:
        d = fa[-1]["test_acc"] - fa[0]["test_acc"]
        print(f"  test 正确率（温度 {args.eval_temp}）  "
              f"{fa[0]['test_acc']:.4f} → {fa[-1]['test_acc']:.4f}   {d:+.4f}")
        print(f"  ★★ 峰值 {BEST['acc']:.4f} 在 step {BEST['step']}"
              f"（比起点 {BEST['acc']-fa[0]['test_acc']:+.4f}）"
              f"  →  {args.out}/ckpt_best.pt")
        print(f"  ★ 注意：基线脚本的 0.456 是温度 1.0 测的，跟这个数不可比")
        print(f"    pass@8 = 0.855（温度 1.0）是能力天花板，RL 动不了它")
print(f"  最终权重 → {args.out}/ckpt_latest.pt   ★ 最佳权重 → {args.out}/ckpt_best.pt")
print("=" * 78)
