#!/usr/bin/env python3
"""
sft_qwen.py —— 给 Qwen 做 SFT：只对回答部分算 loss，存成 grpo_mix_mp.py --init 能直接读的 ckpt。

  臂 1（程序撒种）：
    torchrun --nproc_per_node=4 sft_qwen.py --init out_mix4/ckpt_step300.pt \
        --data sft_cd3.jsonl sft_cd4.jsonl sft_gsm.jsonl --out out_sft1
  臂 2（自蒸馏）：
    torchrun --nproc_per_node=4 sft_qwen.py --init out_mix4/ckpt_step300.pt \
        --data sft_cd3_self.jsonl sft_cd4_self.jsonl sft_gsm.jsonl --out out_sft2
  冒烟：加 --smoke（20 步，写到 <out>_smoke）

设计：
  · lr 1e-5、2 epoch、有效 batch 32、cosine + warmup。★ 只装程序不重写：lr 是 RL 的 2 倍、预训练的 1/30
  · labels：prompt 部分 −100，response 部分照算，末尾接 EOS（★ 学会停，别再幻觉 "User:"）
  · 超过 --max-len 的样本丢掉（穷举器最长 1547 token，占 <3%）
  · 4 卡：各自算梯度，手动 all_reduce 求平均（跟 grpo_mix_mp 一个路子，不用 DDP 包装）
  · val loss 只做 sanity（后训练不用 val loss 选 ckpt —— SFT 阶段的教训），看的是格式和行为
  · 训完 rank0 贪心生成两道题看格式：[2,3,5,7]→31 和一道 GSM
"""
import os, re, sys, json, math, time, random, argparse
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import countdown as CD

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
ap.add_argument("--init", default=None, help="起点 ckpt（不给 = 原始 Base）")
ap.add_argument("--data", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--batch", type=int, default=32, help="有效 batch（所有卡合计）")
ap.add_argument("--micro", type=int, default=1, help="每卡每次前向几条")
ap.add_argument("--max-len", type=int, default=1600, help="prompt+response 超过就丢")
ap.add_argument("--warmup", type=int, default=20)
ap.add_argument("--val-frac", type=float, default=0.05)
ap.add_argument("--clip-grad", type=float, default=1.0)
ap.add_argument("--log-every", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
import shutil as _sh
_free = _sh.disk_usage(".").free / 1024 ** 3
assert args.smoke or _free >= 6, f"盘只剩 {_free:.1f} GB，存 ckpt 要 3 GB。先清盘"

# ── 分布式 ──
WORLD = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))
LOCAL = int(os.environ.get("LOCAL_RANK", 0))
if WORLD > 1:
    dist.init_process_group("nccl")
    torch.cuda.set_device(LOCAL)
DEV = f"cuda:{LOCAL}"
M0 = RANK == 0
def p0(*a, **kw):
    if M0: print(*a, **kw, flush=True)

if args.smoke:
    args.out = args.out.rstrip("/") + "_smoke"
accum = max(1, args.batch // (WORLD * args.micro))
torch.manual_seed(args.seed); random.seed(args.seed)
if M0: os.makedirs(args.out, exist_ok=True)

# ── 数据 ──
tok = AutoTokenizer.from_pretrained(args.model)
EOS = tok.eos_token_id
PAD = tok.pad_token_id if tok.pad_token_id is not None else EOS

RES_RE = re.compile(r"<result>.*?</result>|<reply>.*?</reply>|<obs>.*?</obs>", re.S)   # ★ 注入段：计算器结果 + 专家回复（臂 A），都不学「自己写」

def encode(row):
    p_ids = tok(row["prompt"], add_special_tokens=False).input_ids
    resp = row["response"]
    # ★ 工具版教材：<result>…</result> 是 harness 注入的，模型不该学「自己写结果」→ 这段 label −100
    #   做法：按 <result> 段切开，分段编码再拼，段边界上 token 不会跨段
    r_ids, r_lab, pos = [], [], 0
    for m in RES_RE.finditer(resp):
        seg = tok(resp[pos: m.start()], add_special_tokens=False).input_ids
        res = tok(m.group(0), add_special_tokens=False).input_ids
        r_ids += seg + res; r_lab += seg + [-100] * len(res); pos = m.end()
    seg = tok(resp[pos:], add_special_tokens=False).input_ids
    r_ids += seg + [EOS]; r_lab += seg + [EOS]
    ids = p_ids + r_ids
    if len(ids) > args.max_len:
        return None
    return dict(ids=ids, labels=[-100] * len(p_ids) + r_lab, src=row.get("task", "?"))

rows, per_file = [], {}
for f in args.data:
    n = 0
    for line in open(f):
        r = json.loads(line)
        r.setdefault("task", "cd" if "nums" in r else "gsm")
        rows.append(r); n += 1
    per_file[f] = n
rng = random.Random(args.seed); rng.shuffle(rows)
enc_all = [encode(r) for r in rows]
dropped = sum(1 for e in enc_all if e is None)
enc_all = [e for e in enc_all if e is not None]
n_val = max(WORLD, int(len(enc_all) * args.val_frac)) if len(enc_all) > 50 else 0
val, train = enc_all[:n_val], enc_all[n_val:]
if args.smoke:
    train = train[: 20 * args.batch]
p0(f"数据  " + "  ".join(f"{os.path.basename(f)} {n}" for f, n in per_file.items()))
p0(f"  超 {args.max_len} token 丢 {dropped}   train {len(train)}   val {len(val)}   "
   f"有效 batch {args.batch} = {WORLD} 卡 × micro {args.micro} × 累积 {accum}")
lens = sorted(len(e["ids"]) for e in train)
p0(f"  序列长度  中位 {lens[len(lens)//2]}  P90 {lens[int(.9*(len(lens)-1))]}  最长 {lens[-1]}"
   f"   response token 总数 {sum(sum(1 for l in e['labels'] if l != -100) for e in train):,}")

def collate(items):
    L = max(len(e["ids"]) for e in items)
    ids = torch.full((len(items), L), PAD, dtype=torch.long)
    lab = torch.full((len(items), L), -100, dtype=torch.long)
    att = torch.zeros((len(items), L), dtype=torch.long)
    for i, e in enumerate(items):
        n = len(e["ids"])
        ids[i, :n] = torch.tensor(e["ids"]); lab[i, :n] = torch.tensor(e["labels"]); att[i, :n] = 1
    return ids.to(DEV), lab.to(DEV), att.to(DEV)

# ── 模型 ──
p0(f"载入 {args.model} …")
try:
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
if args.init:
    ck = torch.load(args.init, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"]); del ck
    p0(f"  ★ 起点 {args.init}")
model.to(DEV)
model.gradient_checkpointing_enable()
model.config.use_cache = False
model.train()

try:
    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, eps=1e-8)
    OPT = "AdamW8bit"
except ImportError:
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    OPT = "AdamW(fp32)"

# ★ 末批不足 batch 时丢掉（drop_last）：之前用 ceil，末批只剩 1 条 → 只有 rank 0 分到样本，其他卡没做 backward、梯度全是 None、
#   allreduce_grads 一个集合通信都不发就去做 epoch 末的 val_loss（标量 all_reduce），rank 0 却卡在词嵌入梯度的 all_reduce 上 → NCCL 600 s 超时（2026-09-10 臂 A SFT 第 225 步）
steps_per_epoch = len(train) // args.batch
assert steps_per_epoch >= 1 and args.batch % WORLD == 0, f"batch {args.batch} 要能被卡数 {WORLD} 整除"
total = steps_per_epoch * args.epochs
def lr_at(step):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    t = (step - args.warmup) / max(1, total - args.warmup)
    return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

def allreduce_grads():
    if WORLD == 1: return
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM); p.grad /= WORLD

def loss_on(items):
    ids, lab, att = collate(items)
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
    tgt = lab[:, 1:]
    l = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100, reduction="sum")
    n = (tgt != -100).sum()
    return l, n

@torch.no_grad()
def val_loss():
    if not val: return float("nan")
    model.eval()
    mine = val[RANK::WORLD]
    ls, ns = torch.zeros((), device=DEV), torch.zeros((), device=DEV)
    for i in range(0, len(mine), args.micro):
        l, n = loss_on(mine[i: i + args.micro]); ls += l; ns += n
    if WORLD > 1:
        dist.all_reduce(ls); dist.all_reduce(ns)
    model.train()
    return (ls / ns.clamp(min=1)).item()

p0(f"\nSFT  {OPT}  lr {args.lr:.1e}  epoch {args.epochs}  每 epoch {steps_per_epoch} 步  共 {total} 步"
   f"   {sum(p.numel() for p in model.parameters())/1e9:.2f}B\n" + "=" * 74)
v0 = val_loss(); p0(f"  训前 val loss {v0:.4f}")
step, t0, tok_seen = 0, time.time(), 0
for ep in range(args.epochs):
    order = list(range(len(train))); random.Random(args.seed + ep).shuffle(order)
    for b in range(steps_per_epoch):
        chunk = order[b * args.batch: (b + 1) * args.batch]
        mine = chunk[RANK::WORLD]                       # 每卡分到 batch/WORLD 条
        assert mine, f"rank {RANK} 第 {step} 步分不到样本，四卡集合通信会错位"      # drop_last 后不该发生，真发生就在这儿停，别等 NCCL 超时
        opt.zero_grad(set_to_none=True)
        ls, ns = 0.0, 0
        n_tok = max(1, sum(sum(1 for t in train[j]["labels"] if t != -100) for j in mine))
        for i in range(0, len(mine), args.micro):
            items = [train[j] for j in mine[i: i + args.micro]]
            l, n = loss_on(items)
            (l / n_tok).backward()                      # ★ 按本卡这批的 token 数平均 → 每 token 的梯度，|g| 是 O(1)
            ls += l.item(); ns += n.item()
        allreduce_grads()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad).item()
        lr = lr_at(step)
        for g in opt.param_groups: g["lr"] = lr
        opt.step(); step += 1; tok_seen += ns * WORLD
        if step % args.log_every == 0 or step == total:
            el = time.time() - t0
            p0(f"  ep {ep+1} step {step:4d}/{total}  loss/token {ls/max(1,ns):.4f}  lr {lr:.2e}  |g| {gn:.2f}"
               f"  {tok_seen/el:,.0f} tok/s  剩 {(total-step)*el/step/60:.1f} 分  GB {torch.cuda.max_memory_allocated(DEV)/1024**3:.1f}")
    p0(f"  ── epoch {ep+1} 完  val loss {val_loss():.4f}（训前 {v0:.4f}；只看趋势，不用它选）──")

# ── 存 ──
v1 = val_loss()                                   # ★ 所有 rank 一起算（里面有 all_reduce），不能放进 rank0 的块里
if M0:
    meta = dict(data=args.data, epochs=args.epochs, lr=args.lr, batch=args.batch, n_train=len(train),
                init=args.init, val_before=v0, val_after=v1, dropped=dropped)
    path = os.path.join(args.out, "ckpt.pt")
    torch.save({"model": model.state_dict(), "step": 0, "sft": meta}, path)
    p0(f"\n  ★ 存 {path}   {os.path.getsize(path)/1024**3:.1f} GB   {(time.time()-t0)/60:.1f} 分")

    # ── 看两道题的格式 ──
    model.eval(); model.config.use_cache = True
    tok.padding_side = "left"
    TOOL = any("<calc>" in r["response"] for r in rows[:200])          # ★ 工具版教材 → 演示也走 harness，第一次真闭环
    demo = [(__import__("calc_tool").tool_prompt([2, 3, 5, 7], 31) if TOOL else CD.make_prompt([2, 3, 5, 7], 31)),
            (f"{CD.SYS}\nUser: A baker made 24 muffins and sold 3/4 of them. Each muffin sells for $2. "
             f"How much money did the baker make?\nShow your work in <think> </think> tags. "
             f"Return the final numeric answer in <answer> </answer> tags, for example <answer> 42 </answer>.\n"
             f"Assistant: Let me solve this step by step.\n<think>")]
    for pr in demo:
        if TOOL and "<calc>" in pr:
            import calc_tool
            o = calc_tool.generate_with_tools(model, tok, [pr], max_new=900, temp=0.0, gen_bs=1, dev=DEV)[0]
            txt = o["txt"] + f"\n  [harness: 调用 {o['n_calls']} 次，报错 {o['n_err']}，假结果 {o['fake']}]"
        else:
            enc = tok([pr], return_tensors="pt").to(DEV)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=700, do_sample=False, pad_token_id=PAD)
            txt = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
        print("-" * 74 + "\n" + pr.split("User: ")[1][:80] + "…\n<think>" + txt[:1800])
    print("=" * 74)
    print(f"  下一步（SFT 后测量点）：\n"
          f"    python3 baseline_countdown.py --ckpt {path} --data data/countdown4.json -n 100 -s 8 --offset 95000\n"
          f"    python3 baseline_countdown.py --ckpt {path} -n 100 -s 8 --offset 95000\n"
          f"    python3 crosstask_eval.py --ckpt {path} --out crosstask_{os.path.basename(args.out)}.json")
if WORLD > 1:
    dist.barrier(); dist.destroy_process_group()
