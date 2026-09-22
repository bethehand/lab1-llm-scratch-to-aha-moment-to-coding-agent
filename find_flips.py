#!/usr/bin/env python3
"""
找出训练前后答案翻转的题，把两边的完整推理并排打出来。

  ★ 两个方向都看：
    修好的  base ❌ → RL ✅    ← RL 的战果
    弄坏的  base ✅ → RL ❌    ← ★ 代价，同样重要

    python3 find_flips.py                          # 300 道 holdout 题，约 10 分钟
    python3 find_flips.py --n 100                  # 快一点
    python3 find_flips.py --show 5                 # 每个方向打印几例
"""
import os, json, argparse, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from _reward import reward, extract

ap = argparse.ArgumentParser()
ap.add_argument("--model",  default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--rl",     default="out_grpo_lr5e6/ckpt_latest.pt", help="★ 训练后的权重")
ap.add_argument("--base-ckpt", default=None, help="对照方的权重，不给就用原始 Qwen")
ap.add_argument("--n",      type=int, default=300)
ap.add_argument("--offset", type=int, default=200, help="★ 200 起 = 从没参与过决策的题")
ap.add_argument("--bs",     type=int, default=32)
ap.add_argument("--max-new", type=int, default=1024)
ap.add_argument("--show",   type=int, default=4, help="每个方向打印几例")
ap.add_argument("--out",    default="flips.json")
args = ap.parse_args()

dev = "cuda"
tok = AutoTokenizer.from_pretrained(args.model)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
INSTR = "\n\nSolve this step by step. End your answer with '#### <number>'."

from datasets import load_dataset
D = load_dataset("openai/gsm8k", "main", split="test")
ALL = [(x["question"], x["answer"].split("####")[-1].strip().replace(",", "")) for x in D]
P = ALL[args.offset: args.offset + args.n]
print(f"GSM8K test   取 TEST[{args.offset}:{args.offset+len(P)}]   {len(P)} 题"
      f"   {'★ 从没用过（无偏）' if args.offset >= 200 else '⚠️ 训练时用过'}")


def load(ck):
    try:
        m = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    m = m.to(dev).eval()
    if ck:
        d = torch.load(ck, map_location="cpu", weights_only=False)
        m.load_state_dict(d["model"]); m.to(dev).eval()
        st = d.get("step", "?"); del d; torch.cuda.empty_cache()
        print(f"  ★ 加载 {ck}（step {st}）")
    return m


@torch.no_grad()
def run(m, tag):
    """★ 贪心解码 —— 确定性，翻转必然来自权重，不是采样运气"""
    ps = [tok.apply_chat_template([{"role": "user", "content": q + INSTR}],
                                  tokenize=False, add_generation_prompt=True) for q, _ in P]
    texts, t0 = [], time.time()
    for s in range(0, len(ps), args.bs):
        enc = tok(ps[s: s + args.bs], return_tensors="pt", padding=True).to(dev)
        out = m.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id)
        texts += [tok.decode(o, skip_special_tokens=True)
                  for o in out[:, enc.input_ids.size(1):]]
        print(f"    {tag}  {min(s+args.bs, len(ps)):>4}/{len(ps)}"
              f"   {(time.time()-t0)/60:.1f} 分", end="\r", flush=True)
    print()
    rs = [reward(t, g) for t, (_, g) in zip(texts, P)]
    print(f"    {tag}  正确率 {sum(rs)/len(rs):.4f}")
    return texts, rs


print("\n── 对照模型 ──")
A = load(args.base_ckpt); ta, ra = run(A, "对照")
del A; torch.cuda.empty_cache()

print("\n── 训练后 ──")
B = load(args.rl);        tb, rb = run(B, "训练后")
del B; torch.cuda.empty_cache()

# ── 交叉表 ──
fixed  = [i for i in range(len(P)) if not ra[i] and rb[i]]      # ★ 修好的
broken = [i for i in range(len(P)) if ra[i] and not rb[i]]      # ★ 弄坏的
both_ok = sum(1 for i in range(len(P)) if ra[i] and rb[i])
both_no = sum(1 for i in range(len(P)) if not ra[i] and not rb[i])

print("\n" + "=" * 80)
print("  ★ 翻转分析   贪心解码（温度 0，翻转必然来自权重变化）")
print("=" * 80)
print(f"""
                        训练后 ✅      训练后 ❌
    对照 ✅          {both_ok:>8}       {len(broken):>8}  ← ★ 弄坏的
    对照 ❌          {len(fixed):>8}  ← ★ 修好的   {both_no:>8}

    对照正确率    {sum(ra)/len(P):.4f}
    训练后正确率  {sum(rb)/len(P):.4f}
    ★ 净提升      {(sum(rb)-sum(ra))/len(P):+.4f}   = ({len(fixed)} 修好 − {len(broken)} 弄坏) / {len(P)}

    ★ 注意：净提升是两个方向抵消后的结果。
      修好 {len(fixed)} 道、弄坏 {len(broken)} 道 —— 后者常被忽略，但它是真实代价。
""")


def dump(idx, title, sym):
    print("\n" + "█" * 80)
    print(f"  {title}   共 {len(idx)} 道，展示 {min(args.show, len(idx))} 道")
    print("█" * 80)
    for i in idx[: args.show]:
        q, g = P[i]
        print(f"\n{'━'*80}\n【题 {args.offset+i}】{q}\n  ★ 标准答案 {g}\n{'━'*80}")
        for tag, t, r in (("对照", ta[i], ra[i]), ("训练后", tb[i], rb[i])):
            a, f = extract(t)
            print(f"\n  ── {tag}   {'✅' if r else '❌'}   抽出 {a!r}   "
                  f"{'有 ####' if f else '★ 无 ####'}   {len(tok.encode(t))} token ──")
            print("    " + t.strip().replace("\n", "\n    ")[:900])


dump(fixed,  "★ 修好的（对照 ❌ → 训练后 ✅）", "✅")
dump(broken, "⚠️ 弄坏的（对照 ✅ → 训练后 ❌）", "❌")

json.dump({"offset": args.offset, "n": len(P), "rl": args.rl, "base": args.base_ckpt,
           "acc_base": sum(ra)/len(P), "acc_rl": sum(rb)/len(P),
           "fixed": fixed, "broken": broken,
           "examples": {"fixed": [{"i": i, "q": P[i][0], "gold": P[i][1],
                                   "base": ta[i], "rl": tb[i]} for i in fixed[:20]],
                        "broken": [{"i": i, "q": P[i][0], "gold": P[i][1],
                                    "base": ta[i], "rl": tb[i]} for i in broken[:20]]}},
          open(args.out, "w"), indent=1)
print(f"\n  → {args.out}（含全部翻转题的完整文本）\n")
