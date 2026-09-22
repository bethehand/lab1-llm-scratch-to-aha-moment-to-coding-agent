#!/usr/bin/env python3
"""
在【从没用过的】题上评估，量出「选最优 checkpoint」带来的选择偏差。

  训练时的评估只用 TEST[0:200] —— 而 ckpt_best 就是按那 200 题选出来的
  → ★ 那个分数同时包含「模型好」和「这次测量运气好」
  → 在 TEST[200:500] 上再评一次，才是无偏的真实水平

    python3 holdout_eval.py                       # 全部 checkpoint
    python3 holdout_eval.py --n 300 --offset 200  # 换切片
    python3 holdout_eval.py --only-holdout        # 只跑没用过的那 300 题（快一半）
"""
import os, json, argparse, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from _reward import reward

ap = argparse.ArgumentParser()
ap.add_argument("--model",  default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--n",      type=int, default=300, help="没用过的切片取几道题")
ap.add_argument("--offset", type=int, default=200, help="从第几道开始（★ 训练评估用了前 200）")
ap.add_argument("--used-n", type=int, default=200, help="用过的切片（对照，应复现训练日志）")
ap.add_argument("--bs",     type=int, default=32)
ap.add_argument("--max-new", type=int, default=1024)
ap.add_argument("--only-holdout", action="store_true")
ap.add_argument("--ckpts", nargs="*", default=None)
ap.add_argument("--out",    default="holdout.json")
args = ap.parse_args()

dev = "cuda"
CKPTS = args.ckpts if args.ckpts is not None else [
    "",                                   # 空 = 原始模型
    "out_grpo_best/ckpt_best.pt",
    "out_grpo_best/ckpt_latest.pt",
    "out_grpo_lr5e6/ckpt_latest.pt",
]
CKPTS = [c for c in CKPTS if c == "" or os.path.exists(c)]

from datasets import load_dataset
D = load_dataset("openai/gsm8k", "main", split="test")
ALL = [(x["question"], x["answer"].split("####")[-1].strip().replace(",", "")) for x in D]
USED = ALL[: args.used_n]                                   # 训练时评估用过的
HOLD = ALL[args.offset: args.offset + args.n]               # ★ 从没用过的
print(f"GSM8K test {len(ALL):,} 题")
print(f"  用过的   TEST[0:{args.used_n}]           {len(USED)} 题  ← ckpt_best 是按它选的")
print(f"  ★ 没用过 TEST[{args.offset}:{args.offset+args.n}]   {len(HOLD)} 题  ← 无偏")

tok = AutoTokenizer.from_pretrained(args.model)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
INSTR = "\n\nSolve this step by step. End your answer with '#### <number>'."


def load_base():
    try:
        m = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    return m.to(dev).eval()


print(f"载入 {args.model} …", flush=True)
model = load_base()


@torch.no_grad()
def greedy_acc(probs):
    """★ 贪心解码 —— 完全确定，同样的权重永远给同一个数字，零采样噪声"""
    ps = [tok.apply_chat_template([{"role": "user", "content": q + INSTR}],
                                  tokenize=False, add_generation_prompt=True)
          for q, _ in probs]
    ok, nt = 0, 0
    for s in range(0, len(ps), args.bs):
        enc = tok(ps[s: s + args.bs], return_tensors="pt", padding=True).to(dev)
        out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        new = out[:, enc.input_ids.size(1):]
        nt += int((new != tok.pad_token_id).sum())
        for row, (_, g) in zip(new, probs[s: s + args.bs]):
            ok += reward(tok.decode(row, skip_special_tokens=True), g)
    return ok / len(probs), nt / len(probs)


R, t0 = [], time.time()
for ck in CKPTS:
    name = "★ 原始 Qwen（未训练）" if ck == "" else ck
    if ck:
        d = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(d["model"])
        step = d.get("step", "?")
        rep = d.get("test_acc")
        del d; torch.cuda.empty_cache()
        name = f"{ck.split('/')[0]} step {step}"
    else:
        step, rep = 0, 0.7050
    print(f"\n{'─'*74}\n  {name}", flush=True)

    a_used = None
    if not args.only_holdout:
        a_used, l_used = greedy_acc(USED)
        print(f"    用过的 {len(USED)} 题      {a_used:.4f}   平均 {l_used:.0f} token", flush=True)
    a_hold, l_hold = greedy_acc(HOLD)
    print(f"    ★ 没用过 {len(HOLD)} 题    {a_hold:.4f}   平均 {l_hold:.0f} token", flush=True)
    R.append(dict(name=name, ckpt=ck, step=step, reported=rep,
                  used=a_used, holdout=a_hold, len_holdout=l_hold))
    if ck:
        model = load_base()                     # 换回原始权重，准备加载下一个

print(f"\n  全部完成，用时 {(time.time()-t0)/60:.1f} 分钟")

# ══════════════════════════════════════════════════════════════
W = max(len(r["name"]) for r in R) + 2
print("\n" + "=" * 88)
print("  ★ 选择偏差量化   贪心解码（温度 0，零采样噪声）")
print("=" * 88)
print(f"\n  {'模型':<{W}}{'训练日志报的':>13}{'用过的200题':>13}{'★ 没用过':>11}{'偏差':>10}")
print("  " + "─" * (W + 47))
for r in R:
    u = f"{r['used']:.4f}" if r["used"] is not None else "—"
    b = (f"{r['used']-r['holdout']:+.4f}" if r["used"] is not None else "—")
    rp = f"{r['reported']:.4f}" if r["reported"] is not None else "—"
    print(f"  {r['name']:<{W}}{rp:>13}{u:>13}{r['holdout']:>11.4f}{b:>10}")

base = R[0]["holdout"]
print(f"\n  ── ★ 在没用过的题上，相对原始模型的真实提升 ──")
print(f"  {'模型':<{W}}{'提升':>10}")
print("  " + "─" * (W + 10))
for r in R[1:]:
    print(f"  {r['name']:<{W}}{r['holdout']-base:>+10.4f}")

print(f"""
  ★★ 怎么读：

    「训练日志报的」和「用过的200题」应该几乎一样   ← 验证代码一致性
    「用过的200题」减「没用过」= ★ 选择偏差
       正的越大 → ckpt_best 越是"挑中了运气好的一次"，而不是"模型真的更好"

    ★ 最后那张表才是这次 RL 实验的真实成果 ——
      在从没参与过任何决策的题上，模型到底强了多少
""")
json.dump(R, open(args.out, "w"), indent=1)
print(f"  → {args.out}\n")
