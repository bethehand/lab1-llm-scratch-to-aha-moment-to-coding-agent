#!/usr/bin/env python3
"""
reject_sample.py —— 拒绝采样：让模型自己做题，只留答对的轨迹，做 SFT 数据。

  两条 SFT 臂共用的 GSM8K 数据（防 GSM 塌）：
    python3 reject_sample.py --ckpt out_mix4/ckpt_step300.pt --task gsm --n 2000 --out sft_gsm.jsonl
  臂 2（自蒸馏）的 Countdown 数据 —— ★ 题号取自穷举器的输出，跟臂 1 同题：
    python3 reject_sample.py --ckpt out_mix4/ckpt_step300.pt --task cd --data data/countdown.json  --ids sft_cd3.jsonl --out sft_cd3_self.jsonl
    python3 reject_sample.py --ckpt out_mix4/ckpt_step300.pt --task cd --data data/countdown4.json --ids sft_cd4.jsonl --out sft_cd4_self.jsonl

规则：每题抽 k 条（温度 1），答对的里随机留 1 条。一条都不对的题就没有（★ 不补题，补了就偏简单）。
输出 jsonl 每行 {"prompt", "response", "task", "idx"/"q", "gold", "n_correct"}，格式跟 enum_traces.py 一致。
"""
import os, re, sys, json, time, random, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import countdown as CD
import _reward as RW

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
ap.add_argument("--ckpt", required=True)
ap.add_argument("--task", choices=["gsm", "cd"], required=True)
ap.add_argument("--data", default=None, help="cd：countdown json")
ap.add_argument("--ids", default=None, help="cd：穷举器输出的 jsonl，取里面的 idx 做同题号")
ap.add_argument("--n", type=int, default=2000, help="gsm：要留多少条对的")
ap.add_argument("--gsm-lo", type=int, default=0)
ap.add_argument("--gsm-hi", type=int, default=7000, help="GSM8K train 共 7473，留一段不碰")
ap.add_argument("-k", type=int, default=None, help="每题抽几条；默认 cd 8、gsm 4")
ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--max-new", type=int, default=1024)
ap.add_argument("--bs", type=int, default=32, help="每批多少条（题 × k 展开后）")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
args = ap.parse_args()
k = args.k or (8 if args.task == "cd" else 4)
rng = random.Random(args.seed)
CONT = "\nUser:"

# ── 提示词：跟训练时【逐字相同】 ──
def gsm_prompt(q):
    return (f"{CD.SYS}\nUser: {q}\nShow your work in <think> </think> tags. "
            f"Return the final numeric answer in <answer> </answer> tags, "
            f"for example <answer> 42 </answer>.\n"
            f"Assistant: Let me solve this step by step.\n<think>")

def gsm_ok(txt, gold):
    m = CD.ANS.search(txt)
    if not m:
        return False
    a, _ = RW.extract(m.group(1))
    try:
        return a is not None and abs(float(a) - float(gold)) < 1e-4
    except ValueError:
        return False

def clean(txt):
    """截掉幻觉续写，只留到第一个 </answer>"""
    c = txt.find(CONT)
    if c >= 0:
        txt = txt[:c]
    e = txt.find("</answer>")
    if e >= 0:
        txt = txt[: e + len("</answer>")]
    return txt.rstrip()

# ── 题 ──
if args.task == "cd":
    assert args.data and args.ids, "cd 需要 --data 和 --ids"
    D = json.load(open(args.data))
    ids = [json.loads(l)["idx"] for l in open(args.ids)]
    probs = [dict(idx=i, prompt=CD.make_prompt(D[i]["nums"], D[i]["target"]),
                  nums=D[i]["nums"], target=D[i]["target"]) for i in ids]
    print(f"Countdown  {args.data}  同题号 {len(probs)} 道（来自 {args.ids}）")
else:
    from datasets import load_dataset
    rows = load_dataset("openai/gsm8k", "main")["train"]
    order = list(range(args.gsm_lo, min(args.gsm_hi, len(rows))))
    rng.shuffle(order)
    probs = [dict(idx=i, q=rows[i]["question"], prompt=gsm_prompt(rows[i]["question"]),
                  gold=rows[i]["answer"].split("####")[-1].strip().replace(",", "")) for i in order]
    print(f"GSM8K train[{args.gsm_lo}:{args.gsm_hi}]  打乱后逐题抽，直到留够 {args.n} 条")

# ── 模型 ──
dev = "cuda"
tok = AutoTokenizer.from_pretrained(args.model)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
try:
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
c = torch.load(args.ckpt, map_location="cpu", weights_only=False)
model.load_state_dict(c["model"]); del c
model.to(dev).eval()
print(f"  ★ {args.ckpt}   k={k}  T={args.temp}  max_new={args.max_new}", flush=True)

@torch.no_grad()
def gen(prompts):
    enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
    kw = dict(max_new_tokens=args.max_new, do_sample=args.temp > 0, top_p=1.0, pad_token_id=tok.pad_token_id)
    if args.temp > 0:
        kw["temperature"] = args.temp
    try:
        out = model.generate(**enc, stop_strings=[CONT], tokenizer=tok, **kw)
    except (TypeError, ValueError):
        out = model.generate(**enc, **kw)
    return tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)

# ── 采样 ──
per_batch = max(1, args.bs // k)          # 每批几道题
kept, n_done, n_cov, n_corr_total, t0 = [], 0, 0, 0, time.time()
fout = open(args.out, "w")
for b0 in range(0, len(probs), per_batch):
    batch = probs[b0: b0 + per_batch]
    texts = gen([p["prompt"] for p in batch for _ in range(k)])
    for pi, p in enumerate(batch):
        outs = [clean(t) for t in texts[pi * k: (pi + 1) * k]]
        if args.task == "cd":
            good = [t for t in outs if CD.reward(t, p["nums"], p["target"])[1]["correct"] > 0]
        else:
            good = [t for t in outs if gsm_ok(t, p["gold"])]
        n_done += 1; n_corr_total += len(good)
        if good:
            n_cov += 1
            row = dict(prompt=p["prompt"], response=rng.choice(good), task=args.task,
                       idx=p["idx"], n_correct=len(good))
            if args.task == "cd":
                row.update(nums=p["nums"], target=p["target"])
            else:
                row.update(q=p["q"], gold=p["gold"])
            fout.write(json.dumps(row) + "\n"); kept.append(row)
    if (b0 // per_batch) % 10 == 0 or n_done == len(probs):
        el = time.time() - t0
        print(f"  {n_done}/{len(probs)} 题  留 {len(kept)}  覆盖 {n_cov/n_done:.2f}  "
              f"每题对 {n_corr_total/n_done/k:.2f}  {el/60:.1f} 分", flush=True)
    if args.task == "gsm" and len(kept) >= args.n:
        break
fout.close()

tl = sorted(len(tok(r["response"]).input_ids) for r in kept)
q = lambda f: tl[int(f * (len(tl) - 1))] if tl else 0
print(f"\n  ★ 留 {len(kept)} 条 → {args.out}")
print(f"  覆盖率 {n_cov}/{n_done} = {n_cov/max(1,n_done):.2f}   平均每题 {n_corr_total/max(1,n_done):.1f}/{k} 条对")
print(f"  response token 长度  中位 {q(.5)}  P90 {q(.9)}  最长 {tl[-1] if tl else 0}")
if args.task == "cd":
    print("  ★ 没覆盖的题就是模型自己解不了的 —— 臂 2 的数据天然缺这些，这正是它跟臂 1 的差别")
