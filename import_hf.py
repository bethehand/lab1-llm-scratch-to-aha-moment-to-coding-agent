#!/usr/bin/env python3
"""
import_hf.py —— export_hf.py 的反向：HF 目录 → 我们的 ckpt.pt（{"model": state_dict, "step": …}），给 --init 用。

    python3 import_hf.py --hf-dir hf_armS150 --out out_armS150/ckpt.pt --step 150

只有权重没有优化器状态（--init 会从头建优化器，等于换了池子重新起一段训练）。
"""
import os, json, argparse
import torch
from transformers import AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--hf-dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--step", type=int, default=None, help="记在 ckpt 里的 step；不给则读 export_info.json")
args = ap.parse_args()

try:
    model = AutoModelForCausalLM.from_pretrained(args.hf_dir, dtype=torch.bfloat16)
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(args.hf_dir, torch_dtype=torch.bfloat16)
step = args.step
info = os.path.join(args.hf_dir, "export_info.json")
if step is None and os.path.exists(info):
    step = json.load(open(info)).get("step")
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
torch.save({"model": model.state_dict(), "step": step, "from_hf": args.hf_dir}, args.out)
print(f"  {args.hf_dir} → {args.out}   step {step}   {os.path.getsize(args.out)/1024**3:.2f} GB   {sum(p.numel() for p in model.parameters())/1e9:.2f}B 参数")
