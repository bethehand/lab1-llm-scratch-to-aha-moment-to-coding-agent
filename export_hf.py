#!/usr/bin/env python3
"""
export_hf.py —— 把我们的 ckpt.pt（torch.save 的 {"model": state_dict, "step": …}）导出成 HF 目录，给 vLLM / SGLang 加载。

    python3 export_hf.py --ckpt out_sft_tool/ckpt.pt --out hf_sft_tool
    python3 export_hf.py --ckpt out_armT/ckpt_latest.pt --out hf_armT
    python3 export_hf.py --out hf_base                      # 不给 --ckpt = 原始 Base 也导一份

目录里：config.json、model.safetensors、tokenizer 文件、export_info.json（来源和 step）。CPU 就能跑，十几秒。
"""
import os, json, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
ap.add_argument("--ckpt", default=None)
ap.add_argument("--out", required=True)
args = ap.parse_args()

try:
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
step = None
if args.ckpt:
    c = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(c["model"], strict=False)
    assert not unexpected, f"ckpt 里有模型不认的键：{unexpected[:5]}"
    assert not missing, f"ckpt 缺键：{missing[:5]}"
    step = c.get("step")
    print(f"  ★ 载入 {args.ckpt}（step {step}）")
os.makedirs(args.out, exist_ok=True)
model.save_pretrained(args.out, safe_serialization=True)
AutoTokenizer.from_pretrained(args.model).save_pretrained(args.out)
json.dump(dict(model=args.model, ckpt=args.ckpt, step=step), open(os.path.join(args.out, "export_info.json"), "w"), indent=1)
sz = sum(os.path.getsize(os.path.join(args.out, f)) for f in os.listdir(args.out)) / 1024**3
print(f"  → {args.out}/  {sz:.2f} GB   文件：{sorted(os.listdir(args.out))}")
