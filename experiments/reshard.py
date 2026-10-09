#!/usr/bin/env python3
"""Re-save a single-file safetensors checkpoint as shards plus an index, in the standard Hugging Face layout.

    python3 experiments/reshard.py <src model.safetensors> <dst dir> [max shard size, default 400MB]

Writes model-0000k-of-0000N.safetensors and model.safetensors.index.json into <dst dir>; tensors, dtypes and names are copied
exactly. transformers and vLLM load the sharded layout transparently. Used by upload_checkpoints.sh (RESHARD=400MB) because
on an unreliable link the Hub skips shards that were already uploaded, so an interrupted upload resumes shard by shard
instead of restarting a 3 GB file from zero. Needs torch, safetensors and huggingface_hub (the vLLM environment has all three).
"""
import os, sys, json
from safetensors.torch import load_file, save_file
from huggingface_hub import split_torch_state_dict_into_shards

src, dst = sys.argv[1], sys.argv[2]
size = sys.argv[3] if len(sys.argv) > 3 else "400MB"
os.makedirs(dst, exist_ok=True)
sd = load_file(src)
split = split_torch_state_dict_into_shards(sd, max_shard_size=size, filename_pattern="model{suffix}.safetensors")
for name, keys in split.filename_to_tensors.items():
    save_file({k: sd[k].contiguous() for k in keys}, os.path.join(dst, name), metadata={"format": "pt"})
if split.is_sharded:
    total = sum(t.numel() * t.element_size() for t in sd.values())
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": split.tensor_to_filename}, f, indent=2)
print(f"{src}: {len(sd)} tensors -> {len(split.filename_to_tensors)} file(s) in {dst}")
