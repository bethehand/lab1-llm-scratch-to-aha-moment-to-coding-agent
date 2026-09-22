#!/usr/bin/env python3
"""
第 3 步的第一半：只下载，不分词。

下完之后跑 prepare_data.py，它会发现文件已在本地，直接跳到分词。
建议在 tmux 里跑。中断后重跑会续传。

国内加速（强烈建议先设）:
    pip install hf_transfer
    export HF_ENDPOINT=https://hf-mirror.com
    export HF_HUB_ENABLE_HF_TRANSFER=1
"""
import os, time
from huggingface_hub import hf_hub_download

REPO  = "HuggingFaceFW/fineweb-edu"
# 第二轮：7 个 train 文件（约 53 亿独立 token）+ 1 个 val 文件
# 000/001/013 第一轮已下过，脚本会跳过；实际只需要下 002~006 共 5 个（约 10 GB）
FILES = ["sample/10BT/013_00000.parquet"] + \
        [f"sample/10BT/{i:03d}_00000.parquet" for i in range(7)]

print(f"HF_ENDPOINT              = {os.environ.get('HF_ENDPOINT', '(官方直连)')}")
print(f"HF_HUB_ENABLE_HF_TRANSFER= {os.environ.get('HF_HUB_ENABLE_HF_TRANSFER', '0')}")
print(f"共 {len(FILES)} 个文件；已有的会跳过，实际约需下载 10 GB\n")

t_all = time.time()
total = 0
for i, f in enumerate(FILES, 1):
    t0 = time.time()
    local = os.path.join("data/raw", f)
    if os.path.exists(local) and os.path.getsize(local) > 1024:
        print(f"[{i}/{len(FILES)}] ✓ 已有 {f}  "
              f"({os.path.getsize(local)/1024**3:.2f} GB)\n", flush=True)
        total += os.path.getsize(local)
        continue
    print(f"[{i}/{len(FILES)}] ↓ {f}", flush=True)
    p = hf_hub_download(REPO, f, repo_type="dataset", local_dir="data/raw")
    sz = os.path.getsize(p)
    total += sz
    dt = time.time() - t0
    print(f"        {sz/1024**3:.2f} GB   用时 {dt/60:.1f} 分钟   "
          f"{sz/dt/1024**2:.1f} MB/s\n", flush=True)

dt = time.time() - t_all
print("=" * 54)
print(f"✅ 下载完成  {total/1024**3:.2f} GB   总用时 {dt/60:.1f} 分钟   "
      f"平均 {total/dt/1024**2:.1f} MB/s")
print(f"\n下一步:  python3 prepare_data.py   （会跳过下载，直接分词，约 10~20 分钟）")
