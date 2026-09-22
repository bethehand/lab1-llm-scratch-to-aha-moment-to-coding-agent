#!/usr/bin/env python3
"""
第 3 步：下载 FineWeb-Edu → 用 GPT-2 分词器编码 → 打包成 uint16 的 .bin

train ← 自动扫描 data/raw/sample/10BT/ 下所有 parquet（除 013 留作 val）
        每个 2GB 文件约 7.5 亿 token

★ 注意「独立 token」和「训练 token」是两回事：
    train.bin 里装的是独立 token（不重复的语料）
    config.train_tokens 是训练预算（要看多少个 token-visit）
    第二轮：5.3B 独立  /  10B 训练预算  →  自然跑 1.9 个 epoch
val   ← sample/10BT 的 013（完全不同的文件 → 零文档重叠）

用法:  python3 prepare_data.py
建议在 tmux 里跑，耗时 1~3 小时（主要是下载）。中断后重跑会续传。
"""
import os, sys, time, glob
import numpy as np
import pyarrow.parquet as pq
import tiktoken
from huggingface_hub import hf_hub_download

# ─────────── 配置 ───────────
from config import cfg                    # ← 单一事实来源，别在这里硬编码

REPO         = "HuggingFaceFW/fineweb-edu"
OUT_DIR      = cfg.data_dir
VAL_FILE     =  "sample/10BT/013_00000.parquet"

# 自动扫描本地已下载的 parquet，除 val 文件外全部用作训练数据。
# 这样多下少下都不用改脚本。
_local = sorted(glob.glob("data/raw/sample/10BT/*.parquet"))
TRAIN_FILES = [f"sample/10BT/{os.path.basename(x)}" for x in _local
               if os.path.basename(x) != os.path.basename(VAL_FILE)]
if not TRAIN_FILES:
    raise SystemExit("data/raw/sample/10BT/ 下没找到 parquet —— 先跑 download_data.py")
if not os.path.exists(os.path.join("data/raw", VAL_FILE)):
    raise SystemExit(f"缺 val 文件 {VAL_FILE} —— 先跑 download_data.py")
# 不设上限：把这几个文件全部分词进 train.bin。
# 训练要跑多少 token 由 config.train_tokens 决定，跟这里无关
# （train.py 是随机采样窗口，同一份数据自然会被多次采到 = 多个 epoch）
TRAIN_TOKENS = 10**12
VAL_TOKENS   = cfg.val_tokens
NUM_THREADS  = 64        # tiktoken 内部线程数（你有 128 逻辑核）
ROW_BATCH    = 2000      # 每批送去分词的文档数

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token
assert enc.n_vocab < 65536, "vocab 超出 uint16"


def download(fname):
    """已在本地就直接用；否则才去下载。"""
    local = os.path.join("data/raw", fname)
    if os.path.exists(local) and os.path.getsize(local) > 1024:
        print(f"  ✓ 已有 {fname}  ({os.path.getsize(local)/1024**3:.2f} GB)", flush=True)
        return local
    print(f"  ↓ 下载 {fname}", flush=True)
    return hf_hub_download(REPO, fname, repo_type="dataset", local_dir="data/raw")


def build(files, out_path, target_tokens, label):
    """逐个 parquet 流式读取 → 分词 → 追加写入 .bin，攒够 target_tokens 就停。"""
    paths = [download(f) for f in files]
    # target_tokens 可能是「不设上限」的哨兵值。用 parquet 总大小估算真实规模，
    # 否则进度条会拿 10^12 当分母，显示 0.0% 和荒谬的 ETA。
    pq_gb = sum(os.path.getsize(p) for p in paths) / 1024**3
    est = int(pq_gb / 26.6 * 10e9)          # sample-10BT 全量 26.6 GB ≈ 100 亿 token
    shown = min(target_tokens, est) if target_tokens > est else target_tokens
    print(f"\n{'='*58}\n构建 {label}", flush=True)
    if target_tokens >= 10**11:
        print(f"  不设上限，把 {len(paths)} 个 parquet（{pq_gb:.1f} GB）全部分词",
              flush=True)
        print(f"  按 26.6GB≈100亿token 估算，预计约 {est/1e9:.1f}B token", flush=True)
    else:
        print(f"  目标 {target_tokens:,} token", flush=True)
    print("=" * 58, flush=True)

    written = 0
    n_docs  = 0
    n_bytes = 0                       # 原始 UTF-8 字节数，用来量压缩率
    t0 = time.time()

    with open(out_path, "wb") as fout:
        for path in paths:
            if written >= target_tokens:
                break
            pf = pq.ParquetFile(path)
            print(f"  分词 {os.path.basename(path)} "
                  f"({pf.metadata.num_rows:,} 篇文档)", flush=True)

            for batch in pf.iter_batches(batch_size=ROW_BATCH, columns=["text"]):
                texts = batch.column("text").to_pylist()
                n_bytes += sum(len(t.encode("utf-8")) for t in texts)   # 真实 UTF-8 字节
                n_docs  += len(texts)

                # tiktoken 的 Rust 多线程批量编码，比 multiprocessing 简单也更快
                lists = enc.encode_ordinary_batch(texts, num_threads=NUM_THREADS)

                flat = []
                for ids in lists:
                    flat.extend(ids)
                    flat.append(EOT)          # 文档分隔符

                arr = np.array(flat, dtype=np.uint16)
                if written + len(arr) > target_tokens:
                    arr = arr[: target_tokens - written]

                arr.tofile(fout)
                written += len(arr)

                every = ROW_BATCH * (5 if written < 5e7 else 40)
                if n_docs % every < ROW_BATCH:
                    el = time.time() - t0
                    pct = 100 * written / shown
                    eta = el / max(written, 1) * max(shown - written, 0)
                    print(f"    {written:>13,} / ~{shown:,} token "
                          f"({pct:5.1f}%)  {written/el/1e6:5.2f}M tok/s  "
                          f"剩余 {eta/60:5.1f} 分钟", flush=True)

                if written >= target_tokens:
                    break

    size_gb = os.path.getsize(out_path) / 1024**3
    ratio   = n_bytes / written if written else 0
    print(f"\n  ✅ {out_path}")
    print(f"     {written:,} token   {size_gb:.2f} GB   {n_docs:,} 篇文档")
    print(f"     ★ 真实压缩率 {ratio:.2f} 字节/token   （用时 {(time.time()-t0)/60:.1f} 分钟）")
    return written, ratio


def verify(path, n_samples=5):
    """验收：随机切几个窗口解码回文本，人眼确认读得通。"""
    print(f"\n{'='*58}\n验收 {path}\n{'='*58}")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    print(f"  文件含 {len(data):,} 个 token")
    assert data.max() < enc.n_vocab, "出现越界 token id！"
    print(f"  ✅ 最大 token id = {data.max()} < {enc.n_vocab}")
    print(f"  ✅ 分隔符 <|endoftext|> 出现 {int((data[:5_000_000] == EOT).sum()):,} 次"
          f"（前 500 万 token 内）")

    rng = np.random.default_rng(0)
    for k in range(n_samples):
        i = int(rng.integers(0, len(data) - 300))
        txt = enc.decode([int(t) for t in data[i:i+120]])
        print(f"\n  ── 样本 {k+1} (位置 {i:,}) ──")
        print("   ", txt.replace("\n", " ⏎ ")[:300])


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)

    gb = sum(os.path.getsize(os.path.join("data/raw", f)) for f in TRAIN_FILES) / 1024**3
    print(f"扫描到 {len(TRAIN_FILES)} 个训练文件（{gb:.1f} GB）："
          f" {', '.join(os.path.basename(f)[:3] for f in TRAIN_FILES)}")
    print(f"val 文件：{os.path.basename(VAL_FILE)}\n")

    tr_path  = os.path.join(OUT_DIR, "train.bin")
    val_path = os.path.join(OUT_DIR, "val.bin")

    n_tr, r_tr = build(TRAIN_FILES, tr_path, TRAIN_TOKENS, "train")
    n_va, r_va = build([VAL_FILE],  val_path, VAL_TOKENS,  "val")

    verify(tr_path)

    print(f"\n{'='*58}\n第 3 步完成\n{'='*58}")
    print(f"  train.bin  {n_tr:>15,} 独立 token   压缩率 {r_tr:.2f}   "
          f"{os.path.getsize(tr_path)/1024**3:.1f} GB")
    print(f"  val.bin    {n_va:>15,} 独立 token   压缩率 {r_va:.2f}")
    print(f"\n  训练预算 config.train_tokens = {cfg.train_tokens:,}")
    ep = cfg.train_tokens / max(n_tr, 1)
    flag = "✅ 在 4 遍法则内" if ep <= 4 else "⚠️ 超过 4 遍法则，考虑多下几个文件或缩小模型"
    print(f"  → 会跑 {ep:.2f} 个 epoch   {flag}")
