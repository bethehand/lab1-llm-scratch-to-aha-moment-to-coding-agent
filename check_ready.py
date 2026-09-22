#!/usr/bin/env python3
"""第二轮开跑前的文件就绪检查。在 GS01 上跑：python3 check_ready.py"""
import os, sys, ast

# 每个文件必须包含的「第二轮特征标记」
MARKERS = {
    "config.py":       [("C:          int = 896", "C=896"),
                        ("n_layer:    int = 16", "16 层"),
                        ("d_ff:       int = 2432", "d_ff=2432"),
                        ("train_tokens: int = 10_000_000_000", "10B token"),
                        ("lr:           float = 5e-4", "lr=5e-4"),
                        ("warmup_steps: int   = 500", "warmup=500")],
    "model.py":        [("kv_cache=None, pos=0", "KV cache"),
                        ("repetition_penalty=1.15", "重复惩罚"),
                        ("top_p", "top-p 采样")],
    "train.py":        [("dist.ReduceOp.AVG", "eval 跨卡平均"),
                        ("bf16_compress_hook", "bf16 梯度压缩"),
                        ("save-every", "定期存盘"),
                        ("_sigint", "优雅中断")],
    "prepare_data.py": [("glob.glob(\"data/raw/sample/10BT/*.parquet\")", "自动扫描 parquet"),
                        ("TRAIN_TOKENS = 10**12", "不设 token 上限"),
                        ("cfg.val_tokens", "从 config 读")],
    "download_data.py":[("range(7)", "7+1 文件"),
                        ("已有", "跳过已有文件")],
    "chat.py":         [("预热中", "预热"),
                        ("out_run2", "默认 out_run2")],
    "evaluate.py":     [("out_run2", "默认 out_run2"),
                        ("解码策略对比", "解码对比")],
    "estimate.py":     [("attn_frac", "attention 项")],
    "monitor.py":      [("capture-pane", "tmux 抓取")],
    "inspect_bin.py":  [("训练时的一个样本", "x/y 错位展示")],
}

ok = True
print("=" * 62)
print("  第二轮文件就绪检查")
print("=" * 62)
for fn, marks in MARKERS.items():
    if not os.path.exists(fn):
        print(f"  ❌ {fn:<20} 文件不存在"); ok = False; continue
    src = open(fn, encoding="utf-8").read()
    try:
        ast.parse(src)
    except SyntaxError as e:
        print(f"  ❌ {fn:<20} 语法错误 第{e.lineno}行"); ok = False; continue
    miss = [d for m, d in marks if m not in src]
    if miss:
        print(f"  ⚠️  {fn:<20} 缺: {', '.join(miss)}"); ok = False
    else:
        print(f"  ✅ {fn:<20} {len(marks)} 项标记齐全")

print()
try:
    from config import cfg
    cfg.check()
    n = cfg.n_layer * (4*cfg.C**2 + 3*cfg.C*cfg.d_ff + 2*cfg.C) + cfg.vocab_size*cfg.C + cfg.C
    print(f"  配置自检  参数量 {n:,}  深宽比 {cfg.C/cfg.n_layer:.0f}  "
          f"步数 {cfg.max_steps:,}")
    print(f"            n_head {cfg.n_head} == C/head_dim {cfg.C//cfg.head_dim} ✅")
except Exception as e:
    print(f"  ❌ config 自检失败: {e}"); ok = False

print()
for d, desc in [("data/raw/sample/10BT", "原始 parquet"),
                (os.path.join(cfg.data_dir), "分词后的 .bin")]:
    if os.path.isdir(d):
        fs = sorted(os.listdir(d))
        sz = sum(os.path.getsize(os.path.join(d, f)) for f in fs) / 1024**3
        print(f"  ✅ {desc:<14} {len(fs)} 个文件  {sz:.1f} GB  ({', '.join(f[:3] for f in fs[:10])})")
    else:
        print(f"  ⬜ {desc:<14} 还没有 —— {'跑 download_data.py' if 'raw' in d else '跑 prepare_data.py'}")

print("\n" + ("  ✅ 全部就绪" if ok else "  ⚠️ 有缺失，见上面"))
