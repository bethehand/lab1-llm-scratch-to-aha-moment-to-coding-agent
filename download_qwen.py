#!/usr/bin/env python3
"""
下载 Qwen2.5-1.5B-Instruct，然后把它拆开看看。

    python3 download_qwen.py                          # 下载 + 查看
    python3 download_qwen.py --inspect-only           # 只看，不下载
    python3 download_qwen.py --model Qwen/Qwen2.5-1.5B   # Base 版（第二次实验用）
"""
import os, json, argparse, time

# ★ 必须在 import huggingface_hub 之前设置。
#   Xet 是 HF 新的分块存储后端，在慢速/不稳定线路上会报
#   "CAS Client Error: error decoding response body" 然后整个下载失败。
#   关掉它走老的 HTTP 路径 —— 慢一点，但支持断点续传，稳得多。
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--inspect-only", action="store_true")
ap.add_argument("--xet", action="store_true", help="重新启用 Xet（默认关闭）")
ap.add_argument("--retries", type=int, default=20, help="失败自动重试几次")
args = ap.parse_args()
if args.xet:
    os.environ["HF_HUB_DISABLE_XET"] = "0"

# ══════════════════════════════════════════════════════════
#  ① 下载
# ══════════════════════════════════════════════════════════
if not args.inspect_only:
    from huggingface_hub import snapshot_download
    print("=" * 74)
    print(f"  下载 {args.model}")
    print(f"  缓存目录 {os.environ.get('HF_HOME', '~/.cache/huggingface')}")
    print(f"  Xet 后端 {'开' if os.environ['HF_HUB_DISABLE_XET'] == '0' else '★ 关（用老的 HTTP 路径，可断点续传）'}")
    print("=" * 74, flush=True)
    t0, path = time.time(), None
    for attempt in range(1, args.retries + 1):
        try:
            # max_workers=1：慢速线路上并发反而更容易断
            path = snapshot_download(args.model, max_workers=1)
            break
        except Exception as e:
            print(f"\n  ⚠️ 第 {attempt}/{args.retries} 次失败：{type(e).__name__}: {str(e)[:120]}")
            if attempt == args.retries:
                raise
            print(f"     10 秒后从断点续传…", flush=True)
            time.sleep(10)
    print(f"\n✅ 完成，用时 {(time.time()-t0)/60:.1f} 分钟"
          f"（重试 {attempt-1} 次）\n   → {path}\n")
else:
    from huggingface_hub import snapshot_download
    path = snapshot_download(args.model, local_files_only=True)

# ══════════════════════════════════════════════════════════
#  ② 看看下了些什么
# ══════════════════════════════════════════════════════════
print("─" * 74 + "\n  ① 下载的文件\n" + "─" * 74)
tot = 0
for f in sorted(os.listdir(path)):
    fp = os.path.join(path, f)
    if os.path.isfile(fp):
        sz = os.path.getsize(os.path.realpath(fp))
        tot += sz
        unit = f"{sz/1024**3:>7.2f} GB" if sz > 1024**3 else f"{sz/1024**2:>7.2f} MB"
        print(f"   {unit}   {f}")
print(f"   {'─'*10}\n   {tot/1024**3:>7.2f} GB   合计")

# ══════════════════════════════════════════════════════════
#  ③ 架构配置 —— 跟你自己的 201M 模型并排比
# ══════════════════════════════════════════════════════════
cfg = json.load(open(os.path.join(path, "config.json")))
MINE = dict(vocab_size=50257, hidden_size=896, num_hidden_layers=16,
            num_attention_heads=14, num_key_value_heads=14,
            intermediate_size=2432, max_position_embeddings=1024,
            rope_theta=10000, tie_word_embeddings=True)

print("\n" + "─" * 74 + "\n  ② 架构对比\n" + "─" * 74)
print(f"   {'':<28}{'你的 201M':>14}{'Qwen 1.5B':>14}   倍数")
print("   " + "─" * 64)
ROWS = [("词表 vocab_size", "vocab_size"), ("宽度 hidden_size C", "hidden_size"),
        ("层数 n_layer", "num_hidden_layers"), ("注意力头 n_head", "num_attention_heads"),
        ("★ KV 头 (GQA)", "num_key_value_heads"), ("MLP 宽度 d_ff", "intermediate_size"),
        ("上下文 block_size", "max_position_embeddings"), ("RoPE theta", "rope_theta")]
for label, k in ROWS:
    a, b = MINE.get(k), cfg.get(k)
    r = f"{b/a:>6.1f}×" if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a else "  —"
    print(f"   {label:<28}{str(a):>14}{str(b):>14}   {r}")
print(f"   {'权重共享 tie':<28}{str(MINE['tie_word_embeddings']):>14}"
      f"{str(cfg.get('tie_word_embeddings')):>14}")
print(f"   {'激活函数':<28}{'SwiGLU':>14}{str(cfg.get('hidden_act')):>14}")

hs, nl, ff, vs = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["intermediate_size"], cfg["vocab_size"]
kvh, ah = cfg.get("num_key_value_heads", cfg["num_attention_heads"]), cfg["num_attention_heads"]
hd = hs // ah
emb = vs * hs
attn = nl * (hs*hs + 2*hs*(kvh*hd) + hs*hs)     # q, k, v, o
mlp = nl * 3 * hs * ff                           # SwiGLU 三个矩阵
print(f"\n   参数量估算：词表 {emb/1e6:.0f}M + 注意力 {attn/1e6:.0f}M + MLP {mlp/1e6:.0f}M"
      f"  ≈ {(emb+attn+mlp)/1e9:.2f}B")

kv_mine = 2 * 16 * 14 * 64 * 2
kv_qwen = 2 * nl * kvh * hd * 2
print(f"\n   ★ KV cache 每 token：你 {kv_mine/1024:.0f} KB   Qwen {kv_qwen/1024:.0f} KB"
      f"   → GQA 省 {kv_mine/kv_qwen*(nl*ah)/(16*14):.1f}×（按同规模折算）")
print(f"     Qwen 满上下文 {cfg['max_position_embeddings']:,} token "
      f"= {kv_qwen*cfg['max_position_embeddings']/1024**3:.2f} GB")

# ══════════════════════════════════════════════════════════
#  ④ 分词器 + chat template
# ══════════════════════════════════════════════════════════
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(path)
print("\n" + "─" * 74 + "\n  ③ 分词器\n" + "─" * 74)
s = "Natalia sold 48 clips. 法国的首都是巴黎。"
ids = tok.encode(s)
print(f"   词表 {len(tok):,}（你的 GPT-2 分词器是 50,257，大 {len(tok)/50257:.1f} 倍）")
print(f"   测试串  {s!r}")
print(f"   → {len(ids)} 个 token: {[tok.decode([i]) for i in ids]}")
print(f"   ★ 中文能正常分词 —— 大词表的好处（GPT-2 分词器会把每个汉字拆成 2~3 个字节 token）")
print(f"\n   特殊 token: eos={tok.eos_token!r}({tok.eos_token_id})  pad={tok.pad_token!r}")

print("\n" + "─" * 74 + "\n  ④ ★ Chat template —— GRPO 里每条 prompt 都要走这个\n" + "─" * 74)
demo = tok.apply_chat_template(
    [{"role": "user", "content": "What is 2+2?"}],
    tokenize=False, add_generation_prompt=True)
for line in demo.split("\n"):
    print(f"   │ {line}")
print(f"\n   共 {len(tok.encode(demo))} 个 token（其中 {len(tok.encode(demo))-5} 个是格式脚手架）")
print("   ★ 对比你的 Alpaca 格式：'### Instruction:\\n…\\n\\n### Response:\\n'")
print("     作用完全一样 —— 告诉模型『现在该你说话了』，只是标记不同")
