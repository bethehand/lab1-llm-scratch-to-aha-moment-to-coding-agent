#!/usr/bin/env python3
"""
screen_pool.py —— 用当前模型把训练池筛一遍，只留「有对有错」的题（DAPO dynamic sampling 的离线版）。vllm_env 里跑，一张卡。

    python3 screen_pool.py --hf-dir hf_sft_tool --data data/countdown4.json --n 8000 --lo 0 --hi 90000 -k 8 --out data/countdown4_mixed.json

从 [lo:hi] 随机抽 n 道，每道采 k 条（温度 1，工具 harness），按答对条数分三堆：全对 / 有对有错 / 全错。
输出 --out：有对有错的题（原字段 + 池内下标 idx + 对的条数 c），训练器 --data 直接用；旁边多存一份 *_counts.json（每题的 c）方便以后换池对比。
读法：有对有错的比例 ≈ 评测片上的 14~17%；全对 + 全错的题在 v3 判分下 16 条全零、纯浪费，所以不进池。
"""
import os, json, time, random, argparse
import numpy as np
from transformers import AutoTokenizer
import countdown as CD
import calc_tool_vllm as calc_tool

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
ap.add_argument("--hf-dir", default=None, help="筛的时候必填；--merge 不用")
ap.add_argument("--data", default="data/countdown4.json")
ap.add_argument("--n", type=int, default=8000)
ap.add_argument("--lo", type=int, default=0)
ap.add_argument("--hi", type=int, default=90000, help="别碰 [90000:] 的验证和 [95000:] 的池外测试")
ap.add_argument("-k", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-new", type=int, default=1400)
ap.add_argument("--gpu-mem", type=float, default=0.6)
ap.add_argument("--out", required=True)
ap.add_argument("--ask", action="store_true", help="★ 臂 A：提示词从 prompts_cd[train] 随机抽（说明各 --p-hint 概率出现），harness 开求助（专家 80/20 混），跟 RL 的环境一致。要 DEEPSEEK_API_KEY")
ap.add_argument("--max-asks", type=int, default=3)
ap.add_argument("--ask-weak-frac", type=float, default=0.2)
ap.add_argument("--p-hint", type=float, default=0.7)
ap.add_argument("--shard", default=None, help="★ 多卡：k/N，抽同一批题后只筛第 k 片，输出名带 _shardk；跑完 --merge 合并")
ap.add_argument("--merge", action="store_true", help="★ 把各片 *_shardk.json 合成 --out，不载模型")
args = ap.parse_args()

def shard_name(k):
    return args.out.replace(".json", f"_shard{k}.json")

if args.merge:                                                    # ── 合并模式 ──
    import glob
    files = sorted(glob.glob(args.out.replace(".json", "_shard*.json")))
    files = [f for f in files if "_counts" not in f]
    assert files, f"没找到分片文件 {args.out.replace('.json', '_shard*.json')}：四片还没跑完，或者跑挂了（看 screen_ask*.txt 的尾巴）"
    mixed, counts, n_all = [], {}, 0
    for f in files:
        mixed += json.load(open(f))
        c = json.load(open(f.replace(".json", "_counts.json")))
        counts.update(c["counts"]); k_ = c["k"]
    n_all = len(counts)                                            # 每片记的 n 是原始 --n，不能加；按题数数
    cv = np.array(list(counts.values()))
    allr, allw = int((cv == k_).sum()), int((cv == 0).sum())
    print(f"★ 合并 {len(files)} 片：{n_all} 道   全对 {allr} ({100*allr/n_all:.1f}%)   有对有错 {len(mixed)} ({100*len(mixed)/n_all:.1f}%)   全错 {allw} ({100*allw/n_all:.1f}%)")
    print("  对的条数分布 c=0..k：", np.bincount(cv, minlength=k_ + 1).tolist())
    json.dump(mixed, open(args.out, "w"))
    json.dump({"merged_from": files, "n": n_all, "k": k_, "counts": counts}, open(args.out.replace(".json", "_counts.json"), "w"))
    print(f"  → {args.out}（{len(mixed)} 道，训练器 --data 直接用，配 --val-data data/countdown4.json）")
    raise SystemExit

assert args.hf_dir, "筛池要 --hf-dir（export_hf.py 导出的目录）"
tok = AutoTokenizer.from_pretrained(args.model)
D = json.load(open(args.data))
rng = random.Random(args.seed)
idx = sorted(rng.sample(range(args.lo, min(args.hi, len(D))), args.n))
if args.shard:                                                    # ── 分片：同一批题，各取 1/N ──
    k, N = map(int, args.shard.split("/"))
    per = len(idx) // N
    idx = idx[k * per: (k + 1) * per] if k < N - 1 else idx[k * per:]
    args.out = shard_name(k)
P = [D[i] for i in idx]
be = calc_tool.make_backend("vllm", tok=tok, hf_dir=args.hf_dir, gpu_mem=args.gpu_mem, max_model_len=args.max_new + 400)
if args.ask:                                                       # ★ 跟 grpo --ask 同一套：每题每次抽一份提示词，求助走混合专家
    import prompts_cd as PC, deepseek_tool as DS, random as _r
    DS.get_key()
    prng = _r.Random(args.seed + 7)
    prompts = [PC.pick(prng, p["nums"], p["target"], set="train", p_tool=args.p_hint, p_ask=args.p_hint, max_asks=args.max_asks)[0] for p in P for _ in range(args.k)]
    ask_kw = dict(ask=True, max_asks=args.max_asks,
                  ask_fn=lambda q: calc_tool.ask_expert(q, model="deepseek-chat" if _r.random() < args.ask_weak_frac else "deepseek-reasoner"))
else:
    prompts = [calc_tool.tool_prompt(p["nums"], p["target"]) for p in P for _ in range(args.k)]
    ask_kw = {}
print(f"筛 {len(P)} 道 × {args.k} 条 = {len(prompts)}  ← {args.hf_dir}", flush=True)
t0 = time.time()
res = calc_tool.generate_with_tools(be, tok, prompts, max_new=args.max_new, temp=1.0, progress=True, **ask_kw)
el = time.time() - t0
c = np.zeros(len(P), dtype=int)
for j, o in enumerate(res):
    pi = j // args.k
    c[pi] += CD.reward(o["txt"], P[pi]["nums"], P[pi]["target"])[1]["correct"] > 0
allr, allw = int((c == args.k).sum()), int((c == 0).sum())
mixed = [dict(P[i], idx=idx[i], c=int(c[i])) for i in range(len(P)) if 0 < c[i] < args.k]
print(f"  {el/60:.1f} 分   全对 {allr} ({100*allr/len(P):.1f}%)   有对有错 {len(mixed)} ({100*len(mixed)/len(P):.1f}%)   全错 {allw} ({100*allw/len(P):.1f}%)")
hist = np.bincount(c, minlength=args.k + 1)
print("  对的条数分布 c=0..k：", hist.tolist())
print(f"  有对有错里 c 的分布：", {int(v): int(n) for v, n in zip(*np.unique([m['c'] for m in mixed], return_counts=True))} if mixed else "—")
json.dump(mixed, open(args.out, "w"))
json.dump({"hf_dir": args.hf_dir, "data": args.data, "lo": args.lo, "hi": args.hi, "n": args.n, "k": args.k, "seed": args.seed,
           "counts": {int(i): int(v) for i, v in zip(idx, c)}}, open(args.out.replace(".json", "_counts.json"), "w"))
print(f"  → {args.out}（{len(mixed)} 道，训练器 --data 直接用，配 --val-data {args.data}）")
