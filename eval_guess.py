#!/usr/bin/env python3
"""
eval_guess.py —— E 猜数字的探测 / 评测：n 个秘密数 × s 次采样，走 harness.generate_with_env，打行为读数，落盘 raw。

    # 探测 Base（HF 路径，不用导出）
    CUDA_VISIBLE_DEVICES=0 python3 eval_guess.py --engine hf --model Qwen/Qwen2.5-1.5B -n 100 -s 8 --out guess_base.json
    # 探测 / 评测导出过的模型（vLLM）
    CUDA_VISIBLE_DEVICES=0 python3 eval_guess.py --engine vllm --hf-dir hf_armA --ckpt out_armA/ckpt_latest.pt -n 100 -s 8 --out guess_armA.json
    python3 eval_guess.py --analyze guess_armA_raw.json          # 只读 raw 重算读数，不载模型
读数：写了 guess 的比例 / 猜中率 / 平均轮数（猜中的）/ 轮数分布 / 方向一致率 / 落在可行区间 / 二分得分 / invalid / 重复 / 假观测 / 撞顶 / 有对有错的组
"""
import os, sys, json, time, argparse, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import guess_env as G

ap = argparse.ArgumentParser()
ap.add_argument("--engine", choices=["hf", "vllm"], default="vllm")
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
ap.add_argument("--ckpt", default=None, help="hf 引擎：覆盖权重；vllm 引擎：只用来起名")
ap.add_argument("--hf-dir", default=None)
ap.add_argument("-n", "--n-problems", type=int, default=100)
ap.add_argument("-s", "--samples", type=int, default=8)
ap.add_argument("--N", type=int, default=100, help="秘密数范围 [1, N]")
ap.add_argument("--T", type=int, default=10, help="最多猜几次")
ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--max-new", type=int, default=512)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--gpu-mem", type=float, default=0.6)
ap.add_argument("--prompt-set", default="train", choices=list(G.SETS))
ap.add_argument("--custom-prompt", default=None, help="★ 措辞消融：直接给一段模板文本（含 {N} {T} 占位，结尾自己带 <think>），所有题都用它，pid 记 -1")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--show", type=int, default=3)
ap.add_argument("--out", default=None)
ap.add_argument("--analyze", default=None, help="只分析这份 raw")
args = ap.parse_args()


def report(raw, show=3):
    env = G.GuessEnv(N=raw["N"], T=raw["T"])
    R = raw["records"]; n = len(R)
    S = raw["n_samples"]
    for r in R:                                                       # ★ 用当前的奖励公式重算（瞎答撞中不给分那次改动之后，旧 raw 也能对上）
        r["r"], r["d"] = env.score(r["state"], r["txt"])
    corr = [r["d"]["correct"] for r in R]
    found = [r["state"]["found"] for r in R]
    turns_found = [r["state"]["turns"] for r in R if r["d"]["correct"] > 0]
    beh = [G.behavior(r["txt"], raw["N"]) for r in R]
    agg = lambda k: sum(b[k + "_ok"] for b in beh) / max(1, sum(b[k + "_n"] for b in beh))
    by = collections.defaultdict(list)
    for r in R: by[r["pi"]].append(r["d"]["correct"])
    groups = collections.Counter("全对" if sum(v) == len(v) else "全错" if sum(v) == 0 else "有对有错" for v in by.values())
    hist = collections.Counter(t for t in turns_found)
    print(f"\n{raw.get('name','')}   {raw['n']} 个秘密数 × {S}   N={raw['N']} T={raw['T']}  温度 {raw['temp']}  模板 {raw['prompt_set']}")
    print(f"  写了 <guess> {np.mean([r['d']['fmt'] for r in R]):.3f}   ★ 猜中率（answer == 秘密数）{np.mean(corr):.3f}   环境说 correct 的 {np.mean(found):.3f}   有 <answer> {np.mean([r['d']['has_answer'] for r in R]):.3f}")
    print(f"  ★ 平均轮数（猜中的）{np.mean(turns_found) if turns_found else float('nan'):.2f}（二分 ≤ {int(np.ceil(np.log2(raw['N'])))}，log₂N = {np.log2(raw['N']):.1f}）"
          f"   轮数分布 {dict(sorted(hist.items()))}")
    print(f"  ★ 方向一致率 {agg('dir'):.3f}（{sum(b['dir_n'] for b in beh)} 次）   落在可行区间 {agg('in_range'):.3f}   二分得分 {agg('bisect'):.3f}")
    print(f"  每条：轮数 {np.mean([r['state']['turns'] for r in R]):.2f}   invalid {np.mean([r['state']['invalid'] for r in R]):.2f}   重复 {np.mean([r['state']['repeat'] for r in R]):.2f}"
          f"   假观测 {sum(r['fake'] for r in R)}   撞顶 {sum(r['cut'] for r in R)}   token {np.mean([r['L'] for r in R]):.0f}")
    print(f"  组（每个秘密数 {S} 条）：{dict(groups)}   ← 有对有错的组才有梯度")
    pids = sorted(set(r["pid"] for r in R))
    if len(pids) > 1:                                                 # ★ 按模板拆：哪句话下掉得多
        for pid in pids:
            rr = [r for r in R if r["pid"] == pid]; bb = [G.behavior(r["txt"], raw["N"]) for r in rr]
            dv = lambda k: sum(b[k + "_ok"] for b in bb) / max(1, sum(b[k + "_n"] for b in bb))
            print(f"    模板 {pid:>2}：{len(rr):>4} 条  猜中 {np.mean([r['d']['correct'] for r in rr]):.3f}  方向一致 {dv('dir'):.3f}  二分 {dv('bisect'):.3f}"
                  f"  假观测 {sum(r['fake'] for r in rr):>3}  撞顶 {sum(r['cut'] for r in rr):>3}  中了还猜 {sum(1 for r in rr if r['state']['found'] and r['d']['correct'] == 0):>3}")
    if show:
        ok = sorted([r for r in R if r["d"]["correct"] > 0], key=lambda r: r["state"]["turns"])
        bad = [r for r in R if r["d"]["correct"] == 0]
        for tag, r in [("最短的一条对的", ok[0] if ok else None), ("一条错的", bad[0] if bad else None)] + [("再一条", ok[len(ok)//2] if len(ok) > 2 else None)][: max(0, show - 2)]:
            if r is None: continue
            print("─" * 90); print(f"【{tag}】秘密数 {r['secret']}  模板 {r['pid']}  得分 {r['r']:.2f}  轮数 {r['state']['turns']}  {r['L']} token" + ("  ⚠ 假观测" if r["fake"] else "") + ("  ⚠ 撞顶" if r["cut"] else ""))
            print(r["txt"])


if args.analyze:
    report(json.load(open(args.analyze)), args.show); sys.exit(0)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import calc_tool_vllm as T
import harness as H

tok = AutoTokenizer.from_pretrained(args.model); tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.engine == "vllm":
    assert args.hf_dir, "--engine vllm 要 --hf-dir"
    be = T.make_backend("vllm", tok=tok, hf_dir=args.hf_dir, gpu_mem=args.gpu_mem, max_model_len=args.max_new + 400)
    name = os.path.basename(args.hf_dir.rstrip("/"))
else:
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model = model.to(dev).eval()
    if args.ckpt:
        c = torch.load(args.ckpt, map_location="cpu", weights_only=False); model.load_state_dict(c["model"]); model.to(dev).eval(); del c
    be = T.make_backend("hf", model=model, tok=tok, dev=dev, gen_bs=args.bs)
    name = args.ckpt or args.model
stem = (args.ckpt or args.hf_dir or args.model).replace("/", "_").replace(".pt", "")
args.out = args.out or f"guess_{stem}_N{args.N}_s{args.samples}_p{'custom' if args.custom_prompt else args.prompt_set}.json"

env = G.GuessEnv(N=args.N, T=args.T)
P = G.make_problems(args.n_problems, N=args.N, seed=args.seed)
ids = G.SETS[args.prompt_set]
flat = [(p["idx"], ids[p["idx"] % len(ids)], p["secret"]) for p in P for _ in range(args.samples)]
if args.custom_prompt:
    flat = [(pi, -1, sec) for pi, _, sec in flat]
    _render = lambda pid, N, T: args.custom_prompt.replace("{N}", str(N)).replace("{T}", str(T)).replace("\\n", "\n")
    args.prompt_set = "custom"
else:
    _render = G.render
print(f"{name}   {len(P)} 个秘密数 × {args.samples}   N={args.N} T={args.T} 温度 {args.temp}   模板 {args.prompt_set}\n  提示词样例（模板 {flat[0][1]}）：\n{_render(flat[0][1], args.N, args.T)}\n", flush=True)
rec, t0 = [], time.time()
for s in range(0, len(flat), args.bs if args.engine == "hf" else len(flat)):
    chunk = flat[s: s + (args.bs if args.engine == "hf" else len(flat))]
    states = [env.reset(sec) for _, _, sec in chunk]
    outs = H.generate_with_env(be, tok, [_render(pid, args.N, args.T) for _, pid, _ in chunk], env, states, max_new=args.max_new, temp=args.temp,
                               gen_bs=args.bs, dev=dev, progress=(args.engine == "vllm"))
    for (pi, pid, sec), o in zip(chunk, outs):
        r, d = env.score(o["state"], o["txt"])
        rec.append(dict(pi=pi, pid=pid, secret=sec, txt=o["txt"], r=r, d=d, L=len(o["ids"]), fake=o["fake"], cut=o["cut"], cont=o["cont"], state=o["state"]))
    print(f"  {len(rec)}/{len(flat)}  {time.time()-t0:.0f} s", flush=True)
raw = dict(name=name, n=len(P), n_samples=args.samples, N=args.N, T=args.T, temp=args.temp, prompt_set=args.prompt_set, seed=args.seed, max_new=args.max_new,
           time=time.time() - t0, records=rec)
json.dump(raw, open(args.out.replace(".json", "_raw.json"), "w"))
print(f"\n  ★ raw → {args.out.replace('.json', '_raw.json')}")
report(raw, args.show)
