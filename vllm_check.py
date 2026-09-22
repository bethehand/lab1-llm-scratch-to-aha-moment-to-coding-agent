#!/usr/bin/env python3
"""
vllm_check.py —— 换引擎的等价性测试：同一个 ckpt，HF 后端 vs vLLM 后端。要在装了 vLLM 的 venv 里跑，占一张空卡。

    python3 vllm_check.py --ckpt out_sft_tool/ckpt.pt --hf-dir hf_sft_tool
    python3 vllm_check.py --ckpt out_sft_tool/ckpt.pt --hf-dir hf_sft_tool --n-greedy 20 --n-sample 100 -s 8

① 贪心：n 道题，两个后端各生成一条，逐 token 比。bf16 算子不同允许在某个 token 之后分叉，报分叉率和平均分叉位置
② 温度 1：n 道 × s 条，vLLM 跑一遍报 pass@1 / 调用/条 / 报错 / 假结果 / 撞顶 / 用时；HF 可选（--hf-sample）跑同样的量比速度
"""
import time, json, argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import countdown as CD
import calc_tool_vllm as calc_tool

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
ap.add_argument("--ckpt", default=None)
ap.add_argument("--hf-dir", required=True)
ap.add_argument("--data", default="data/countdown4.json")
ap.add_argument("--offset", type=int, default=95000)
ap.add_argument("--n-greedy", type=int, default=20)
ap.add_argument("--n-sample", type=int, default=100)
ap.add_argument("-s", "--samples", type=int, default=8)
ap.add_argument("--max-new", type=int, default=1400)
ap.add_argument("--gpu-mem", type=float, default=0.45, help="vLLM 占显存比例（HF 模型也在同一张卡上，别开太大）")
ap.add_argument("--hf-sample", action="store_true", help="温度 1 那段也用 HF 跑一遍比速度")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model); tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
D = json.load(open(args.data))
P = D[args.offset: args.offset + max(args.n_greedy, args.n_sample)]
prompts = [calc_tool.tool_prompt(p["nums"], p["target"]) for p in P]

print("载入 HF 模型 …", flush=True)
try:
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
if args.ckpt:
    c = torch.load(args.ckpt, map_location="cpu", weights_only=False); model.load_state_dict(c["model"]); del c
model = model.cuda().eval()
hf = calc_tool.HFBackend(model, tok)
print("载入 vLLM …", flush=True)
vl = calc_tool.VLLMBackend(args.hf_dir, tok, gpu_mem=args.gpu_mem, max_model_len=args.max_new + 400)


def stats(res, probs):
    ok = [CD.reward(o["txt"], p["nums"], p["target"])[1]["correct"] for o, p in zip(res, probs)]
    L = np.array([len(o["ids"]) for o in res])
    return dict(pass1=float(np.mean(ok)), calls=float(np.mean([o["n_calls"] for o in res])),
                err=sum(o["n_err"] for o in res), fake=sum(o["fake"] for o in res),
                cut=int((L >= args.max_new - 2).sum()), len_med=float(np.median(L)))


# ── ① 贪心等价 ──
n = args.n_greedy
print(f"\n① 贪心 {n} 道：HF vs vLLM 逐 token 比")
t0 = time.time(); a = calc_tool.generate_with_tools(hf, tok, prompts[:n], max_new=args.max_new, temp=0.0); ta = time.time() - t0
t0 = time.time(); b = calc_tool.generate_with_tools(vl, tok, prompts[:n], max_new=args.max_new, temp=0.0); tb = time.time() - t0
same, first_div, shown = 0, [], 0
for x, y in zip(a, b):
    if x["ids"] == y["ids"]:
        same += 1
    else:
        k = next((i for i, (u, v) in enumerate(zip(x["ids"], y["ids"])) if u != v), min(len(x["ids"]), len(y["ids"])))
        first_div.append(k)
        if shown < 3:                                   # ★ 看分叉处：是中途算出不同的字，还是只差结尾的 EOS 记法
            shown += 1
            print(f"   分叉样例 {shown}：位置 {k}  HF 长 {len(x['ids'])} / vLLM 长 {len(y['ids'])}")
            print(f"      HF   ids[{k-2}:{k+3}] = {x['ids'][max(0,k-2):k+3]}  尾 {repr(tok.decode(x['ids'][max(0,k-2):], skip_special_tokens=False))[:80]}")
            print(f"      vLLM ids[{k-2}:{k+3}] = {y['ids'][max(0,k-2):k+3]}  尾 {repr(tok.decode(y['ids'][max(0,k-2):], skip_special_tokens=False))[:80]}")
sa, sb = stats(a, P[:n]), stats(b, P[:n])
print(f"   完全相同 {same}/{n}   分叉的 {len(first_div)} 条：首个分叉位置 中位 {int(np.median(first_div)) if first_div else '—'}（token）")
print(f"   HF   pass@1 {sa['pass1']:.2f} 调用/条 {sa['calls']:.1f} 报错 {sa['err']} 假 {sa['fake']} 撞顶 {sa['cut']} 长度中位 {sa['len_med']:.0f}   {ta:.0f}s")
print(f"   vLLM pass@1 {sb['pass1']:.2f} 调用/条 {sb['calls']:.1f} 报错 {sb['err']} 假 {sb['fake']} 撞顶 {sb['cut']} 长度中位 {sb['len_med']:.0f}   {tb:.0f}s   ← 提速 {ta/max(tb,1e-9):.1f}×")

# ── ② 温度 1 ──
n, s = args.n_sample, args.samples
if n == 0:
    raise SystemExit("（--n-sample 0，跳过温度 1）")
flat = [(i, prompts[i]) for i in range(n) for _ in range(s)]
print(f"\n② 温度 1：{n} 道 × {s} 条 = {len(flat)}")
t0 = time.time(); b = calc_tool.generate_with_tools(vl, tok, [p for _, p in flat], max_new=args.max_new, temp=1.0); tb = time.time() - t0
sb = stats(b, [P[i] for i, _ in flat])
print(f"   vLLM pass@1 {sb['pass1']:.3f} 调用/条 {sb['calls']:.1f} 报错 {sb['err']} 假 {sb['fake']} 撞顶 {sb['cut']} ({100*sb['cut']/len(flat):.1f}%) 长度中位 {sb['len_med']:.0f}   {tb/60:.1f} 分")
if args.hf_sample:
    t0 = time.time(); a = calc_tool.generate_with_tools(hf, tok, [p for _, p in flat], max_new=args.max_new, temp=1.0, gen_bs=32); ta = time.time() - t0
    sa = stats(a, [P[i] for i, _ in flat])
    print(f"   HF   pass@1 {sa['pass1']:.3f} 调用/条 {sa['calls']:.1f} 报错 {sa['err']} 假 {sa['fake']} 撞顶 {sa['cut']} ({100*sa['cut']/len(flat):.1f}%) 长度中位 {sa['len_med']:.0f}   {ta/60:.1f} 分   ← 提速 {ta/max(tb,1e-9):.1f}×")
print("\n读法：贪心分叉率低、温度 1 的 pass@1 和调用/条落在 ±0.05 内 → 换引擎成功；假结果、报错应为 0")
