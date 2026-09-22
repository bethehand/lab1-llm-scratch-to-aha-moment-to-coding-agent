#!/usr/bin/env python3
"""
chat_guess_vllm.py —— 猜数字的原样对话：你贴什么模型就看什么（不加任何提示词），harness 只管把 <obs> 贴回去。
    CUDA_VISIBLE_DEVICES=0 python3 chat_guess_vllm.py --hf-dir hf_E2 [--temp 1 -s 1 --N 100 --T 10]
每局：先输秘密数（回车 = 随机；「230 1000」= 秘密数 230 且这局范围 1~1000），再粘贴提示词（空行回车发送），q 退出。打原样 + 标注版（⟦⟧ 是环境注入的观测），尾巴给轮数 / 猜中 / 无效 / 得分。
提示词里的范围和轮数要和 --N --T 一致，模型看不到环境的设置。
"""
import argparse, random, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoTokenizer
import calc_tool_vllm as T
import harness as H
import guess_env as G

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
ap.add_argument("--hf-dir", required=True)
ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("-s", "--samples", type=int, default=1)
ap.add_argument("--N", type=int, default=100)
ap.add_argument("--T", type=int, default=10)
ap.add_argument("--max-new", type=int, default=512)
ap.add_argument("--gpu-mem", type=float, default=0.6)
ap.add_argument("--no-annot", action="store_true")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
be = T.make_backend("vllm", tok=tok, hf_dir=args.hf_dir, gpu_mem=args.gpu_mem, max_model_len=args.max_new + 600)
env = G.GuessEnv(N=args.N, T=args.T)
print(f"\n★ 引擎就绪 ← {args.hf_dir}   温度 {args.temp}   每次 {args.samples} 条   范围 1~{args.N}   最多 {args.T} 轮   先输秘密数（回车随机），再粘贴提示词，空行回车发送，q 退出\n")


def annotate(txt, spans_txt):
    out, pos = "", 0
    for a, b in spans_txt:
        out += txt[pos:a] + "⟦" + txt[a:b] + "⟧"; pos = b
    return out + txt[pos:]


while True:
    try:
        sec = input("秘密数> ").strip()
    except EOFError:
        break
    if sec == "q":
        break
    parts = sec.split()
    N = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else args.N          # ★「230 1000」= 秘密数 230、这局范围 1~1000；只写一个数沿用 --N
    env = G.GuessEnv(N=N, T=args.T)
    secret = int(parts[0]) if parts and parts[0].isdigit() else random.randint(1, N)
    print(f"（这局的秘密数 {secret}，范围 1~{N}，最多 {args.T} 轮）\n提示词> ", end="", flush=True)
    lines = []
    while True:
        try:
            l = input()
        except EOFError:
            l = "q"
        if l == "q":
            sys.exit(0)
        if l == "" and lines:
            break
        lines.append(l)
    prompt = "\n".join(lines)
    states = [env.reset(secret) for _ in range(args.samples)]
    outs = H.generate_with_env(be, tok, [prompt] * args.samples, env, states, max_new=args.max_new, temp=args.temp)
    for i, o in enumerate(outs):
        r, d = env.score(o["state"], o["txt"])
        print("\n" + "─" * 78 + (f"  第 {i+1} 条" if args.samples > 1 else "")); print(o["txt"])
        if not args.no_annot and o["spans"]:
            spans_txt = []
            for s0, l0 in o["spans"]:
                a = len(tok.decode(o["ids"][:s0], skip_special_tokens=True)); b = len(tok.decode(o["ids"][:s0 + l0], skip_special_tokens=True))
                spans_txt.append((a, b))
            print("· · · 标注版 · · ·"); print(annotate(o["txt"], spans_txt))
        st = o["state"]
        print(f"  {len(o['ids'])} token   轮数 {st['turns']}   猜过 {st['guesses']}   观测 {st['obs']}   猜中 {'是' if d['correct'] else '否'}   得分 {r:.2f}"
              + ("   ⚠ 自己编了 <obs>" if o["fake"] else "") + ("   ⚠ 撞顶" if o["cut"] else ""))
    print()
