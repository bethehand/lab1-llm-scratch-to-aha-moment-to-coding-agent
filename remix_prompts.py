#!/usr/bin/env python3
"""
remix_prompts.py —— 把教材的提示词换成 prompts_cd 的随机改写（response 不动），教「做法不绑模板、没说明也会用工具」。

    python3 remix_prompts.py sft_cd4_ask.jsonl --out sft_cd4_ask_mix.jsonl
    python3 remix_prompts.py sft_cd4_tool.jsonl --out sft_cd4_tool_mix.jsonl
    python3 remix_prompts.py sft_cd3_tool.jsonl --out sft_cd3_tool_mix.jsonl
每行：随机模板（--set train）、工具说明以 --p-tool 概率出现、求助说明以 --p-ask 概率出现、措辞随机；求助次数上限写 --max-asks。
没有 nums/target 的行（GSM）原样保留。多出 pid / tool_hint / ask_hint 三个字段。
"""
import os, sys, json, random, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompts_cd as P

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("--out", required=True)
ap.add_argument("--set", default="train", choices=list(P.SETS))
ap.add_argument("--p-tool", type=float, default=0.7)
ap.add_argument("--p-ask", type=float, default=0.7)
ap.add_argument("--max-asks", type=int, default=3)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
rng = random.Random(args.seed)
rows = [json.loads(l) for l in open(args.src)]
cnt = collections.Counter(); n_cd = 0
with open(args.out, "w") as f:
    for r in rows:
        if "nums" in r and "target" in r:
            r["prompt"], meta = P.pick(rng, r["nums"], r["target"], set=args.set, p_tool=args.p_tool, p_ask=args.p_ask, max_asks=args.max_asks)
            r.update(meta); n_cd += 1
            cnt[f"pid{meta['pid']}"] += 1; cnt["tool_hint"] += meta["tool_hint"]; cnt["ask_hint"] += meta["ask_hint"]
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{args.src} → {args.out}：{len(rows)} 行，换了提示词 {n_cd} 行（集合 {args.set}，上限 {args.max_asks} 次）")
print("  模板分布：", {k: v for k, v in sorted(cnt.items()) if k.startswith("pid")})
print(f"  带工具说明 {cnt['tool_hint']}/{n_cd} = {cnt['tool_hint']/max(1,n_cd):.2f}   带求助说明 {cnt['ask_hint']}/{n_cd} = {cnt['ask_hint']/max(1,n_cd):.2f}")
