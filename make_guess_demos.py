#!/usr/bin/env python3
"""
make_guess_demos.py —— E 撒种臂的教材：二分搜索的示范轨迹（<obs> 段训练时 −100）。
    python3 make_guess_demos.py --n 40 --out sft_guess.jsonl
默认 80% 严格二分（每次猜当前区间中点）、20% 「粗糙」的（中点 ± 随机偏移，仍然只在可行区间内、仍用观测），给 RL 留「哪种更好」的选择。
每条：随机训练模板；think 里每轮写「The number is between lo and hi. Guess the middle: <guess>m</guess><obs>…</obs>」，猜中后 </think> + <answer>。
"""
import json, random, argparse
import guess_env as G

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--N", type=int, default=100)
ap.add_argument("--T", type=int, default=10)
ap.add_argument("--sloppy", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
args = ap.parse_args()
rng = random.Random(args.seed)

def episode(secret, sloppy):
    lo, hi, lines, turns = 1, args.N, [], 0
    lines.append(f"The number is between 1 and {args.N}. I will narrow the range with each guess.")
    while True:
        mid = (lo + hi) // 2
        if sloppy and hi - lo >= 4:
            mid = max(lo, min(hi, mid + rng.randint(-(hi - lo) // 4, (hi - lo) // 4)))
        turns += 1
        if mid == secret:
            lines.append(f"The number is between {lo} and {hi}. Guess the middle: <guess>{mid}</guess><obs>correct</obs>")
            lines.append(f"Found it in {turns} guesses.")
            return "\n".join(lines) + f"\n</think>\n<answer>{secret}</answer>", turns
        obs = "higher" if secret > mid else "lower"
        lines.append(f"The number is between {lo} and {hi}. Guess the middle: <guess>{mid}</guess><obs>{obs}</obs>")
        if obs == "higher":
            lo = mid + 1; lines.append(f"So the number is larger than {mid}.")
        else:
            hi = mid - 1; lines.append(f"So the number is smaller than {mid}.")
        if turns >= args.T:
            return None, turns

rows, tries = [], 0
while len(rows) < args.n and tries < 10 * args.n:
    tries += 1
    secret = rng.randint(1, args.N); sloppy = rng.random() < args.sloppy
    resp, turns = episode(secret, sloppy)
    if resp is None: continue
    pid = rng.choice(G.SETS["train"])
    rows.append(dict(prompt=G.render(pid, args.N, args.T), response=resp, secret=secret, pid=pid, turns=turns, sloppy=sloppy, task="guess"))
with open(args.out, "w") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{len(rows)} 条 → {args.out}   平均轮数 {sum(r['turns'] for r in rows)/len(rows):.2f}   粗糙 {sum(r['sloppy'] for r in rows)}   模板 {sorted(set(r['pid'] for r in rows))}")
print("─── 样例 ───"); print(rows[0]["prompt"]); print(rows[0]["response"])
