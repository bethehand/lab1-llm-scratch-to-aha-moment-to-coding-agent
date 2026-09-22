#!/usr/bin/env python3
"""
GRPO 第 1 步：测基线。不训练，只采样 + 判分。

它回答三个决定后续所有配置的问题：
  ① 正确率落在 20~60% 的黄金区间吗？（太低没信号，太高没空间）
  ② ★ 有多少题能产生梯度？（8 次采样全对或全错的题 → 优势全 0 → 白跑）
  ③ 采样有多快？（RL 80% 的时间花在这，直接决定训练要几小时）

    python3 baseline_gsm8k.py                    # 200 题 × 8 采样
    python3 baseline_gsm8k.py -n 50              # 快速试跑
    python3 baseline_gsm8k.py --temp 0.7         # 换温度看信号率
"""
import os, re, time, json, argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--ckpt", default=None,
                help="★ GRPO 训练出的 checkpoint（out_grpo_lr5e6/ckpt_latest.pt）"
                     "，不给就用原始模型")
ap.add_argument("-n", "--n-problems", type=int, default=200)
ap.add_argument("--offset", type=int, default=0,
                help="★ 从第几道题开始。训练时的评估用了 TEST[0:200]，"
                     "所以无偏测量要用 --offset 200")
ap.add_argument("-s", "--samples", type=int, default=8,
                help="★ 每题采样几次（统计学里的 n）。注意这【不是】pass@k 的 k —— "
                     "k 是'允许交几份答案'，是从同一批采样里算出来的多个指标")
ap.add_argument("-k", "--k", type=int, default=None,
                help="⚠️ 旧名，等同 --samples。保留只为兼容旧命令")
ap.add_argument("--temp",     type=float, default=1.0, help="★ 训练时的温度，不是评估温度")
ap.add_argument("--top-p",    type=float, default=1.0)
ap.add_argument("--max-new",  type=int,   default=1024)  # ★ RL 会让回答变长，留余量
ap.add_argument("--bs",       type=int,   default=32,  help="一次并行几条序列")
ap.add_argument("--out",      default=None, help="不给就自动按 ckpt 命名")
args = ap.parse_args()

if args.k is not None:                     # 兼容旧命令行
    args.samples = args.samples
    print(f"⚠️ -k 是旧名（它是每题采样次数 n，不是 pass@k 的 k）→ 已按 --samples {args.samples} 处理")

dev = "cuda" if torch.cuda.is_available() else "cpu"
assert dev == "cuda", "需要 GPU"
if args.out is None:
    stem = ("base" if not args.ckpt else
            args.ckpt.replace("/", "_").replace(".pt", ""))
    args.out = f"pk_{stem}_k{args.samples}_off{args.offset}.json"

# ── 数据 ──
from datasets import load_dataset
ds = load_dataset("openai/gsm8k", "main", split="test")
ALL = [(d["question"], d["answer"].split("####")[-1].strip().replace(",", "")) for d in ds]
probs = ALL[args.offset: args.offset + args.n_problems]
tag = ("★ 从没用过（无偏）" if args.offset >= 200 else "⚠️ 训练时评估用过（有偏）")
print(f"GSM8K test 共 {len(ds):,} 题，取 TEST[{args.offset}:{args.offset+len(probs)}]"
      f"  {len(probs)} 题   {tag}")

# ── 模型 ──
def load_model(path, dtype, device):
    """
    不用 device_map —— 模型只有 3 GB，单卡装得下，device_map 需要 accelerate 还没好处。
    dtype 这个参数名在新版 transformers 里从 torch_dtype 改成了 dtype，两个都试一遍。
    """
    try:
        m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
    return m.to(device).eval()


print(f"载入 {args.model} …", flush=True)
tok = AutoTokenizer.from_pretrained(args.model)
tok.padding_side = "left"                       # ★ 生成必须左填充
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = load_model(args.model, torch.bfloat16, dev)
if args.ckpt:
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.to(dev).eval()
    print(f"  ★ 覆盖成 GRPO 权重 {args.ckpt}（step {ck.get('step','?')}）")
    del ck; torch.cuda.empty_cache()
n_param = sum(p.numel() for p in model.parameters())
print(f"  {n_param/1e9:.2f}B 参数   显存 {torch.cuda.memory_allocated()/1024**3:.1f} GB")

INSTR = "\n\nSolve this step by step. End your answer with '#### <number>'."


def make_prompt(q):
    return tok.apply_chat_template([{"role": "user", "content": q + INSTR}],
                                   tokenize=False, add_generation_prompt=True)


from _reward import reward, extract, has_format    # ★ 判分器抽成独立模块，见 _reward.py


# ── 采样 ──
@torch.no_grad()
def gen(prompts):
    enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
    out = model.generate(**enc, max_new_tokens=args.max_new,
                         do_sample=args.temp > 0, temperature=max(args.temp, 1e-5),
                         top_p=args.top_p, pad_token_id=tok.pad_token_id)
    new = out[:, enc.input_ids.size(1):]
    return [tok.decode(o, skip_special_tokens=True) for o in new], new


print(f"\n采样中：{len(probs)} 题 × {args.samples} 次   T={args.temp}   "
      f"每批 {args.bs} 条\n" + "=" * 72)

records, t_start, n_tok = [], time.time(), 0
per_prob = []                                   # 每题 k 次的分数

flat = [(i, make_prompt(q)) for i, (q, _) in enumerate(probs) for _ in range(args.samples)]
for s in range(0, len(flat), args.bs):
    chunk = flat[s: s + args.bs]
    texts, ids = gen([p for _, p in chunk])
    lens = (ids != tok.pad_token_id).sum(1).tolist()      # 每条各自的 token 数
    n_tok += sum(lens)
    for (pi, _), t, L in zip(chunk, texts, lens):
        records.append((pi, t, reward(t, probs[pi][1]), has_format(t), L))
    done = s + len(chunk)
    el = time.time() - t_start
    print(f"  {done:>5}/{len(flat)}   {el/60:>5.1f} 分   "
          f"{n_tok/el:>6.0f} tok/s   剩余 {el/done*(len(flat)-done)/60:>5.1f} 分", flush=True)

t_total = time.time() - t_start

# ══════════════════════════════════════════════════════════════
#  ★ 生成结束，先把原始数据落盘再算统计量。
#    教训：上一版把 numpy 导入漏了，崩在统计那一步，
#    14 分钟的 GPU 生成全丢了。★ 贵的东西先存。
# ══════════════════════════════════════════════════════════════
_raw = args.out.replace(".json", "_raw.json")
json.dump({"model": args.model, "ckpt": args.ckpt, "n": len(probs),
           "n_samples": args.samples, "offset": args.offset, "temp": args.temp,
           "records": [(pi, r, f, L) for pi, _t, r, f, L in records],
           "t_total": t_total, "n_tok": n_tok},
          open(_raw, "w"))
print(f"\n  ★ 原始结果已落盘 → {_raw}（统计出错也不用重跑）", flush=True)

# ── 汇总 ──
scores = [[] for _ in probs]
for pi, _, r, _f, _L in records:
    scores[pi].append(r)
fmt_rate = sum(rec[3] for rec in records) / len(records)

# ★ 交叉表：有 #### / 无 #### 分别的正确率 —— 直接量化抽取器的假阴性
with_f  = [rec for rec in records if rec[3]]
wo_f    = [rec for rec in records if not rec[3]]
acc_w   = sum(rec[2] for rec in with_f) / max(len(with_f), 1)
acc_wo  = sum(rec[2] for rec in wo_f)   / max(len(wo_f), 1)

L = np.array([rec[4] for rec in records])
n_cut   = int((L >= args.max_new - 2).sum())

COUNTS = [int(sum(s)) for s in scores]          # ★ 每题对了几次 —— 原始数据，务必存下来
N = args.samples


def pass_at_k(n, c, k):
    """
    ★ Codex 论文的无偏估计：pass@k = 1 − C(n−c, k) / C(n, k)
      = 1 − 「随机抽 k 份全是错的」的概率
    数值稳定的连乘写法，不用算组合数。
    """
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


KS = [k for k in (1, 2, 4, 8, 16, 32, 64, 128) if k <= N]
if N not in KS:
    KS.append(N)
CURVE = {k: float(np.mean([pass_at_k(N, c, k) for c in COUNTS])) for k in KS}

pass1 = CURVE[1]
passk = CURVE[N]
all_right = sum(1 for s in scores if all(s))
all_wrong = sum(1 for s in scores if not any(s))
useful = len(probs) - all_right - all_wrong

NOTE = {1: "   ← 随手一答的正确率", N: "   ← k=n，用满全部采样"}
CURVE_TXT = "\n".join(
    f"    pass@{k:<5}{CURVE[k]:.4f}{NOTE.get(k, '')}" for k in KS)
CURVE_TXT += "\n    " + " " * 5 + "         ★ k 越大越接近「能力的支撑集」，RL 动不了它"

print("\n" + "=" * 72)
print(f"  基线结果   {args.model}")
print("=" * 72)
print(f"""
  ── ★ pass@k 曲线（同一批 n={N} 次采样，算出的多个指标）──
    ★ n = 每题采样几次（成本，决定精度）
      k = 假设允许交几份答案（问题本身，k ≤ n）
{CURVE_TXT}
  ── ★ 梯度信号（最关键的一项）──
    全对（{args.samples}/{args.samples}）   {all_right:>4} 题  {100*all_right/len(probs):>5.1f}%   优势全 0，白跑
    全错（0/{args.samples}）   {all_wrong:>4} 题  {100*all_wrong/len(probs):>5.1f}%   优势全 0，白跑
    ★ 有对有错   {useful:>4} 题  {100*useful/len(probs):>5.1f}%   ← 只有这些产生梯度

  ── ★ 格式合规 × 正确率（量化抽取器的假阴性）──
    输出了 '#### N'         {fmt_rate:.3f}   ★ 低于 0.8 该加格式奖励
    有 ####  {len(with_f):>5} 条   正确率 {acc_w:.3f}
    无 ####  {len(wo_f):>5} 条   正确率 {acc_wo:.3f}
    ★ 两者差距 {acc_w/max(acc_wo,1e-9):.1f} 倍 —— 差得越多，说明「取最后一个数」误判越严重

  ── 长度 / 截断 ──
    中位 {np.median(L):.0f}   均值 {L.mean():.0f}   P95 {np.percentile(L,95):.0f}   最长 {L.max()}
    撞上 {args.max_new} 上限被截断   {n_cut} 条 ({100*n_cut/len(records):.1f}%)   ★ 截断=假阴性

  ── 速度（用来估训练时长）──
    总耗时          {t_total/60:.1f} 分钟，{len(flat)} 条轨迹
    吞吐            {n_tok/t_total:.0f} tok/s
    平均每条        {n_tok/len(flat):.0f} token
    ★ 每题（{args.samples} 条）  {t_total/len(probs):.1f} 秒
""")

if useful / len(probs) < 0.4:
    print("  ⚠️ 有效题目不足 40% —— 大部分采样在做无用功。考虑：")
    if all_right > all_wrong:
        print("     题目太简单 → 换更难的数据集（MATH）或更小的模型")
    else:
        print("     题目太难 → 换更大的模型，或提高温度增加多样性")
else:
    print("  ✅ 信号充足，可以进入训练配置")

json.dump({"model": args.model, "n": len(probs), "k": args.samples, "temp": args.temp,
           "offset": args.offset, "ckpt": args.ckpt,
           "n_samples": N,
           "counts": COUNTS,          # ★ 每题对了几次 —— 存原始数据，随时能重算任何 pass@k
           "curve": {str(k): v for k, v in CURVE.items()},
           "max_new": args.max_new,
           "pass1": pass1, "passk": passk, "useful_frac": useful / len(probs),
           "all_right": all_right, "all_wrong": all_wrong,
           "fmt_rate": fmt_rate, "acc_with_fmt": acc_w, "acc_without_fmt": acc_wo,
           "len_median": float(np.median(L)), "len_p95": float(np.percentile(L, 95)),
           "trunc_frac": n_cut / len(records),
           "tok_per_s": n_tok / t_total, "sec_per_problem": t_total / len(probs),
           "avg_tokens": n_tok / len(flat)}, open(args.out, "w"), indent=2)
print(f"  → {args.out}\n")

if args.ckpt and os.path.exists("baseline.json"):
    b = json.load(open("baseline.json"))
    if b.get("k") != args.samples or b.get("n") != len(probs) or b.get("offset", 0) != args.offset:
        print(f"  ⚠️ baseline.json 是 k={b.get('k')} / n={b.get('n')} 测的，"
              f"跟本次 k={args.samples} / n={len(probs)} 不可比 —— 跳过自动对账")
        print(f"     用 compare_passk.py 按 k 分组看\n")
        raise SystemExit(0)
    print("─" * 72 + "\n  ★★ 跟训练前的基线对账（同温度 1.0，同 k，同题目）\n" + "─" * 72)
    print(f"""
    {'指标':<26}{'训练前':>10}{'训练后':>10}{'变化':>11}
    {'─'*57}
    {'pass@1（随便问一次对的概率）':<20}{b['pass1']:>10.4f}{pass1:>10.4f}{pass1-b['pass1']:>+11.4f}
    {'★ pass@' + str(args.samples) + '（能力天花板）':<22}{b['passk']:>10.4f}{passk:>10.4f}{passk-b['passk']:>+11.4f}
    {'有信号的题目比例':<22}{b['useful_frac']:>10.4f}{useful/len(probs):>10.4f}{useful/len(probs)-b['useful_frac']:>+11.4f}
    {'平均长度 (token)':<22}{b['avg_tokens']:>10.0f}{n_tok/len(flat):>10.0f}{n_tok/len(flat)-b['avg_tokens']:>+11.0f}

    ★ 怎么读 pass@{args.samples} 那一行：
      ≈ 0        RL 只重排概率，没创造新能力   ← 理论预测
      明显 +     打破理论，值得深挖
      ★ 明显 −   熵坍缩的代价：变稳了，探索能力下降了
""")

# 抽三条看看长什么样
print("─" * 72 + "\n  样本（前 3 题各取 2 条）\n" + "─" * 72)
for pi in range(min(3, len(probs))):
    print(f"\n【题 {pi}】{probs[pi][0][:110]}…\n  标准答案 {probs[pi][1]}")
    shown = [r for r in records if r[0] == pi][:2]
    for _, t, r, f, _L in shown:
        a, _ = extract(t)
        print(f"  {'✅' if r else '❌'} 抽出 {a!r}  {'有####' if f else '无####'}  {t.strip()[:220]}")
