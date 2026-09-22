#!/usr/bin/env python3
"""
★ 跨任务测试：在 Countdown 上 RL 训练过的模型，去做 GSM8K 还行不行？

验证「环境过拟合」假说 —————————————————————————————————
  ★ RL 在 Countdown 上学到了「闭嘴，直接给答案」
      eval 长度 186 → 48，纠错措辞 0.146 → 0.005，正确率 0.035 → 0.405
  ★★ 如果这是【环境过拟合】，这个策略会迁移到 GSM8K 并【破坏】它
      —— 因为 GSM8K 必须写出中间步骤才算得对

★★ 2×2 设计 —— 把「模板变量」和「权重变量」分开 ————————————
                     R1 模板（★ 训练过的）   中性模板（★ 没训练过）
    ① Base               A                        B
    ② RL 后              C                        D

  ★ A → C   只改权重（模板固定成训练过的那个）
  ★ B → D   只改权重（模板固定成没训练过的那个）
  ★ A → B   只改模板（权重固定）

  ★★ 判读
     C < A 且 D < B   → ★ 能力真的被破坏了 —— 环境过拟合成立
     C ≈ A 但 D << B  → ★ 只是变得依赖训练时的模板，底层能力没坏
     C ≈ A 且 D ≈ B   → ★ RL 完全没有副作用（最好的情况）

    python3 crosstask_eval.py                # 200 题，贪心
    python3 crosstask_eval.py -n 50          # 快速试跑
    python3 crosstask_eval.py --bs 48        # 显存够就调大
"""
import os, re, json, time, argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from _reward import extract                      # ★ 判准只有一份定义

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
ap.add_argument("--ckpt",  default=None, help="不传就只测 --model 那一行（原始模型），用来填 Instruct 系的格子")
ap.add_argument("-n", "--n-problems", type=int, default=200)
ap.add_argument("--offset",   type=int, default=0)
ap.add_argument("--max-new",  type=int, default=512)
ap.add_argument("--bs",       type=int, default=32)
ap.add_argument("--out",      default="crosstask.json")
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
assert dev == "cuda", "需要 GPU"

# ════════════════════════════════════════════════════════════
#  两个提示模板
# ════════════════════════════════════════════════════════════
SYS = ("A conversation between User and Assistant. The user asks a question, "
       "and the Assistant solves it. The Assistant first thinks about the reasoning "
       "process in the mind and then provides the user with the answer. The reasoning "
       "process and answer are enclosed within <think> </think> and <answer> </answer> "
       "tags, respectively.")


def tpl_r1(q):
    """★ Countdown 训练时用的那个模板，只把题目换成 GSM8K"""
    return (f"{SYS}\nUser: {q}\nShow your work in <think> </think> tags. "
            f"Return the final numeric answer in <answer> </answer> tags, "
            f"for example <answer> 42 </answer>.\n"
            f"Assistant: Let me solve this step by step.\n<think>")


def tpl_plain(q):
    """★ 中性模板 —— 训练时从没见过"""
    return f"Question: {q}\nAnswer: Let's solve this step by step.\n"


#            名字        模板函数    ★ 截断标记（Base 模型不会好好停）
TEMPLATES = [("R1 模板",   tpl_r1,    ["</answer>", "\nUser:", "\nQuestion:"]),
             ("中性模板",  tpl_plain, ["\nQuestion:", "\nAnswer:", "\nUser:"])]

REFLECT = re.compile(
    r"\b(wait|hold on|hmm|actually|alternatively|instead|let me (?:try|check|recheck|reconsider)"
    r"|that (?:doesn'?t|does not) work|not (?:right|correct)|recheck|reconsider)\b", re.I)


def cut(txt, stops):
    """★ 在第一个截断标记处切掉 —— 否则模型会自问自答，最后一个数字抓错"""
    best = len(txt)
    for s in stops:
        i = txt.find(s)
        if i >= 0:
            best = min(best, i + (len(s) if s == "</answer>" else 0))
    return txt[:best]


# ════════════════════════════════════════════════════════════
#  数据
# ════════════════════════════════════════════════════════════
ds = load_dataset("openai/gsm8k", "main", split="test")
ALL = [(d["question"], d["answer"].split("####")[-1].strip().replace(",", "")) for d in ds]
PROB = ALL[args.offset: args.offset + args.n_problems]
print(f"GSM8K test[{args.offset}:{args.offset+len(PROB)}]  {len(PROB)} 题   ★ 贪心解码")

# ════════════════════════════════════════════════════════════
#  模型（★ 只装一份，中途覆盖权重，省显存）
# ════════════════════════════════════════════════════════════
tok = AutoTokenizer.from_pretrained(args.model)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print(f"载入 {args.model} …", flush=True)
try:
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
model = model.to(dev).eval()
BASE_SD = {k: v.clone() for k, v in model.state_dict().items()}   # ★ 备份原始权重


@torch.no_grad()
def run(mk, stops):
    """跑完一整套题 → 每题一个 dict"""
    res = []
    for i in range(0, len(PROB), args.bs):
        chunk = PROB[i:i + args.bs]
        enc = tok([mk(q) for q, _ in chunk], return_tensors="pt", padding=True).to(dev)
        out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        new = out[:, enc.input_ids.size(1):]
        for j, (q, gold) in enumerate(chunk):
            raw = tok.decode(new[j], skip_special_tokens=True)
            txt = cut(raw, stops)
            a, _ = extract(txt)
            try:
                ok = a is not None and abs(float(a) - float(gold)) < 1e-4
            except ValueError:
                ok = False
            res.append(dict(ok=bool(ok), pred=a, gold=gold,
                            ntok=len(tok(txt).input_ids),
                            nref=len(REFLECT.findall(txt)), txt=txt[:800]))
        print(f"    {min(i+args.bs, len(PROB))}/{len(PROB)}", end="\r", flush=True)
    return res


# ════════════════════════════════════════════════════════════
#  四格
# ════════════════════════════════════════════════════════════
CELLS, t0 = {}, time.time()
ROWS = [("① Base", False)] + ([("② RL 后", True)] if args.ckpt else [])
for wname, use_ckpt in ROWS:
    if use_ckpt:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); model.to(dev).eval()
        print(f"\n★ 覆盖成 {args.ckpt}（step {ck.get('step','?')}）")
        del ck; torch.cuda.empty_cache()
    else:
        model.load_state_dict(BASE_SD); model.to(dev).eval()
    for tname, mk, stops in TEMPLATES:
        print(f"\n  {wname} × {tname}")
        CELLS[f"{wname}|{tname}"] = run(mk, stops)
        # ★ 贵的东西先存 —— 每跑完一格就落盘
        json.dump(CELLS, open(args.out, "w"), ensure_ascii=False)

print(f"\n\n★ 原始数据已存 {args.out}   用时 {(time.time()-t0)/60:.1f} 分钟")


# ════════════════════════════════════════════════════════════
#  报告
# ════════════════════════════════════════════════════════════
def stat(rs):
    return (float(np.mean([r["ok"] for r in rs])),
            float(np.mean([r["ntok"] for r in rs])),
            float(np.mean([r["nref"] > 0 for r in rs])))


if not args.ckpt:                                   # ★ 只有 Base 行：打两格就走
    for t, _, _ in TEMPLATES:
        c, L, r = stat(CELLS[f"① Base|{t}"])
        print(f"  {args.model}  {t:<8}  正确率 {c:.4f}   长度 {L:.0f}   纠错 {r:.3f}")
    raise SystemExit

print("\n" + "=" * 84)
print(f"  ★★ 2×2：Countdown 上 RL 训练后，在 GSM8K 上的表现   ({len(PROB)} 题，贪心)")
print("=" * 84)
for metric, idx, fmt in [("正确率", 0, "{:.4f}"), ("长度", 1, "{:.0f}"), ("纠错措辞", 2, "{:.3f}")]:
    print(f"\n  ── {metric} ──")
    print(f"  {'':<10}{'R1 模板(训练过)':>18}{'中性模板(没训过)':>20}")
    for w in ("① Base", "② RL 后"):
        row = "".join(fmt.format(stat(CELLS[f'{w}|{t}'])[idx]).rjust(19 if t == "R1 模板" else 20)
                      for t, _, _ in TEMPLATES)
        print(f"  {w:<10}{row}")
    a, c = stat(CELLS["① Base|R1 模板"])[idx], stat(CELLS["② RL 后|R1 模板"])[idx]
    b, d = stat(CELLS["① Base|中性模板"])[idx], stat(CELLS["② RL 后|中性模板"])[idx]
    print(f"  {'Δ (RL−Base)':<10}{(fmt.format(c-a) if idx else f'{c-a:+.4f}').rjust(19)}"
          f"{(fmt.format(d-b) if idx else f'{d-b:+.4f}').rjust(20)}")

# ── McNemar：同一批题上配对比较 ──
print("\n  ── ★ McNemar 检验（配对，只看正确率）──")
for tname, _, _ in TEMPLATES:
    A = CELLS[f"① Base|{tname}"]; C = CELLS[f"② RL 后|{tname}"]
    b_ = sum(1 for x, y in zip(A, C) if x["ok"] and not y["ok"])    # Base 对 → RL 错
    c_ = sum(1 for x, y in zip(A, C) if not x["ok"] and y["ok"])    # Base 错 → RL 对
    z = (b_ - c_) / np.sqrt(b_ + c_) if (b_ + c_) else 0.0
    verdict = ("★★ RL 显著变差" if z > 1.96 else
               "★ RL 显著变好" if z < -1.96 else "无显著差异")
    print(f"  {tname:<10} 训坏 {b_:>3} 题 / 训好 {c_:>3} 题   z = {z:+.2f}   {verdict}")

# ── 判读 ──
S = lambda w, t: stat(CELLS[f"{w}|{t}"])
ca = S("② RL 后", "R1 模板")[0]   - S("① Base", "R1 模板")[0]
db = S("② RL 后", "中性模板")[0] - S("① Base", "中性模板")[0]
lca = S("② RL 后", "R1 模板")[1]   - S("① Base", "R1 模板")[1]
ldb = S("② RL 后", "中性模板")[1] - S("① Base", "中性模板")[1]

print("\n  ── ★★★ 判读 ──")
print(f"  正确率 Δ   R1 模板 {ca:+.4f}   中性模板 {db:+.4f}")
print(f"  长度   Δ   R1 模板 {lca:+.0f}     中性模板 {ldb:+.0f}")
print()
if ca < -0.03 and db < -0.03:
    print("  ★★ 两个模板下都变差 → 【能力被真正破坏】")
    print("     底层推理能力受损，不只是策略问题")
elif ca < -0.03 and db >= -0.03:
    print("  ★★★ 只在【训练过的模板】下变差 → 【条件性策略绑定】")
    print("     ★ 这是环境过拟合最精确的形式：")
    print("       · 能力没坏 —— 中性模板下正确率和长度都没变")
    print("       · ★★ 但训练时的模板变成了【触发器】，一看到就切换到退化策略")
    if lca < -50:
        print(f"     · ★★★ 长度是铁证：R1 模板下掉了 {-lca:.0f} token，中性模板下没动")
    print("     → 下一步：多环境混训，让同一个模板下不存在通吃的退化策略")
elif ca >= -0.03 and db < -0.03:
    print("  ★ 只在【没训过的模板】下变差 → 变得依赖训练时的模板")
else:
    print("  ★ 两个模板下都没变差 → RL 没有跨任务副作用（最好的情况）")

if S("① Base", "R1 模板")[2] == 0 and S("① Base", "中性模板")[2] == 0:
    print("\n  ⚠️ 纠错措辞在 Base 上就是 0 —— GSM8K 本来不诱发这类措辞，此列无信息量")
print("=" * 84 + "\n")
