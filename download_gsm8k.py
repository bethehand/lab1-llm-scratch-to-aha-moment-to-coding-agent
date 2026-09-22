#!/usr/bin/env python3
"""
下载 GSM8K，然后把它拆开看看 —— 这就是 GRPO 的全部『环境』。

    python3 download_gsm8k.py
    python3 download_gsm8k.py -n 5        # 看 5 条样本
"""
import re, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-n", type=int, default=3)
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
args = ap.parse_args()

from datasets import load_dataset
print("=" * 74 + "\n  下载 GSM8K（约 10 MB）\n" + "=" * 74, flush=True)
tr = load_dataset("openai/gsm8k", "main", split="train")
te = load_dataset("openai/gsm8k", "main", split="test")
print(f"\n✅ train {len(tr):,} 题   test {len(te):,} 题\n")

# ── ① 原始格式 ──
print("─" * 74 + "\n  ① 原始数据（一条完整的）\n" + "─" * 74)
d = tr[0]
print(f"\n   question:\n   │ " + "\n   │ ".join(d["question"].split("\n")))
print(f"\n   answer:")
for line in d["answer"].split("\n"):
    print(f"   │ {line}")
print("""
   ★ 注意两件事：
     · <<48/2=24>> 是数据集自带的『计算器标注』，训练时通常剥掉
     · #### 后面那个数才是标准答案 —— 判分器只认它""")

# ── ② 判分器怎么解析 ──
print("\n" + "─" * 74 + "\n  ② ★ 判分器（这就是全部的『环境』）\n" + "─" * 74)
from _reward import reward, extract          # ★ 判分器统一在 _reward.py


def gold_of(ans):
    return ans.split("####")[-1].strip().replace(",", "")


G = gold_of(d["answer"])
print(f"""
   def extract(text):
       t = text.replace(",", "").replace("$", "").replace("*", "")
       if "####" in t:                          # ① 优先 '#### N'
           m = NUM.findall(t.split("####")[-1])
           if m: return m[0], True
       m = NUM.findall(t)                       # ② ★ 否则取【最后一个】数字
       return (m[-1] if m else None), False

   标准答案 = {G!r}

   模型可能的输出                                              抽出   判分
   ──────────────────────────────────────────────────────────────────────""")
for t, note in [
        ("48 + 24 = 72. #### 72",                    "标准格式"),
        ("The answer is 72.",                        "没有 ####，取末尾"),
        ("She sold 48 in April... total is **72**.", "★ 开头有 48，取第一个就错了"),
        ("#### 24",                                  "格式对但答错"),
        ("I need more information.",                 "没给答案"),
        ("Maybe 24, 48, 72, 96.",                    "★ 堆数字作弊 → 只认末尾 96")]:
    a, f = extract(t)
    print(f"   {t!r:<46}{str(a):>7}  {'✅ 1.0' if reward(t, G) else '❌ 0.0'}   {note}")
print("""
   ★ 两条关键设计（都是踩坑踩出来的）：
     ① 没有 #### 时取【最后一个】数字，不是第一个
        模型必然先复述题干的数字再推理 —— 取第一个必然抓到题干的数
        （原来那版就是这个 bug，把答对的判成了错的：系统性假阴性）
     ② 取最后一个也抗作弊：堆一串数字蒙答案，只有最后那个算数""")

# ── ③ 套上 chat template ──
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    INSTR = "\n\nSolve this step by step. End your answer with '#### <number>'."
    p = tok.apply_chat_template([{"role": "user", "content": d["question"] + INSTR}],
                                tokenize=False, add_generation_prompt=True)
    print("\n" + "─" * 74 + "\n  ③ 模型实际收到的 prompt\n" + "─" * 74)
    for line in p.split("\n"):
        print(f"   │ {line}")
    print(f"\n   {len(tok.encode(p))} 个 token")
except Exception as e:
    print(f"\n   （跳过 chat template：{e}）")

# ── ④ 统计 ──
print("\n" + "─" * 74 + "\n  ④ 统计\n" + "─" * 74)
ql = np.array([len(x.split()) for x in tr["question"]])
al = np.array([len(x.split()) for x in tr["answer"]])
steps = np.array([x.count("\n") + 1 for x in tr["answer"]])
print(f"   题目长度   中位 {np.median(ql):.0f}  均值 {ql.mean():.0f}  P95 {np.percentile(ql,95):.0f} 词")
print(f"   解答长度   中位 {np.median(al):.0f}  均值 {al.mean():.0f}  P95 {np.percentile(al,95):.0f} 词")
print(f"   推理步数   中位 {np.median(steps):.0f}  均值 {steps.mean():.1f}  最多 {steps.max()} 步")
print(f"\n   ★ 中位 {np.median(steps):.0f} 步推理 —— 这就是为什么它能训练『多步推理』能力")
print(f"     单步题（1+1=2）没有信号可学，20 步题信用分配又太难，{np.median(steps):.0f} 步刚好\n")

# ── ⑤ 再看几条 ──
print("─" * 74 + f"\n  ⑤ 再看 {args.n} 条\n" + "─" * 74)
for i in range(1, args.n + 1):
    d = tr[i]
    print(f"\n   【{i}】{d['question'][:150]}")
    print(f"        标准答案 → {gold_of(d['answer'])}")
