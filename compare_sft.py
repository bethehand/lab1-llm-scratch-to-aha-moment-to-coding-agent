#!/usr/bin/env python3
"""
SFT 对照实验，一次跑完两部分。

  第一部分  分布内（Alpaca 式问题）· 2×2 网格
            ①base+原始  ②base+格式  ③sft+格式  ④sft+原始
            ①→② 只改提示  ②→③ 只改权重  ③→④ 测格式依赖

  第二部分  分布外（5 类刁钻问题）· 欠训 vs 过训 同屏
            ★ 过拟合的代价只在分布外才看得出来

    python3 compare_sft.py                      # 全跑
    python3 compare_sft.py --only in            # 只跑分布内
    python3 compare_sft.py --only ood           # 只跑分布外
"""
import argparse, os, torch, tiktoken
from config import Config
from model import GPT

ap = argparse.ArgumentParser()
ap.add_argument("--base",   default="out_run2/ckpt_final.pt")
ap.add_argument("--sft",    default="out_sft/ckpt.pt",       help="欠训（val 最低）")
ap.add_argument("--sft2",   default="out_sft/ckpt_final.pt", help="过训（跑满 3 epoch）")
ap.add_argument("--only",   choices=["in", "ood", "all"], default="all")
ap.add_argument("--tokens", type=int,   default=90)
ap.add_argument("--temp",   type=float, default=0.7)
ap.add_argument("--width",  type=int,   default=300)
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config(**{k: v for k, v in ck["cfg"].items() if k in Config.__dataclass_fields__})
    m = GPT(cfg); m.load_state_dict(ck["model"])
    return m.to(dev).eval(), ck


base, bck = load(args.base)
sftA, ack = load(args.sft)
sftB, cck = (load(args.sft2) if os.path.exists(args.sft2) else (None, None))

if dev == "cuda":                                   # 预热
    with torch.no_grad():
        z = torch.zeros(1, 4, dtype=torch.long, device=dev)
        for m in (base, sftA, sftB):
            if m is not None: m(z)
    torch.cuda.synchronize()


@torch.no_grad()
def gen(m, prompt):
    """→ (文本, 是否自己停下, 停在第几个 token)"""
    ids = torch.tensor([enc.encode_ordinary(prompt)], dtype=torch.long, device=dev)
    out = m.generate(ids, args.tokens, temperature=args.temp,
                     top_p=0.9, repetition_penalty=1.15)
    new = out[0].tolist()[ids.size(1):]
    if EOT in new:
        n = new.index(EOT)
        return enc.decode(new[:n]).strip(), True, n
    return enc.decode(new).strip(), False, len(new)


def alpaca(q):
    return f"### Instruction:\n{q}\n\n### Response:\n"


def show(label, m, prompt, stat=None, key=None):
    txt, stopped, n = gen(m, prompt)
    if stat is not None:
        stat[key]["stop"] += stopped; stat[key]["len"].append(n)
    flag = f"⏹ {n:>2} token 自己停" if stopped else f"✂️ 写满 {n} 被砍断"
    print(f"\n  ── {label}   [{flag}]")
    print("    ", txt.replace("\n", " ⏎ ")[: args.width])


print("=" * 78)
print(f"  base  {args.base:<28} step {bck['step']:>7,}  val {bck['best_val']:.4f}")
print(f"  欠训  {args.sft:<28} step {ack['step']:>7,}  val {ack['best_val']:.4f}")
if sftB is not None:
    print(f"  过训  {args.sft2:<28} step {cck['step']:>7,}  val {cck['best_val']:.4f}")
print(f"  {base.num_params()/1e6:.1f}M 参数   T={args.temp}  top_p=0.9  rep=1.15  上限 {args.tokens} token")
print("=" * 78)


# ══════════════════════════════════════════════════════════════════
#  第一部分：分布内 · 2×2 网格
# ══════════════════════════════════════════════════════════════════
IN_DIST = [
    ("常见事实", "What is the capital of France?"),
    ("常见事实", "At what temperature does water boil?"),
    ("解释",     "Explain what photosynthesis is."),
    ("列举",     "List three healthy breakfast foods."),
    ("生成",     "Write a short poem about the ocean."),
    ("建议",     "Give me three tips for learning a new language."),
    ("推理",     "If a train travels 60 miles per hour for 2 hours, how far does it go?"),
]
CELLS = [
    ("①", "BASE + 原始问题      它本来的样子",           lambda: base, lambda q: q),
    ("②", "BASE + Alpaca 格式   只加格式，权重没变",     lambda: base, alpaca),
    ("③", "SFT  + Alpaca 格式   训练过，格式也给了",     lambda: sftA, alpaca),
    ("④", "SFT  + 原始问题      ★ 不给格式还当助手吗",   lambda: sftA, lambda q: q),
]
stat = {c[0]: {"stop": 0, "len": []} for c in CELLS}

if args.only in ("in", "all"):
    print(f"\n\n{'█'*78}\n  第一部分  分布内 —— Alpaca 训练数据里到处都是的那类问题\n{'█'*78}")
    for tag, q in IN_DIST:
        print(f"\n{'━'*78}\n【{tag}】{q}\n{'━'*78}")
        for num, title, getm, mk in CELLS:
            show(f"{num} {title}", getm(), mk(q), stat, num)

    print(f"\n{'='*78}\n  量化对账（分布内）\n")
    print(f"  {'条件':<36}{'自己停下':>10}{'平均长度':>12}")
    print(f"  {'─'*58}")
    for num, title, _, _ in CELLS:
        s = stat[num]
        print(f"  {num} {title[:20].strip():<34}"
              f"{s['stop']}/{len(IN_DIST):<9}{sum(s['len'])/len(s['len']):>10.1f}")
    print("""
    ① → ②   只改提示   光靠 "### Response:" 能不能变助手？
    ② → ③   只改权重   ★ 变化出现在这里，才证明是训练的功劳
    ③ → ④   只改提示   离了格式还行不行 —— 「格式依赖」""")


# ══════════════════════════════════════════════════════════════════
#  第二部分：分布外 · 欠训 vs 过训
# ══════════════════════════════════════════════════════════════════
PARA = ("The Amazon rainforest produces about 6% of the world's oxygen. It spans "
        "nine countries and covers 5.5 million square kilometres. Deforestation has "
        "removed roughly 17% of it since 1970, mostly for cattle ranching and soy "
        "farming. Scientists warn that losing another 20% could push the forest past "
        "a tipping point, after which it would dry out and become savanna.")

OOD = [
    ("① 格式约束",
     alpaca("Answer in exactly five words: what is gravity?"),
     "能不能遵守「恰好五个词」—— 模板会不会盖过指令"),

    ("② 假前提",
     alpaca("Why is the sky green?"),
     "会顺着编，还是会指出前提是错的"),

    ("③ 多轮追问",
     "### Instruction:\nWhat is 2 + 2?\n\n### Response:\n4\n\n"
     "### Instruction:\nWhy?\n\n### Response:\n",
     "Alpaca 全是单轮，多轮完全没见过"),

    ("④ 超长输入",
     f"### Instruction:\nSummarize the following in one sentence.\n\n"
     f"### Input:\n{PARA}\n\n### Response:\n",
     "Alpaca 的 input 都很短，这段 110+ token"),

    ("⑤ 中文",
     alpaca("法国的首都是哪里？"),
     "训练数据 100% 英文"),
]

if args.only in ("ood", "all") and sftB is not None:
    print(f"\n\n{'█'*78}\n  第二部分  分布外 —— Alpaca 里几乎没有的问题类型\n"
          f"  ★ 过拟合的代价只在这里才看得出来\n{'█'*78}")
    for tag, prompt, why in OOD:
        print(f"\n{'━'*78}\n【{tag}】看点：{why}\n{'━'*78}")
        print("  提示：", repr(prompt if len(prompt) < 160 else prompt[:120] + " …"))
        show(f"BASE          （对照）",              base, prompt)
        show(f"欠训 step {ack['step']:,}  val {ack['best_val']:.4f}",  sftA, prompt)
        show(f"过训 step {cck['step']:,}  val {cck['best_val']:.4f}", sftB, prompt)
elif args.only == "ood":
    print("\n⚠️ 找不到 --sft2，分布外部分需要两个 checkpoint 才有对比意义")

print()
