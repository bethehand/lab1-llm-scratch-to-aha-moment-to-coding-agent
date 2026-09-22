#!/usr/bin/env python3
"""
Countdown 基线测量 —— R1-Zero 复现的第 1 步。

★ 跟 GSM8K 那次最大的不同：Base 模型没做过任何后训练
  → 不会遵循指令、不会输出结构
  → ★ 所以要额外量「格式合规率」和「自我纠错措辞频率」（H2 的起点）

    python3 baseline_countdown.py                      # 100 题 × 8 次
    python3 baseline_countdown.py -n 50 -s 64          # 算完整 pass@k 曲线
    python3 baseline_countdown.py --model Qwen/Qwen2.5-1.5B-Instruct   # 对照

★ 多卡并行（工具版一条要几十轮生成，单卡 800 条太慢）：
    CUDA_VISIBLE_DEVICES=k python3 baseline_countdown.py … --shard k/4      # 4 张卡各跑 1/4 的题，输出名带 _shardk
    python3 baseline_countdown.py … --merge cd_…_shard0_…_raw.json cd_…_shard1_…_raw.json …   # 不载模型，合并成标准名
"""
import os, re, json, time, argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import countdown as CD

ap = argparse.ArgumentParser()
ap.add_argument("--model",   default="Qwen/Qwen2.5-1.5B", help="★ Base 版，没有 -Instruct")
ap.add_argument("--ckpt",    default=None)
ap.add_argument("--data",    default="data/countdown.json")
ap.add_argument("-n", "--n-problems", type=int, default=100)
ap.add_argument("--offset",  type=int, default=0)
ap.add_argument("-s", "--samples", type=int, default=8, help="每题采样几次（统计的 n）")
ap.add_argument("--temp",    type=float, default=1.0)
ap.add_argument("--max-new", type=int, default=1024)
ap.add_argument("--bs",      type=int, default=32)
ap.add_argument("--out",     default=None)
ap.add_argument("--show",    type=int, default=3)
ap.add_argument("--shard",   default=None, help="★ 多卡并行：k/N，只跑第 k 片题（0 起），输出名带 _shardk；跑完用 --merge 合并")
ap.add_argument("--merge",   nargs="*", default=None, help="★ 传各分片的 _raw.json，不载模型不采样，直接合并出汇总和标准名的 raw")
ap.add_argument("--tool",    action="store_true",
                help="★ 计算器：提示词加工具说明，生成走 calc_tool.generate_with_tools（停、注入、续）。评工具版模型必须开")
ap.add_argument("--plain",   action="store_true",
                help="★ 常规模板：去掉系统说明 / User:Assistant / <think>，只留题目和 <answer> 要求。"
                     "跟 crosstask_eval 的中性模板同一用途 —— 测 RL 的收益离了训练模板还在不在")
args = ap.parse_args()

dev = "cuda"


def out_name():
    stem = "base" if not args.ckpt else args.ckpt.replace("/", "_").replace(".pt", "")
    if args.plain:
        stem += "_plain"
    if args.tool:
        stem += "_tool"
    if args.shard:
        stem += f"_shard{args.shard.split('/')[0]}"
    if args.data != "data/countdown.json":                    # ★ 换了数据集就带上名字，别覆盖 CD-3 的结果
        stem += "_" + os.path.splitext(os.path.basename(args.data))[0]
    return f"cd_{stem}_s{args.samples}_off{args.offset}.json"


# ── 数据 ──
if args.merge:                                            # ★ 合并模式：数据路径以分片里记的为准（CD-3 / CD-4 别拿错）
    shards = [json.load(open(f)) for f in args.merge]
    args.data = shards[0]["data"]
if not os.path.exists(args.data):
    raise SystemExit(f"没找到 {args.data}\n  先跑：python3 countdown.py --gen 100000")
D = json.load(open(args.data))
if args.shard:                                            # ★ 第 k 片：偏移往后挪，题数按 N 等分（最后一片兜余数）
    k, N_SH = map(int, args.shard.split("/"))
    per = args.n_problems // N_SH
    args.offset, args.n_problems = args.offset + k * per, (per if k < N_SH - 1 else args.n_problems - k * per)
P = D[args.offset: args.offset + args.n_problems]
if not args.merge:                                        # 合并模式的题号范围由分片决定，下面再打
    print(f"Countdown  共 {len(D):,} 道，取 [{args.offset}:{args.offset+len(P)}]  {len(P)} 道")

# ★ 自我纠错措辞 —— H2 的检测器
REFLECT = re.compile(
    r"\b(wait|hold on|hmm|actually|alternatively|instead|let me (?:try|check|recheck|reconsider)"
    r"|that (?:doesn'?t|does not) work|not (?:right|correct)|recheck|reconsider|另一种|不对)\b", re.I)

# ★★ 搜索措辞 —— 混训模型实测会写「(too high)」「(too low)」「(perfect match)」这种带反馈的试错，
#    REFLECT 那套 wait/hmm 词表完全抓不到。这是另一种形态的「反思」，单独计。
SEARCH = re.compile(
    r"\b(too (?:high|low|big|small|large)|not (?:equal|right|correct)|doesn'?t (?:work|equal)"
    r"|does not (?:work|equal)|perfect|close|try (?:another|again|different|a different|the next)"
    r"|let'?s try|next,? try|another (?:combination|approach|way|try)|nope|wrong|incorrect)\b", re.I)

# ★ Base 模型写完 </answer> 不发 EOS，接着幻觉出下一轮 "User: …" 直到撞 max_new。
#   实测 85% 的贪心输出被这样撑到 512。判分不受影响（判分器取第一个 <answer>），但长度指标全废。
#   → 生成时把它当停止串；统计时再按它截一次算「有效长度」。
CONT = "\nUser:"


def plain_prompt(nums, target):
    """常规模板。判分器靠 <answer> 标签，所以标签要求要留，其他包装全去"""
    return (f"Numbers: {nums}\nTarget: {target}\n"
            f"Using each number exactly once with +, -, *, /, write an equation that equals the target. "
            f"Put the final equation in <answer> </answer> tags.\n")


TOOLST = dict(calls=0, err=0, fake=0, rounds=0, batches=0)
TOOLREC = []                                              # 跟 rec 对齐：工具模式每条 (calls, err, fake)，落盘给 --merge 用

if args.merge:
    # ── ★ 合并模式：读各分片 _raw.json，不载模型不采样，pi 按分片偏移重排，直接进统计 ──
    base = min(sh["offset"] for sh in shards); n_all = sum(sh["n"] for sh in shards)
    covered = sorted((sh["offset"], sh["offset"] + sh["n"]) for sh in shards)
    for (a0, a1), (b0, b1) in zip(covered, covered[1:]):
        assert a1 == b0, f"分片不连续：[{a0}:{a1}] 后面接的是 [{b0}:{b1}]"
    args.offset, args.n_problems, args.samples = base, n_all, shards[0]["n_samples"]
    args.max_new, args.ckpt, args.data = shards[0]["max_new"], shards[0]["ckpt"], shards[0]["data"]
    args.tool = args.tool or "calls" in shards[0]["records"][0]     # 分片是工具版就按工具版汇总
    P = D[base: base + n_all]
    rec = []
    for sh in shards:
        off = sh["offset"] - base
        for r in sh["records"]:
            rec.append((r["pi"] + off, r["txt"], r["r"], r["d"], r["L"], r["reflect"], r["search"], r["L_eff"], r["cont"]))
            if "calls" in r:
                TOOLREC.append((r["calls"], r["err"], r["fake"]))
                TOOLST["calls"] += r["calls"]; TOOLST["err"] += r["err"]; TOOLST["fake"] += r["fake"]
        ts = sh.get("toolst", {})
        TOOLST["rounds"] += ts.get("rounds", 0); TOOLST["batches"] += ts.get("batches", 0)
    T = sum(sh.get("time", 0.0) for sh in shards) or 1e-9
    ntok = sum(sh.get("ntok", 0) for sh in shards)
    if args.out is None:
        args.out = out_name()
    print(f"Countdown  共 {len(D):,} 道   ★ 合并 {len(shards)} 片：{len(rec)} 条，题 [{base}:{base+n_all}]  → {args.out}")
else:
    if args.out is None:
        args.out = out_name()
    # ── 模型 ──
    print(f"载入 {args.model} …", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model = model.to(dev).eval()
    if args.ckpt:
        c = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(c["model"]); model.to(dev).eval()
        print(f"  ★ 覆盖成 {args.ckpt}（step {c.get('step','?')}）")
        del c; torch.cuda.empty_cache()
    print(f"  {sum(p.numel() for p in model.parameters())/1e9:.2f}B 参数"
          f"   显存 {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    # ── 采样 ──
    if args.tool:
        import calc_tool
    MK = plain_prompt if args.plain else (calc_tool.tool_prompt if args.tool else CD.make_prompt)
    flat = [(i, MK(p["nums"], p["target"]))
            for i, p in enumerate(P) for _ in range(args.samples)]
    print(f"\n采样中：{len(P)} 题 × {args.samples} 次   T={args.temp}   每批 {args.bs} 条")
    print("=" * 74)

    rec, t0, ntok = [], time.time(), 0
    with torch.no_grad():
        for s in range(0, len(flat), args.bs):
            chunk = flat[s: s + args.bs]
            if args.tool:                                   # ★ 计算器 harness：停、注入、续
                res = calc_tool.generate_with_tools(model, tok, [p for _, p in chunk], max_new=args.max_new,
                                                    temp=args.temp, gen_bs=len(chunk), dev=dev)
                TOOLST["batches"] += 1; TOOLST["rounds"] += res[0]["rounds"]
                for (pi, _), o in zip(chunk, res):
                    L = len(o["ids"]); ntok += L
                    te, cont = o["txt"], o["cont"]
                    TOOLST["calls"] += o["n_calls"]; TOOLST["err"] += o["n_err"]; TOOLST["fake"] += o["fake"]
                    TOOLREC.append((o["n_calls"], o["n_err"], int(o["fake"])))
                    r, d = CD.reward(te, P[pi]["nums"], P[pi]["target"])
                    rec.append((pi, te, r, d, L, len(REFLECT.findall(te)), len(SEARCH.findall(te)), L, cont))
                done = s + len(chunk); el = time.time() - t0
                print(f"  {done:>5}/{len(flat)}  {el/60:>5.1f} 分  {ntok/el:>6.0f} tok/s  剩 {el/done*(len(flat)-done)/60:>5.1f} 分"
                      f"  调用/条 {TOOLST['calls']/done:.1f}  报错 {TOOLST['err']}  假结果 {TOOLST['fake']}", flush=True)
                continue
            enc = tok([p for _, p in chunk], return_tensors="pt", padding=True).to(dev)
            gen_kw = dict(max_new_tokens=args.max_new, do_sample=args.temp > 0,
                          temperature=max(args.temp, 1e-5), top_p=1.0, pad_token_id=tok.pad_token_id)
            try:                                            # ★ 撞到 "\nUser:" 就停，省掉几百个幻觉 token
                out = model.generate(**enc, stop_strings=[CONT], tokenizer=tok, **gen_kw)
            except (TypeError, ValueError):                 # 老版 transformers 没有 stop_strings
                out = model.generate(**enc, **gen_kw)
            new = out[:, enc.input_ids.size(1):]
            lens = (new != tok.pad_token_id).sum(1).tolist()
            ntok += sum(lens)
            for (pi, _), row, L in zip(chunk, new, lens):
                t = tok.decode(row, skip_special_tokens=True)
                cont = CONT in t                            # ★ 有没有幻觉出下一轮
                te = t.split(CONT)[0]                       # ★ 有效部分
                Le = len(tok(te).input_ids) if cont else L  # ★ 有效长度
                r, d = CD.reward(te, P[pi]["nums"], P[pi]["target"])
                rec.append((pi, te, r, d, L, len(REFLECT.findall(te)), len(SEARCH.findall(te)), Le, cont))
            done = s + len(chunk); el = time.time() - t0
            print(f"  {done:>5}/{len(flat)}  {el/60:>5.1f} 分  {ntok/el:>6.0f} tok/s"
                  f"  剩 {el/done*(len(flat)-done)/60:>5.1f} 分", flush=True)
    T = time.time() - t0

# ★ 先落盘原始数据（分片跑的也落，--merge 就是读这些）
json.dump({"model": args.model, "ckpt": args.ckpt, "n": len(P), "n_samples": args.samples,
           "offset": args.offset, "temp": args.temp, "data": args.data, "max_new": args.max_new,   # ★ 记数据路径和上限，分析脚本要用
           "time": T, "ntok": ntok, "toolst": TOOLST,
           # ★ 现在存文本了 —— 之前只存分数，回头想查措辞查不了
           "records": [dict(pi=pi, r=r, d=d, L=L, L_eff=Le, reflect=rf, search=sf, cont=ct, txt=t,
                            **(dict(calls=tr[0], err=tr[1], fake=tr[2]) if tr else {}))
                       for (pi, t, r, d, L, rf, sf, Le, ct), tr in zip(rec, TOOLREC or [None] * len(rec))]},
          open(args.out.replace(".json", "_raw.json"), "w"))
print(f"\n  ★ 原始结果已落盘 → {args.out.replace('.json','_raw.json')}", flush=True)

# ── 统计 ──
N = args.samples
COR = [[0] * 0 for _ in P]
for pi in range(len(P)):
    COR[pi] = [d["correct"] for p2, _t, _r, d, *_ in rec if p2 == pi]
counts = [int(sum(c)) for c in COR]


def pass_at_k(n, c, k):
    if n - c < k: return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


KS = [k for k in (1, 2, 4, 8, 16, 32, 64, 128) if k <= N]
if N not in KS: KS.append(N)
CURVE = {k: float(np.mean([pass_at_k(N, c, k) for c in counts])) for k in KS}

M = lambda key: float(np.mean([d[key] for *_, d, _L, _f in [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rec]]))
fmt   = float(np.mean([r[3]["fmt"]     for r in rec]))
parse = float(np.mean([r[3]["parse"]   for r in rec]))
nok   = float(np.mean([r[3]["nums_ok"] for r in rec]))
nfr   = float(np.mean([r[3].get("nums_frac", 0.0) for r in rec]))
cor   = float(np.mean([r[3]["correct"] for r in rec]))
refl  = float(np.mean([r[5] > 0 for r in rec]))          # 宽松：有没有出现
refl_n = float(np.mean([r[5] for r in rec]))             # ★ 严格：平均出现几次
srch   = float(np.mean([r[6] > 0 for r in rec]))         # ★★ 搜索措辞：有没有
srch_n = float(np.mean([r[6] for r in rec]))             # ★★ 搜索措辞：几次/条
Le     = np.array([r[7] for r in rec])                    # ★ 有效长度（截掉幻觉续写）
cont_frac = float(np.mean([r[8] for r in rec]))           # ★ 幻觉出下一轮的比例


def gap(idx):
    """★差：带某措辞的轨迹平均分 − 不带的。正 = 跟奖励挂钩，会被放大；负 = 会被压"""
    y = [r[2] for r in rec if r[idx] > 0]; n_ = [r[2] for r in rec if r[idx] == 0]
    return (float(np.mean(y)) - float(np.mean(n_))) if (y and n_) else float("nan")


ref_gap, srch_gap = gap(5), gap(6)
L = np.array([r[4] for r in rec])
n_cut = int((L >= args.max_new - 2).sum())
all_r = sum(1 for c in counts if c == N); all_w = sum(1 for c in counts if c == 0)
useful = len(P) - all_r - all_w

# ★★ 真正决定有没有梯度的，是【总分】的组内方差，不是「答案正确」这一层
#    阶梯奖励下，同一组里 0.00 / 0.10 / 0.20 混着 → 优势非零 → 有梯度
SCORE = [[] for _ in P]
for r in rec:
    SCORE[r[0]].append(r[2])
sd = np.array([np.std(g) for g in SCORE])
useful_r = int((sd > 1e-9).sum())                 # ★ 组内总分有差异 = 有梯度
sd_mean = float(sd.mean())

print("\n" + "=" * 74)
print(f"  Countdown 基线   {args.model}" + (f"  +  {args.ckpt}" if args.ckpt else ""))
print("=" * 74)
print(f"""
  ── ★ 奖励拆解（Base 模型的瓶颈在哪一层）──
    有 <answer> 标签      {fmt:.4f}   ★ 格式 —— Base 模型的第一道坎
    表达式能解析          {parse:.4f}
    ★★ 数字用对的【比例】  {nfr:.4f}   ← ★ 连续值，部分分（新增）
    数字恰好各用一次      {nok:.4f}   ← 0/1，旧口径
    ★ 答案正确            {cor:.4f}   ← 这就是 pass@1

  ── ★ pass@k 曲线（n={N} 次采样）──""")
for k in KS:
    tail = "   ← 随手一答" if k == 1 else ("   ← k=n" if k == N else "")
    print(f"    pass@{k:<5}{CURVE[k]:.4f}{tail}")
print(f"""
  ── ★★ 梯度信号（★ 按【总分】算，不是只看"答案正确"）──
    ★ 组内总分有差异   {useful_r:>4} 题 {100*useful_r/len(P):>5.1f}%   ← ★ 这些才真正产生梯度
      组内总分标准差   {sd_mean:.4f}         ← 越大梯度越强

    ── 仅看「答案正确」这一层（旧口径，会低估）──
    全对 {N}/{N}   {all_r:>4} 题 {100*all_r/len(P):>5.1f}%
    全错 0/{N}     {all_w:>4} 题 {100*all_w/len(P):>5.1f}%
    有对有错   {useful:>4} 题 {100*useful/len(P):>5.1f}%

  ── ★★ H2 基线：自我纠错措辞 ──
    宽松口径：至少出现一次            {refl:.4f}
      ⚠️ 这些词在普通解释文本里也常见，涨了不一定是纠错
    ★★ 严格口径：平均出现【几次】/条   {refl_n:.3f}
      ★ 真在搜索的模型一条会说好几次 → 训练后 >1.0 才算数
    ★差（带 wait 的 − 不带的）         {ref_gap:+.3f}

  ── ★★ 搜索措辞（too high / too low / perfect / try another …）──
    至少出现一次                      {srch:.4f}
    平均出现【几次】/条               {srch_n:.3f}
    ★★ ★差（带的 − 不带的）           {srch_gap:+.3f}   ← ★ 这才是「试错有没有用」的直接读数
      正 → 试错的轨迹更容易对，RL 会放大它；负 → 会被压掉

  ── 工具 ──""" + (f"""
    调用/条 {TOOLST['calls']/len(rec):.2f}   报错 {TOOLST['err']}   ★ 假结果轨迹 {TOOLST['fake']} ({100*TOOLST['fake']/len(rec):.1f}%)   平均轮数/批 {TOOLST['rounds']/max(1,TOOLST['batches']):.1f}""" if args.tool else "    （没开 --tool）") + f"""
  ── 长度 ──
    原始   中位 {np.median(L):.0f}  均值 {L.mean():.0f}  P95 {np.percentile(L,95):.0f}  最长 {L.max()}
    ★ 有效 中位 {np.median(Le):.0f}  均值 {Le.mean():.0f}  P95 {np.percentile(Le,95):.0f}   ← 截掉幻觉的 "User:" 续写
    幻觉出下一轮 "User:"   {100*cont_frac:.1f}%      撞上 {args.max_new}   {n_cut} 条 ({100*n_cut/len(rec):.1f}%)

  ── 速度 ──
    {T/60:.1f} 分钟  {len(rec)} 条  {ntok/T:.0f} tok/s  每题 {T/len(P):.1f} 秒{"   （合并：各片时间之和）" if args.merge else ""}
""")

# ★ 判据：阶梯奖励下，看的是「组内总分有差异的比例」，不是正确率
uf = useful_r / len(P)
if uf < 0.30:
    print(f"  ❌ 只有 {100*uf:.0f}% 的题组内总分有差异 —— ★ 大部分组优势全 0，训不动")
    print(f"     对策：再降难度（--hi 20 --tmax 50），或把奖励阶梯拆得更细")
elif uf < 0.60:
    print(f"  ⚠️ {100*uf:.0f}% 的题有梯度 —— 能训，但一半算力在空转")
else:
    print(f"  ✅ {100*uf:.0f}% 的题有梯度信号，★ 可以开训")
print(f"     组内总分标准差 {sd_mean:.4f}   （0 = 完全没信号）")

if cor < 0.02:
    print(f"  ⚠️ 但「答案正确」只有 {cor:.4f} —— ★ 80% 的奖励权重几乎拿不到")
    print(f"     → 训练早期会主要在爬「格式/解析/数字对」这三级阶梯（H4 会很明显）")
elif cor > 0.40:
    print(f"  ⚠️ 正确率 {cor:.3f} 偏高 —— 提升空间小")
else:
    print(f"  ✅ 「答案正确」{cor:.4f}，落在可训区间")

json.dump({"model": args.model, "ckpt": args.ckpt, "n": len(P), "n_samples": N,
           "offset": args.offset, "temp": args.temp, "max_new": args.max_new,
           "fmt": fmt, "parse": parse, "nums_ok": nok, "nums_frac": nfr, "correct": cor,
           "reflect": refl, "reflect_n": refl_n, "reflect_gap": ref_gap,
           "search": srch, "search_n": srch_n, "search_gap": srch_gap,
           "len_eff_median": float(np.median(Le)), "len_eff_mean": float(Le.mean()),
           "cont_frac": cont_frac, "counts": counts,
           "curve": {str(k): v for k, v in CURVE.items()},
           "useful_frac": useful/len(P), "all_right": all_r, "all_wrong": all_w,
           "useful_by_score": useful_r/len(P), "score_sd": sd_mean,
           "len_median": float(np.median(L)), "len_p95": float(np.percentile(L, 95)),
           "trunc_frac": n_cut/len(rec), "tok_per_s": ntok/T}, open(args.out, "w"), indent=1)
print(f"  → {args.out}\n")

print("─" * 74 + f"\n  样例（前 {args.show} 条）\n" + "─" * 74)
for r in rec[: args.show]:
    pi, t, sc, d, L_, rf, sf, Le_, ct = r
    tags = (f"  ★ 纠错 ×{rf}" if rf else "") + (f"  ★★ 搜索 ×{sf}" if sf else "") + ("  ⚠ 幻觉续写" if ct else "")
    print(f"\n  【题 {pi}】nums {P[pi]['nums']}  target {P[pi]['target']}"
          f"   得分 {sc:.2f}  {Le_} token{tags}")
    print("  " + t.strip().replace("\n", "\n  ")[:700])
