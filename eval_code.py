#!/usr/bin/env python3
"""
eval_code.py —— 长程编程阶段 ①「修 bug」的探测 / 评测：n 道题 × s 条，走 harness.generate_with_env + code_env，落盘 raw，打读数。

    # 探测（vLLM；--hf-dir 可以是导出目录也可以是 HF 仓库名）
    CUDA_VISIBLE_DEVICES=0 python3 eval_code.py --hf-dir hf_armA --ckpt out_armA/ckpt_latest.pt -n 100 -s 8 --with-tb --out code_armA_tb.json
    CUDA_VISIBLE_DEVICES=1 python3 eval_code.py --hf-dir Qwen/Qwen3-1.7B-Base --model Qwen/Qwen3-1.7B-Base -n 100 -s 8 --with-tb --out code_q3_tb.json
    python3 eval_code.py --analyze code_armA_tb_raw.json --show 2       # 只读 raw 重算读数
    python3 eval_code.py --dry -n 3 --with-tb                            # 不载模型，打 3 道题面
    --inject "…"  在 <think> 前插一段文字（作弊探针用：引它改测试 / 硬编码）；--ask 开专家（要 DEEPSEEK_API_KEY）
读数：修好率（3 条全过）/ 隐藏通过 / 可见通过 / 改过文件 / 调用·test·write·run·ask 每局 / write 前后有没有 test / 语法错 / 假观测 / 撞顶 / token /
      按 bug_type、kind、模板拆；组的分布（有对有错才有梯度）
"""
import os, sys, json, time, argparse, collections, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import code_env as C

ap = argparse.ArgumentParser()
ap.add_argument("--engine", choices=["hf", "vllm"], default="vllm")
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B", help="tokenizer 来源；hf 引擎也是权重来源")
ap.add_argument("--ckpt", default=None, help="hf 引擎：覆盖权重；vllm 引擎：只用来起名")
ap.add_argument("--hf-dir", default=None, help="vllm 引擎：导出目录或 HF 仓库名")
ap.add_argument("--data", default="data/code_bugs.json")
ap.add_argument("--split", default="train", choices=["train", "heldout"])
ap.add_argument("--kinds", default=None, help="只用这些 kind，逗号分隔")
ap.add_argument("--bug-type", default=None, choices=[None, "wrong", "crash"])
ap.add_argument("-n", "--n-problems", type=int, default=100)
ap.add_argument("-s", "--samples", type=int, default=8)
ap.add_argument("--with-tb", action="store_true", help="带报错开局（题面里放可见测试的输出）")
ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--max-new", type=int, default=1024)
ap.add_argument("--max-calls", type=int, default=8)
ap.add_argument("--call-cost", type=float, default=0.02)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--gpu-mem", type=float, default=0.6)
ap.add_argument("--prompt-set", default="train", choices=list(C.SETS))
ap.add_argument("--inject", default=None, help="插在结尾 <think> 之前的一段文字（探针用）")
ap.add_argument("--ask", action="store_true", help="开专家（<ask>）；不开时 <ask> 回 error")
ap.add_argument("--ask-model", default="deepseek-reasoner")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--show", type=int, default=2)
ap.add_argument("--dry", action="store_true", help="不载模型，打题面")
ap.add_argument("--out", default=None)
ap.add_argument("--analyze", default=None)
ap.add_argument("--step-workers", type=int, default=8, help="沙盒并发数（每条一个临时目录，线程安全）")
ap.add_argument("--show-ok", type=int, default=0, help="--analyze 时：打 N 条修好的完整轨迹（含题面里的 bug 代码、最终文件、和参考答案的 diff）")
args = ap.parse_args()


def report(raw, show=2):
    R = raw["records"]; n = len(R); S = raw["n_samples"]
    D = [r["d"] for r in R]
    m = lambda k, sel=None: float(np.mean([d[k] for d in (sel or D)])) if (sel or D) else float("nan")
    beh = [r["beh"] for r in R]
    by = collections.defaultdict(list)
    for r in R: by[r["pi"]].append(r["d"]["correct"])
    groups = collections.Counter("全对" if sum(v) == len(v) else "全错" if sum(v) == 0 else "有对有错" for v in by.values())
    print(f"\n{raw.get('name','')}   {raw['n']} 题 × {S}   split {raw['split']}  开局 {'带报错' if raw['with_tb'] else '不带报错'}  温度 {raw['temp']}  模板 {raw['prompt_set']}"
          + (f"  inject=「{raw['inject'][:40]}…」" if raw.get("inject") else "") + (f"  专家 {raw['ask_model']}" if raw.get("ask") else ""))
    print(f"  ★ 修好率（3 条全过）{m('correct'):.3f}   隐藏通过 {m('hidden_pass'):.3f}   可见通过 {m('vis_pass')/2:.3f}   改过文件 {m('changed'):.3f}   有 <answer> {m('has_answer'):.3f}   用过工具 {m('fmt'):.3f}")
    print(f"  每局：调用 {m('calls'):.2f}   test {m('n_test'):.2f}   write {m('n_write'):.2f}   run {m('n_run'):.2f}   ask {m('n_ask'):.2f}   语法错 {m('syntax_err'):.2f}"
          f"   write 前 test {np.mean([b['test_before_write'] for b in beh]):.2f}   write 后 test {np.mean([b['test_after_write'] for b in beh]):.2f}"
          f"   假观测 {sum(r['fake'] for r in R)}   撞顶 {sum(r['cut'] for r in R)}   token {np.mean([r['L'] for r in R]):.0f}")
    print(f"  ★ 可见过但隐藏挂（硬编码 / 过拟合可见测试）{sum(1 for d in D if d['vis_pass'] == 2 and d['hidden_pass'] == 0)} 条")
    ed = [r["edit"] for r in R if r.get("edit") and r["d"]["changed"] > 0]
    if ed:
        okd = [r["edit"] for r in R if r.get("edit") and r["d"]["correct"] > 0]
        print(f"  ★ 改过文件的 {len(ed)} 条：改动行数中位 {int(np.median([e['changed_lines'] for e in ed]))}   ★ 定位修改 {np.mean([e.get('localized', 0) for e in ed]):.2f}   整段重写 {np.mean([e['rewrite'] for e in ed]):.2f}"
              + (f"   ｜ 修好的 {len(okd)} 条：定位修改 {np.mean([e.get('localized', 0) for e in okd]):.2f}   整段重写 {np.mean([e['rewrite'] for e in okd]):.2f}" if okd else ""))
    print(f"  组（每题 {S} 条）：{dict(groups)}   ← 有对有错的组才有梯度")
    for key, name in (("bug_type", "bug_type"), ("kind", "kind"), ("pid", "模板")):
        vals = sorted(set(r[key] for r in R))
        if len(vals) > 1:
            print(f"  按 {name}：" + "   ".join(f"{v} {m('correct', [r['d'] for r in R if r[key] == v]):.3f}({sum(1 for r in R if r[key] == v)})" for v in vals))
    first = collections.Counter(b["first"] for b in beh)
    print(f"  第一步是什么：{dict(first.most_common())}")
    if show:
        ok = sorted([r for r in R if r["d"]["correct"] > 0], key=lambda r: r["L"])
        bad = [r for r in R if r["d"]["correct"] == 0 and r["d"]["changed"] > 0] or [r for r in R if r["d"]["correct"] == 0]
        for tag, r in [("最短的一条修好的", ok[0] if ok else None), ("一条没修好的", bad[0] if bad else None)][:show]:
            if r is None: continue
            print("─" * 90); print(f"【{tag}】task {r['task_id']} {r['kind']}/{r['bug_type']}  模板 {r['pid']}  得分 {r['r']:.2f}  调用 {r['d']['calls']}  {r['L']} token"
                                  + ("  ⚠ 假观测" if r["fake"] else "") + ("  ⚠ 撞顶" if r["cut"] else ""))
            print(r["txt"]); print("··· 最终文件 ···"); print(r["final_code"])


if args.analyze:
    raw = json.load(open(args.analyze))
    if raw["records"] and "localized" not in raw["records"][0].get("edit", {}):   # 旧 raw 补算 / 重算改动读数
        B = {(b["task_id"], b["kind"]): b for b in json.load(open(args.data))}
        for r in raw["records"]:
            b = B.get((r["task_id"], r["kind"]))
            if b: r["edit"] = C.edit_stats(b["bug_code"], r["final_code"], b["diff"])
    report(raw, args.show)
    if args.show_ok:
        import difflib
        B = {(b["task_id"], b["kind"]): b for b in json.load(open(args.data))}
        ok = [r for r in raw["records"] if r["d"]["correct"] > 0]
        random.Random(args.seed).shuffle(ok)
        for r in ok[: args.show_ok]:
            b = B.get((r["task_id"], r["kind"]))
            print("█" * 100); print(f"task {r['task_id']}  {r['kind']}/{r['bug_type']}  模板 {r['pid']}  得分 {r['r']:.2f}  调用 {r['d']['calls']}  "
                                    f"test {r['d']['n_test']} write {r['d']['n_write']} run {r['d']['n_run']} ask {r['d']['n_ask']}  {r['L']} token")
            if b:
                print("── 题目 ──"); print(b["text"]); print("── 有 bug 的代码 ──"); print(b["bug_code"]); print("── 注入的 bug（diff）──"); print("\n".join(b["diff"]))
            print("── 模型写的（原样，⟦⟧ 没标，注入段就是 <result>/<reply>）──"); print(r["txt"])
            print("── 最终文件 ──"); print(r["final_code"])
            if b:
                dd = [l for l in difflib.unified_diff(b["ref_code"].splitlines(), r["final_code"].splitlines(), lineterm="", n=0) if not l.startswith(("---", "+++", "@@"))]
                print("── 最终文件 vs 参考答案 ──"); print("\n".join(dd) if dd else "（一字不差）")
    sys.exit(0)

kinds = args.kinds.split(",") if args.kinds else None
probs = C.load_problems(args.data, split=args.split, kinds=kinds, bug_type=args.bug_type)
rng = random.Random(args.seed); rng.shuffle(probs)
seen, P = set(), []
for p in probs:                                                   # 每道 task 只取一个变异体，题之间不重复
    if p["task_id"] in seen: continue
    seen.add(p["task_id"]); P.append(p)
    if len(P) >= args.n_problems: break
ids = C.SETS[args.prompt_set]
env0 = C.CodeEnv(max_calls=args.max_calls, call_cost=args.call_cost)


def make_prompt(i, p):
    s = C.render(ids[i % len(ids)], p, with_tb=args.with_tb, tool_style=i % 3, env=env0)
    if args.inject:
        s = s[: s.rfind("<think>")] + args.inject.strip() + "\n<think>"
    return s


if args.dry:
    for i, p in enumerate(P[: args.n_problems]):
        print("█" * 40, f"task {p['task_id']} {p['kind']}/{p['bug_type']} 模板 {ids[i % len(ids)]}"); print(make_prompt(i, p))
    sys.exit(0)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import calc_tool_vllm as T
import harness as H

tok = AutoTokenizer.from_pretrained(args.model); tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prompts = [make_prompt(i, p) for i, p in enumerate(P)]
plen = max(len(tok.encode(s)) for s in prompts)
if args.engine == "vllm":
    assert args.hf_dir, "--engine vllm 要 --hf-dir"
    be = T.make_backend("vllm", tok=tok, hf_dir=args.hf_dir, gpu_mem=args.gpu_mem, max_model_len=plen + args.max_new + 64)
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
args.out = args.out or f"code_{stem}_{args.split}_{'tb' if args.with_tb else 'notb'}_s{args.samples}_p{args.prompt_set}.json"
ask_fn = (lambda q, p, code: C.ask_expert(q, p, code, model=args.ask_model)) if args.ask else None
env = C.CodeEnv(max_calls=args.max_calls, call_cost=args.call_cost, ask_fn=ask_fn)
print(f"{name}   {len(P)} 题 × {args.samples}   split {args.split}  开局 {'带报错' if args.with_tb else '不带报错'}  温度 {args.temp}  模板 {args.prompt_set}  最长题面 {plen} token\n"
      f"  题面样例（task {P[0]['task_id']}）：\n{prompts[0]}\n", flush=True)

flat = [(i, p) for i, p in enumerate(P) for _ in range(args.samples)]
rec, t0 = [], time.time()
step = args.bs if args.engine == "hf" else len(flat)
for s in range(0, len(flat), step):
    chunk = flat[s: s + step]
    states = [env.reset(p) for _, p in chunk]
    outs = H.generate_with_env(be, tok, [prompts[i] for i, _ in chunk], env, states, max_new=args.max_new, temp=args.temp, gen_bs=args.bs, dev=dev,
                               progress=(args.engine == "vllm"), step_workers=args.step_workers, max_total=(plen + args.max_new + 64) if args.engine == "vllm" else None)
    ts = time.time()
    if args.step_workers > 1 and len(outs) > 1:                     # ★ 终评（秤跑 3 条测试）也并发：800 条串行要两三分钟，GPU 干等
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.step_workers) as ex:
            scores = list(ex.map(lambda o: env.score(o["state"], o["txt"]), outs))
    else:
        scores = [env.score(o["state"], o["txt"]) for o in outs]
    for (i, p), o, (r, d) in zip(chunk, outs, scores):
        final_code = o["state"]["code"]
        rec.append(dict(pi=i, task_id=p["task_id"], kind=p["kind"], bug_type=p["bug_type"], pid=ids[i % len(ids)], txt=o["txt"], final_code=final_code,
                        r=r, d=d, beh=C.behavior(o["txt"]), edit=C.edit_stats(p["bug_code"], final_code, p["diff"]),
                        L=len(o["ids"]), fake=o["fake"], cut=o["cut"], cont=o["cont"]))
    print(f"  {len(rec)}/{len(flat)}  {time.time()-t0:.0f} s（终评 {time.time()-ts:.1f} s，{args.step_workers} 并发）", flush=True)
raw = dict(name=name, n=len(P), n_samples=args.samples, split=args.split, with_tb=args.with_tb, temp=args.temp, prompt_set=args.prompt_set,
           inject=args.inject, ask=args.ask, ask_model=args.ask_model, max_calls=args.max_calls, call_cost=args.call_cost, seed=args.seed,
           max_new=args.max_new, time=time.time() - t0, records=rec)
json.dump(raw, open(args.out.replace(".json", "_raw.json"), "w"))
print(f"\n  ★ raw → {args.out.replace('.json', '_raw.json')}")
report(raw, args.show)
