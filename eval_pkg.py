#!/usr/bin/env python3
"""
eval_pkg.py —— 阶段 ②-A 的评测器：多文件包（eval_code 的多文件版）。
    CUDA_VISIBLE_DEVICES=0 python3 eval_pkg.py --hf-dir hf_codeRL_dyn --model hf_coder_1.5b -K 3 -n 100 -s 8 --out dyn_K3.json
    python3 eval_pkg.py --analyze dyn_K3_raw.json --p1 0.73          # 只读 raw 重算读数，--p1 给单文件的修好率就打 K 次方基线
    python3 eval_pkg.py --dry -K 3 -n 2                                # 不载模型，打题面
读数：修好率（全部通过）/ 文件修好比例 / 测试通过比例 / 可见全绿但隐藏挂 / ★ 按题面位置的修好率 / ★ 按写的顺序的修好率 / 修好文件数分布 /
      每局调用·test·write·run·ask / 不带文件名的 write / 语法错 / 撞顶 / 状态行 / 组分布 / 按 kind、模板 / 死法分类
"""
import os, sys, json, time, argparse, collections, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import code_env as C
import pkg_env as P


def report(raw, show=2, p1=None):
    R = raw["records"]; n = len(R); S = raw["n_samples"]; K = raw["K"]
    D = [r["d"] for r in R]
    m = lambda k: float(np.mean([d[k] for d in D])) if D else float("nan")
    beh = [r["beh"] for r in R]
    by = collections.defaultdict(list)
    for r in R: by[r["pi"]].append(r["d"]["correct"])
    groups = collections.Counter("全对" if sum(v) == len(v) else "全错" if sum(v) == 0 else "有对有错" for v in by.values())
    print(f"\n{raw.get('name','')}   {raw['n']} 包 × {S}   K={K}   split {raw['split']}  开局 {'带报错' if raw['with_tb'] else '不带报错'}  温度 {raw['temp']}  模板 {raw['prompt_set']}"
          + (f"  inject=「{raw['inject'][:40]}…」" if raw.get("inject") else "") + (f"  奖励 {raw.get('reward')}" if raw.get("reward") not in (None, "bin") else ""))
    fixed = m("correct")
    print(f"  ★ 修好率（全部通过）{fixed:.3f}   文件修好比例 {m('frac_files'):.3f}   测试通过比例 {m('frac_tests'):.3f}   可见全绿 {m('vis_all'):.3f}   隐藏全过 {m('hidden_pass'):.3f}"
          f"   有 <answer> {m('has_answer'):.3f}   用过工具 {m('fmt'):.3f}"
          + (f"   ｜ K 次方基线 {p1 ** K:.3f}（p1={p1}）→ 长度代价 {p1 ** K - fixed:+.3f}" if p1 else ""))
    print(f"  ★ 可见全绿但隐藏挂（蒙过 2 条可见、被 20 条隐藏抓住）{sum(1 for d in D if d['vis_all'] and not d['hidden_pass'])} 条")
    print(f"  每局：调用 {m('calls'):.2f}   test {m('n_test'):.2f}   write {m('n_write'):.2f}   run {m('n_run'):.2f}   ask {m('n_ask'):.2f}   坏 write {m('n_bad_write'):.2f}"
          f"   语法错 {m('syntax_err'):.2f}   write 前 test {np.mean([b['test_before_write'] for b in beh]):.2f}   write 后 test {np.mean([b['test_after_write'] for b in beh]):.2f}"
          f"   假观测 {sum(r['fake'] for r in R)}   撞顶 {sum(r['cut'] for r in R)}   token {np.mean([r['L'] for r in R]):.0f}   碰过文件 {m('files_touched'):.2f}/{K}"
          f"   状态行 {np.mean([b['n_state_lines'] for b in beh]):.2f}")
    # ★ 按位置 / 按写的顺序
    pos = collections.defaultdict(list); wro = collections.defaultdict(list); never = []
    for r in R:
        per = r["d"]["per_file"]; names = [p["name"] for p in per]
        for i, p in enumerate(per): pos[i].append(p["correct"])
        order = []
        for f in r["beh"]["files_written"]:
            if f in names and f not in order: order.append(f)
        for rank, f in enumerate(order): wro[rank].append(per[names.index(f)]["correct"])
        for p in per:
            if p["name"] not in order: never.append(p["correct"])
    print("  ★ 按题面位置的修好率：" + "   ".join(f"第{i+1}个 {np.mean(v):.3f}" for i, v in sorted(pos.items()))
          + "   ｜ 按写的顺序：" + "   ".join(f"第{i+1}写 {np.mean(v):.3f}({len(v)})" for i, v in sorted(wro.items())) + (f"   没碰过 {np.mean(never):.3f}({len(never)})" if never else ""))
    hist = collections.Counter(d["files_fixed"] for d in D)
    print("  修好文件数分布：" + "   ".join(f"{k}/{K}: {hist.get(k, 0)}" for k in range(K + 1)) + f"   ← 全修好才得分；差一个的 {hist.get(K - 1, 0)} 条")
    ed = [p for r in R for p in r["d"]["per_file"] if p["changed"]]
    if ed:
        print(f"  ★ 改过的文件 {len(ed)} 个：定位修改 {np.mean([e['localized'] for e in ed]):.2f}   整段重写 {np.mean([e['rewrite'] for e in ed]):.2f}   改中注入行 {np.mean([e['hit_bug_line'] for e in ed]):.2f}")
    # write 的形态（探针里见过 CDATA 包裹、把题面的 === 文件头写进 write、空 write、多文件塞进一个 write）
    import re as _re
    W = [(a, b) for r in R for a, b in P.WRITE_RE.findall(r["txt"])]
    if W:
        nf = sum(1 for a, _ in W if P.FILE_IN_ATTR.search(a)); cd = sum(1 for _, b in W if "CDATA" in b); hd = sum(1 for _, b in W if "=== " in b)
        em = sum(1 for _, b in W if not b.strip()); ml = sum(1 for _, b in W if b.count("=== ") >= 2)
        errs = collections.Counter(_re.sub(r"\(line \d+\)", "", m).strip() for r in R for m in _re.findall(r"but SyntaxError: ([^<]*)", r["txt"]))
        print(f"  write 形态（共 {len(W)} 次）：带文件名 {nf/len(W):.2f}   CDATA 包裹 {cd}   把 === 文件头写进去 {hd}（其中一次写多个文件 {ml}）   空 write {em}"
              + (f"   ｜ 语法错前三种：{errs.most_common(3)}" if errs else ""))
    print(f"  组（每包 {S} 条）：{dict(groups)}   ← 有对有错的组才有梯度")
    # 死法
    dead = collections.Counter()
    for r in R:
        d = r["d"]
        if d["correct"]: continue
        if r["fake"]: dead["自己编观测"] += 1
        elif d["n_bad_write"] > 0 and d["files_touched"] < K: dead["write 没带对文件名"] += 1
        elif r["cut"]: dead["撞顶(token)"] += 1
        elif d["calls"] >= raw.get("max_calls", 16): dead["撞调用上限"] += 1
        elif d["n_write"] == 0: dead["一个文件都没改"] += 1
        elif d["has_answer"] and d["files_fixed"] < K: dead["没修完就交卷"] += 1
        elif d["syntax_err"] > 0: dead["留下语法错"] += 1
        else: dead["其它"] += 1
    print(f"  没修好的 {n - int(fixed * n)} 条死法：{dict(dead.most_common())}")
    byk = collections.defaultdict(list)
    for r in R:
        for p, kd in zip(r["d"]["per_file"], r["kinds"]): byk[kd].append(p["correct"])
    print("  按 kind（每文件）：" + "   ".join(f"{k} {np.mean(v):.3f}({len(v)})" for k, v in sorted(byk.items())))
    byt = collections.defaultdict(list)
    for r in R: byt[r["pid"]].append(r["d"]["correct"])
    print("  按模板：" + "   ".join(f"{k} {np.mean(v):.3f}({len(v)})" for k, v in sorted(byt.items())))
    print(f"  第一步是什么：{dict(collections.Counter(b['first'] for b in beh))}")
    if show:
        ok = sorted([r for r in R if r["d"]["correct"]], key=lambda r: r["L"])
        bad = [r for r in R if not r["d"]["correct"]]
        for tag, rr in (("最短的一条修好的", ok[:1]), ("一条没修好的", bad[:1])):
            for r in rr:
                print("─" * 90); print(f"【{tag}】{r['pkg_id']}  {r['kinds']}  模板 {r['pid']}  得分 {r['r']:.2f}  调用 {r['d']['calls']}  {r['L']} token"
                                      + ("  ⚠ 撞顶" if r["cut"] else "") + f"  文件 {r['d']['files_fixed']}/{K} 修好")
                print(r["txt"])
                print("··· 每个文件 ···" + "   ".join(f"{p['name']} {'✓' if p['correct'] else '✗'}（可见 {p['vis_pass']}/2 隐藏 {'过' if p['hidden_pass'] else '挂'}）" for p in r["d"]["per_file"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hf_coder_1.5b", help="tokenizer 来源")
    ap.add_argument("--hf-dir", default=None, help="vllm 引擎：导出目录或 HF 仓库名")
    ap.add_argument("--data", default="data/code_bugs.json")
    ap.add_argument("--split", default="train", choices=["train", "heldout"])
    ap.add_argument("-K", type=int, default=3)
    ap.add_argument("-n", "--n-pkgs", type=int, default=100)
    ap.add_argument("-s", "--samples", type=int, default=8)
    ap.add_argument("--with-tb", action="store_true")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--max-calls", type=int, default=16)
    ap.add_argument("--call-cost", type=float, default=0.01)
    ap.add_argument("--reward", default="bin", choices=["bin", "frac"])
    ap.add_argument("--gpu-mem", type=float, default=0.6)
    ap.add_argument("--prompt-set", default="train", choices=list(P.SETS))
    ap.add_argument("--inject", default=None)
    ap.add_argument("--ask", action="store_true")
    ap.add_argument("--ask-model", default="deepseek-reasoner")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", type=int, default=2)
    ap.add_argument("--p1", type=float, default=None, help="单文件修好率 → 打 K 次方基线")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--analyze", default=None)
    ap.add_argument("--step-workers", type=int, default=32)
    ap.add_argument("--inflight", type=int, default=128, help="异步 harness 同时在引擎里的轨迹数（KV cache 装得下的量级；800 条全塞进去前缀会被挤掉、prefill 重算）")
    args = ap.parse_args()

    if args.analyze:
        report(json.load(open(args.analyze)), args.show, args.p1); return

    recs = C.load_problems(args.data, split=args.split)
    ext = P.load_ext()
    pkgs = P.make_pkgs(recs, args.K, args.n_pkgs, seed=args.seed, ext=ext)
    ids = P.SETS[args.prompt_set]
    env0 = P.PkgEnv(max_calls=args.max_calls, call_cost=args.call_cost)

    def make_prompt(i, pkg):
        s = P.render(ids[i % len(ids)], pkg, with_tb=args.with_tb, tool_style=i % 3, env=env0)
        if args.inject:
            s = s[: s.rfind("<think>")] + args.inject.strip() + "\n<think>"
        return s

    if args.dry:
        for i, pkg in enumerate(pkgs):
            print("█" * 40, f"{pkg['pkg_id']} {pkg['kinds']} 模板 {ids[i % len(ids)]}"); print(make_prompt(i, pkg))
        return

    import torch
    from transformers import AutoTokenizer
    import calc_tool_vllm as T
    import harness as H
    from concurrent.futures import ThreadPoolExecutor
    assert args.hf_dir, "要 --hf-dir"
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [make_prompt(i, pkg) for i, pkg in enumerate(pkgs)]
    plen = max(len(tok.encode(s)) for s in prompts)
    be = T.make_backend("vllm", tok=tok, hf_dir=args.hf_dir, gpu_mem=args.gpu_mem, max_model_len=plen + args.max_new + 64)
    name = os.path.basename(args.hf_dir.rstrip("/"))
    args.out = args.out or f"pkg_{name}_K{args.K}_{args.split}_{'tb' if args.with_tb else 'notb'}_s{args.samples}_p{args.prompt_set}.json"
    ask_fn = (lambda q, pkg, files: P.ask_expert(q, pkg, files, model=args.ask_model)) if args.ask else None
    env = P.PkgEnv(max_calls=args.max_calls, call_cost=args.call_cost, ask_fn=ask_fn, reward=args.reward)
    print(f"{name}   {len(pkgs)} 包 × {args.samples}   K={args.K}   split {args.split}  开局 {'带报错' if args.with_tb else '不带报错'}  温度 {args.temp}  模板 {args.prompt_set}"
          f"  最长题面 {plen} token  隐藏测试 {np.mean([len(f['hidden_tests']) for pkg in pkgs for f in pkg['files']]):.1f} 条/文件\n  题面样例（{pkgs[0]['pkg_id']}）：\n{prompts[0]}\n", flush=True)

    flat = [(i, pkg) for i, pkg in enumerate(pkgs) for _ in range(args.samples)]
    t0 = time.time()
    states = [env.reset(pkg) for _, pkg in flat]
    outs = H.generate_with_env(be, tok, [prompts[i] for i, _ in flat], env, states, max_new=args.max_new, temp=args.temp,
                               progress=True, step_workers=args.step_workers, max_total=plen + args.max_new + 64, inflight=args.inflight)
    ts = time.time()
    with ThreadPoolExecutor(max_workers=args.step_workers) as ex:
        scores = list(ex.map(lambda o: env.score(o["state"], o["txt"]), outs))
    rec = []
    for (i, pkg), o, (r, d) in zip(flat, outs, scores):
        rec.append(dict(pi=i, pkg_id=pkg["pkg_id"], K=pkg["K"], task_ids=pkg["task_ids"], kinds=pkg["kinds"], pid=ids[i % len(ids)], txt=o["txt"],
                        files=dict(o["state"]["files"]), r=r, d=d, beh=P.behavior(o["txt"]), L=len(o["ids"]), fake=o["fake"], cut=o["cut"], cont=o["cont"]))
    print(f"  {len(rec)} 条  {time.time()-t0:.0f} s（终评 {time.time()-ts:.1f} s）", flush=True)
    raw = dict(name=name, n=len(pkgs), n_samples=args.samples, K=args.K, split=args.split, with_tb=args.with_tb, temp=args.temp, prompt_set=args.prompt_set,
               inject=args.inject, ask=args.ask, max_calls=args.max_calls, call_cost=args.call_cost, reward=args.reward, seed=args.seed, max_new=args.max_new,
               time=time.time() - t0, records=rec)
    json.dump(raw, open(args.out.replace(".json", "_raw.json"), "w"))
    print(f"\n  ★ raw → {args.out.replace('.json', '_raw.json')}")
    report(raw, args.show, args.p1)


if __name__ == "__main__":
    main()
