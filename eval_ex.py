#!/usr/bin/env python3
"""
eval_ex.py —— 阶段 ①′ / ③ 的评测器：从头写 exercism 题（ex_env）；--obs weak = 阶段 ③ 弱观测（不给测试，<run> 自己验证）。
    CUDA_VISIBLE_DEVICES=0 python3 eval_ex.py --hf-dir hf_pkgRL_bin --model hf_coder_1.5b -n 100 -s 8 --out rlb_ex.json
    CUDA_VISIBLE_DEVICES=0 python3 eval_ex.py --hf-dir hf_exRL_bin --model hf_coder_1.5b -n 100 -s 8 --obs weak --out rlex_weak.json
    python3 eval_ex.py --analyze rlb_ex_raw.json --show 1          # 只读 raw 重算读数
    python3 eval_ex.py --dry -n 2                                    # 不载模型，打题面
读数：通过率（全部测试全过）/ 可见全绿 / 隐藏全过 / ★ 首版通过率与首版通过比例 / ★ 改了几轮、每轮多过几条 / 可见全绿但隐藏挂 /
      每局调用·test·write·run·ask / 语法错 / 撞顶 / 撞调用上限 / 没写完就交卷 / 组分布 / 按模板 / 按参考行数、带不带类 / 死法分类
      ★ 首版（全秤）/ 改一轮效果（全秤，按版本）/ 弱观测四读数：自测率、交卷前验过最终版、真阳·假阴·假阳·真阴、自测全绿但隐藏挂 / 赢家的调用数分布
"""
import os, sys, json, time, argparse, collections, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import code_env as C
import ex_env as E


def report(raw, show=2):
    R = raw["records"]; n = len(R); S = raw["n_samples"]
    D = [r["d"] for r in R]
    m = lambda k, sel=None: float(np.mean([d[k] for d in (sel if sel is not None else D)])) if (sel if sel is not None else D) else float("nan")
    beh = [r["beh"] for r in R]
    by = collections.defaultdict(list)
    for r in R: by[r["pi"]].append(r["d"]["correct"])
    groups = collections.Counter("全对" if sum(v) == len(v) else "全错" if sum(v) == 0 else "有对有错" for v in by.values())
    weak = raw.get("obs", "strong") == "weak"
    print(f"\n{raw.get('name','')}   {raw['n']} 题 × {S}   split {raw['split']}  温度 {raw['temp']}  模板 {raw['prompt_set']}  max_new {raw.get('max_new')}  观测 {'弱（不给测试）' if weak else '强（可见七成）'}"
          + (f"  inject=「{raw['inject'][:40]}…」" if raw.get("inject") else ""))
    solved = m("correct")
    print(f"  ★ 通过率（全部测试全过）{solved:.3f}   可见全绿 {m('vis_all'):.3f}   隐藏全过 {m('hid_all'):.3f}   测试通过比例 {m('frac_tests'):.3f}   有 <answer> {m('has_answer'):.3f}   用过工具 {m('fmt'):.3f}")
    if not weak:
        print(f"  ★ 可见全绿但隐藏挂（只对给它看的测试，被藏起来的抓住）{sum(1 for d in D if d['vis_all'] and not d['hid_all'])} 条")
    with_first = [d for d in D if d["first_pass_frac"] is not None]
    if with_first:
        print(f"  ★ 首版（第一次 write 后第一次 test）：有首版 {len(with_first)}/{n}   首版可见全过 {m('first_all', with_first):.3f}   首版通过比例 {m('first_pass_frac', with_first):.3f}"
              f"   改了几轮 {m('n_rounds'):.2f}   模块 import 失败 {m('load_error'):.2f}   最终代码行数 {m('code_lines'):.0f}")
    wf = [d for d in D if d.get("first_ver_frac") is not None]
    if wf:                                                              # ★ 首版（全秤）：第一次 write 的版本跑全部测试 —— 强弱两种观测下同一把尺
        print(f"  ★ 首版（全秤，第一次 write 的版本跑全部测试）：有首版 {len(wf)}/{n}   首版全过 {m('first_ver_all', wf):.3f}   首版通过比例 {m('first_ver_frac', wf):.3f}"
              f"   改了几轮 {m('n_rounds'):.2f}   模块 import 失败 {m('load_error'):.2f}   最终代码行数 {m('code_lines'):.0f}")
    vg = []                                                             # 改一轮的效果（全秤）：相邻两个都评过的版本
    for d in D:
        vf = dict((v, f) for v, f in d.get("ver_frac", []))
        vg += [vf[v + 1] - vf[v] for v in vf if v + 1 in vf]
    if vg:
        g = np.array(vg)
        print(f"  ★ 改一轮的效果（全秤，{len(g)} 轮）：平均多过 {g.mean():+.3f}（占全部测试比例）   变好 {np.mean(g > 0):.2f}   没变 {np.mean(g == 0):.2f}   变差 {np.mean(g < 0):.2f}")
    if weak or any(d.get("n_selftest") for d in D):
        n_st = sum(1 for d in D if d.get("n_selftest", 0) > 0)
        direct = np.mean([d.get("n_selftest", 0) == 0 and d["has_answer"] for d in D])
        print(f"  ★ 自测：自测率（交卷前 ≥1 次带 assert 的 run）{n_st / n:.3f}   交卷前验过最终版 {m('verified_final'):.3f}   不自测就交卷 {direct:.3f}   试图 <test> {m('n_test'):.2f}/局")
        tp, fn, fp, tn = (sum(d.get(k, 0) for d in D) for k in ("n_tp", "n_fn", "n_fp", "n_tn")); tot = tp + fn + fp + tn
        if tot:
            print(f"  ★ 自测质量（{tot} 次断言 run）：真阳（自测挂·隐藏也挂）{tp / tot:.2f}   假阴（自测绿·隐藏挂 = 断言太弱）{fn / tot:.2f}   假阳（自测挂·隐藏过 = 断言写错）{fp / tot:.2f}   真阴 {tn / tot:.2f}"
                  f"   自测全绿但隐藏挂 {sum(d.get('selfgreen_hidfail', 0) for d in D)} 条")
    # 每轮多过几条：hist 里相邻两次 test 之间通过数的变化（只算中间有 write 的）
    gains = []
    for d in D:
        h = d["hist"]
        for a, b in zip(h, h[1:]):
            if b["after_writes"] > a["after_writes"]:
                gains.append((b["k"] - a["k"]) / max(1, a["n"]))
    if gains:
        g = np.array(gains)
        print(f"  ★ 改一轮的效果（可见测试，{len(g)} 轮）：平均多过 {g.mean():+.3f}（占可见比例）   变好 {np.mean(g > 0):.2f}   没变 {np.mean(g == 0):.2f}   变差 {np.mean(g < 0):.2f}")
    print(f"  每局：调用 {m('calls'):.2f}   test {m('n_test'):.2f}   write {m('n_write'):.2f}   run {m('n_run'):.2f}   ask {m('n_ask'):.2f}   语法错 {m('syntax_err'):.2f}"
          f"   write 前 test {np.mean([b['test_before_write'] for b in beh]):.2f}   write 后 test {np.mean([b['test_after_write'] for b in beh]):.2f}"
          f"   假观测 {sum(r['fake'] for r in R)}（其中最终代码碰巧对 {sum(1 for r in R if r['fake'] and r['d']['correct'])}）   撞顶 {sum(r['cut'] for r in R)}   token {np.mean([r['L'] for r in R]):.0f}")
    print(f"  组（每题 {S} 条）：{dict(groups)}   ← 有对有错的组才有梯度")
    dead = collections.Counter()
    for r in R:
        d = r["d"]
        if d["correct"]: continue
        if r["fake"]: dead["自己编观测"] += 1
        elif r["cut"]: dead["撞顶(token)"] += 1
        elif d["calls"] >= raw.get("max_calls", 12): dead["撞调用上限"] += 1
        elif d["n_write"] == 0: dead["没写就交卷"] += 1
        elif d["load_error"]: dead["最终文件 import 不了"] += 1
        elif weak and d["has_answer"] and d.get("selfgreen_hidfail"): dead["自测全绿但隐藏挂"] += 1
        elif weak and d["has_answer"] and d.get("n_selftest", 0) == 0: dead["没自测就交卷"] += 1
        elif weak and d["has_answer"] and not d.get("verified_final"): dead["改完没再验就交卷"] += 1
        elif weak and d["has_answer"]: dead["自测挂着就交卷"] += 1
        elif d["has_answer"] and d["vis_all"] and not d["hid_all"]: dead["可见全绿隐藏挂"] += 1
        elif d["has_answer"]: dead["可见没全绿就交卷"] += 1
        else: dead["其它"] += 1
    print(f"  没通过的 {n - int(round(solved * n))} 条死法：{dict(dead.most_common())}")
    wins = collections.Counter(d["calls"] for d in D if d["correct"])
    if wins:
        print("  赢家的调用数分布（最短赢法有多长）：" + "  ".join(f"{k}次 {v}" for k, v in sorted(wins.items())))
    byt = collections.defaultdict(list)
    for r in R: byt[r["pid"]].append(r["d"]["correct"])
    print("  按模板：" + "   ".join(f"{k} {np.mean(v):.3f}({len(v)})" for k, v in sorted(byt.items())))
    bins = [("参考 ≤15 行", lambda r: r["ref_lines"] <= 15), ("16~30", lambda r: 15 < r["ref_lines"] <= 30), ("31~60", lambda r: 30 < r["ref_lines"] <= 60), (">60", lambda r: r["ref_lines"] > 60)]
    print("  按参考实现长度：" + "   ".join(f"{name} {np.mean([r['d']['correct'] for r in R if f(r)]):.3f}({sum(1 for r in R if f(r))})" for name, f in bins if any(f(r) for r in R))
          + "   ｜ " + "   ".join(f"{name} {np.mean([r['d']['correct'] for r in R if f(r)]):.3f}({sum(1 for r in R if f(r))})"
                                 for name, f in (("带类", lambda r: r["n_classes"] > 0), ("不带类", lambda r: r["n_classes"] == 0)) if any(f(r) for r in R)))
    byslug = collections.defaultdict(list)
    for r in R: byslug[r["slug"]].append(r["d"]["correct"])
    top = sorted(byslug.items(), key=lambda kv: -np.mean(kv[1]))
    print("  最容易的 5 题：" + "  ".join(f"{k} {np.mean(v):.2f}" for k, v in top[:5]) + "   最难的 5 题：" + "  ".join(f"{k} {np.mean(v):.2f}" for k, v in top[-5:]))
    print(f"  第一步是什么：{dict(collections.Counter(b['first'] for b in beh))}")
    if show:
        ok = sorted([r for r in R if r["d"]["correct"] and not r["fake"]], key=lambda r: r["L"])   # 假观测的不算最短赢家（代码碰巧对的运气局）
        bad = [r for r in R if not r["d"]["correct"] and r["d"]["n_write"] > 0]
        for tag, rr in (("最短的一条通过的", ok[:1]), ("一条没通过的", bad[:1])):
            for r in rr:
                print("─" * 90); print(f"【{tag}】{r['slug']}  模板 {r['pid']}  得分 {r['r']:.2f}  调用 {r['d']['calls']}  {r['L']} token" + ("  ⚠ 撞顶" if r["cut"] else "")
                                      + f"  可见 {r['d']['vis_pass']}/{r['d']['n_vis']} 隐藏 {r['d']['hidden_pass']}/{r['d']['n_hid']}")
                print(r["txt"][:6000])
                print("··· 最终文件 ···"); print(r["final_code"][:2500])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hf_coder_1.5b", help="tokenizer 来源")
    ap.add_argument("--hf-dir", default=None)
    ap.add_argument("--data", default="data/exercism.json")
    ap.add_argument("--split", default="train", choices=["train", "heldout"])
    ap.add_argument("-n", "--n-problems", type=int, default=100)
    ap.add_argument("-s", "--samples", type=int, default=8)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=4096)
    ap.add_argument("--max-calls", type=int, default=12)
    ap.add_argument("--call-cost", type=float, default=0.01)
    ap.add_argument("--gpu-mem", type=float, default=0.6)
    ap.add_argument("--prompt-set", default="train", choices=list(E.SETS))
    ap.add_argument("--inject", default=None)
    ap.add_argument("--obs", default="strong", choices=["strong", "weak"], help="weak = 阶段 ③：题面不给测试，<test> 不可用，模型用 <run> 自己验证")
    ap.add_argument("--ask", action="store_true")
    ap.add_argument("--ask-model", default="deepseek-reasoner")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", type=int, default=2)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--analyze", default=None)
    ap.add_argument("--step-workers", type=int, default=32)
    ap.add_argument("--inflight", type=int, default=128)
    ap.add_argument("--shard", default=None, help="i/N：把抽出的 n 道题按顺序切 N 份只跑第 i 份（四卡并行用）；raw 存成 *_shard{i}of{N}_raw.json，--merge 合并")
    ap.add_argument("--merge", nargs="*", default=None, help="把多份 raw 合成一份再出报告：--merge a_raw.json b_raw.json … --out merged.json")
    args = ap.parse_args()

    if args.merge:
        raws = [json.load(open(f)) for f in args.merge]
        base = dict(raws[0]); base["records"] = [r for raw in raws for r in raw["records"]]
        base["n"] = len({r["slug"] for r in base["records"]}); base["time"] = sum(r.get("time", 0) for r in raws)
        out = (args.out or "merged.json").replace(".json", "_raw.json")
        json.dump(base, open(out, "w")); print(f"合并 {len(raws)} 份 → {out}（{len(base['records'])} 条，{base['n']} 题）")
        report(base, args.show); return
    if args.analyze:
        report(json.load(open(args.analyze)), args.show); return

    probs = E.load_problems(args.data, split=args.split)
    rng = random.Random(args.seed); rng.shuffle(probs)
    P = probs[: args.n_problems]
    if args.shard:
        i, N = map(int, args.shard.split("/"))
        P = P[i::N]                                                    # 先抽 n 道再切，和不分片时同一批题
    ids = E.SETS[args.prompt_set]
    env0 = E.ExEnv(max_calls=args.max_calls, call_cost=args.call_cost, obs=args.obs)

    def make_prompt(i, p):
        s = E.render(ids[i % len(ids)], p, tool_style=i % 3, env=env0)
        if args.inject:
            s = s[: s.rfind("<think>")] + args.inject.strip() + "\n<think>"
        return s

    if args.dry:
        for i, p in enumerate(P):
            print("█" * 40, f"{p['slug']}  模板 {ids[i % len(ids)]}  可见 {p['n_vis']} 隐藏 {p['n_hid']}"); print(make_prompt(i, p))
        return

    from transformers import AutoTokenizer
    import calc_tool_vllm as T
    import harness as H
    from concurrent.futures import ThreadPoolExecutor
    assert args.hf_dir, "要 --hf-dir"
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [make_prompt(i, p) for i, p in enumerate(P)]
    plen = max(len(tok.encode(s)) for s in prompts)
    be = T.make_backend("vllm", tok=tok, hf_dir=args.hf_dir, gpu_mem=args.gpu_mem, max_model_len=plen + args.max_new + 64)
    name = os.path.basename(args.hf_dir.rstrip("/"))
    args.out = args.out or f"ex_{name}_{args.split}_s{args.samples}_p{args.prompt_set}{'_weak' if args.obs == 'weak' else ''}.json"
    if args.shard:
        i, N = map(int, args.shard.split("/")); args.out = args.out.replace(".json", f"_shard{i}of{N}.json")
    ask_fn = (lambda q, p, code: E.ask_expert(q, p, code, model=args.ask_model, weak=(args.obs == "weak"))) if args.ask else None
    env = E.ExEnv(max_calls=args.max_calls, call_cost=args.call_cost, ask_fn=ask_fn, obs=args.obs)
    print(f"{name}   {len(P)} 题 × {args.samples}   split {args.split}  温度 {args.temp}  模板 {args.prompt_set}  观测 {args.obs}  最长题面 {plen} token"
          f"   可见/隐藏测试 中位 {np.median([p['n_vis'] for p in P]):.0f}/{np.median([p['n_hid'] for p in P]):.0f}\n  题面样例（{P[0]['slug']}）：\n{prompts[0]}\n", flush=True)
    prompt_list = prompts

    off = int(args.shard.split("/")[0]) * 1000 if args.shard else 0    # 分片时 pi 错开，合并后组不撞
    flat = [(off + i, p) for i, p in enumerate(P) for _ in range(args.samples)]
    prompts = {off + i: s for i, s in enumerate(prompt_list)}          # 题面也按错开后的号存（之前用错开的号去索引 25 条的列表，第 1~3 片越界）
    t0 = time.time()
    states = [env.reset(p) for _, p in flat]
    outs = H.generate_with_env(be, tok, [prompts[i] for i, _ in flat], env, states, max_new=args.max_new, temp=args.temp,
                               progress=True, step_workers=args.step_workers, max_total=plen + args.max_new + 64, inflight=args.inflight)
    ts = time.time()
    with ThreadPoolExecutor(max_workers=args.step_workers) as ex:
        scores = list(ex.map(lambda o: env.score(o["state"], o["txt"]), outs))
    rec = []
    for (i, p), o, (r, d) in zip(flat, outs, scores):
        rec.append(dict(pi=i, slug=p["slug"], pid=ids[(i - off) % len(ids)], ref_lines=p["ref_lines"], n_classes=p["n_classes"], txt=o["txt"], final_code=o["state"]["code"],
                        r=r, d=d, beh=E.behavior(o["txt"]), L=len(o["ids"]), fake=o["fake"], cut=o["cut"], cont=o["cont"]))
    print(f"  {len(rec)} 条  {time.time()-t0:.0f} s（终评 {time.time()-ts:.1f} s）   沙盒真跑 {env.n_run} / 缓存命中 {env.n_hit}", flush=True)
    raw = dict(name=name, n=len(P), n_samples=args.samples, split=args.split, temp=args.temp, prompt_set=args.prompt_set, inject=args.inject, ask=args.ask, obs=args.obs,
               max_calls=args.max_calls, call_cost=args.call_cost, seed=args.seed, max_new=args.max_new, time=time.time() - t0, records=rec)
    json.dump(raw, open(args.out.replace(".json", "_raw.json"), "w"))
    print(f"\n  ★ raw → {args.out.replace('.json', '_raw.json')}")
    report(raw, args.show)


if __name__ == "__main__":
    main()
