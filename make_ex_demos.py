#!/usr/bin/env python3
"""
make_ex_demos.py —— 阶段 ①′ 的教材：强模型（DeepSeek）当老师，在真环境里「写 → 跑可见测试 → 看报错 → 改」，轨迹记下来做 SFT。
    export DEEPSEEK_API_KEY=sk-...
    python3 make_ex_demos.py --out sft_ex.jsonl [--n 100 --variants 2 --per-prob 3 --model deepseek-chat --max-rounds 3 --workers 8]
    python3 make_ex_demos.py --out /tmp/x.jsonl --n 10 --show 1          # 先看 10 道：老师通过率、轨迹长什么样
规矩   老师只看题面 + 空壳 + 可见测试（藏起来的三成老师也看不见）；每条轨迹最后用秤（含隐藏）验过全过才落盘 —— 拒绝采样，蒸馏的是「会做的题」
轨迹   「Let me implement x.py and run the tests.」<write>…</write>⟦written⟧<test></test>⟦passed k/n …⟧ →
       挂了：「The failing tests point at …: <老师一句话>. Let me fix x.py and run the tests again.」<write>…</write>⟦⟧<test></test>⟦⟧ → …（最多 --max-rounds 轮）
       改坏了（通过数比上一版少）：「That made it worse (k2 < k1). Let me go back to the previous version and fix only …」<write>上一版</write> → 再改
       全过：「All tests pass.」</think><answer>done</answer>
       开局就写（不先在空壳上跑测试 —— 探针里 76% 的局先测，白花一次调用）
变体   --variants 2：同一题老师温度 0 和 0.7 各来一遍（轨迹不同）；每条轨迹配 --per-prob 个不同模板 → 行数 ≈ 题数 × variants × per-prob
★ --weak（阶段 ③ 弱观测教材）  老师只看题面 + 空壳，一次给三段：EXPLANATION / CHECKS（从题面的例子和规则推出的断言，末尾 print("checks passed")）/ CODE。
       轨迹：「Let me write checks from the task …」<write>CODE</write>⟦written⟧<run>CHECKS</run>⟦checks passed | Traceback⟧ →
       挂了：老师看观测，先判「断言错还是代码错」再改：「The check failed at …: <一句话>. Let me fix the check / x.py and run the checks again.」→ …（最多 --max-rounds 轮）
       过了：「Checks passed.」</think><answer>done</answer>；最后用秤（全部测试）验过才落盘。顺手用 ExEnv 弱模式给老师的断言打分：真阳 / 假阴 / 假阳 / 真阴、自测全绿但隐藏挂
       --fix-from 配 --weak：第一版是学生（弱模式探针 raw）写的错版，老师写断言抓它、再改
    python3 make_ex_demos.py --weak --out sft_exw.jsonl --n 100 --variants 3
    python3 make_ex_demos.py --weak --out sft_exw_fix.jsonl --fix-from rlex_weak_raw.json sftex_weak_raw.json
"""
import os, re, sys, json, time, random, argparse, collections
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import code_env as C
import ex_env as E

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/exercism.json")
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=100, help="用多少道训练题")
ap.add_argument("--variants", type=int, default=2, help="每题老师来几遍（温度 0 / 0.7 / 1.0）")
ap.add_argument("--per-prob", type=int, default=3, help="每条轨迹配几套模板")
ap.add_argument("--model", default="deepseek-chat")
ap.add_argument("--max-rounds", type=int, default=3, help="首版之后最多改几轮")
ap.add_argument("--max-tokens", type=int, default=4000)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--show", type=int, default=0)
ap.add_argument("--mock", action="store_true", help="不调 API：用参考实现当老师（第一版故意坏一处），只验流水线")
ap.add_argument("--fix-from", nargs="*", default=None, help="★ 「修」教材：从这些评测 raw 里取学生自己写的、没全过的首版当第一版，老师只负责改。教材缺的那一半")
ap.add_argument("--fix-per-slug", type=int, default=3, help="每道题最多取几条学生首版（不同的错法）")
ap.add_argument("--weak", action="store_true", help="★ 阶段 ③：弱观测教材（不给老师测试，老师自己写断言）")
args = ap.parse_args()
rng = random.Random(args.seed)

SYS = ("You are an expert Python programmer solving small exercises. You get the task description, a stub file with the required signatures, "
       "and the unit tests the module must pass. Reply in exactly this format and nothing else:\n"
       "EXPLANATION: <one sentence: what you implemented, or what was wrong and how you fixed it>\n"
       "CODE:\n<the complete module, plain Python, no markdown fences>")


def teacher_call(p, code, obs, temperature):
    """→ (explanation, code)。obs=None 是首版；否则是上一版的测试观测"""
    if args.mock:
        return _mock_teacher(p, code, obs)
    import deepseek_tool as DS
    q = f"Task:\n{E.clean_text(p['text'])[:E.PROMPT_TEXT_CHARS]}\n\nStub ({p['module']}.py):\n{p['stub']}\n\nTests ({p['module']}_test.py):\n{p['visible_src']}\n"
    if obs is not None:
        q += f"\nYour current {p['module']}.py:\n{code}\n\nRunning the tests gave:\n{obs}\n\nFix the module so all tests pass."
    else:
        q += f"\nImplement {p['module']}.py so all tests pass."
    r = DS.ask(q, model=args.model, system=SYS, max_tokens=args.max_tokens, temperature=temperature)
    c = (r.get("content") or "").strip()
    m = re.search(r"EXPLANATION:\s*(.*?)\n\s*CODE:\s*\n(.*)$", c, re.S)
    if not m:
        code_only = re.sub(r"^```(?:python)?\s*|\s*```\s*$", "", c, flags=re.S).strip()
        return "", code_only
    expl = m.group(1).strip().replace("\n", " ")
    body = re.sub(r"^```(?:python)?\s*|\s*```\s*$", "", m.group(2).strip(), flags=re.S).strip()
    return expl, body


SYS_WEAK = ("You are an expert Python programmer solving small exercises. You get ONLY the task description and a stub file with the required signatures; "
            "no tests are provided, and there is no test tool. You must (1) derive checks from the task's rules and examples, (2) implement the module. "
            "Reply in exactly this format and nothing else:\n"
            "EXPLANATION: <ONE short sentence (max 40 words): what you implemented, or which line was wrong (the check or the code) and why>\n"
            "CHECKS:\n<plain Python, 6 to 25 lines: assert statements covering every rule, every example, the edge cases and the exact error messages stated in the task. "
            "Expected values must be computed by applying the task's rules step by step; never take them from outside knowledge (a familiar-looking case can contradict the task's rule). "
            "Every assert carries a message showing the actual value, e.g. assert f(2) == 5, f(2). The module's names are already available (no import needed); "
            "the last line must be print(\"checks passed\")>\n"
            "CODE:\n<the complete module, plain Python, no markdown fences>")
MAX_EXPL_CHARS = 300                                                    # 解释超过这个长度 = 老师在胡说，整局丢掉


def teacher_call_weak(p, code, checks, obs, temperature):
    """弱模式：→ (explanation, checks, code)。obs=None 是首版；否则是上一版跑断言的观测，老师要先判断言错还是代码错"""
    if args.mock:
        return _mock_teacher_weak(p, code, obs)
    import deepseek_tool as DS
    q = f"Task:\n{E.clean_text(p['text'])[:E.PROMPT_TEXT_CHARS]}\n\nStub ({p['module']}.py):\n{p['stub']}\n"
    if obs is None and code is None:
        q += f"\nWrite CHECKS from the task and implement {p['module']}.py."
    elif obs is None:
        q += f"\nA student wrote this {p['module']}.py, which may be wrong:\n{code}\n\nWrite CHECKS from the task that would expose any mistake, and give the corrected CODE."
    else:
        q += (f"\nYour current {p['module']}.py:\n{code}\n\nYour checks:\n{checks}\n\nRunning the checks gave:\n{obs}\n\n"
              "Re-read the task and decide whether the check or the code is wrong. Return the corrected CHECKS and CODE (keep unchanged parts verbatim).")
    r = DS.ask(q, model=args.model, system=SYS_WEAK, max_tokens=args.max_tokens, temperature=temperature, retries=6, timeout=300)   # 长正文 + 多路并发：多等、多试
    c = (r.get("content") or "").strip()
    m = re.search(r"EXPLANATION:\s*(.*?)\n\s*CHECKS:\s*\n(.*?)\n\s*CODE:\s*\n(.*)$", c, re.S)
    if not m:
        return "", "", ""
    strip = lambda t: re.sub(r"^```(?:python)?\s*|\s*```\s*$", "", t.strip(), flags=re.S).strip()
    return m.group(1).strip().replace("\n", " "), strip(m.group(2)), strip(m.group(3))


def _mock_teacher_weak(p, code, obs):
    """本机验流水线：首版 = 参考实现改坏一处 + 一条必挂的断言；第二版 = 参考实现 + 只打印的断言"""
    if obs is None:
        bad = re.sub(r"return (.+)$", "return None  # oops", p["ref_code"], count=1, flags=re.M)
        return "implemented the module", 'raise AssertionError("mock: first version returns None")\nprint("checks passed")', (bad if bad != p["ref_code"] else p["ref_code"])
    return "the code was wrong: the first return statement returned None instead of the computed value", 'print("checks passed")', p["ref_code"]


def _first_fail_line(obs):
    """观测里第一条 assert / 异常行，给轨迹的「The check failed at …」用"""
    for l in obs.splitlines():
        if l.strip().startswith("assert") or "Error" in l:
            return l.strip()[:120]
    return "the first check"


def run_one_weak(p, temperature, env, first_code=None):
    """弱模式老师走一局 → (parts, st, ok, rounds, first_all)。first_code 给了 = 「修」教材：第一版是学生写的"""
    st = env.reset(p); parts = []
    def add(t): parts.append(t)
    def call(tag, arg=""):
        piece = f"<{tag}>{arg}</{tag}>"; add(piece)
        inj, done = env.step(st, "".join(parts)); add(inj)
        return inj[len(C.RESULT_OPEN):-len(C.RESULT_CLOSE)]
    expl, checks, code = teacher_call_weak(p, first_code, None, None, temperature)
    if not checks.strip() or not code.strip():
        return parts, st, False, 0, 0
    fixed = code if first_code is not None else None                  # 修教材：老师第一次就给了修好的版本，等断言抓住学生的错再用
    code = first_code if first_code is not None else code
    add(f"Let me write checks from the task's rules and examples, implement {p['module']}.py, and run the checks.\n")
    call("write", "\n" + code.strip("\n") + "\n"); add("\n")
    obs = call("run", "\n" + checks.strip("\n") + "\n"); add("\n")
    passed = lambda o: ("Traceback" not in o) and ("checks passed" in o)
    first_all = int(passed(obs)); rounds = 0
    while not passed(obs) and rounds < args.max_rounds:
        rounds += 1
        if fixed is not None:
            expl2, checks2, code2 = expl, checks, fixed; fixed = None
        else:
            expl2, checks2, code2 = teacher_call_weak(p, st["code"], checks, obs, temperature)
        if not checks2.strip() or not code2.strip():
            break
        if len(expl2) > MAX_EXPL_CHARS:                                  # ★ 啰嗦 = 判断混乱，这局不要
            st["rambling"] = True; break
        why = (expl2[0].lower() + expl2[1:]).rstrip(".") if expl2 else "something does not match the task"
        what = f"{p['module']}.py" if code2.strip() != st["code"].strip() else "the check"
        add(f"The check failed at `{_first_fail_line(obs)}`: {why}. Let me fix {what} and run the checks again.\n")
        if code2.strip() != st["code"].strip():
            call("write", "\n" + code2.strip("\n") + "\n"); add("\n")
        checks = checks2
        obs = call("run", "\n" + checks.strip("\n") + "\n"); add("\n")
        if st["calls"] >= env.max_calls - 1:
            break
    ok = passed(obs) and not st.get("rambling")
    if ok:
        add("Checks passed.\n</think>\n<answer>done</answer>")
    return parts, st, ok, rounds, first_all


def _mock_teacher(p, code, obs):
    """本机流水线验证用：首版 = 参考实现改坏一处（把第一个 return 换成 return None）；第二版 = 参考实现"""
    if obs is None:
        bad = re.sub(r"return (.+)$", "return None  # oops", p["ref_code"], count=1, flags=re.M)
        return "implemented the module", bad if bad != p["ref_code"] else p["ref_code"]
    return "the first return statement returned None instead of the computed value", p["ref_code"]


def failing_names(obs):
    return re.findall(r"^(?:FAIL|ERROR|TIMEOUT) (\S+):", obs, re.M)


def student_first_versions(paths, per_slug):
    """从评测 raw 里挑「写了首版、可见没全过」的局 → {slug: [首版代码…]}（按错法去重，每题最多 per_slug 条）"""
    out = collections.defaultdict(list)
    for path in paths:
        for r in json.load(open(path))["records"]:
            m = C.TAG_RE["write"].findall(r["txt"])
            if not m or r["d"].get("first_all") or r["d"].get("first_ver_all"):
                continue
            code = C.normalize_code(m[0])
            if not code.strip() or code in out[r["slug"]]:
                continue
            out[r["slug"]].append(code)
    return {k: v[:per_slug] for k, v in out.items()}


def run_one(p, temperature, env, first_code=None):
    """老师在真环境里走一局 → (parts, st, ok, rounds, first_all)。first_code 给了 = 「修」教材：第一版是学生写的，老师只改"""
    st = env.reset(p); parts = []
    def add(t): parts.append(t)
    def call(tag, arg=""):
        piece = f"<{tag}>{arg}</{tag}>"; add(piece)
        inj, done = env.step(st, "".join(parts)); add(inj)
        return inj[len(C.RESULT_OPEN):-len(C.RESULT_CLOSE)]
    if first_code is not None:
        code = first_code
    else:
        expl, code = teacher_call(p, None, None, temperature)
    if not code.strip():
        return parts, st, False, 0, 0
    add(f"Let me implement {p['module']}.py and run the tests.\n")
    call("write", "\n" + code.strip("\n") + "\n"); add("\n")
    obs = call("test"); add("\n")
    k = int(obs.split("\n", 1)[0].split()[1].split("/")[0]); n = p["n_vis"]
    first_all = int(k == n)
    best_k, best_code, rounds = k, st["code"], 0
    while k < n and rounds < args.max_rounds:
        rounds += 1
        names = failing_names(obs)
        expl, code = teacher_call(p, st["code"], obs, temperature)
        if not code.strip():
            break
        why = (expl[0].lower() + expl[1:]).rstrip(".") if expl else "the implementation needs a fix"
        add(f"The failing tests point at {', '.join(names[:3]) if names else 'the remaining cases'}: {why}. Let me fix {p['module']}.py and run the tests again.\n")
        call("write", "\n" + code.strip("\n") + "\n"); add("\n")
        obs = call("test"); add("\n")
        k = int(obs.split("\n", 1)[0].split()[1].split("/")[0])
        if k < best_k and rounds < args.max_rounds:                    # ★ 改坏了：退回上一版再改
            add(f"That made it worse ({k}/{n} < {best_k}/{n}). Let me go back to the previous version and fix only what those tests need.\n")
            call("write", "\n" + best_code.strip("\n") + "\n"); add("\n")
            obs = call("test"); add("\n")
            k = int(obs.split("\n", 1)[0].split()[1].split("/")[0])
        if k > best_k:
            best_k, best_code = k, st["code"]
        if st["calls"] >= env.max_calls - 1:
            break
    ok = k == n
    if ok:
        add("All tests pass.\n</think>\n<answer>done</answer>")
    return parts, st, ok, rounds, first_all


def main():
    probs = E.load_problems(args.data, "train")
    rng.shuffle(probs); probs = probs[: args.n]
    env = E.ExEnv(max_calls=12, call_cost=0.01, obs="weak" if args.weak else "strong")
    temps = [0.0, 0.7, 1.0][: args.variants]
    if args.fix_from:                                                  # ★ 「修」教材：学生的坏首版 × 老师改；温度只用 0
        firsts = student_first_versions(args.fix_from, args.fix_per_slug)
        jobs = [(p, 0.0, fc) for p in probs for fc in firsts.get(p["slug"], [])]
        print(f"从 {args.fix_from} 取到 {sum(len(v) for v in firsts.values())} 条学生首版（{len(firsts)} 道题），本批用 {len(jobs)} 条", flush=True)
    else:
        jobs = [(p, t, None) for p in probs for t in temps]
    rows, stat, t0 = [], collections.Counter(), time.time()
    lock = __import__("threading").Lock()

    def work(job):
        p, t, fc = job
        try:
            parts, st, ok, rounds, first_all = (run_one_weak if args.weak else run_one)(p, t, env, first_code=fc)
        except Exception as e:
            return dict(slug=p["slug"], ok=False, err=f"{type(e).__name__}: {e}"[:200])
        resp = "".join(parts)
        r_, d = env.score(st, resp) if (ok or args.weak) else (0.0, dict(correct=0))   # 弱模式每局都过秤：老师的断言质量要算（没过的也算）
        return dict(slug=p["slug"], p=p, ok=ok and d["correct"] > 0, vis_ok=ok, rounds=rounds, first_all=first_all, resp=resp, calls=st["calls"], temp=t,
                    hidden_fail=bool(ok and d["correct"] == 0), d=d, rambling=st.get("rambling", False))

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(work, jobs))
    errs = collections.Counter()
    for res in results:
        stat["局"] += 1
        if res.get("err"):
            stat["异常"] += 1; errs[res["err"][:70]] += 1; continue
        stat["可见全过"] += res["vis_ok"]; stat["隐藏也过"] += res["ok"]; stat["可见过隐藏挂"] += res["hidden_fail"]
        stat["首版全过"] += res["first_all"]; stat["改过后过"] += (res["ok"] and not res["first_all"])
        if args.weak and res.get("d"):
            for k in ("n_tp", "n_fn", "n_fp", "n_tn"): stat[k] += res["d"].get(k, 0)
            stat["自测全绿隐藏挂"] += res["d"].get("selfgreen_hidfail", 0); stat["秤全过"] += int(res["d"].get("correct", 0) > 0)
            stat["啰嗦丢弃"] += int(bool(res.get("rambling")))
        if not res["ok"]:
            continue
        p = res["p"]
        pids = rng.sample(E.SETS["train"], min(args.per_prob, len(E.SETS["train"])))
        for pid in pids:
            prompt = E.render(pid, p, tool_style=rng.randrange(3), env=env)
            rows.append(dict(prompt=prompt, response=res["resp"], task="ex", slug=p["slug"], pid=pid, temp=res["temp"], rounds=res["rounds"],
                             first_all=res["first_all"], n_calls=res["calls"], teacher=args.model, kind="fix" if args.fix_from else "write",
                             obs="weak" if args.weak else "strong"))
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    L = sorted(len(r["prompt"]) + len(r["response"]) for r in rows) or [0]
    slugs_ok = len({r["slug"] for r in rows})
    if args.weak:
        tot = stat["n_tp"] + stat["n_fn"] + stat["n_fp"] + stat["n_tn"]
        print(f"\n老师 {args.model} 弱模式  {'「修」教材（学生错版 × 老师写断言抓 + 改）' if args.fix_from else f'{len(probs)} 题 × {len(temps)} 温度'} = {stat['局']} 局："
              f"断言全过 {stat['可见全过']}  秤也过 {stat['隐藏也过']}  断言过秤挂 {stat['可见过隐藏挂']}  首版断言就过 {stat['首版全过']}  改过后过 {stat['改过后过']}  异常 {stat['异常']}   {time.time()-t0:.0f} s")
        if tot:
            print(f"  老师的断言质量（{tot} 次 run）：真阳 {stat['n_tp']/tot:.2f}  假阴 {stat['n_fn']/tot:.2f}  假阳 {stat['n_fp']/tot:.2f}  真阴 {stat['n_tn']/tot:.2f}   自测全绿但隐藏挂 {stat['自测全绿隐藏挂']} 局   啰嗦丢弃 {stat['啰嗦丢弃']} 局")
    else:
        print(f"\n老师 {args.model}  {'「修」教材（学生首版 × 老师改）' if args.fix_from else f'{len(probs)} 题 × {len(temps)} 温度'} = {stat['局']} 局：可见全过 {stat['可见全过']}  隐藏也过 {stat['隐藏也过']}  可见过隐藏挂 {stat['可见过隐藏挂']}"
              f"  首版全过 {stat['首版全过']}  改过后过 {stat['改过后过']}  异常 {stat['异常']}   {time.time()-t0:.0f} s")
    print(f"落盘 {len(rows)} 条（{slugs_ok} 道题）→ {args.out}   字符中位 {L[len(L)//2]} P90 {L[int(.9*(len(L)-1))]} 最长 {L[-1]}（≈ token /3.5）")
    if errs:
        print("  异常都是什么：" + "   ".join(f"{k} ×{v}" for k, v in errs.most_common(5)))
    for r in rows[: args.show]:
        print("█" * 100); print(r["prompt"][-800:]); print("── response ──"); print(r["response"])


if __name__ == "__main__":
    main()
