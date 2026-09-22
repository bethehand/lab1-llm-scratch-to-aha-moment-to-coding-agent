#!/usr/bin/env python3
"""
ex_env.py —— 阶段 ①′ 的环境：从头写 exercism 题，给测试（可见七成），修自己写的 bug。
题     data/exercism.json（collect_exercism.py 收的 133 道）：题面 / 空壳 stub / 参考实现 / unittest 测试文件。
       测试按方法名切：~七成可见（模型能跑、写进题面），~三成隐藏（只进秤）；按 slug 定种子，每题固定
环境   桌上一份文件（开局 = 空壳）。工具 <test></test>（在沙盒里跑可见测试，按方法名报：passed k/n + 前两条挂的断言/异常 + 其余名字）
       <run>代码</run>（装好模块再跑这段）  <write>代码</write>（整文件替换）  <ask>问题</ask>（专家，可选）  <answer>done</answer> 结束，或调用上限
验证器 子进程里把模块和测试文件都造在内存（sys.modules），不落盘；每条测试单独限时；linecache 喂了源码，traceback 能显示出错行
秤     全部测试（可见 + 隐藏）全过 → 1 − call_cost × 调用，否则 0
弱观测（阶段 ③，ExEnv(obs="weak")）：题面不给任何测试，<test> 不可用（回一句提示、照样计一次调用），模型只能用 <run> 写断言自己验证；秤不变（全部测试藏着）。
       环境记下每次 run（第几版代码、有没有 assert、有没有报错），交卷时对首版 / 最终版 / 自测过的版本跑全秤 → 自测率、真阳、假阴、假阳、交卷前验没验（四读数）
harness 一行不改：stops / fake_tags / step(state, full) / score(state, txt) 的接口和 CodeEnv / PkgEnv 一样

    python3 ex_env.py --demo clock          # 打一份题面（不载模型）
"""
import os, re, ast, json, math, random, shutil, tempfile, collections
import warnings; warnings.filterwarnings("ignore", category=SyntaxWarning)
import code_env as C

RESULT_OPEN, RESULT_CLOSE = C.RESULT_OPEN, C.RESULT_CLOSE
REPLY_OPEN, REPLY_CLOSE = C.REPLY_OPEN, C.REPLY_CLOSE
TAGS = C.TAGS
ANS_RE = C.ANS_RE
HIDDEN_FRAC = 0.3
PER_TEST_TIMEOUT = 2.0
SHOW_FAILS = 2                 # 观测里给完整信息的挂测试条数
NAME_FAILS = 8                 # 之后只报名字的条数
PROMPT_TEXT_CHARS = 3000       # 题面说明截到多少字
PROMPT_TESTS_CHARS = 3500      # 题面里可见测试文件截到多少字
NO_TEST_MSG = "no test tool in this task: no tests are provided. Check your code yourself with <run> (for example: assert f(x) == y)."
_ASSERT_RE = re.compile(r"\bassert\w*\b|AssertionError|unittest|pytest")


# ══════════════════════════ 造题：切测试 ══════════════════════════
def _test_methods(src):
    """→ [(class, method, lineno, end_lineno)]，按出现顺序"""
    tree = ast.parse(src); out = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name.startswith("test"):
                    start = min([d.lineno for d in f.decorator_list] + [f.lineno])
                    out.append((node.name, f.name, start, f.end_lineno))
    return out


def split_tests(prob, frac=HIDDEN_FRAC, seed=0):
    """按方法名切可见 / 隐藏（按 slug 定种子，每题固定）。→ (visible_src, visible_ids, hidden_ids)；id = 'Class.method'"""
    meths = _test_methods(prob["test_src"])
    ids = [f"{c}.{m}" for c, m, _, _ in meths]
    n = len(ids); n_hid = max(1, int(round(n * frac))) if n >= 3 else 0
    rng = random.Random(seed * 7919 + hash(prob["slug"]) % (1 << 30))
    hidden = set(rng.sample(ids, n_hid)) if n_hid else set()
    lines = prob["test_src"].splitlines(True)
    drop = set()
    for (c, m, a, b) in meths:
        if f"{c}.{m}" in hidden:
            drop.update(range(a - 1, b))
    visible_src = "".join(l for i, l in enumerate(lines) if i not in drop)
    return visible_src, [i for i in ids if i not in hidden], [i for i in ids if i in hidden]


def make_prob(rec, seed=0):
    p = dict(rec)
    p["visible_src"], p["visible_ids"], p["hidden_ids"] = split_tests(rec, seed=seed)
    p["n_vis"], p["n_hid"] = len(p["visible_ids"]), len(p["hidden_ids"])
    return p


def load_problems(path="data/exercism.json", split="train", seed=0):
    return [make_prob(r, seed) for r in json.load(open(path)) if r["split"] == split]


# ══════════════════════════ 验证器：内存里装模块 + 跑 unittest ══════════════════════════
RUNNER = r"""
import sys as _sys, types as _types, unittest as _ut, traceback as _tb, io as _io, json as _json, linecache as _lc, signal as _sig
_mod, _msrc, _tsrc, _only, _per, _maxto = _MOD_, _MSRC_, _TSRC_, _ONLY_, _PER_, _MAXTO_
_lc.cache[_mod + ".py"] = (len(_msrc), None, _msrc.splitlines(True), _mod + ".py")
_lc.cache[_mod + "_test.py"] = (len(_tsrc), None, _tsrc.splitlines(True), _mod + "_test.py")
def _fmt(exc):
    lines = _tb.format_exception(*exc)
    keep = [l for l in lines if ('File "' + _mod + '.py"' in l) or ('File "' + _mod + '_test.py"' in l) or not l.startswith("  File")]
    return "".join(keep)[-1200:]
m = _types.ModuleType(_mod); m.__file__ = _mod + ".py"; _sys.modules[_mod] = m
try:
    exec(compile(_msrc, _mod + ".py", "exec"), m.__dict__)
except BaseException:
    _sys.stdout.write("__EX__" + _json.dumps(dict(load_error=_fmt(_sys.exc_info()))) + chr(10)); raise SystemExit(0)
t = _types.ModuleType(_mod + "_test"); t.__file__ = _mod + "_test.py"; _sys.modules[_mod + "_test"] = t
try:
    exec(compile(_tsrc, _mod + "_test.py", "exec"), t.__dict__)
except BaseException:
    _sys.stdout.write("__EX__" + _json.dumps(dict(load_error="test file failed to import: " + _fmt(_sys.exc_info()))) + chr(10)); raise SystemExit(0)
class _TO(BaseException): pass
def _alarm(signum, frame): raise _TO("test ran longer than %.0f s" % _per)
_sig.signal(_sig.SIGALRM, _alarm)
class _R(_ut.TestResult):
    def __init__(s): super().__init__(); s.rows = []
    def startTest(s, test): super().startTest(test); _sig.setitimer(_sig.ITIMER_REAL, _per)
    def stopTest(s, test): _sig.setitimer(_sig.ITIMER_REAL, 0); super().stopTest(test)
    def _id(s, test): return type(test).__name__ + "." + test._testMethodName
    def addSuccess(s, test): s.rows.append(dict(id=s._id(test), status="ok"))
    def addFailure(s, test, err): s.rows.append(dict(id=s._id(test), status="fail", msg=_fmt(err)))
    def addError(s, test, err):
        if isinstance(err[1], _TO): s.rows.append(dict(id=s._id(test), status="timeout", msg="Timeout: " + str(err[1])))
        else: s.rows.append(dict(id=s._id(test), status="error", msg=_fmt(err)))
    def addSkip(s, test, reason): s.rows.append(dict(id=s._id(test), status="ok"))
    def addExpectedFailure(s, test, err): s.rows.append(dict(id=s._id(test), status="ok"))
    def addUnexpectedSuccess(s, test): s.rows.append(dict(id=s._id(test), status="fail", msg="unexpected success"))
def _flat(suite):
    for x in suite:
        if isinstance(x, _ut.TestSuite):
            yield from _flat(x)
        else:
            yield x
tests = [x for x in _flat(_ut.TestLoader().loadTestsFromModule(t)) if _only is None or (type(x).__name__ + "." + x._testMethodName) in _only]
r = _R(); _real = _sys.stdout; _sys.stdout = _io.StringIO(); _sys.stderr = _io.StringIO()
_nto = 0
for x in tests:
    _tid = type(x).__name__ + "." + x._testMethodName
    if _maxto and _nto >= _maxto:                                   # ★ 连着超时够 _maxto 次：多半是共用代码里的死循环，剩下的不再跑，直接标超时（27 s → ~5 s）
        r.rows.append(dict(id=_tid, status="timeout", msg="Timeout: skipped after %d timeouts (infinite loop?)" % _nto)); continue
    _before = len(r.rows)
    try:
        x.run(r)
    except _TO as e:
        r.rows.append(dict(id=_tid, status="timeout", msg="Timeout: " + str(e)))
    _nto = _nto + 1 if (len(r.rows) > _before and r.rows[-1]["status"] == "timeout") else 0
_sys.stdout = _real
_sys.stdout.write("__EX__" + _json.dumps(dict(rows=r.rows)) + chr(10))
"""


MAX_CONSEC_TIMEOUTS = 0        # ★ 连着几条超时就不再跑剩下的（0 = 关，老行为）。阶段 ①′ 整段用 0 保持一致；下一阶段起设 2


def run_unittests(module, code, test_src, only, cwd, per_test=PER_TEST_TIMEOUT, total=None, max_timeouts=None):
    """→ dict(rows=[{id,status,msg}], load_error=None|str, timeout=bool)。only = 要跑的 id 列表（None = 全部）"""
    mt = MAX_CONSEC_TIMEOUTS if max_timeouts is None else max_timeouts
    script = (RUNNER.replace("_MOD_", repr(module)).replace("_MSRC_", repr(code)).replace("_TSRC_", repr(test_src))
              .replace("_ONLY_", repr(list(only) if only is not None else None)).replace("_PER_", repr(float(per_test))).replace("_MAXTO_", repr(int(mt))))
    n = len(only) if only is not None else 100
    total = total or min(30.0, 3.0 + per_test * min(n, 12))
    rc, out, err = C.run_in_sandbox(script, cwd, total, raw=True)
    i = (out or "").find("__EX__")
    if i < 0:
        return dict(rows=[], load_error=None, timeout=True, raw_err=(err or "")[-400:])
    d = json.loads(out[i + 6:].splitlines()[0])
    d.setdefault("rows", []); d.setdefault("load_error", None); d["timeout"] = False
    return d


def _short_name(tid, ids):
    m = tid.split(".", 1)[1]
    return m if sum(1 for x in ids if x.endswith("." + m)) == 1 else tid


def format_obs(res, ids):
    """观测：passed k/n + 前 SHOW_FAILS 条挂的（断言 / 异常 + 出错行）+ 其余名字（最多 NAME_FAILS 个）"""
    n = len(ids)
    if res.get("timeout"):
        return f"passed 0/{n}\nTimeout: the test run did not finish (infinite loop?)"
    if res.get("load_error"):
        return f"passed 0/{n}\nThe module failed to import:\n{res['load_error'].strip()}"
    st = {r["id"]: r for r in res["rows"]}
    bad = [i for i in ids if st.get(i, {}).get("status") != "ok"]
    k = n - len(bad)
    lines = [f"passed {k}/{n}"]
    for i in bad[:SHOW_FAILS]:
        r = st.get(i) or dict(status="error", msg="not run")
        tag = {"fail": "FAIL", "error": "ERROR", "timeout": "TIMEOUT"}.get(r["status"], "FAIL")
        msg = (r.get("msg") or "").strip()
        if r["status"] == "fail":                                          # 断言：只留最后的断言行 + 测试里那一行
            keep = [l for l in msg.splitlines() if l.strip() and not l.startswith("Traceback")]
            msg = "\n".join(keep[-3:])
        else:
            keep = [l for l in msg.splitlines() if l.strip() and not l.startswith("Traceback")]
            msg = "\n".join(keep[-4:])
        lines.append(f"{tag} {_short_name(i, ids)}:\n" + "\n".join("  " + l for l in msg.splitlines()))
    rest = bad[SHOW_FAILS:]
    if rest:
        names = [_short_name(i, ids) for i in rest[:NAME_FAILS]]
        lines.append("also failing: " + ", ".join(names) + (f" (+{len(rest) - NAME_FAILS} more)" if len(rest) > NAME_FAILS else ""))
    return "\n".join(lines)


RUN_SNIPPET = r"""
import sys as _sys, types as _types, traceback as _tb, linecache as _lc
_mod, _msrc, _snip = _MOD_, _MSRC_, _SNIP_
_lc.cache[_mod + ".py"] = (len(_msrc), None, _msrc.splitlines(True), _mod + ".py")
_lc.cache["<snippet>"] = (len(_snip), None, _snip.splitlines(True), "<snippet>")     # ★ 片段挂了 traceback 带上那一行源码（不然只有 line N，数行数容易数错）
m = _types.ModuleType(_mod); m.__file__ = _mod + ".py"; _sys.modules[_mod] = m
try:
    exec(compile(_msrc, _mod + ".py", "exec"), m.__dict__)
except BaseException:
    _sys.stderr.write("[while loading %s.py]" % _mod + chr(10)); _tb.print_exc(); raise SystemExit(0)
g = dict(m.__dict__)
try:
    exec(compile(_snip, "<snippet>", "exec"), g)
except BaseException:
    _tb.print_exc()
"""


def run_snippet(module, code, snippet, cwd, timeout=5):
    script = RUN_SNIPPET.replace("_MOD_", repr(module)).replace("_MSRC_", repr(code)).replace("_SNIP_", repr(snippet))
    rc, out, err = C.run_in_sandbox(script, cwd, timeout, raw=True)
    out = (out or "").strip()
    err = "\n".join(l for l in (err or "").splitlines() if 'File "<stdin>"' not in l and not (l.startswith("    ") and "_sys" in l)).strip()
    if "Traceback" in err:
        err = err[err.index("Traceback"):]
    txt = out + ("\n" if out and err else "") + err
    return (txt.strip() or "(no output)")[: C.MAX_OUT]


def _ok_map(res, ids):
    """秤的结果 → {id: 过没过}；整体超时或 import 失败 = 全挂"""
    if res.get("timeout") or res.get("load_error"):
        return {i: False for i in ids}
    stt = {r["id"]: r for r in res.get("rows", [])}
    return {i: (stt.get(i, {}).get("status") == "ok") for i in ids}


# ══════════════════════════ 环境 ══════════════════════════
class ExEnv:
    stops = tuple(f"</{t}>" for t in TAGS)
    fake_tags = (RESULT_OPEN, REPLY_OPEN)

    def __init__(self, max_calls=12, call_cost=0.01, per_test=PER_TEST_TIMEOUT, ask_fn=None, workdir=None, cache=True, max_timeouts=None,
                 obs="strong", run_timeout=5, track_versions=True):
        self.max_calls, self.call_cost, self.per_test, self.ask_fn, self.workdir = max_calls, call_cost, per_test, ask_fn, workdir
        self.max_timeouts = max_timeouts                              # None = 用模块级 MAX_CONSEC_TIMEOUTS
        assert obs in ("strong", "weak"), obs
        self.obs, self.run_timeout, self.track_versions = obs, run_timeout, track_versions   # weak = 不给测试、<test> 不可用（阶段 ③）
        self.cache = {} if cache else None
        self.n_run = self.n_hit = 0

    def _tests(self, p, code, hidden, cwd):
        key = (p["slug"], code, hidden) if self.cache is not None else None
        if key is not None and key in self.cache:
            self.n_hit += 1; return self.cache[key]
        only = None if hidden else p["visible_ids"]
        r = run_unittests(p["module"], code, p["test_src"], only, cwd, self.per_test, max_timeouts=self.max_timeouts); self.n_run += 1
        if key is not None:
            if len(self.cache) > 100000: self.cache.clear()
            self.cache[key] = r
        return r

    def reset(self, prob):
        d = tempfile.mkdtemp(dir=self.workdir)
        return dict(prob=prob, code=prob["stub"], dir=d, calls=0, n_test=0, n_run=0, n_write=0, n_ask=0, syntax_err=0, hist=[], log=[],
                    versions=[prob["stub"]], runs=[])                 # versions[k] = 第 k 次 write 之后的文件；runs = 每次 <run> 的 (版本, 有无 assert, 有无报错)

    def step(self, st, full):
        tag = None
        for t in TAGS:
            if full.rstrip().endswith(f"</{t}>"):
                tag = t; break
        if tag is None:
            return None, True
        st["calls"] += 1
        p = st["prob"]
        if tag == "test":
            st["n_test"] += 1
            if self.obs == "weak":                                     # ★ 弱观测：没有测试工具，照样计一次调用，不起沙盒
                obs = NO_TEST_MSG
            else:
                res = self._tests(p, st["code"], False, st["dir"])
                obs = format_obs(res, p["visible_ids"])
                k = int(obs.split("\n", 1)[0].split()[1].split("/")[0])
                st["hist"].append(dict(after_writes=st["n_write"], k=k, n=p["n_vis"]))
        elif tag == "run":
            st["n_run"] += 1
            m = C.TAG_RE["run"].findall(full); arg = (m[-1] if m else "").strip("\n")
            snip = C.normalize_code(arg)
            obs = run_snippet(p["module"], st["code"], snip, st["dir"], timeout=self.run_timeout)
            st["runs"].append(dict(ver=st["n_write"], has_assert=int(bool(_ASSERT_RE.search(snip))), err=int("Traceback" in obs)))
        elif tag == "write":
            st["n_write"] += 1
            m = C.TAG_RE["write"].findall(full); body = (m[-1] if m else "")
            code = C.normalize_code(body)
            if not code.strip():
                obs = "not written: empty code"
            else:
                try:
                    compile(code, p["module"] + ".py", "exec"); obs = f"written {p['module']}.py"
                except SyntaxError as e:
                    st["syntax_err"] += 1; obs = f"written {p['module']}.py, but SyntaxError: {e.msg} (line {e.lineno})"
                st["code"] = code
            st["versions"].append(st["code"])                          # 空 write 也占一个版本号（内容不变），让 versions[n_write] 始终对得上
        else:
            st["n_ask"] += 1
            m = C.TAG_RE["ask"].findall(full); q = (m[-1] if m else "").strip()
            a = self.ask_fn(q, p, st["code"]) if self.ask_fn else "error"
            st["log"].append(("ask", q, a))
            return f"{REPLY_OPEN}{a}{REPLY_CLOSE}", st["calls"] >= self.max_calls
        st["log"].append((tag, obs[:200]))
        obs = obs[:1500]
        return f"{RESULT_OPEN}{obs}{RESULT_CLOSE}", st["calls"] >= self.max_calls

    def score(self, st, txt):
        """全部测试（可见 + 隐藏）全过 → 1 − call_cost × 调用，否则 0"""
        p = st["prob"]
        res = self._tests(p, st["code"], True, st["dir"])
        ids = p["visible_ids"] + p["hidden_ids"]
        ok = _ok_map(res, ids)
        vis_pass = sum(ok[i] for i in p["visible_ids"]); hid_pass = sum(ok[i] for i in p["hidden_ids"])
        allpass = all(ok.values())
        r = max(0.0, 1.0 - self.call_cost * st["calls"]) if allpass else 0.0
        h = st["hist"]
        first = next((x for x in h if x["after_writes"] >= 1), None)
        # ★ 版本评估（全秤）：首版、最终版、自测过的版本 —— 首版就对 / 改一轮效果 / 自测四读数 都从这里来（缓存去重）
        nw = st["n_write"]; runs = st.get("runs", []); vers = st.get("versions", [p["stub"]])
        want = ({1, nw} | {x["ver"] for x in runs if x["has_assert"]}) if self.track_versions else set()
        ver_ok = {}
        for v in sorted(want):
            if not (1 <= v <= nw): continue
            rv = res if vers[v] == st["code"] else self._tests(p, vers[v], True, st["dir"])
            okv = _ok_map(rv, ids); ver_ok[v] = (sum(okv.values()) / max(1, len(ids)), all(okv.values()))
        tp = fn = fp = tn = 0
        for x in runs:
            if not x["has_assert"]: continue
            hid_ok = ver_ok[x["ver"]][1] if x["ver"] in ver_ok else False   # 第 0 版是空壳，必挂
            if x["err"] and not hid_ok: tp += 1
            elif (not x["err"]) and not hid_ok: fn += 1
            elif x["err"] and hid_ok: fp += 1
            else: tn += 1
        selftests = [x for x in runs if x["has_assert"]]
        verified_final = int(nw >= 1 and any(x["ver"] == nw for x in selftests))
        last_err = selftests[-1]["err"] if selftests else None
        selfgreen_hidfail = int(bool(verified_final and last_err == 0 and not allpass))
        d = dict(correct=float(allpass), passed=sum(ok.values()), n_tests=len(ids), vis_pass=vis_pass, n_vis=p["n_vis"], hidden_pass=hid_pass, n_hid=p["n_hid"],
                 vis_all=int(vis_pass == p["n_vis"]), hid_all=int(hid_pass == p["n_hid"]), frac_tests=sum(ok.values()) / max(1, len(ids)),
                 calls=st["calls"], n_test=st["n_test"], n_run=st["n_run"], n_write=st["n_write"], n_ask=st["n_ask"], syntax_err=st["syntax_err"],
                 has_answer=float(bool(ANS_RE.search(txt))), changed=float(st["code"].strip() != p["stub"].strip()), fmt=float(st["calls"] > 0),
                 first_pass_frac=(first["k"] / first["n"]) if first else None, first_all=int(bool(first and first["k"] == first["n"])),
                 n_rounds=max(0, st["n_write"] - 1), load_error=int(bool(res.get("load_error"))), hist=h, code_lines=len(st["code"].strip().splitlines()),
                 obs=self.obs, first_ver_frac=(ver_ok[1][0] if 1 in ver_ok else None), first_ver_all=int(bool(1 in ver_ok and ver_ok[1][1])),
                 ver_frac=[[v, ver_ok[v][0]] for v in sorted(ver_ok)], n_selftest=len(selftests), n_tp=tp, n_fn=fn, n_fp=fp, n_tn=tn,
                 verified_final=verified_final, last_selftest_err=last_err, selfgreen_hidfail=selfgreen_hidfail)
        shutil.rmtree(st["dir"], ignore_errors=True)
        return r, d


# ══════════════════════════ 行为读数 / 专家 ══════════════════════════
def behavior(txt):
    return C.behavior(txt)


EX_SYS = ("You solve small Python exercises. You are given the task description, a stub file with the required signatures, and unit tests. "
          "Reply with ONLY the complete Python module code that makes the tests pass, no explanation, no markdown fences.")


def ask_expert(question, prob, code, model="deepseek-reasoner", weak=False):
    try:
        import deepseek_tool as DS
        tests = "" if weak else f"\nVisible tests:\n{prob['visible_src'][:PROMPT_TESTS_CHARS]}"
        q = (f"Task:\n{prob['text'][:PROMPT_TEXT_CHARS]}\nStub:\n{prob['stub']}\nCurrent code:\n{code}{tests}\nQuestion: {question}")
        r = DS.ask(q, model=model, system=EX_SYS, max_tokens=8000)
        c = (r.get("content") or "").strip()
        c = re.sub(r"^```(?:python)?\s*|\s*```$", "", c, flags=re.S).strip()
        return c[:3000] if c else "error"
    except Exception:
        return "error"


# ══════════════════════════ 题面模板（8 训练 + 2 留出 × 3 种工具措辞）══════════════════════════
KEEP_WORDS = ("this particular exercise", "this exercise", "the tests for", "tests expect", "the tests will", "is not required", "please ignore",
              "you will need to", "should be implemented", "expected to", "must be implemented")


def clean_text(text):
    """题面说明去教程。Introduction + Instructions 原样（只去掉底部的链接定义行）；「Instructions append」是 Python 轨道加的说明，多半是给初学者的教程
    （什么是 __repr__、怎么 raise、REPL 演示），白名单式只留：小标题、非 REPL 的代码块（raise ValueError("…") 这种是测试要求的原文）、明确讲本题要求的段落"""
    head, sep, tail = text.partition("# Instructions append")
    head = "\n".join(l for l in head.splitlines() if not re.match(r"^\[[^\]]+\]:\s*\S+", l)).strip()
    if not sep:
        return re.sub(r"\n{3,}", "\n\n", head)
    kept, in_code, buf = [], False, []
    def flush_para():
        para = "\n".join(buf).strip(); buf.clear()
        if not para or re.match(r"^\[[^\]]+\]:\s*\S+$", para):
            return
        low = para.lower()
        if para.startswith("#") or any(w in low for w in KEEP_WORDS):
            kept.append(para)
    for l in tail.splitlines():
        if l.strip().startswith("```"):
            if in_code:
                buf.append(l); block = "\n".join(buf); buf.clear(); in_code = False
                if ">>>" not in block:                                  # REPL 演示是教程，不要
                    kept.append(block)
            else:
                flush_para(); in_code = True; buf.append(l)
            continue
        if in_code:
            buf.append(l); continue
        if not l.strip():
            flush_para(); continue
        buf.append(l)
    flush_para()
    res = [para for i, para in enumerate(kept) if not (para.startswith("#") and (i + 1 >= len(kept) or kept[i + 1].startswith("#")))]
    body = "\n\n".join(res).strip()
    return re.sub(r"\n{3,}", "\n\n", head + ("\n\n" + body if body else ""))


def _text(p):
    t = clean_text(p["text"])
    return t if len(t) <= PROMPT_TEXT_CHARS else t[:PROMPT_TEXT_CHARS].rsplit("\n", 1)[0] + "\n[...]"


def _tests_block(p):
    t = p["visible_src"].strip()
    if len(t) <= PROMPT_TESTS_CHARS:
        return t
    cut = t[:PROMPT_TESTS_CHARS].rsplit("\n    def test", 1)[0]
    return cut + f"\n    # ... {p['n_vis']} visible tests in total; <test></test> runs all of them"


def _tools_en(p, ts):
    mod, n = p["module"], p["n_vis"]
    return [
        f"Tools: <test></test> runs the {n} visible tests and returns <result>passed k/{n} plus the first failing assertions</result>. <run>code</run> executes code with your module imported and returns its output. <write>code</write> replaces the whole file {mod}.py with your code. <ask>question</ask> asks an expert, who replies with code in <reply></reply>; always verify with <test> before answering. At most {{mc}} tool calls. ",
        f"You can call tools: <write>...</write> to write the whole {mod}.py, <test></test> to run the {n} visible tests (you get <result>passed k/{n}</result> and the failing assertions), <run>...</run> to execute a snippet with your module loaded, <ask>...</ask> to consult an expert (reply comes in <reply></reply>, verify it with <test>). {{mc}} calls maximum. ",
        f"TOOLS: <write>CODE</write> -> replaces {mod}.py | <test></test> -> <result>passed k/{n}, failing assertions</result> | <run>CODE</run> -> <result>output</result> | <ask>Q</ask> -> <reply>code</reply> (verify before answering). Budget: {{mc}} calls. ",
    ][ts]


def _tools_zh(p):
    mod, n = p["module"], p["n_vis"]
    return (f"工具：<write>代码</write> 用你的代码整个写入 {mod}.py；<test></test> 跑 {n} 条可见测试，返回 <result>passed k/{n} 和前几条失败的断言</result>；"
            f"<run>代码</run> 在装好模块后执行一段代码；<ask>问题</ask> 问专家，回复在 <reply></reply> 里，作答前必须用 <test> 验证。最多 {{mc}} 次调用。")


def e0(p, ts):
    return (f"A conversation between User and Assistant. The user gives a programming exercise with a stub file and unit tests; the Assistant implements it so the tests pass.\n"
            f"User: {_text(p)}\nStub ({p['module']}.py):\n{p['stub']}\nTests ({p['module']}_test.py):\n{_tests_block(p)}\n" + _tools_en(p, ts) +
            "Think inside <think> </think>, write the module, run the tests, fix what fails, then write <answer>done</answer>.\nAssistant: <think>")
def e1(p, ts):
    return (f"Task:\n{_text(p)}\nStub ({p['module']}.py):\n{p['stub']}\nVisible tests:\n{_tests_block(p)}\n" + _tools_en(p, ts) +
            "Reason in <think></think>, use <write> to implement it and <test> to check, finish with <answer>done</answer>.\n<think>")
def e2(p, ts):
    return (f"You are a careful Python programmer.\nSpecification:\n{_text(p)}\nThe file {p['module']}.py currently contains only stubs:\n{p['stub']}\nIt must pass:\n{_tests_block(p)}\n" +
            _tools_en(p, ts) + "Work inside <think> </think>; when the tests pass, write <answer>done</answer>.\n<think>")
def e3(p, ts):
    return (f"Implement {p['module']}.py so the tests pass.\n{_text(p)}\n{p['stub']}\n{_tests_block(p)}\n" + _tools_en(p, ts) + "Think in <think></think>, end with <answer>done</answer>.\n<think>")
def e4(p, ts):
    return (f"Exercise\n- description:\n{_text(p)}\n- stub ({p['module']}.py):\n{p['stub']}\n- tests:\n{_tests_block(p)}\n- tools: " + _tools_en(p, ts).strip() +
            "\n- output: reasoning in <think> </think>, then <answer>done</answer>\n<think>")
def e5(p, ts):
    return (_tools_en(p, ts) + f"\nProblem:\n{_text(p)}\nStart from this stub ({p['module']}.py):\n{p['stub']}\nTests:\n{_tests_block(p)}\n"
            "Think step by step inside <think> </think> and finish with <answer>done</answer>.\n<think>")
def e6(p, ts):
    return (f"题目：\n{_text(p)}\n空壳文件 {p['module']}.py：\n{p['stub']}\n它需要通过：\n{_tests_block(p)}\n" + _tools_zh(p) +
            "思考写在 <think> </think> 里，用 <write> 写、用 <test> 验证、挂了就改，全部通过后写 <answer>done</answer>。\n<think>")
def e7(p, ts):
    return (f"User: Can you implement this exercise? The tests are below.\n{_text(p)}\n{p['stub']}\n{_tests_block(p)}\n" + _tools_en(p, ts) +
            "Please think in <think> </think> and write <answer>done</answer> once the tests pass.\nAssistant: Sure. <think>")
def h8(p, ts):
    return (f"Here is a small programming exercise. The description is given first, followed by a stub file that only contains the required signatures, and then the unit tests it has to pass.\n"
            f"{_text(p)}\n{p['stub']}\n{_tests_block(p)}\n" + _tools_en(p, ts) +
            "Please reason inside <think> </think>, implement the module using the tools, confirm with <test>, and finally write <answer>done</answer>.\n<think>")
def h9(p, ts):
    return (f"写代码：\n{_text(p)}\n空壳：\n{p['stub']}\n测试：\n{_tests_block(p)}\n" + _tools_zh(p) + "思考写在 <think> </think> 里，全部通过并验证后写 <answer>done</answer>。\n<think>")

TEMPLATES = [e0, e1, e2, e3, e4, e5, e6, e7, h8, h9]
SETS = {"train": list(range(8)), "heldout": [8, 9], "all": list(range(10)), "t0": [0]}


# ══════════════════════════ 弱观测模板（阶段 ③：不给测试，<run> 自己验证；8 训练 + 2 留出 × 3 种工具措辞）══════════════════════════
def _tools_en_weak(p, ts):
    mod = p["module"]
    return [
        f"Tools: <run>code</run> executes code with your module {mod} already imported and returns its output; there are NO provided tests, so use it to check your own work (for example: assert ...). <write>code</write> replaces the whole file {mod}.py with your code. <ask>question</ask> asks an expert, who replies with code in <reply></reply>; always check the reply with <run> before answering. Hidden tests are run only when you answer. At most {{mc}} tool calls. ",
        f"You can call tools: <write>...</write> to write the whole {mod}.py, <run>...</run> to execute a snippet with your module loaded (no tests are given: write your own asserts to check it), <ask>...</ask> to consult an expert (reply comes in <reply></reply>, check it with <run>). The score comes from hidden tests. {{mc}} calls maximum. ",
        f"TOOLS: <write>CODE</write> -> replaces {mod}.py | <run>CODE</run> -> <result>output</result> (your own checks; no tests provided) | <ask>Q</ask> -> <reply>code</reply> (check before answering). Score = hidden tests. Budget: {{mc}} calls. ",
    ][ts]


def _tools_zh_weak(p):
    mod = p["module"]
    return (f"工具：<write>代码</write> 用你的代码整个写入 {mod}.py；<run>代码</run> 在装好模块后执行一段代码，没有给你测试，用它写断言自己验证；"
            f"<ask>问题</ask> 问专家，回复在 <reply></reply> 里，作答前用 <run> 验证。评分用隐藏测试。最多 {{mc}} 次调用。")


def w0(p, ts):
    return (f"A conversation between User and Assistant. The user gives a programming exercise with a stub file; no tests are provided, so the Assistant implements it and checks it by itself.\n"
            f"User: {_text(p)}\nStub ({p['module']}.py):\n{p['stub']}\n" + _tools_en_weak(p, ts) +
            "Think inside <think> </think>, write the module, check it with your own asserts in <run>, fix what fails, then write <answer>done</answer>.\nAssistant: <think>")
def w1(p, ts):
    return (f"Task:\n{_text(p)}\nStub ({p['module']}.py):\n{p['stub']}\n" + _tools_en_weak(p, ts) +
            "Reason in <think></think>, use <write> to implement it and <run> to check it yourself, finish with <answer>done</answer>.\n<think>")
def w2(p, ts):
    return (f"You are a careful Python programmer.\nSpecification:\n{_text(p)}\nThe file {p['module']}.py currently contains only stubs:\n{p['stub']}\nNo tests are given; hidden tests will check it.\n" +
            _tools_en_weak(p, ts) + "Work inside <think> </think>; when you have verified it yourself, write <answer>done</answer>.\n<think>")
def w3(p, ts):
    return (f"Implement {p['module']}.py from the description. There are no tests to run; verify it yourself.\n{_text(p)}\n{p['stub']}\n" + _tools_en_weak(p, ts) + "Think in <think></think>, end with <answer>done</answer>.\n<think>")
def w4(p, ts):
    return (f"Exercise\n- description:\n{_text(p)}\n- stub ({p['module']}.py):\n{p['stub']}\n- tests: none provided (hidden)\n- tools: " + _tools_en_weak(p, ts).strip() +
            "\n- output: reasoning in <think> </think>, then <answer>done</answer>\n<think>")
def w5(p, ts):
    return (_tools_en_weak(p, ts) + f"\nProblem:\n{_text(p)}\nStart from this stub ({p['module']}.py):\n{p['stub']}\n"
            "Think step by step inside <think> </think>, check your work with <run>, and finish with <answer>done</answer>.\n<think>")
def w6(p, ts):
    return (f"题目：\n{_text(p)}\n空壳文件 {p['module']}.py：\n{p['stub']}\n没有给测试，评分用隐藏测试。\n" + _tools_zh_weak(p) +
            "思考写在 <think> </think> 里，用 <write> 写、用 <run> 写断言自己验证、挂了就改，验证通过后写 <answer>done</answer>。\n<think>")
def w7(p, ts):
    return (f"User: Can you implement this exercise? There are no tests, please check it yourself.\n{_text(p)}\n{p['stub']}\n" + _tools_en_weak(p, ts) +
            "Please think in <think> </think> and write <answer>done</answer> once you have verified it.\nAssistant: Sure. <think>")
def hw8(p, ts):
    return (f"Here is a small programming exercise. The description is given first, followed by a stub file that only contains the required signatures. No unit tests are provided; hidden tests will grade the result.\n"
            f"{_text(p)}\n{p['stub']}\n" + _tools_en_weak(p, ts) +
            "Please reason inside <think> </think>, implement the module using the tools, confirm it with your own checks in <run>, and finally write <answer>done</answer>.\n<think>")
def hw9(p, ts):
    return (f"写代码：\n{_text(p)}\n空壳：\n{p['stub']}\n没有测试，自己验证。\n" + _tools_zh_weak(p) + "思考写在 <think> </think> 里，自己验证通过后写 <answer>done</answer>。\n<think>")

WEAK_TEMPLATES = [w0, w1, w2, w3, w4, w5, w6, w7, hw8, hw9]


def render(pid, prob, tool_style=0, env=None, with_tb=False):
    env = env or ExEnv()
    T = WEAK_TEMPLATES if getattr(env, "obs", "strong") == "weak" else TEMPLATES
    return T[pid](prob, tool_style % 3).replace("{mc}", str(env.max_calls))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", default="clock")
    ap.add_argument("--pid", type=int, default=0)
    ap.add_argument("--data", default="data/exercism.json")
    ap.add_argument("--obs", default="strong", choices=["strong", "weak"])
    a = ap.parse_args()
    P = {p["slug"]: p for sp in ("train", "heldout") for p in load_problems(a.data, sp)}
    p = P[a.demo]
    print(render(a.pid, p, env=ExEnv(obs=a.obs)))
    print(f"\n--- {p['slug']}  split {p['split']}  可见 {p['n_vis']} / 隐藏 {p['n_hid']}  参考 {p['ref_lines']} 行  题面 {len(p['text'])} 字")
