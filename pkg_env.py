#!/usr/bin/env python3
"""
pkg_env.py —— 阶段 ②-A 的环境：多文件包（独立拼包）。
题     抽 K 个现成的 bug（data/code_bugs.json，都验过、都有老师 diff）拼成一个包：一文件一 bug，文件名取被测函数名；
       测试 = 每文件 2 条可见 + 隐藏（原来那 1 条 + 严秤造的 ≤ 20 条，data/tests_ext_all.json 有就用）
环境   工具 <test></test>（一次跑全部，按文件报：每文件 passed k/2 + 第一条挂的 traceback，其余只给计数 —— 强但有界）
       <write file="x.py">代码</write>（整个换掉那个文件；K>1 时不写文件名 / 写错文件名 → 报错观测，也算一次调用）
       <run>代码</run>（K 个文件都装好再跑这段）  <ask>问题</ask>（专家）  <answer>done</answer> 结束，或调用上限
       每个文件的测试各起一个沙盒子进程（vfork 之后 11 ms 一个）：一个文件死循环不拖累别的文件；盘上仍然不落任何文件
秤     全部可见 + 隐藏测试通过 → 1 − call_cost × 调用，否则 0（终局二进制，与 ① 同形）；同时算出 r_frac = 通过测试比例 − 价格，给 E3（奖励形状）用
harness 一行不改：stops / fake_tags / step(state, full) / score(state, txt) 的接口和 CodeEnv 一样

    python3 pkg_env.py --demo 3          # 打一份 K=3 的题面（不载模型）
"""
import os, re, ast, json, random, shutil, tempfile, collections
import warnings; warnings.filterwarnings("ignore", category=SyntaxWarning)   # 模型写的怪代码（生成器取下标之类）编译时的提醒，只刷屏，不影响任何东西
import code_env as C

RESULT_OPEN, RESULT_CLOSE = C.RESULT_OPEN, C.RESULT_CLOSE
REPLY_OPEN, REPLY_CLOSE = C.REPLY_OPEN, C.REPLY_CLOSE
TAGS = C.TAGS
ANS_RE = C.ANS_RE
WRITE_RE = re.compile(r"<write\b([^>]*)>(.*?)</write>", re.S)          # 属性里找文件名：file="x.py" / file=x.py / 光秃秃的 x.py
FILE_IN_ATTR = re.compile(r"([A-Za-z_][A-Za-z0-9_]*\.py)")
TB_PER_FILE = 300                                                      # 每个文件的 traceback 最多留多少字
EXT_PATH = "data/tests_ext_all.json"


# ══════════════════════════ 造题 ══════════════════════════
def _top_names(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    names = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name): names.add(t.id)
    return names                                                   # import 的名字不算：两个文件都 import math 不冲突


def _tested_fn(rec):
    for t in rec["visible_tests"]:
        m = re.search(r"assert\s+(?:not\s+)?\(?\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", t)
        if m: return m.group(1)
    names = sorted(_top_names(rec["ref_code"]))
    return names[0] if names else f"task{rec['task_id']}"


def _fname(fn):
    s = re.sub(r"[^A-Za-z0-9]+", "_", fn).strip("_").lower() or "mod"
    if s[0].isdigit(): s = "f_" + s
    return s


def load_ext(path=EXT_PATH):
    try:
        return json.load(open(path))
    except Exception:
        return {}


def make_pkg(recs, ext=None, pkg_id=None):
    """K 条 bug 记录 → 一个包。文件名唯一、顶层名字两两不撞（<run> 时要装进同一个命名空间）"""
    ext = ext or {}
    files, used, seen_names = [], set(), set()
    for r in recs:
        names = _top_names(r["bug_code"]) | _top_names(r["ref_code"])
        if names & seen_names:
            return None
        seen_names |= names
        base = _fname(_tested_fn(r)); name = base; k = 2
        while name in used:
            name = f"{base}_{k}"; k += 1
        used.add(name)
        hidden = [r["hidden_test"]] + [t for t in ext.get(str(r["task_id"]), []) if t != r["hidden_test"]][:20]
        files.append(dict(name=name + ".py", task_id=r["task_id"], kind=r["kind"], bug_type=r["bug_type"], text=r["text"],
                          bug_code=r["bug_code"].strip("\n"), ref_code=r["ref_code"], diff=r["diff"], setup=r.get("setup", "") or "",
                          visible_tests=list(r["visible_tests"]), hidden_tests=hidden, fn=_tested_fn(r)))
    return dict(pkg_id=pkg_id if pkg_id is not None else "+".join(str(f["task_id"]) for f in files), K=len(files), files=files,
                task_ids=[f["task_id"] for f in files], kinds=[f["kind"] for f in files])


def make_pkgs(recs, K, n, seed=0, ext=None, tries=50):
    """从记录里按 seed 抽 n 个 K 文件的包（评测按 seed 固定；训练每步换 seed）。同一个包里不重复用同一道 MBPP 题的不同变异"""
    rng = random.Random(seed); out = []
    while len(out) < n:
        for _ in range(tries):
            pick = rng.sample(recs, K)
            if len({p["text"] for p in pick}) < K:                    # 同一原题的两个变异体：题面一样，丢
                continue
            pkg = make_pkg(pick, ext, pkg_id=f"s{seed}_{len(out)}")
            if pkg: out.append(pkg); break
        else:
            raise RuntimeError("凑不出无名字冲突的包")
    return out


# ══════════════════════════ 沙盒：按文件跑测试、<run> 装全部文件 ══════════════════════════
def run_file_tests(code, tests, setup, cwd, timeout, filename):
    k, oks, tb = C.run_tests(code, tests, setup, cwd, timeout, filename=filename)
    return k, oks, tb


PKG_RUN = r"""
import sys as _sys, traceback as _tb
_files = _FILES_PLACEHOLDER
_snip = _SNIP_PLACEHOLDER
_g = globals()
for _name, _src in _files:
    try:
        exec(compile(_src, _name, "exec"), _g)
    except BaseException:
        _sys.stderr.write("[while loading %s]" % _name + chr(10)); _tb.print_exc(); raise SystemExit(0)
try:
    exec(compile(_snip, "<snippet>", "exec"), _g)
except BaseException:
    _tb.print_exc()
"""


def _clean_run_tb(err):
    """去掉 runner 那层 <stdin> 的帧（连同它下一行源码），其余原样（文件帧已经是各自的文件名）"""
    if "Traceback" in err:
        head = err[: err.index("Traceback")]; err = head + err[err.index("Traceback"):]
    out, skip = [], False
    for l in err.splitlines():
        if 'File "<stdin>"' in l:
            skip = True; continue
        if skip:
            skip = False
            if l.startswith("    "): continue
        out.append(l)
    return "\n".join(out).strip()


def run_snippet(files, snippet, cwd, timeout):
    """files = [(name, src)]；→ 观测文本（stdout + 整理过的 stderr）"""
    script = PKG_RUN.replace("_FILES_PLACEHOLDER", repr([(n, s) for n, s in files])).replace("_SNIP_PLACEHOLDER", repr(snippet))
    rc, out, err = C.run_in_sandbox(script, cwd, timeout, raw=True)
    out = (out or "").strip(); err = _clean_run_tb(err or "")
    txt = out + ("\n" if out and err else "") + err
    return (txt.strip() or "(no output)")[: C.MAX_OUT]


# ══════════════════════════ 环境 ══════════════════════════
class PkgEnv:
    stops = tuple(f"</{t}>" for t in TAGS)
    fake_tags = (RESULT_OPEN, REPLY_OPEN)

    def __init__(self, max_calls=16, call_cost=0.01, timeout=5, ask_fn=None, workdir=None, reward="bin", cache=True):
        self.max_calls, self.call_cost, self.timeout, self.ask_fn, self.workdir, self.reward = max_calls, call_cost, timeout, ask_fn, workdir, reward
        self.cache = {} if cache else None                          # ★ (文件名, 代码, 测试) → 结果。同一份代码跑同一组测试结果是定的：
        self.n_run = self.n_hit = 0                                 #   <test> 每次重跑 K 个文件里没改过的那些、16 条采样反复撞同一个死循环，全是白跑；命中就不起子进程

    def _tests(self, code, tests, setup, cwd, name):
        key = (name, code, tuple(tests), setup) if self.cache is not None else None
        if key is not None:
            r = self.cache.get(key)
            if r is not None:
                self.n_hit += 1; return r
        r = run_file_tests(code, tests, setup, cwd, self.timeout, name); self.n_run += 1
        if key is not None:
            if len(self.cache) > 200000: self.cache.clear()
            self.cache[key] = r
        return r

    def reset(self, pkg):
        d = tempfile.mkdtemp(dir=self.workdir)
        return dict(pkg=pkg, files={f["name"]: f["bug_code"] for f in pkg["files"]}, dir=d, calls=0, n_test=0, n_run=0, n_write=0, n_ask=0,
                    n_bad_write=0, syntax_err=0, written=collections.Counter(), log=[], last=None)

    # ── 跑全部可见测试，按文件报 ──
    def _run_all(self, st, hidden=False):
        """→ [(name, k, oks, tb)]；hidden=True 时把隐藏测试也带上（只在秤里用）"""
        res = []
        for f in st["pkg"]["files"]:
            tests = list(f["visible_tests"]) + (list(f["hidden_tests"]) if hidden else [])
            k, oks, tb = self._tests(st["files"][f["name"]], tests, f["setup"], st["dir"], f["name"])
            res.append((f["name"], k, oks, tb))
        return res

    @staticmethod
    def format_obs(res):
        tot = sum(len(o) for _, _, o, _ in res); ok = sum(k for _, k, _, _ in res)
        lines = [f"passed {ok}/{tot}"]
        for name, k, oks, tb in res:
            lines.append(f"{name}: {k}/{len(oks)}")
            if tb and k < len(oks):
                lines += ["  " + l for l in tb[:TB_PER_FILE].splitlines()]
        return "\n".join(lines)

    def initial_obs(self, pkg):
        if "initial_obs" in pkg:
            return pkg["initial_obs"]
        st = self.reset(pkg)
        try:
            pkg["initial_obs"] = self.format_obs(self._run_all(st))
        finally:
            shutil.rmtree(st["dir"], ignore_errors=True)
        return pkg["initial_obs"]

    # ── 一步 ──
    def step(self, st, full):
        tag = None
        for t in TAGS:
            if full.rstrip().endswith(f"</{t}>"):
                tag = t; break
        if tag is None:
            return None, True
        st["calls"] += 1
        pkg = st["pkg"]; names = [f["name"] for f in pkg["files"]]
        if tag == "test":
            st["n_test"] += 1
            res = self._run_all(st); st["last"] = {n: (k, len(o)) for n, k, o, _ in res}
            obs = self.format_obs(res)
        elif tag == "run":
            st["n_run"] += 1
            m = C.TAG_RE["run"].findall(full); arg = (m[-1] if m else "").strip("\n")
            obs = run_snippet([(n, st["files"][n]) for n in names], C.normalize_code(arg), st["dir"], self.timeout)
        elif tag == "write":
            st["n_write"] += 1
            m = WRITE_RE.findall(full); attr, body = (m[-1] if m else ("", ""))
            fm = FILE_IN_ATTR.search(attr); fname = fm.group(1) if fm else (names[0] if len(names) == 1 else None)
            if fname is None:
                st["n_bad_write"] += 1; obs = "not written: say which file, e.g. <write file=\"" + names[0] + "\">…</write>. Files: " + ", ".join(names)
            elif fname not in st["files"]:
                st["n_bad_write"] += 1; obs = f"not written: no file named {fname}. Files: " + ", ".join(names)
            elif not C.normalize_code(body).strip():
                st["n_bad_write"] += 1; obs = f"not written: empty code for {fname}"          # 探针里见过 <write></write>：空文件不是任何人想要的
            else:
                code = C.normalize_code(body)
                try:
                    compile(code, fname, "exec"); obs = f"written {fname}"
                except SyntaxError as e:
                    st["syntax_err"] += 1; obs = f"written {fname}, but SyntaxError: {e.msg} (line {e.lineno})"
                st["files"][fname] = code; st["written"][fname] += 1
        else:                                                      # ask
            st["n_ask"] += 1
            m = C.TAG_RE["ask"].findall(full); q = (m[-1] if m else "").strip()
            a = self.ask_fn(q, pkg, dict(st["files"])) if self.ask_fn else "error"
            st["log"].append(("ask", q, a))
            return f"{REPLY_OPEN}{a}{REPLY_CLOSE}", st["calls"] >= self.max_calls
        st["log"].append((tag, obs[:200]))
        obs = obs[: max(C.MAX_OUT, 200 + TB_PER_FILE * len(names))]
        return f"{RESULT_OPEN}{obs}{RESULT_CLOSE}", st["calls"] >= self.max_calls

    # ── 秤 ──
    def score(self, st, txt):
        """全部可见 + 隐藏测试通过 → 1 − call_cost × 调用，否则 0。d 里同时给 r_frac（通过比例 − 价格）、按文件的对错、每个文件的改动读数"""
        pkg = st["pkg"]; res = self._run_all(st, hidden=True)
        per, n_pass, n_tot = [], 0, 0
        for f, (name, k, oks, _) in zip(pkg["files"], res):
            nv = len(f["visible_tests"])
            per.append(dict(name=name, correct=int(all(oks)), vis_pass=sum(oks[:nv]), hidden_pass=int(all(oks[nv:])) if len(oks) > nv else 1,
                            changed=int(st["files"][name].strip() != f["bug_code"].strip()),
                            **C.edit_stats(f["bug_code"], st["files"][name], f["diff"])))
            n_pass += k; n_tot += len(oks)
        allpass = all(p["correct"] for p in per)
        price = self.call_cost * st["calls"]
        r_bin = max(0.0, 1.0 - price) if allpass else 0.0
        r_frac = max(0.0, n_pass / max(1, n_tot) - price)
        d = dict(correct=float(allpass), r_bin=r_bin, r_frac=r_frac, files_fixed=sum(p["correct"] for p in per), K=pkg["K"],
                 frac_files=sum(p["correct"] for p in per) / pkg["K"], frac_tests=n_pass / max(1, n_tot),
                 passed=n_pass, n_tests=n_tot, vis_pass=sum(p["vis_pass"] for p in per), hidden_pass=int(all(p["hidden_pass"] for p in per)),
                 vis_all=int(all(p["vis_pass"] == len(f["visible_tests"]) for p, f in zip(per, pkg["files"]))),
                 calls=st["calls"], n_test=st["n_test"], n_run=st["n_run"], n_write=st["n_write"], n_ask=st["n_ask"], n_bad_write=st["n_bad_write"],
                 syntax_err=st["syntax_err"], has_answer=float(bool(ANS_RE.search(txt))), changed=float(any(p["changed"] for p in per)),
                 fmt=float(st["calls"] > 0), per_file=per, files_touched=len(st["written"]))
        shutil.rmtree(st["dir"], ignore_errors=True)
        return (r_frac if self.reward == "frac" else r_bin), d


# ══════════════════════════ 行为读数 ══════════════════════════
STATE_RE = re.compile(r"(?<![A-Za-z])(Status|Progress|Remaining|Done|状态|进度|已修|剩)[:：]", re.I)   # E2：老师写的状态行长这样（可能紧跟在 </result> 后面，不要求行首），评测时数模型写没写


CALL_RE = {t: re.compile(rf"<{t}\b[^>]*>(?:(?!<{t}\b|<result>).)*?</{t}>", re.S) for t in TAGS}   # 开标签可带属性（<write file="x.py">）


def behavior(txt):
    order = sorted((m.start(), t) for t in TAGS for m in CALL_RE[t].finditer(txt))
    seq = [t for _, t in order]
    fw = seq.index("write") if "write" in seq else None
    return dict(seq=seq, first=seq[0] if seq else None, n_calls=len(seq),
                test_before_write=int(fw is not None and "test" in seq[:fw]), test_after_write=int(fw is not None and "test" in seq[fw + 1:]),
                files_written=[FILE_IN_ATTR.search(a).group(1) if FILE_IN_ATTR.search(a) else None for a, _ in WRITE_RE.findall(txt)],
                n_state_lines=len(STATE_RE.findall(txt)))


# ══════════════════════════ 专家 ══════════════════════════
def ask_expert(question, pkg, files, model="deepseek-reasoner"):
    try:
        import deepseek_tool as DS
        body = "\n".join(f"=== {n} ===\n{c}" for n, c in files.items())
        tests = "\n".join(t for f in pkg["files"] for t in f["visible_tests"])
        q = "Package files:\n" + body + "\nVisible tests:\n" + tests + f"\nQuestion: {question}"
        r = DS.ask(q, model=model, system=C.CODE_SYS.replace("function", "file"), max_tokens=8000)
        c = (r.get("content") or "").strip()
        c = re.sub(r"^```(?:python)?\s*|\s*```$", "", c, flags=re.S).strip()
        return c[:1500] if c else "error"
    except Exception:
        return "error"


# ══════════════════════════ 题面模板（8 训练 + 2 留出，两种开局，三种工具措辞）══════════════════════════
def files_block(pkg, zh=False):
    out = []
    for f in pkg["files"]:
        out.append(f"=== {f['name']} ===\n# {f['text']}\n{f['bug_code']}")
    return "\n".join(out)


def tests_block(pkg):
    return "\n".join(t for f in pkg["files"] for t in f["visible_tests"])


def _tools_en(pkg, ts):
    K = pkg["K"]; n = 2 * K; ex = pkg["files"][0]["name"]
    return [
        f"Tools: <test></test> runs all visible tests and returns <result>passed k/{n} plus, per file, the first failing traceback</result>. <run>code</run> executes code with every file loaded and returns its output. <write file=\"{ex}\">code</write> replaces that whole file with your code. <ask>question</ask> asks an expert, who replies with code in <reply></reply>; always verify with <test> before answering. At most {{mc}} tool calls. ",
        f"You can call tools: write <test></test> to run the tests of all files (you get <result>passed k/{n}</result> and the failing traceback of each file), <run>...</run> to execute a snippet, <write file=\"{ex}\">...</write> to overwrite one file with new code, <ask>...</ask> to consult an expert (reply comes in <reply></reply>, verify it with <test>). {{mc}} calls maximum. ",
        f"TOOLS: <test></test> -> <result>passed k/{n}, per-file traceback</result> | <run>CODE</run> -> <result>output</result> | <write file=\"{ex}\">CODE</write> -> replaces that file | <ask>Q</ask> -> <reply>code</reply> (verify before answering). Budget: {{mc}} calls. ",
    ][ts]


def _tools_zh(pkg):
    K = pkg["K"]; n = 2 * K; ex = pkg["files"][0]["name"]
    return (f"工具：<test></test> 跑全部测试，返回 <result>passed k/{n} 和每个文件第一条失败的 traceback</result>；<run>代码</run> 执行一段代码（所有文件已装载）；"
            f"<write file=\"{ex}\">代码</write> 用你的代码整个替换那个文件；<ask>问题</ask> 问专家，回复在 <reply></reply> 里，作答前必须用 <test> 验证。最多 {{mc}} 次调用。")


def _obs(obs, zh=False):
    return "" if obs is None else (("运行测试的输出：\n" if zh else "Output of running the tests:\n") + obs + "\n")


def p0(p, obs, ts):
    return (f"A conversation between User and Assistant. The user gives a small Python package where each file has one bug and the Assistant fixes all of them.\n"
            f"User: Files:\n{files_block(p)}\nTests:\n{tests_block(p)}\n" + _obs(obs) + _tools_en(p, ts) +
            "Think inside <think> </think>, fix every file with the tools, then write <answer>done</answer>.\nAssistant: <think>")
def p1(p, obs, ts):
    return (f"Task: each file below contains one bug. Make all tests pass.\n{files_block(p)}\nVisible tests:\n{tests_block(p)}\n" + _obs(obs) + _tools_en(p, ts) +
            "Reason in <think></think>, use <write> to fix files and <test> to check, finish with <answer>done</answer>.\n<think>")
def p2(p, obs, ts):
    return (f"You are a careful Python debugger.\nThe package below has {p['K']} files; each one has a bug.\n{files_block(p)}\nIt must pass:\n{tests_block(p)}\n" +
            _obs(obs) + _tools_en(p, ts) + "Work inside <think> </think>; when all tests pass, write <answer>done</answer>.\n<think>")
def p3(p, obs, ts):
    return (f"Fix these files so the tests pass.\n{files_block(p)}\n{tests_block(p)}\n" + _obs(obs) + _tools_en(p, ts) + "Think in <think></think>, end with <answer>done</answer>.\n<think>")
def p4(p, obs, ts):
    return (f"Bug report\n- package ({p['K']} files, one bug each):\n{files_block(p)}\n- failing tests:\n" + "\n".join("  " + t for f in p["files"] for t in f["visible_tests"]) + "\n" +
            ("- test output:\n" + obs + "\n" if obs else "") + "- tools: " + _tools_en(p, ts).strip() + "\n- output: reasoning in <think> </think>, then <answer>done</answer>\n<think>")
def p5(p, obs, ts):
    return (_tools_en(p, ts) + f"\nThe package below fails its tests; every file has exactly one bug. Find and fix them all.\n{files_block(p)}\nTests:\n{tests_block(p)}\n" +
            _obs(obs) + "Think step by step inside <think> </think> and finish with <answer>done</answer>.\n<think>")
def p6(p, obs, ts):
    return (f"下面这个包有 {p['K']} 个文件，每个文件里有一个 bug：\n{files_block(p)}\n它需要通过：\n{tests_block(p)}\n" + _obs(obs, zh=True) + _tools_zh(p) +
            "思考写在 <think> </think> 里，用 <write> 改、用 <test> 验证，全部通过后写 <answer>done</answer>。\n<think>")
def p7(p, obs, ts):
    return (f"User: These files are supposed to work but the tests fail. Can you fix them all?\n{files_block(p)}\nTests:\n{tests_block(p)}\n" + _obs(obs) + _tools_en(p, ts) +
            "Please think in <think> </think> and write <answer>done</answer> once every test passes.\nAssistant: Sure. <think>")
def h8(p, obs, ts):
    return (f"Here is a small debugging exercise with {p['K']} files. Each file was written to do what its comment says, but each contains one mistake and the tests after them fail.\n"
            f"{files_block(p)}\n{tests_block(p)}\n" + _obs(obs) + _tools_en(p, ts) +
            "Please reason inside <think> </think>, repair each file using the tools, confirm with <test>, and finally write <answer>done</answer>.\n<think>")
def h9(p, obs, ts):
    return (f"修 bug（{p['K']} 个文件，每个一处）：\n{files_block(p)}\n测试：\n{tests_block(p)}\n" + _obs(obs, zh=True) + _tools_zh(p) +
            "思考写在 <think> </think> 里，全部修好并验证后写 <answer>done</answer>。\n<think>")

TEMPLATES = [p0, p1, p2, p3, p4, p5, p6, p7, h8, h9]
SETS = {"train": list(range(8)), "heldout": [8, 9], "all": list(range(10)), "t0": [0]}


def render(pid, pkg, with_tb=True, tool_style=0, env=None):
    env = env or PkgEnv()
    obs = env.initial_obs(pkg) if with_tb else None
    return TEMPLATES[pid](pkg, obs, tool_style % 3).replace("{mc}", str(env.max_calls))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=int, default=3, help="K")
    ap.add_argument("--split", default="train")
    ap.add_argument("--pid", type=int, default=0)
    ap.add_argument("--tb", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    recs = C.load_problems("data/code_bugs.json", a.split)
    pkg = make_pkgs(recs, a.demo, 1, seed=a.seed, ext=load_ext())[0]
    print(render(a.pid, pkg, with_tb=a.tb))
    print("\n--- 文件 / 隐藏测试条数：", [(f["name"], len(f["hidden_tests"])) for f in pkg["files"]])
