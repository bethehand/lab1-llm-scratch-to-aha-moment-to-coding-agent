#!/usr/bin/env python3
"""
code_env.py —— 长程编程阶段 ①「修 bug」的环境：沙盒 + 四个工具 + 秤 + 题面模板。走 harness.generate_with_env（stops / fake_tags / step / score）。

一局   题 = data/code_bugs.json 里一条（有 bug 的函数 + 2 条可见测试 + 1 条隐藏测试）。环境状态 = 当前函数文件内容（模型 <write> 会改它）
工具   <test></test>            用当前函数文件跑 2 条可见测试（每条单独跑，测试文件永远从母本重铺）→ <result>passed k/2 …第一条挂的 traceback（截 600 字）</result>
       <run>代码</run>          沙盒里执行：先 exec 当前函数文件，再跑这段代码 → <result>stdout / stderr（截 600 字）</result>
       <write>代码</write>      整个函数文件替换成这段（只能写这一个文件；先 compile，语法错也写进去并把错报回来）→ <result>written</result>
       <ask>问题</ask>          问专家（DeepSeek，看得到题面 + 当前代码；回一段代码）→ <reply>代码</reply>；模型得自己 <write> 再 <test>
       结束：写 <answer>done</answer>（然后 EOS），或工具调用到 max_calls 次
秤     最终函数文件跑全部 3 条（2 可见 + 1 隐藏）：全过 1.0，否则 0；− call_cost × 调用次数；假 <result>/<reply> 由训练器罚。只奖终局，不做过程分
沙盒   子进程、cwd = 临时目录、5 s 超时、Linux 上 RLIMIT_AS 512 MB、环境变量只留 PATH、start_new_session 超时杀整组、socket 被换成抛异常、输出截 600 字
开局   with_tb=True：题面里带「运行可见测试的输出」（挂的那条的 traceback）—— 强观测；False：不带，模型得先 <test>
模板   8 训练 + 2 留出（8 啰嗦英文、9 中文）；两种开局各自渲染
"""
import os, re, sys, json, random, tempfile, subprocess, shutil, platform, traceback as _tb
import warnings; warnings.filterwarnings("ignore", category=SyntaxWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")      # 沙盒每次 fork 子进程，tokenizers 会刷「forked after parallelism」警告

RESULT_OPEN, RESULT_CLOSE = "<result>", "</result>"
REPLY_OPEN, REPLY_CLOSE = "<reply>", "</reply>"
TAGS = ("test", "run", "write", "ask")
TAG_RE = {t: re.compile(rf"<{t}>(.*?)</{t}>", re.S) for t in TAGS}
ANS_RE = re.compile(r"<answer>(.*?)</answer>", re.S)
MAX_OUT = 600

SANDBOX_PRELUDE = (
    "import socket as _s\n"
    "def _no(*a, **k): raise OSError('network disabled in sandbox')\n"
    "_s.socket = _no; _s.create_connection = _no\n"
    "import builtins as _b; _b.input = lambda *a, **k: ''\n"
    "try:\n"                                                                                    # ★ 资源上限在子进程自己身上设（原来用 preexec_fn 在 fork 后 exec 前设）：
    "    import resource as _r, platform as _p\n"                                              #   preexec_fn 逼 Python 走真 fork，要复制父进程整张页表；训练进程十几 GB，一次 fork 0.1~0.3 s，
    "    if _p.system() == 'Linux': _r.setrlimit(_r.RLIMIT_AS, (512 << 20, 512 << 20)); _r.setrlimit(_r.RLIMIT_NPROC, (64, 64))\n"   #   还串行。去掉后 Python ≥ 3.10 走 vfork，起进程不再随父进程大小涨。
    "except Exception: pass\n"                                                                 #   软硬上限一起设，用户代码升不回去；效果和原来一样
)


N_TIMEOUT = 0                                              # ★ 诊断：本进程沙盒超时次数（训练器每步读差值，看哪张卡在等死循环）
_TLOCK = __import__("threading").Lock()


def run_in_sandbox(src, cwd, timeout=5, raw=False):
    global N_TIMEOUT
    """src 走 stdin 喂给 `python -I -`，盘上不落任何文件 → (returncode, stdout, stderr)；超时 → (-9, '', 'Timeout: …')。
    ★ 之前写成 _run.py 放在 cwd：秤跑隐藏测试时断言就在盘上，函数运行时 open('_run.py') 就能读到期望值 —— 一个作弊洞，改成 stdin 堵掉。
    子进程 start_new_session=True 自己当组长，超时 killpg(p.pid) 只杀它那一组"""
    p = subprocess.Popen([sys.executable, "-I", "-S", "-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd,   # -S：不加载 site，冷启动快几百毫秒
                         env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"}, start_new_session=True)   # ★ 不用 preexec_fn：让 Python 能走 vfork（见 SANDBOX_PRELUDE）
    try:
        out, err = p.communicate(SANDBOX_PRELUDE + src, timeout=timeout)
        if raw:
            return p.returncode, out[-20000:], err[-20000:]
        return p.returncode, out[-MAX_OUT:], err[-MAX_OUT:]
    except subprocess.TimeoutExpired:
        with _TLOCK:
            N_TIMEOUT += 1
        try:
            os.killpg(p.pid, 9)
        except Exception:
            p.kill()
        p.communicate()
        return -9, "", f"Timeout: code ran longer than {timeout} s"


def normalize_code(arg):
    """<write>/<run> 里的代码：去 markdown 围栏、去整体缩进（模型常写「<write> def f():」首行带一个空格 → IndentationError，编辑器都会容忍的格式噪声）"""
    import textwrap
    c = re.sub(r"^\s*```[a-zA-Z]*\s*\n?|\n?\s*```\s*$", "", arg.strip("\n"))
    c = textwrap.dedent(c).strip("\n")
    lines = c.split("\n")
    if lines and lines[0] != lines[0].lstrip():                                  # ★ 只有首行带空格（「<write> n = 1\ndef f():」）：dedent 救不了，首行单独去掉。模块级第一条语句本来就该顶格
        lines[0] = lines[0].lstrip()
    return "\n".join(lines)


def _clean_tb(stderr, shell=False):
    """只留 Traceback 之后；去掉 runner 那层 <stdin> 的壳（exec 出来的帧已经标成文件名）；直接喂 stdin 的（<run>）把 <stdin> 行号减去 prelude。
    shell=True：来自 run_tests 的 runner，<stdin> 帧一律是壳（模块级 SyntaxError 时 stderr 里没有 <test>，以前会把壳帧漏成 solution.py 第 7 行）"""
    if "Traceback" in stderr:
        stderr = stderr[stderr.index("Traceback"):]
    out = []
    skip = False
    for l in stderr.splitlines():
        if 'File "<stdin>"' in l and (shell or "<test>" in stderr):
            skip = True; continue                              # runner 壳的帧（下一行是它的源码行，一并跳）
        if skip:
            skip = False
            if l.startswith("    "): continue
        out.append(re.sub(r'File "<stdin>", line (\d+)', lambda m: f'File "solution.py", line {int(m.group(1)) - SANDBOX_PRELUDE.count(chr(10))}', l))
    return "\n".join(out).strip()[-MAX_OUT:]


RUNNER = r"""
import sys as _sys, traceback as _tb
_res = []
_src = _SRC_PLACEHOLDER
_tests = _TESTS_PLACEHOLDER
try:
    exec(compile(_src, _FN_PLACEHOLDER, "exec"), globals())
except BaseException as _e:
    _tb.print_exc()
    _sys.stdout.write("__TESTS__" + repr([(False, _e.__class__.__name__)] * len(_tests)) + chr(10))
    raise SystemExit(0)
for _i, _t in enumerate(_tests):
    try:
        exec(compile(_t, "<test>", "exec"), globals())
        _res.append((True, ""))
    except BaseException as _e:
        if not any(ok for ok, _ in _res) and all(not ok for ok, _ in _res):
            _sys.stderr.write("__TB__%d__" % _i + chr(10)); _tb.print_exc(); _sys.stderr.write("__ENDTB__" + chr(10))
        _res.append((False, _e.__class__.__name__))
_sys.stdout.write("__TESTS__" + repr(_res) + chr(10))
"""


def run_tests(code, tests, setup, cwd, timeout=5, filename="solution.py"):
    """★ 一个子进程跑全部测试（每条单独 try/except 记结果），不再每条起一个进程 → 子进程数从 N 变 1。
    → (通过数, 每条 ok?, 第一条挂的整理过的 traceback)。整个进程超时 → 全算挂。filename：traceback 里代码帧显示的文件名（多文件包传各自的名字）"""
    import ast as _ast
    src = ((setup + "\n") if setup else "") + code                    # ★ 没有 setup 就别拼换行：否则 traceback 的行号比题面里的代码大 1（① 一直如此，多文件里行号要准）
    script = RUNNER.replace("_SRC_PLACEHOLDER", repr(src)).replace("_TESTS_PLACEHOLDER", repr(list(tests))).replace("_FN_PLACEHOLDER", repr(filename))
    rc, out, err = run_in_sandbox(script, cwd, timeout, raw=True)
    m = re.search(r"__TESTS__(\[.*\])", out or "")
    if not m:                                                          # 超时 / 被杀 / 输出被截：全算挂
        oks = [False] * len(tests)
        tb = f"{tests[0]}\n{_clean_tb(err, shell=True) if err else 'Timeout or no output'}"
        return 0, oks, tb
    res = _ast.literal_eval(m.group(1))
    oks = [ok for ok, _ in res]
    first_tb = ""
    for i, (ok, _) in enumerate(res):
        if not ok:
            seg = re.search(r"__TB__%d__\n(.*?)__ENDTB__" % i, err or "", re.S)
            body = _clean_tb(seg.group(1), shell=True) if seg else _clean_tb(err, shell=True) if err else "no output"
            first_tb = f"{tests[i]}\n{body}"
            break
    return sum(oks), oks, first_tb


class CodeEnv:
    stops = [f"</{t}>" for t in TAGS]
    fake_tags = [RESULT_OPEN, REPLY_OPEN]

    def __init__(self, max_calls=8, call_cost=0.02, timeout=5, ask_fn=None, workdir=None):
        self.max_calls, self.call_cost, self.timeout, self.ask_fn = max_calls, call_cost, timeout, ask_fn
        self.workdir = workdir or tempfile.mkdtemp(prefix="code_env_")

    # ── 一局的状态 ──
    def reset(self, prob):
        d = tempfile.mkdtemp(dir=self.workdir)
        return dict(prob=prob, code=prob["bug_code"], dir=d, calls=0, n_test=0, n_run=0, n_write=0, n_ask=0, log=[], syntax_err=0)

    def initial_obs(self, prob):
        """带报错开局用：可见测试的输出（跑一次，结果可缓存在题上）"""
        if "initial_obs" in prob:
            return prob["initial_obs"]
        d = tempfile.mkdtemp(dir=self.workdir)
        try:
            k, oks, tb = run_tests(prob["bug_code"], prob["visible_tests"], prob.get("setup", ""), d, self.timeout)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        prob["initial_obs"] = f"passed {k}/{len(oks)}\n{tb}".strip()
        return prob["initial_obs"]

    def step(self, st, full):
        """harness 停在某个 </tag> 上 → 执行 → (注入文本, done)"""
        tag = None
        for t in TAGS:
            if full.rstrip().endswith(f"</{t}>"):
                tag = t; break
        if tag is None:
            return None, True
        m = TAG_RE[tag].findall(full)
        arg = (m[-1] if m else "").strip("\n")
        st["calls"] += 1
        p = st["prob"]
        if tag == "test":
            st["n_test"] += 1
            k, oks, tb = run_tests(st["code"], p["visible_tests"], p.get("setup", ""), st["dir"], self.timeout)
            obs = f"passed {k}/{len(oks)}" + (f"\n{tb}" if tb else "")
        elif tag == "run":
            st["n_run"] += 1
            rc, out, err = run_in_sandbox((p.get("setup", "") or "") + "\n" + st["code"] + "\n" + normalize_code(arg), st["dir"], self.timeout)
            obs = (out.strip() + ("\n" if out.strip() and err.strip() else "") + (_clean_tb(err) if err.strip() else "")).strip() or "(no output)"
        elif tag == "write":
            st["n_write"] += 1
            code = normalize_code(arg)
            try:
                compile(code, "solution.py", "exec"); obs = "written"
            except SyntaxError as e:
                st["syntax_err"] += 1; obs = f"written, but SyntaxError: {e.msg} (line {e.lineno})"
            st["code"] = code
        else:                                                      # ask
            st["n_ask"] += 1
            q = arg
            a = self.ask_fn(q, p, st["code"]) if self.ask_fn else "error"
            st["log"].append(("ask", q, a))
            inj = f"{REPLY_OPEN}{a}{REPLY_CLOSE}"
            return inj, st["calls"] >= self.max_calls
        st["log"].append((tag, arg[:200], obs[:200]))
        obs = obs[:MAX_OUT]
        return f"{RESULT_OPEN}{obs}{RESULT_CLOSE}", st["calls"] >= self.max_calls

    def score(self, st, txt):
        """最终函数文件跑全部 3 条：全过 1 − call_cost × 调用；否则 0。返回 (r, d)"""
        p = st["prob"]
        tests = list(p["visible_tests"]) + [p["hidden_test"]]
        k, oks, _ = run_tests(st["code"], tests, p.get("setup", ""), st["dir"], self.timeout)
        allpass = all(oks)
        r = max(0.0, 1.0 - self.call_cost * st["calls"]) if allpass else 0.0
        d = dict(correct=float(allpass), passed=k, n_tests=len(oks), vis_pass=sum(oks[:-1]), hidden_pass=int(oks[-1]),
                 calls=st["calls"], n_test=st["n_test"], n_run=st["n_run"], n_write=st["n_write"], n_ask=st["n_ask"],
                 syntax_err=st["syntax_err"], has_answer=float(bool(ANS_RE.search(txt))), changed=float(st["code"].strip() != p["bug_code"].strip()),
                 fmt=float(st["calls"] > 0))
        shutil.rmtree(st["dir"], ignore_errors=True)
        return r, d


# ── 改动读数：最终文件 vs 原 bug 文件 ──
def edit_stats(bug_code, final_code, diff):
    """→ dict(changed_lines=最终文件相对原文件改了几行（+ 与 − 各算）, hit_bug_line=改的行里有没有注入的那行, rewrite=改动 ≥ 一半的行)
    诊断「定位修改」（1~2 行）还是「整段重写」（一大片）；hit_bug_line 看它找没找到 bug"""
    import difflib
    a, b = bug_code.strip().splitlines(), final_code.strip().splitlines()
    d = [l for l in difflib.unified_diff(a, b, lineterm="", n=0) if not l.startswith(("---", "+++", "@@"))]
    bug_lines = {l[1:].strip() for l in diff if l.startswith("+")}          # 注入器的 diff：+ 行 = bug 版里那一行
    removed = {l[1:].strip() for l in d if l.startswith("-")}
    n_removed = sum(1 for l in d if l.startswith("-"))
    survived = (len(a) - n_removed) / max(1, len(a))                         # 原文件的行有多少原样留下
    rewrite = int(len(a) >= 3 and survived < 0.5)
    hit = int(bool(bug_lines & removed))
    return dict(changed_lines=len(d), hit_bug_line=hit, rewrite=rewrite,
                localized=int(hit and not rewrite and n_removed <= 2))          # ★ 定位修改：改中 bug 行、不是整段重写、最多动两行 —— 「用了观测」的读数


# ── 行为读数（从轨迹文本算，评测和 analyze 用）──
CALL_RE = {t: re.compile(rf"<{t}>(?:(?!<{t}>|<result>).)*?</{t}>", re.S) for t in TAGS}   # 不跨越另一个开标签或 <result>：观测里的 File "<test>" 不会被配成调用


def behavior(txt):
    order = [(m.start(), t) for t in TAGS for m in CALL_RE[t].finditer(txt)]
    order.sort()
    seq = [t for _, t in order]
    first_write = seq.index("write") if "write" in seq else None
    return dict(seq=seq, first=seq[0] if seq else None,
                test_before_write=int(first_write is not None and "test" in seq[:first_write]),
                test_after_write=int(first_write is not None and "test" in seq[first_write + 1:]),
                n_calls=len(seq))


# ── 专家 ──
CODE_SYS = ("You fix Python functions. You are given a task description, the current (buggy) code and its failing tests. "
            "Reply with ONLY the complete corrected Python function code, no explanation, no markdown fences.")


def ask_expert(question, prob, code, model="deepseek-reasoner"):
    try:
        import deepseek_tool as DS
        q = (f"Task: {prob['text']}\nCurrent code:\n{code}\nVisible tests:\n" + "\n".join(prob["visible_tests"]) + f"\nQuestion: {question}")
        r = DS.ask(q, model=model, system=CODE_SYS, max_tokens=8000)
        c = (r.get("content") or "").strip()
        c = re.sub(r"^```(?:python)?\s*|\s*```$", "", c, flags=re.S).strip()
        return c[:1500] if c else "error"
    except Exception:
        return "error"


# ── 题面模板：8 训练 + 2 留出，两种开局 ──
TOOLS_EN = [
    "Tools: <test></test> runs the visible tests and returns <result>passed k/2 + the first failing traceback</result>. <run>code</run> executes code with your function available and returns its output. <write>code</write> replaces the whole function file with your code. <ask>question</ask> asks an expert, who replies with code in <reply></reply>; always verify with <test> before answering. At most 8 tool calls. ",
    "You can call tools: write <test></test> to run the given tests (you get <result>passed k/2</result> and the failing traceback), <run>...</run> to execute a snippet, <write>...</write> to overwrite the function with new code, <ask>...</ask> to consult an expert (reply comes in <reply></reply>, verify it with <test>). 8 calls maximum. ",
    "TOOLS: <test></test> -> <result>passed k/2, traceback</result> | <run>CODE</run> -> <result>output</result> | <write>CODE</write> -> replaces the function | <ask>Q</ask> -> <reply>code</reply> (verify before answering). Budget: 8 calls. ",
]
TOOLS_ZH = "工具：<test></test> 跑给定的测试，返回 <result>passed k/2 和第一条失败的 traceback</result>；<run>代码</run> 执行一段代码（函数已定义）；<write>代码</write> 用你的代码整个替换函数文件；<ask>问题</ask> 问专家，回复在 <reply></reply> 里，作答前必须用 <test> 验证。最多 8 次调用。"


def _obs_block(obs, zh=False):
    if obs is None:
        return ""
    return ("运行测试的输出：\n" if zh else "Output of running the tests:\n") + obs + "\n"


def t0(p, obs, ts):
    return (f"A conversation between User and Assistant. The user gives a Python function with a bug and the Assistant fixes it.\n"
            f"User: {p['text']}\nThe following implementation is buggy:\n{p['bug_code']}\nTests:\n" + "\n".join(p["visible_tests"]) + "\n" + _obs_block(obs) +
            TOOLS_EN[ts] + "Think inside <think> </think>, fix the function with the tools, then write <answer>done</answer>.\nAssistant: <think>")
def t1(p, obs, ts):
    return (f"Task: {p['text']}\nBuggy code:\n{p['bug_code']}\nVisible tests:\n" + "\n".join(p["visible_tests"]) + "\n" + _obs_block(obs) +
            TOOLS_EN[ts] + "Reason in <think></think>, use <write> to fix it and <test> to check, finish with <answer>done</answer>.\n<think>")
def t2(p, obs, ts):
    return (f"You are a careful Python debugger.\nSpecification: {p['text']}\nCurrent implementation (it has a bug):\n{p['bug_code']}\nIt must pass:\n" + "\n".join(p["visible_tests"]) + "\n" +
            _obs_block(obs) + TOOLS_EN[ts] + "Work inside <think> </think>; when the tests pass, write <answer>done</answer>.\n<think>")
def t3(p, obs, ts):
    return (f"Fix this function so the tests pass.\n{p['text']}\n{p['bug_code']}\n" + "\n".join(p["visible_tests"]) + "\n" + _obs_block(obs) +
            TOOLS_EN[ts] + "Think in <think></think>, end with <answer>done</answer>.\n<think>")
def t4(p, obs, ts):
    return (f"Bug report\n- description: {p['text']}\n- code:\n{p['bug_code']}\n- failing tests:\n" + "\n".join("  " + t for t in p["visible_tests"]) + "\n" +
            ("- test output:\n" + obs + "\n" if obs else "") + "- tools: " + TOOLS_EN[ts].strip() + "\n- output: reasoning in <think> </think>, then <answer>done</answer>\n<think>")
def t5(p, obs, ts):
    return (TOOLS_EN[ts] + f"\nProblem: {p['text']}\nThe implementation below fails its tests. Find and fix the bug.\n{p['bug_code']}\nTests:\n" + "\n".join(p["visible_tests"]) + "\n" +
            _obs_block(obs) + "Think step by step inside <think> </think> and finish with <answer>done</answer>.\n<think>")
def t6(p, obs, ts):
    return (f"问题：{p['text']}\n下面这个实现有一个 bug：\n{p['bug_code']}\n它需要通过：\n" + "\n".join(p["visible_tests"]) + "\n" + _obs_block(obs, zh=True) +
            TOOLS_ZH + "思考写在 <think> </think> 里，用 <write> 改、用 <test> 验证，通过后写 <answer>done</answer>。\n<think>")
def t7(p, obs, ts):
    return (f"User: This function is supposed to do the following but a test fails. Can you fix it?\n{p['text']}\n{p['bug_code']}\nTests:\n" + "\n".join(p["visible_tests"]) + "\n" +
            _obs_block(obs) + TOOLS_EN[ts] + "Please think in <think> </think> and write <answer>done</answer> once the tests pass.\nAssistant: Sure. <think>")
def h8(p, obs, ts):
    return (f"Here is a small debugging exercise. The function below was written to accomplish the following: {p['text']} Unfortunately it contains a mistake and does not pass the tests listed after it.\n"
            f"{p['bug_code']}\n" + "\n".join(p["visible_tests"]) + "\n" + _obs_block(obs) + TOOLS_EN[ts] +
            "Please reason inside <think> </think>, repair the function using the tools, confirm with <test>, and finally write <answer>done</answer>.\n<think>")
def h9(p, obs, ts):
    return (f"修 bug：{p['text']}\n代码：\n{p['bug_code']}\n测试：\n" + "\n".join(p["visible_tests"]) + "\n" + _obs_block(obs, zh=True) + TOOLS_ZH +
            "思考写在 <think> </think> 里，修好并验证后写 <answer>done</answer>。\n<think>")

TEMPLATES = [t0, t1, t2, t3, t4, t5, t6, t7, h8, h9]
SETS = {"train": list(range(8)), "heldout": [8, 9], "all": list(range(10)), "t0": [0]}


def render(pid, prob, with_tb=True, tool_style=0, env=None):
    obs = (env or CodeEnv()).initial_obs(prob) if with_tb else None
    return TEMPLATES[pid](prob, obs, tool_style % 3)


def load_problems(path="data/code_bugs.json", split="train", kinds=None, bug_type=None):
    B = json.load(open(path))
    return [b for b in B if b["split"] == split and (kinds is None or b["kind"] in kinds) and (bug_type is None or b["bug_type"] == bug_type)]
