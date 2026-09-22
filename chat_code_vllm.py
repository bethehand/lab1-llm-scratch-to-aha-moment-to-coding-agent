#!/usr/bin/env python3
"""
chat_code_vllm.py —— 修 bug 的原样对话：提示词全由你写，脚本一个字不加。
你粘贴的内容里，assert 开头的行当可见测试，剩下的 Python 代码当待修文件；模型写 <test>/<run>/<write>/<ask> 就真的在沙盒里执行，观测再贴回去。
    CUDA_VISIBLE_DEVICES=0 python3 chat_code_vllm.py --hf-dir hf_codeRL_dyn [--temp 1 -s 1 --max-new 1024 --ask]
粘完在单独一行输入 /go 发送（或跟在最后一行末尾；Ctrl-D 也行）。/r 用上一段提示词再采一次（/r 4 = 采 4 条）；/s 看上局最终文件；q 退出。
打原样 + 标注版（⟦⟧ 是环境注入的观测，训练时不算模型输出），尾巴给调用次数 / 可见通过 / 隐藏通过 / 改了几行 / 真正的 bug 在哪。
粘的题若在 data/code_bugs.json 里（按 assert + 代码认），自动带上那道题的隐藏测试和注入记录，能看出「可见全绿但其实没修对」。--ask 要 export DEEPSEEK_API_KEY。
"""
import argparse, sys, os, re, json, shutil, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="hf_coder_1.5b")
ap.add_argument("--hf-dir", required=True)
ap.add_argument("--data", default="data/code_bugs.json", help="只用来认出隐藏测试和注入记录；文件不在也能跑")
ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("-s", "--samples", type=int, default=1)
ap.add_argument("--max-new", type=int, default=1024)
ap.add_argument("--max-calls", type=int, default=8)
ap.add_argument("--gpu-mem", type=float, default=0.6)
ap.add_argument("--ask", action="store_true", help="开 <ask>（要 DEEPSEEK_API_KEY）")
ap.add_argument("--ask-model", default="deepseek-reasoner")
ap.add_argument("--no-annot", action="store_true")
ap.add_argument("--no-echo", action="store_true", help="发送前不回显整段提示词")
args = ap.parse_args()

ESC = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
CODE_START = re.compile(r"^\s*(def |class |import |from |@|[A-Za-z_][A-Za-z0-9_]*\s*=)")
WS = re.compile(r"\s+")


def norm(s):
    return WS.sub("", s or "")


# ── 题库：按可见测试认题，再按代码认出具体是哪个变异（同一道 MBPP 题有好几个 bug 版本，隐藏测试一样、注入的行不一样）──
BANK = {}                                                       # norm(assert) → [题]
try:
    for b in json.load(open(args.data)):
        for t in b["visible_tests"]:
            BANK.setdefault(norm(t), []).append(b)
except Exception:
    pass


def lookup(tests, code):
    cands = [b for t in tests for b in BANK.get(norm(t), [])]
    if not cands:
        return None, None
    exact = next((b for b in cands if norm(b["bug_code"]) == norm(code)), None)
    return exact, (exact or cands[0])["hidden_test"]


TAIL = re.compile(r"^\s*(Tests?:|Visible tests:|It must pass:|Output of running the tests:|Tools:|TOOLS:|You can call tools:|- (tools|test output|failing tests|output):"
                  r"|Assistant:|Think step by step|<think>|它需要通过|运行测试的输出|工具[:：]|测试[:：]|passed \d+/\d+|Traceback \(most recent|Timeout: code ran)")


def compiles(code):
    try:
        compile(code, "solution.py", "exec"); return True
    except SyntaxError:
        return False


def split_paste(text):
    """粘贴的一段 → (代码, [assert...], 编译过没)。assert 行挑出来当测试；从第一行像 Python 的地方起，到第一行「模板尾巴」（Tests:/Tools:/<think>/工具：…）为止当代码。
    代码自己编不过就原样交给环境（<test> 会报 SyntaxError，那正是要让模型看见的），不再从尾巴往回削 —— 否则用户故意粘一段坏代码会被削成只剩前两行"""
    lines = text.split("\n")
    tests, seen = [], set()
    for l in lines:                                             # 带报错开局的题面会把挂掉的那条 assert 再回显一遍 → 按去空白后的文本去重
        if l.strip().startswith("assert") and norm(l) not in seen:
            seen.add(norm(l)); tests.append(l.strip())
    body = [l for l in lines if not l.strip().startswith("assert")]
    i = next((k for k, l in enumerate(body) if CODE_START.match(l)), None)
    if i is None:
        return "", tests, False
    j = next((k for k in range(i + 1, len(body)) if TAIL.match(body[k])), len(body))
    code = "\n".join(body[i:j]).strip("\n")
    if compiles(code):
        return code, tests, True
    trimmed = code                                              # 尾巴没认出来的散文（「请把 bug 改好。」）：只削那些明显不是代码的行
    while trimmed.strip() and not compiles(trimmed):
        last = trimmed.split("\n")[-1]
        if re.search(r"[(){}\[\]=:]", last) and not re.search(r"[\u4e00-\u9fff]", last):
            break                                               # 像代码的行不削 → 整段原样交出去
        trimmed = "\n".join(trimmed.split("\n")[:-1]).rstrip()
    if trimmed.strip() and compiles(trimmed):
        return trimmed, tests, True
    return code, tests, False


def read_prompt(LAST):
    """读到 /go 为止。返回 (提示词, 采样数覆盖) ；命令 /s /r /q 在这里处理。兼容两种粘贴：一行一行来，或整段一次来（bracketed paste）"""
    print("提示词>（粘完单独一行 /go 发送；/r 重发上一段；/s 看上局文件；q 退出）", flush=True)
    lines = []
    while True:
        try:
            raw = input()
        except EOFError:                                        # Ctrl-D：有内容就发，没内容就退出
            if not lines: sys.exit(0)
            return "\n".join(lines), None
        for l in ESC.sub("", raw).split("\n"):                  # 整段粘进来也按行处理
            t = l.strip()
            if not lines and t == "q":
                sys.exit(0)
            if not lines and t == "/s":
                print("\n".join("    " + x for x in LAST.get("code", "（还没跑过）").splitlines()) + "\n"); return None, None
            if not lines and (t == "/r" or t.startswith("/r ")):
                if "prompt" not in LAST:
                    print("  还没有上一段\n"); return None, None
                n = t[2:].strip(); return LAST["prompt"], (int(n) if n.isdigit() else None)
            if t == "/go" or t.endswith("/go"):                 # ★ 显式发送：提示词里本来就有空行，不能拿空行当发送
                rest = l[: l.rstrip().rfind("/go")].rstrip()
                if rest: lines.append(rest)
                return "\n".join(lines), None
            lines.append(l)


try:
    import readline                                              # 让 input() 支持方向键/行编辑，否则箭头会变成 ^[[A
    readline.parse_and_bind("set disable-completion on")         # 别把粘进来的 Tab 当补全
    readline.parse_and_bind("set enable-bracketed-paste off")    # 整段粘贴按行交给 input()
except Exception:
    pass

from transformers import AutoTokenizer
import calc_tool_vllm as T
import harness as H
import code_env as C

tok = AutoTokenizer.from_pretrained(args.model)
CTX = args.max_new + 1600
be = T.make_backend("vllm", tok=tok, hf_dir=args.hf_dir, gpu_mem=args.gpu_mem, max_model_len=CTX)
ask_fn = (lambda q, p, code: C.ask_expert(q, p, code, model=args.ask_model)) if args.ask else None
env = C.CodeEnv(max_calls=args.max_calls, ask_fn=ask_fn)
LAST = {}
print(f"\n★ 引擎就绪 ← {args.hf_dir}   温度 {args.temp}   每次 {args.samples} 条   调用上限 {args.max_calls}   求助 {'开' if args.ask else '关'}"
      f"   题库 {len(BANK)} 条测试可认\n")


def annotate(txt, ids, spans):
    out, pos = "", 0
    for s0, l0 in spans:
        a = len(tok.decode(ids[:s0], skip_special_tokens=True)); b = len(tok.decode(ids[:s0 + l0], skip_special_tokens=True))
        out += txt[pos:a] + "⟦" + txt[a:b] + "⟧"; pos = b
    return out + txt[pos:]


while True:
    prompt, n_over = read_prompt(LAST)
    if prompt is None:
        continue
    prompt = prompt.strip("\n")
    if not prompt.strip():
        continue
    n_samples = n_over or args.samples
    code, tests, compiled = split_paste(prompt)
    if not code:
        print("  ⚠ 没认出代码（要有 def / import / 赋值 开头的行），这一局不发\n"); continue
    exact, hidden = lookup(tests, code)
    ptoks = len(tok.encode(prompt, add_special_tokens=False))
    if not args.no_echo:
        print(f"── 发给模型的提示词（一字不差，{ptoks} token）──"); print(prompt); print("─" * 30)
    print(f"（认出 {len(tests)} 条可见测试、{len(code.splitlines())} 行代码"
          + ("" if compiled else "；⚠ 代码编译不过，原样交给环境")
          + ("；⚠ 没有 assert，<test> 会跑 0 条" if not tests else "")
          + (f"；题库 task {exact['task_id']} {exact['kind']}，隐藏测试 {hidden}" if exact else f"；题库同源题，隐藏测试 {hidden}" if hidden else "；题库里没找到，只按可见测试报")
          + (f"；⚠ 题面 {ptoks} token，留给生成的只剩 {CTX - ptoks - 8}" if CTX - ptoks - 8 < args.max_new else "") + "）")
    LAST["prompt"] = prompt
    prob = dict(task_id=exact["task_id"] if exact else -1, split="adhoc", kind=exact["kind"] if exact else "?", bug_type="?", text="", setup="",
                bug_code=code, ref_code="", diff=exact["diff"] if exact else [], visible_tests=tests, hidden_test=hidden)
    states = [env.reset(prob) for _ in range(n_samples)]
    outs = H.generate_with_env(be, tok, [prompt] * n_samples, env, states, max_new=args.max_new, temp=args.temp,
                               step_workers=min(8, n_samples), max_total=CTX)
    for i, o in enumerate(outs):
        st = o["state"]; LAST["code"] = st["code"]
        print("\n" + "─" * 78 + (f"  第 {i+1} 条" if n_samples > 1 else "")); print(o["txt"])
        if not args.no_annot and o["spans"]:
            print("· · · 标注版（⟦⟧ 是环境注入的） · · ·"); print(annotate(o["txt"], o["ids"], o["spans"]))
        alltests = tests + ([hidden] if hidden else [])
        d = tempfile.mkdtemp()
        try:
            _, oks, _ = C.run_tests(st["code"], alltests, "", d, 5) if alltests else (0, [], "")
        finally:
            shutil.rmtree(d, ignore_errors=True); shutil.rmtree(st["dir"], ignore_errors=True)
        kv = sum(oks[:len(tests)]); okh = oks[len(tests):]
        ed = C.edit_stats(code, st["code"], prob["diff"])
        print("··· 最终文件 ···"); print("\n".join("    " + l for l in st["code"].splitlines()))
        print(f"  {len(o['ids'])} token   调用 {st['calls']}（test {st['n_test']} write {st['n_write']} run {st['n_run']} ask {st['n_ask']}）"
              f"   可见 {kv}/{len(tests)}   隐藏 {'过' if okh and all(okh) else '挂' if okh else '无'}"
              f"   改了 {ed['changed_lines']} 行" + ("（整段重写）" if ed["rewrite"] else "（改中了注入的那行）" if ed["hit_bug_line"] else "")
              + ("   ⚠ 语法错" if st["syntax_err"] else "") + ("   ⚠ 自己编了 <result>" if o["fake"] else "") + ("   ⚠ 撞顶" if o["cut"] else ""))
        if exact:
            print("  真正的 bug（注入记录，− 原行 / + 改坏后的行）：" + "   ".join(exact["diff"]))
    print()
