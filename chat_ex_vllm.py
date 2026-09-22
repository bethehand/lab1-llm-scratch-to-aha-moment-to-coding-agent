#!/usr/bin/env python3
"""
chat_ex_vllm.py —— 和 exercism 两个模型聊：②-B 强观测（hf_exRL_bin，题面带七成测试，有 <test>）/ ③ 弱观测（hf_exwRL_bin，不给测试，只能 <run> 自验）。
模型写 <write>/<test>/<run>/<ask> 就真的在沙盒里执行，观测贴回去；交卷后用秤（全部测试）打分，打出完整轨迹、标注版、最终文件、读数、挂的是哪几条测试。
    CUDA_VISIBLE_DEVICES=0 python3 chat_ex_vllm.py --hf-dir hf_exwRL_bin --obs weak            # ③ 的模型，弱模式
    CUDA_VISIBLE_DEVICES=0 python3 chat_ex_vllm.py --hf-dir hf_exRL_bin  --obs strong          # ②-B 的模型，强模式
    python3 chat_ex_vllm.py --dry --obs weak --p clock                                          # 不载模型，只打题面
两种出题法：
  /p <slug>      库题：exercism 的 130 道（clock、leap、bob…）和 MBPP 的 961 道（mbpp-12）；题面按训练模板渲染（--pid 选模板，--tool-style 选工具措辞）
  直接粘贴       自定义题：说明 + 空壳（def/class … pass）+ 可选的 assert 行；assert 在强模式里是给模型看的可见测试，在弱模式里只进秤。没有 assert 就没有秤，只看它怎么走
命令：/go 发送（粘完单独一行）；/r [n] 用上一段再采 n 条；/s 看上局最终文件；/ref 看库题参考实现；/tests 看库题全部测试；/obs weak|strong 切模式；/p random 随机一道；q 退出
"""
import argparse, sys, os, re, json, shutil, tempfile, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="hf_coder_1.5b")
ap.add_argument("--hf-dir", default=None)
ap.add_argument("--obs", choices=["strong", "weak"], default="weak")
ap.add_argument("--data", nargs="+", default=["data/exercism.json", "data/mbpp_weak.json"], help="库题来源（不存在的文件跳过）")
ap.add_argument("--pid", type=int, default=0, help="题面模板 0~7 训练、8~9 留出")
ap.add_argument("--tool-style", type=int, default=0)
ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("-s", "--samples", type=int, default=1)
ap.add_argument("--max-new", type=int, default=4096)
ap.add_argument("--max-calls", type=int, default=12)
ap.add_argument("--call-cost", type=float, default=0.01)
ap.add_argument("--gpu-mem", type=float, default=0.6)
ap.add_argument("--ask", action="store_true", help="开 <ask>（要 DEEPSEEK_API_KEY）")
ap.add_argument("--ask-model", default="deepseek-reasoner")
ap.add_argument("--no-annot", action="store_true")
ap.add_argument("--no-echo", action="store_true", help="发送前不回显整段题面")
ap.add_argument("--dry", action="store_true", help="不载模型：只渲染题面（配 --p）")
ap.add_argument("--p", default=None, help="--dry 时打哪道题")
args = ap.parse_args()

import ex_env as E
import code_env as C

ESC = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
CODE_START = re.compile(r"^\s*(def |class |import |from |@)")

# ── 库题 ──
PROBS = {}
for f in args.data:
    if not os.path.exists(f):
        continue
    for sp in ("train", "heldout"):
        for p in E.load_problems(f, sp):
            p["_src"] = os.path.basename(f); PROBS[p["slug"]] = p


def make_test_src(module, asserts):
    lines = ["import unittest", "import math", "import re", "import collections", "import itertools", f"from {module} import *", "", "", "class AdhocTest(unittest.TestCase):"]
    for i, t in enumerate(asserts):
        lines += [f"    def test_{i:02d}(self):", f"        {t}", ""]
    return "\n".join(lines).rstrip() + "\n"


def parse_paste(text, obs):
    """粘贴 → 自定义题。assert 行 = 测试；从第一行 def/class/import 起到末尾 = 空壳；之前的散文 = 说明"""
    lines = text.split("\n")
    asserts = [l.strip() for l in lines if l.strip().startswith("assert")]
    body = [l for l in lines if not l.strip().startswith("assert")]
    i = next((k for k, l in enumerate(body) if CODE_START.match(l)), None)
    if i is None:
        return None, "没认出空壳（要有 def / class 开头的行）"
    desc = "\n".join(body[:i]).strip(); stub = "\n".join(body[i:]).strip("\n")
    try:
        compile(stub, "stub.py", "exec")
    except SyntaxError as e:
        return None, f"空壳编译不过：{e.msg}（第 {e.lineno} 行）"
    module = "solution"
    test_src = make_test_src(module, asserts)
    ids = [f"AdhocTest.test_{i:02d}" for i in range(len(asserts))]
    p = dict(slug="adhoc", module=module, split="adhoc", text=desc or "(no description)", stub=stub, ref_code="", test_src=test_src, test_names=ids,
             n_tests=len(ids), ref_lines=0, n_classes=int("class " in stub), n_funcs=stub.count("def "), instr_chars=len(desc), _src="paste")
    if obs == "strong":                                          # 强模式：贴的 assert 全给模型看
        p["visible_src"], p["visible_ids"], p["hidden_ids"] = test_src, ids, []
    else:                                                        # 弱模式：全藏，只进秤
        p["visible_src"], p["visible_ids"], p["hidden_ids"] = "", [], ids
    p["n_vis"], p["n_hid"] = len(p["visible_ids"]), len(p["hidden_ids"])
    return p, None


def render(p, env):
    return E.render(args.pid, p, tool_style=args.tool_style, env=env)


def failing_names(p, code):
    d = tempfile.mkdtemp()
    try:
        res = E.run_unittests(p["module"], code, p["test_src"], None, d, per_test=2.0, max_timeouts=2)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    if res.get("load_error"):
        return ["(import 失败) " + res["load_error"].strip().splitlines()[-1][:120]]
    if res.get("timeout"):
        return ["(整体超时)"]
    return [f"{r['id'].split('.', 1)[1]}[{r['status']}]" for r in res["rows"] if r["status"] != "ok"]


if args.dry:
    env = E.ExEnv(max_calls=args.max_calls, call_cost=args.call_cost, obs=args.obs)
    p = PROBS.get(args.p or "clock")
    if p is None:
        print(f"没有这道题；库里 {len(PROBS)} 道，如 clock / leap / mbpp-12"); sys.exit(1)
    print(render(p, env)); print(f"\n--- {p['slug']}  {p['_src']}  split {p['split']}  测试 {p['n_tests']}（可见 {p['n_vis']} 隐藏 {p['n_hid']}）  参考 {p['ref_lines']} 行  观测 {args.obs}")
    sys.exit(0)

assert args.hf_dir, "要 --hf-dir（或 --dry）"


def read_input(LAST):
    print(f"题面>（/p <slug> 或粘自定义题，单独一行 /go 发送；/r 重发；/s 文件；/ref /tests 看库题；/obs weak|strong；q 退出）", flush=True)
    lines = []
    while True:
        try:
            raw = input()
        except EOFError:
            if not lines: sys.exit(0)
            return ("paste", "\n".join(lines), None)
        for l in ESC.sub("", raw).split("\n"):
            t = l.strip()
            if not lines:
                if t == "q": sys.exit(0)
                if t == "/s":
                    print("\n".join("    " + x for x in LAST.get("code", "（还没跑过）").splitlines()) + "\n"); return ("noop", None, None)
                if t == "/ref":
                    p = LAST.get("prob"); print(("\n".join("    " + x for x in p["ref_code"].splitlines()) if p and p.get("ref_code") else "（没有参考实现）") + "\n"); return ("noop", None, None)
                if t == "/tests":
                    p = LAST.get("prob"); print((p["test_src"] if p else "（还没出题）") + "\n"); return ("noop", None, None)
                if t.startswith("/obs "):
                    return ("obs", t.split(None, 1)[1].strip(), None)
                if t == "/r" or t.startswith("/r "):
                    if "prob" not in LAST: print("  还没有上一段\n"); return ("noop", None, None)
                    n = t[2:].strip(); return ("redo", None, int(n) if n.isdigit() else None)
                if t.startswith("/p "):
                    return ("slug", t[3:].strip(), None)
            if t == "/go" or t.endswith("/go"):
                rest = l[: l.rstrip().rfind("/go")].rstrip()
                if rest: lines.append(rest)
                return ("paste", "\n".join(lines), None)
            lines.append(l)


try:
    import readline
    readline.parse_and_bind("set disable-completion on")
    readline.parse_and_bind("set enable-bracketed-paste off")
except Exception:
    pass

from transformers import AutoTokenizer
import calc_tool_vllm as T
import harness as H

tok = AutoTokenizer.from_pretrained(args.model)
CTX = args.max_new + 2600
be = T.make_backend("vllm", tok=tok, hf_dir=args.hf_dir, gpu_mem=args.gpu_mem, max_model_len=CTX)
OBS = args.obs


def make_env(obs):
    ask_fn = (lambda q, p, code: E.ask_expert(q, p, code, model=args.ask_model, weak=(obs == "weak"))) if args.ask else None
    return E.ExEnv(max_calls=args.max_calls, call_cost=args.call_cost, obs=obs, ask_fn=ask_fn)


env = make_env(OBS)
LAST = {}
print(f"\n★ 引擎就绪 ← {args.hf_dir}   观测 {OBS}   温度 {args.temp}   每次 {args.samples} 条   调用上限 {args.max_calls}   求助 {'开' if args.ask else '关'}"
      f"   库题 {len(PROBS)} 道   模板 {args.pid} 措辞 {args.tool_style}\n")


def annotate(txt, ids, spans):
    out, pos = "", 0
    for s0, l0 in spans:
        a = len(tok.decode(ids[:s0], skip_special_tokens=True)); b = len(tok.decode(ids[:s0 + l0], skip_special_tokens=True))
        out += txt[pos:a] + "⟦" + txt[a:b] + "⟧"; pos = b
    return out + txt[pos:]


def play(p, n_samples):
    prompt = render(p, env)
    ptoks = len(tok.encode(prompt, add_special_tokens=False))
    if not args.no_echo:
        print(f"── 发给模型的题面（一字不差，{ptoks} token）──"); print(prompt); print("─" * 30)
    print(f"（{p['slug']}  {p['_src']}  观测 {OBS}  秤 {p['n_tests']} 条测试" + (f"，模型看得到 {p['n_vis']} 条" if OBS == "strong" else "，模型一条都看不到")
          + ("；⚠ 没有 assert = 没有秤，只看它怎么走" if p["n_tests"] == 0 else "")
          + (f"；⚠ 题面 {ptoks} token，留给生成的只剩 {CTX - ptoks - 8}" if CTX - ptoks - 8 < args.max_new else "") + "）", flush=True)
    LAST["prob"] = p
    states = [env.reset(p) for _ in range(n_samples)]
    outs = H.generate_with_env(be, tok, [prompt] * n_samples, env, states, max_new=args.max_new, temp=args.temp, step_workers=min(8, n_samples), max_total=CTX)
    for i, o in enumerate(outs):
        st = o["state"]; LAST["code"] = st["code"]
        print("\n" + "─" * 78 + (f"  第 {i+1} 条" if n_samples > 1 else "")); print(o["txt"])
        if not args.no_annot and o["spans"]:
            print("· · · 标注版（⟦⟧ 是环境注入的，训练时不算模型输出） · · ·"); print(annotate(o["txt"], o["ids"], o["spans"]))
        print("··· 最终文件 ···"); print("\n".join("    " + l for l in st["code"].splitlines()))
        bad = failing_names(p, st["code"]) if p["n_tests"] else []
        r, d = env.score(st, o["txt"])                             # 会删临时目录
        line = (f"  {len(o['ids'])} token   调用 {st['calls']}（write {st['n_write']} test {st['n_test']} run {st['n_run']} ask {st['n_ask']}）"
                + (f"   秤 {d['passed']}/{d['n_tests']}{'  ★ 全过' if d['correct'] else ''}   得分 {r:.2f}" if p["n_tests"] else "   （无秤）")
                + (f"   首版{'全过' if d['first_ver_all'] else '没全过'}" if d.get("first_ver_frac") is not None else "   没写过")
                + ("   ⚠ 语法错" if st["syntax_err"] else "") + ("   ⚠ 自己编了 <result>" if o["fake"] else "") + ("   ⚠ 撞顶" if o["cut"] else ""))
        print(line)
        if OBS == "weak" or d.get("n_selftest"):
            print(f"  自测 {d['n_selftest']} 次（真阳 {d['n_tp']} 假阴 {d['n_fn']} 假阳 {d['n_fp']} 真阴 {d['n_tn']}）   交卷前验过最终版 {'是' if d['verified_final'] else '否'}"
                  + ("   ⚠ 自测全绿但秤挂" if d.get("selfgreen_hidfail") else ""))
        if bad:
            print(f"  秤上挂的：{', '.join(bad[:8])}" + (f" …共 {len(bad)} 条" if len(bad) > 8 else ""))
    print()


while True:
    kind, arg, n_over = read_input(LAST)
    if kind == "noop":
        continue
    if kind == "obs":
        if arg not in ("strong", "weak"):
            print("  /obs weak 或 /obs strong\n"); continue
        OBS = arg; env = make_env(OBS); print(f"  观测切到 {OBS}\n"); continue
    n_samples = n_over or args.samples
    if kind == "redo":
        play(LAST["prob"], n_samples); continue
    if kind == "slug":
        p = random.choice(list(PROBS.values())) if arg == "random" else PROBS.get(arg)
        if p is None:
            near = [s for s in PROBS if arg in s][:8]
            print(f"  没有 {arg}" + (f"，像的有：{' '.join(near)}" if near else "") + "\n"); continue
        play(p, n_samples); continue
    text = (arg or "").strip("\n")
    if not text.strip():
        continue
    p, err = parse_paste(text, OBS)
    if err:
        print(f"  ⚠ {err}，这一局不发\n"); continue
    play(p, n_samples)
