"""假后端（脚本化的模型）+ 真沙盒：验 ex_env 的切测试、验证器、观测、write/run/test/ask、秤（隐藏测试）、单条限时、调用上限、模板；
弱观测模式（阶段 ③）：模板无测试、<test> 被拒计费、自测四读数、交卷前验没验、假观测、clock 真题。python3 tests/test_ex_env.py"""
import sys, os, tempfile, shutil, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness as H
import ex_env as E
import code_env as C


class Tok:
    eos_token_id, pad_token_id = 0, 1
    def encode(self, s, add_special_tokens=False): return [ord(c) for c in s]
    def decode(self, ids, skip_special_tokens=True): return "".join(chr(i) for i in ids if i > 1)


class FakeBackend:
    def __init__(self, scripts): self.scripts, self.ptr = scripts, {k: 0 for k in scripts}
    def generate(self, rows, budgets, temp, stops):
        tk = Tok(); outs = []
        for row, b in zip(rows, budgets):
            cur = tk.decode(row); key = next(pr for pr in self.scripts if cur.startswith(pr))
            segs = self.scripts[key]; i = self.ptr[key]
            if i >= len(segs):
                outs.append([0]); continue
            seg = segs[i]; self.ptr[key] += 1
            cut = min([seg.find(st) + len(st) for st in stops if st in seg] + [len(seg)])
            ids = tk.encode(seg[:cut][:b])
            if i == len(segs) - 1 and cut >= len(seg): ids.append(0)
            outs.append(ids)
        return outs


tok = Tok()

# ── 一道合成题：两个测试类、一个辅助方法、assertRaises ──
TEST_SRC = '''import unittest

from adder import (
    add,
    safe_div,
)


class AddTest(unittest.TestCase):
    def check(self, a, b, want):
        self.assertEqual(add(a, b), want)

    def test_small(self):
        self.check(1, 2, 3)

    def test_zero(self):
        self.check(0, 0, 0)

    def test_negative(self):
        self.check(-1, -2, -3)

    def test_big(self):
        self.check(10**6, 1, 10**6 + 1)


class DivTest(unittest.TestCase):
    def test_div(self):
        self.assertEqual(safe_div(6, 3), 2)

    def test_div_zero_raises(self):
        with self.assertRaises(ValueError) as err:
            safe_div(1, 0)
        self.assertEqual(err.exception.args[0], "division by zero")
'''
REC = dict(slug="adder", module="adder", split="train", text="Write add(a, b) and safe_div(a, b); safe_div raises ValueError('division by zero') on zero.",
           stub="def add(a, b):\n    pass\n\n\ndef safe_div(a, b):\n    pass", ref_code="def add(a, b):\n    return a + b\n\n\ndef safe_div(a, b):\n    if b == 0:\n        raise ValueError('division by zero')\n    return a // b",
           test_src=TEST_SRC, test_names=[], n_tests=6, ref_lines=7, n_classes=0, n_funcs=2, instr_chars=90)

# ① 切测试：确定、隐藏 ≈ 三成、辅助方法和 assertRaises 都留在可见文件里、隐藏的方法从可见文件里消失
p = E.make_prob(REC); p2 = E.make_prob(REC)
assert p["hidden_ids"] == p2["hidden_ids"] and p["n_hid"] == 2 and p["n_vis"] == 4, (p["hidden_ids"], p["n_vis"])
assert "def check(self" in p["visible_src"] and "import unittest" in p["visible_src"], "辅助方法 / import 必须留着"
for hid in p["hidden_ids"]:
    assert f"def {hid.split('.')[1]}(" not in p["visible_src"], hid
for vid in p["visible_ids"]:
    assert f"def {vid.split('.')[1]}(" in p["visible_src"], vid
assert set(p["visible_ids"]) | set(p["hidden_ids"]) == {"AddTest.test_small", "AddTest.test_zero", "AddTest.test_negative", "AddTest.test_big", "DivTest.test_div", "DivTest.test_div_zero_raises"}

env = E.ExEnv(max_calls=6, call_cost=0.01, per_test=1.0, ask_fn=lambda q, pr, code: REC["ref_code"])
FIX = REC["ref_code"]
table = {"test_small": (1, 2), "test_zero": (0, 0), "test_negative": (-1, -2), "test_big": (10**6, 1)}
_vis_add = next(v.split(".")[1] for v in p["visible_ids"] if v.startswith("AddTest."))                       # 挑一条可见的 add 测试，让第一版恰好在它上挂
HALF = f"def add(a, b):\n    return a + b + (1 if (a, b) == {table[_vis_add]!r} else 0)\n\n\ndef safe_div(a, b):\n    if b == 0:\n        raise ValueError('division by zero')\n    return a // b"
# 只让可见过、隐藏挂的实现：按隐藏的是哪几条来造
hid_names = {h.split(".")[1] for h in p["hidden_ids"]}

# ② 标准一局：write → test（部分挂）→ write 修 → test 全过 → answer
P1 = "Q1:"; S1 = [P1 + "<write>" + HALF + "</write>", "<test></test>", " the zero case\n<write>" + FIX + "</write>", "<test></test>", "\n</think>\n<answer>done</answer>"]
# ③ 空 write、语法错 write、run 片段、盘上无文件、ask、调用上限（6 次）
P2 = "Q2:"; S2 = [P2 + "<write></write>", "<write>def add(a, b)\n    return a + b</write>", "<run>print(1)</run>",
                  "<write>" + FIX + "</write>", "<run>import os; print(sorted(os.listdir('.')), add(2, 3), safe_div(9, 3))</run>", "<ask>how?</ask>", " keep going <answer>done</answer>"]
# ④ 自己编 <result>
P3 = "Q3:"; S3 = [P3 + "<write>" + FIX + "</write>", "<result>passed 4/4</result><answer>done</answer>"]
# ⑤ 没写就交卷
P4 = "Q4:"; S4 = [P4 + "<test></test>", "<answer>done</answer>"]
be = FakeBackend({P1: S1, P2: S2, P3: S3, P4: S4})
r = H.generate_with_env(be, tok, [P1, P2, P3, P4], env, [env.reset(p) for _ in range(4)], max_new=6000, temp=0.0)

a = r[0]; st = a["state"]; t = a["txt"]
assert f"passed {p['n_vis']}/{p['n_vis']}" in t, t
sc, d = env.score(st, t)
assert d["correct"] == 1 and d["hidden_pass"] == p["n_hid"] and abs(sc - 0.96) < 1e-9 and d["n_rounds"] == 1, (sc, d)
assert d["first_pass_frac"] is not None and d["first_pass_frac"] < 1 and d["first_all"] == 0, d          # 首版没全过
assert len(a["spans"]) == 4 and not a["fake"], a["spans"]
bh = E.behavior(t); assert bh["seq"] == ["write", "test", "write", "test"] and bh["first"] == "write", bh

b = r[1]; st = b["state"]; t = b["txt"]
assert "not written: empty code" in t and "but SyntaxError" in t and st["syntax_err"] == 1, t
assert "<run>print(1)</run><result>Traceback" in t and "SyntaxError" in t.split("<run>print(1)</run>")[1].split("</result>")[0], t   # 语法错的模块上 <run> 报 import 失败（措辞随 Python 版本变，只认 SyntaxError）
assert "<result>[] 5 3</result>" in t and "<reply>" in t and st["n_ask"] == 1, t                            # ★ 盘上没有文件；模块装好后片段能用它
assert st["calls"] == 6 and "keep going" not in t, (st["calls"], t)                                       # 调用上限后结束

c = r[2]; assert c["fake"] and c["state"]["calls"] == 1, (c["fake"], c["state"]["calls"])

e = r[3]; sc, d = env.score(e["state"], e["txt"]); assert d["correct"] == 0 and d["changed"] == 0 and d["first_pass_frac"] is None, d

# ⑥ 秤：可见全过但隐藏挂 → 0 分（用一份「只对可见测试作弊」的实现）
vis_calls = {v.split(".")[1] for v in p["visible_ids"]}
cheat = "def add(a, b):\n    return a + b if (a, b) in {(1, 2), (0, 0), (-1, -2), (10**6, 1)} - set() else 0\n\n\ndef safe_div(a, b):\n    if b == 0:\n        raise ValueError('division by zero')\n    return a // b"
# 让隐藏的 add 测试挂：把隐藏那条对应的输入从「作弊表」里去掉
keep = [repr(table[n]) for n in table if n in vis_calls]
cheat = f"def add(a, b):\n    return a + b if (a, b) in {{{', '.join(keep)}}} else -999\n\n\ndef safe_div(a, b):\n    if b == 0:\n        raise ValueError('division by zero')\n    return a // b"   # 未知输入回 -999（回 0 会和 test_zero 撞上）
st = env.reset(p); st["code"] = cheat; sc, d = env.score(st, "")
if any(n in hid_names for n in table):                                                                     # 隐藏里有 add 的测试才能验这条
    assert d["vis_all"] == 1 and d["hid_all"] == 0 and d["correct"] == 0 and sc == 0.0, d
    print("  ⑥ 可见全过隐藏挂 → 0 分 ✓")

# ⑦ 单条限时：死循环只拖累自己那条
loop = "def add(a, b):\n    while True: pass\n\n\ndef safe_div(a, b):\n    return a // b"
d0 = tempfile.mkdtemp(); res = E.run_unittests("adder", loop, TEST_SRC, p["visible_ids"], d0, per_test=1.0); shutil.rmtree(d0, ignore_errors=True)
sts = {x["id"]: x["status"] for x in res["rows"]}
assert not res["timeout"] and any(v == "timeout" for v in sts.values()) and ("DivTest.test_div" not in sts or sts["DivTest.test_div"] == "ok"), sts

# ⑦b 连续超时早停（默认关；开了之后死循环文件从「每条等满」变成「两条之后直接标超时」，结果一样都是挂，只是快）
import time as _time
loop_all = "def add(a, b):\n    while True: pass\n\n\ndef safe_div(a, b):\n    while True: pass"
d0 = tempfile.mkdtemp(); t0 = _time.time(); res_off = E.run_unittests("adder", loop_all, TEST_SRC, None, d0, per_test=1.0, max_timeouts=0); t_off = _time.time() - t0
t0 = _time.time(); res_on = E.run_unittests("adder", loop_all, TEST_SRC, None, d0, per_test=1.0, max_timeouts=2); t_on = _time.time() - t0; shutil.rmtree(d0, ignore_errors=True)
assert all(x["status"] == "timeout" for x in res_on["rows"]) and len(res_on["rows"]) == 6 and not res_on["timeout"], res_on
assert sum(1 for x in res_on["rows"] if "skipped" in x["msg"]) == 4, res_on["rows"]
assert t_on < t_off, (t_on, t_off)
print(f"  ⑦b 早停：6 条全死循环 关 {t_off:.1f} s → 开 {t_on:.1f} s，结果都是 6 条 timeout ✓")
# ⑧ 缓存：同样的代码不再起子进程
NEW = FIX + "\n# fresh\n"; n0, h0 = env.n_run, env.n_hit; st = env.reset(p)
env._tests(p, NEW, False, st["dir"]); env._tests(p, NEW, False, st["dir"]); assert env.n_run == n0 + 1 and env.n_hit == h0 + 1, (env.n_run - n0, env.n_hit - h0)

# ⑨ 模板：10 套 × 3 措辞，都以 <think> 收尾、mc 替换、题面含空壳和可见测试、不含隐藏测试的方法名
for pid in E.SETS["all"]:
    for ts in range(3):
        s = E.render(pid, p, tool_style=ts, env=env)
        assert s.endswith("<think>") and "{mc}" not in s and " 6 " in s + " " or "6 calls" in s or "6 次" in s or "6 tool" in s, (pid, s[-300:])
        assert "def add(a, b):" in s and "def check(self" in s, pid
        for hid in p["hidden_ids"]:
            assert f"def {hid.split('.')[1]}(" not in s, (pid, hid)

# ⑩ 真题：clock 的参考实现在秤上得满分减价格；空壳 0 分
P = {q["slug"]: q for q in E.load_problems("data/exercism.json", "train")} | {q["slug"]: q for q in E.load_problems("data/exercism.json", "heldout")}
ck = P["clock"]; st = env.reset(ck); st["code"] = ck["ref_code"]; st["calls"] = 3; sc, d = env.score(st, "<answer>done</answer>")
assert d["correct"] == 1 and abs(sc - 0.97) < 1e-9 and d["n_tests"] == 55, (sc, d["n_tests"])
st = env.reset(ck); sc, d = env.score(st, ""); assert d["correct"] == 0 and d["vis_pass"] < ck["n_vis"], d["vis_pass"]
print("ex_env 假后端 + 真沙盒单测：全部通过（切测试 / 标准一局 / 空写·语法错·run·盘上无文件·ask·调用上限 / 假观测 / 没写交卷 / 隐藏测试抓作弊 / 单条限时 / 缓存 / 模板 10×3 / clock 真题）")

# ══════════════════════════ 弱观测模式（阶段 ③）══════════════════════════
wenv = E.ExEnv(max_calls=6, call_cost=0.01, per_test=1.0, obs="weak", ask_fn=lambda q, pr, code: REC["ref_code"])
# ⑪ 模板：10 套 × 3 措辞，无测试文本、有空壳、提到 <run>、不提 <test>、mc 替换；强模式模板不受影响
for pid in E.SETS["all"]:
    for ts in range(3):
        s = E.render(pid, p, tool_style=ts, env=wenv)
        assert s.endswith("<think>") and "{mc}" not in s and "def add(a, b):" in s, (pid, ts, s[-200:])
        assert "def test_" not in s and "class AddTest" not in s and "unittest" not in s, (pid, ts)
        assert "<run>" in s and "<test>" not in s, (pid, ts)
assert "def test_" in E.render(0, p, tool_style=0, env=env), "强模式模板不能变"
# ⑫ 四局脚本：W1 真阳 + 真阴 → 对；W2 假阴（断言太弱）→ 错；W3 假阳（断言写错）→ 对；W4 <test> 被拒计费 + 不自测直接交卷
va, vb = table[_vis_add]                                                                                   # HALF 在这组输入上算错
W1 = "W1:"; SW1 = [W1 + "<write>" + HALF + "</write>", f"<run>assert add({va}, {vb}) == {va + vb}, add({va}, {vb})</run>",
                   "<write>" + FIX + "</write>", f"<run>assert add({va}, {vb}) == {va + vb}; print('ok')</run>", "<answer>done</answer>"]
W2 = "W2:"; SW2 = [W2 + "<write>" + HALF + "</write>", "<run>assert add(5, 5) == 10; print('fine')</run>", "<answer>done</answer>"]
W3 = "W3:"; SW3 = [W3 + "<write>" + FIX + "</write>", "<run>assert add(1, 2) == 4</run>", "<answer>done</answer>"]
W4 = "W4:"; SW4 = [W4 + "<test></test>", "<write>" + FIX + "</write>", "<answer>done</answer>"]
wr = H.generate_with_env(FakeBackend({W1: SW1, W2: SW2, W3: SW3, W4: SW4}), tok, [W1, W2, W3, W4], wenv, [wenv.reset(p) for _ in range(4)], max_new=6000, temp=0.0)
a = wr[0]; sc, d = wenv.score(a["state"], a["txt"])
assert "AssertionError" in a["txt"] and "<result>ok</result>" in a["txt"], a["txt"]
assert f"assert add({va}, {vb}) == {va + vb}" in a["txt"].split("</run><result>")[1], a["txt"]     # ★ traceback 带出错那行源码
assert d["correct"] == 1 and abs(sc - 0.96) < 1e-9 and d["n_selftest"] == 2 and (d["n_tp"], d["n_tn"], d["n_fn"], d["n_fp"]) == (1, 1, 0, 0), d
assert d["verified_final"] == 1 and d["selfgreen_hidfail"] == 0 and d["first_ver_all"] == 0 and d["first_ver_frac"] < 1 and d["obs"] == "weak", d
assert [v for v, _ in d["ver_frac"]] == [1, 2] and d["ver_frac"][1][1] == 1.0, d["ver_frac"]
b = wr[1]; sc, d = wenv.score(b["state"], b["txt"])
assert "<result>fine</result>" in b["txt"] and d["correct"] == 0 and d["n_fn"] == 1 and d["selfgreen_hidfail"] == 1 and d["verified_final"] == 1, d
c = wr[2]; sc, d = wenv.score(c["state"], c["txt"])
assert d["correct"] == 1 and d["n_fp"] == 1 and d["n_tp"] == 0 and d["first_ver_all"] == 1 and d["last_selftest_err"] == 1, d
e = wr[3]; sc, d = wenv.score(e["state"], e["txt"])
assert "no test tool" in e["txt"] and d["n_test"] == 1 and d["calls"] == 2 and d["correct"] == 1 and d["n_selftest"] == 0 and d["verified_final"] == 0 and abs(sc - 0.98) < 1e-9, (sc, d)
# ⑬ 弱模式 <test> 不起沙盒、计一次调用、不结束
st = wenv.reset(p); k0 = wenv.n_run; obs, done = wenv.step(st, "<test></test>")
assert "no test tool" in obs and wenv.n_run == k0 and st["calls"] == 1 and not done, (obs, wenv.n_run - k0, st["calls"])
# ⑭ 假观测在弱模式照样被抓
W5 = "W5:"; SW5 = [W5 + "<write>" + FIX + "</write>", "<result>ok</result><answer>done</answer>"]
wr5 = H.generate_with_env(FakeBackend({W5: SW5}), tok, [W5], wenv, [wenv.reset(p)], max_new=6000, temp=0.0)
assert wr5[0]["fake"] and wr5[0]["state"]["calls"] == 1, wr5[0]["fake"]
# ⑮ 真题 clock 弱模式：参考实现满分减价格，首版（全秤）全过
st = wenv.reset(ck); st["code"] = ck["ref_code"]; st["versions"].append(ck["ref_code"]); st["n_write"] = 1; st["calls"] = 3
sc, d = wenv.score(st, "<answer>done</answer>"); assert d["correct"] == 1 and abs(sc - 0.97) < 1e-9 and d["first_ver_all"] == 1 and d["n_tests"] == 55, (sc, d["first_ver_all"])
# ⑯ 强模式也有首版（全秤）读数，且旧读数一字不变
st = env.reset(p); r1 = H.generate_with_env(FakeBackend({"Z1:": ["Z1:<write>" + HALF + "</write>", "<test></test>", "<write>" + FIX + "</write>", "<answer>done</answer>"]}), tok, ["Z1:"], env, [st], max_new=6000, temp=0.0)
sc, d = env.score(r1[0]["state"], r1[0]["txt"]); assert d["first_ver_all"] == 0 and d["first_all"] == 0 and d["correct"] == 1 and d["obs"] == "strong" and d["n_selftest"] == 0, d
print("弱观测模式单测：全部通过（模板 10×3 无测试 / <test> 被拒计费不起沙盒 / 真阳·真阴·假阴·假阳 / 交卷前验没验 / 假观测 / clock 真题 / 强模式旧读数不变）")
