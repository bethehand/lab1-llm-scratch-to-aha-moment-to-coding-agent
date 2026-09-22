"""假后端（脚本化的模型）+ 真沙盒：验 pkg_env 的造题、按文件报的观测、带文件名的 write、<run> 装全部文件、秤（含严秤隐藏测试）、超时隔离、模板。python3 tests/test_pkg_env.py"""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness as H
import pkg_env as P
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
R1 = dict(task_id=1, split="train", kind="ret_wrong", bug_type="wrong", text="Write a python function to count true booleans in the given list.",
          bug_code="def count(lst):\n    return lst", ref_code="def count(lst):\n    return sum(lst)", diff=["-    return sum(lst)", "+    return lst"],
          visible_tests=["assert count([False,False]) == 0", "assert count([True,True,True]) == 3"], hidden_test="assert count([True,False]) == 1", setup="")
R2 = dict(task_id=2, split="train", kind="arith_swap", bug_type="crash", text="Write a python function to find the first digit of a given number.",
          bug_code="def first_Digit(n):\n    while n >= 10:\n        n = n * 10\n    return int(n)", ref_code="def first_Digit(n):\n    while n >= 10:\n        n = n / 10\n    return int(n)",
          diff=["-        n = n / 10", "+        n = n * 10"], visible_tests=["assert first_Digit(123) == 1", "assert first_Digit(456) == 4"], hidden_test="assert first_Digit(12) == 1", setup="")
R3 = dict(task_id=3, split="train", kind="ret_wrong", bug_type="crash", text="Write a python function to get the last element of each sublist.",
          bug_code="def Extract(lst):\n    return item", ref_code="def Extract(lst):\n    return [item[-1] for item in lst]", diff=["-    return [item[-1] for item in lst]", "+    return item"],
          visible_tests=["assert Extract([[1, 2, 3], [4, 5]]) == [3, 5]", "assert Extract([['x', 'y'], ['m']]) == ['y', 'm']"], hidden_test="assert Extract([[7]]) == [7]", setup="")
EXT = {"1": ["assert count([True,True,False,True]) == 3", "assert count([]) == 0"], "3": ["assert Extract([[1],[2],[3]]) == [1, 2, 3]"]}
FIX1 = "def count(lst):\n    return sum(lst)"
FIX2 = "def first_Digit(n):\n    while n >= 10:\n        n = n / 10\n    return int(n)"
FIX3 = "def Extract(lst):\n    return [item[-1] for item in lst]"

pkg = P.make_pkg([R1, R2, R3], EXT)
assert [f["name"] for f in pkg["files"]] == ["count.py", "first_digit.py", "extract.py"], [f["name"] for f in pkg["files"]]
assert len(pkg["files"][0]["hidden_tests"]) == 3 and len(pkg["files"][1]["hidden_tests"]) == 1, "隐藏测试 = 原 1 条 + 严秤的"
env = P.PkgEnv(max_calls=16, call_cost=0.01, timeout=1, ask_fn=lambda q, p, files: FIX3)

# ① 标准一局：test → 修 count → test → 修另两个 → test → answer
P1 = "Q1:"; S1 = [P1 + "<test></test>", " count returns the list.\n<write file=\"count.py\">" + FIX1 + "</write>", "<test></test>",
                  "Status: count.py done; remaining first_digit.py, extract.py\n<write file=\"first_digit.py\">" + FIX2 + "</write>",
                  "<write file=\"extract.py\">" + FIX3 + "</write>", "<test></test>", "\n</think>\n<answer>done</answer>"]
# ② K=3 不写文件名 / 写错文件名 → 报错观测，文件不动
P2 = "Q2:"; S2 = [P2 + "<write>" + FIX1 + "</write>", "<write file=\"foo.py\">" + FIX1 + "</write>", "<write file=\"count.py\"></write>", "<answer>done</answer>"]
# ③ 只修两个就交卷 → 0 分，但 r_frac > 0
P3 = "Q3:"; S3 = [P3 + "<write file=\"count.py\">" + FIX1 + "</write>", "<write file=\"extract.py\">" + FIX3 + "</write>", "<answer>done</answer>"]
# ④ 硬编码 count：可见过、隐藏挂
HARD = "def count(lst):\n    return 0 if lst == [False,False] else 3"
P4 = "Q4:"; S4 = [P4 + "<write file=\"count.py\">" + HARD + "</write>", "<write file=\"first_digit.py\">" + FIX2 + "</write>", "<write file=\"extract.py\">" + FIX3 + "</write>", "<answer>done</answer>"]
# ⑤ <run>：装了全部文件；traceback 带文件名；盘上没文件；语法错的 write 报回来
P5 = "Q5:"; S5 = [P5 + "<run>print(Extract([[1,2]]))</run>", "<run>import os; print(sorted(os.listdir('.')))</run>",
                  "<write file=\"count.py\">def count(lst)\n    return sum(lst)</write>", "<test></test>", "<ask>how?</ask>", "<answer>done</answer>"]

be = FakeBackend({P1: S1, P2: S2, P3: S3, P4: S4, P5: S5})
states = [env.reset(pkg) for _ in range(5)]
r = H.generate_with_env(be, tok, [P1, P2, P3, P4, P5], env, states, max_new=4000, temp=0.0)

a = r[0]; st = a["state"]; sc, d = env.score(st, a["txt"])
t = a["txt"]
assert "passed 0/6" in t and "count.py: 0/2" in t and "first_digit.py: 0/2" in t and "extract.py: 0/2" in t, t
assert "Timeout" in t and 'File "extract.py", line 2, in Extract' in t and "NameError" in t, t                # 超时隔离：别的文件照常报；行号要和题面里的代码对上
assert "written count.py" in t and "count.py: 2/2" in t and "passed 6/6" in t, t
assert st["calls"] == 6 and st["n_test"] == 3 and st["n_write"] == 3 and not a["fake"], st
assert d["correct"] == 1 and d["files_fixed"] == 3 and d["hidden_pass"] == 1 and abs(sc - 0.94) < 1e-9 and d["n_tests"] == 6 + 3 + 1 + 2, (sc, d)
assert all(p["hit_bug_line"] for p in d["per_file"]) and d["files_touched"] == 3, d["per_file"]
assert len(a["spans"]) == 6, a["spans"]
bh = P.behavior(t); assert bh["files_written"] == ["count.py", "first_digit.py", "extract.py"] and bh["n_state_lines"] == 1 and bh["test_before_write"] == 1, bh

b = r[1]; st = b["state"]; sc, d = env.score(st, b["txt"])
assert "not written: say which file" in b["txt"] and "not written: no file named foo.py" in b["txt"] and "not written: empty code for count.py" in b["txt"], b["txt"]
assert st["n_bad_write"] == 3 and st["calls"] == 3 and st["files"]["count.py"] == R1["bug_code"] and d["correct"] == 0, (st["n_bad_write"], d)

c = r[2]; sc, d = env.score(c["state"], c["txt"])
assert sc == 0.0 and d["files_fixed"] == 2 and abs(d["frac_files"] - 2 / 3) < 1e-9 and 0 < d["r_frac"] < 1 and abs(d["r_frac"] - (d["passed"] / d["n_tests"] - 0.02)) < 1e-9, d

e = r[3]; sc, d = env.score(e["state"], e["txt"])
pf = d["per_file"][0]; assert pf["vis_pass"] == 2 and pf["hidden_pass"] == 0 and d["correct"] == 0 and sc == 0.0, (sc, d)   # ★ 硬编码被隐藏测试抓住

f = r[4]; st = f["state"]; t = f["txt"]
assert 'File "extract.py", line 2, in Extract' in t and 'File "<snippet>", line 1' in t and "NameError" in t, t
assert "<result>[]</result>" in t, t                                                                                  # ★ 盘上没有文件
assert "written count.py, but SyntaxError" in t and st["syntax_err"] == 1, t
seg = t.split("count.py: 0/2")[1][:400]; assert "SyntaxError" in seg and "solution.py" not in seg and "<stdin>" not in seg, seg   # 语法错的文件在 <test> 里照报，壳的帧不漏
assert "<reply>" in t and st["n_ask"] == 1, t

# ⑤b 结果缓存：同一份代码 + 同一组测试不再起子进程；改了的文件才重跑
envc = P.PkgEnv(max_calls=16, timeout=1); stc = envc.reset(pkg)
envc._run_all(stc); n1 = envc.n_run; envc._run_all(stc); assert envc.n_run == n1 and envc.n_hit == 3, (envc.n_run, envc.n_hit)      # 第二次 3 个文件全命中
stc["files"]["count.py"] = FIX1; envc._run_all(stc); assert envc.n_run == n1 + 1, envc.n_run                                            # 只重跑改了的那个
st2 = envc.reset(pkg); envc._run_all(st2); assert envc.n_run == n1 + 1, "另一条轨迹、同样的坏文件 → 全命中（16 条采样反复撞同一个死循环不再各等一次）"
sc, d = envc.score(stc, ""); assert d["files_fixed"] == 1 and envc.n_run == n1 + 1 + 3, "秤的隐藏测试是另一组 key，第一次要跑"
sc, d = envc.score(envc.reset(pkg) | {"files": dict(stc["files"])}, ""); assert envc.n_run == n1 + 1 + 3, "同样的最终文件再判一次 → 全命中"
shutil_dir = stc["dir"]
# ⑥ 秤可切换成比例奖励
env_f = P.PkgEnv(reward="frac", call_cost=0.01, timeout=1); st = env_f.reset(pkg); st["files"]["count.py"] = FIX1; st["calls"] = 3
sc_f, d = env_f.score(st, ""); assert abs(sc_f - (d["passed"] / d["n_tests"] - 0.03)) < 1e-9 and d["r_bin"] == 0.0, (sc_f, d)

# ⑦ K=1：不写文件名也行
pk1 = P.make_pkg([R1], EXT); env1 = P.PkgEnv(timeout=1)
P6 = "Q6:"; S6 = [P6 + "<write>" + FIX1 + "</write>", "<test></test>", "<answer>done</answer>"]
r6 = H.generate_with_env(FakeBackend({P6: S6}), tok, [P6], env1, [env1.reset(pk1)], max_new=2000, temp=0.0)[0]
sc, d = env1.score(r6["state"], r6["txt"]); assert "written count.py" in r6["txt"] and "passed 2/2" in r6["txt"] and d["correct"] == 1, r6["txt"]

# ⑧ 造题：顶层名字撞车的不进同一个包；文件名撞车加后缀；seed 可复现
R1b = dict(R1, task_id=11, text="Another count task.", bug_code="def count(x):\n    return 0", ref_code="def count(x):\n    return len(x)")
assert P.make_pkg([R1, R1b], EXT) is None, "两个 count 不能同包"
Rc = dict(R1, task_id=12, text="Count upper.", bug_code="def Count(x):\n    return 0", ref_code="def Count(x):\n    return 1",
          visible_tests=["assert Count([1]) == 1"], hidden_test="assert Count([]) == 1")
pk = P.make_pkg([R1, Rc], EXT); assert [f["name"] for f in pk["files"]] == ["count.py", "count_2.py"], pk["files"]
pool = [R1, R2, R3, R1b, Rc, dict(R2, task_id=22, text="digit again", bug_code="def fd2(n):\n    return 0", ref_code="def fd2(n):\n    return 1", visible_tests=["assert fd2(1) == 1"], hidden_test="assert fd2(2) == 1")]
A = P.make_pkgs(pool, 3, 5, seed=3, ext=EXT); B = P.make_pkgs(pool, 3, 5, seed=3, ext=EXT); Cc = P.make_pkgs(pool, 3, 5, seed=4, ext=EXT)
assert [x["task_ids"] for x in A] == [x["task_ids"] for x in B] and [x["task_ids"] for x in A] != [x["task_ids"] for x in Cc]
for x in A:
    assert len({f["text"] for f in x["files"]}) == 3 and len({f["name"] for f in x["files"]}) == 3
    names = [P._top_names(f["bug_code"]) for f in x["files"]]; assert not (names[0] & names[1]) and not (names[1] & names[2]) and not (names[0] & names[2])

# ⑨ 模板：10 套 × 2 开局 × 3 措辞，K=1 和 K=3
for pk_ in (pkg, pk1):
    for pid in P.SETS["all"]:
        for tb in (False, True):
            for ts in range(3):
                s = P.render(pid, pk_, with_tb=tb, tool_style=ts, env=env)
                assert s.endswith("<think>") and "{mc}" not in s and "16" in s, (pid, s[-200:])
                assert all(f["name"] in s for f in pk_["files"]) and all(t in s for f in pk_["files"] for t in f["visible_tests"]), pid
                assert (f"passed 0/{2 * pk_['K']}" in s) == tb, (pid, tb)
print("pkg_env 假后端 + 真沙盒单测：全部通过（造题 / 按文件报 / 带文件名的 write / 超时隔离 / 结果缓存 / <run> 装全部文件 / 盘上无文件 / 语法错 / 严秤隐藏测试 / 比例奖励 / K=1 / 模板 10×2×3）")
