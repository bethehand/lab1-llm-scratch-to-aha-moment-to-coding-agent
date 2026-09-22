"""假 tokenizer + 假后端（脚本化的模型）+ 真沙盒：验 code_env 的四个工具、秤、假结果、调用上限、mask、行为读数、沙盒、模板。python3 tests/test_code_env.py"""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness as H
import code_env as C
import calc_tool_vllm as T


class Tok:
    eos_token_id, pad_token_id = 0, 1
    def encode(self, s, add_special_tokens=False): return [ord(c) for c in s]
    def decode(self, ids, skip_special_tokens=True): return "".join(chr(i) for i in ids if i > 1)


class FakeBackend:
    """脚本 = 模型自己要写的段的列表（注入的观测不在里面）；每次 generate 按顺序吐下一段，碰停止串就停；最后一段末尾补 EOS"""
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
BUG = "def add(a, b):\n    return a - b"
FIX = "def add(a, b):\n    return a + b"
prob = dict(task_id=1, split="train", kind="arith_swap", bug_type="wrong", text="Write a function to add two numbers.", bug_code=BUG, ref_code=FIX,
            diff=["-    return a + b", "+    return a - b"], visible_tests=["assert add(1, 2) == 3", "assert add(0, 0) == 0"], hidden_test="assert add(5, 7) == 12", setup="")
env = C.CodeEnv(max_calls=4, call_cost=0.02, ask_fn=lambda q, p, code: FIX)

# ① 标准一局：test（挂）→ write 修好 → test（过）→ answer
P1 = "Q1:"; S1 = [P1 + "<test></test>", " the sign is wrong\n<write>" + FIX + "</write>", "\n<test></test>", "\n</think>\n<answer>done</answer>"]
# ② 语法错的 write + run + 调用上限（4 次）后结束，第 5 段永远吐不出来
P2 = "Q2:"; S2 = [P2 + "<write>def add(a, b)\n    return a + b</write>", "<run>print(add(2, 2))</run>", "<test></test>", "<run>print(1)</run>", " keep going <answer>done</answer>"]
# ③ 自己编 <result> = 假
P3 = "Q3:"; S3 = [P3 + "<test></test>", "<result>passed 2/2</result> <answer>done</answer>"]
# ④ ask → reply → write → test → answer
P4 = "Q4:"; S4 = [P4 + "<ask>how to fix?</ask>", "\n<write>" + FIX + "</write>", "<test></test>", "</think><answer>done</answer>"]
# ⑤ 硬编码可见测试：可见过、隐藏挂 → 0 分
HARD = "def add(a, b):\n    if (a, b) == (1, 2): return 3\n    return 0"
P5 = "Q5:"; S5 = [P5 + "<write>" + HARD + "</write>", "<test></test>", "<answer>done</answer>"]

be = FakeBackend({P1: S1, P2: S2, P3: S3, P4: S4, P5: S5})
states = [env.reset(prob) for _ in range(5)]
r = H.generate_with_env(be, tok, [P1, P2, P3, P4, P5], env, states, max_new=2000, temp=0.0)

a = r[0]; st = a["state"]; sc, d = env.score(st, a["txt"])
assert "passed 1/2" in a["txt"] and "AssertionError" in a["txt"] and a["txt"].endswith("<answer>done</answer>"), a["txt"]
assert st["n_test"] == 2 and st["n_write"] == 1 and st["calls"] == 3 and not a["fake"], st
assert d["correct"] == 1 and d["hidden_pass"] == 1 and abs(sc - 0.94) < 1e-9 and d["changed"] == 1, (sc, d)
assert len(a["spans"]) == 3, a["spans"]
m = T.response_mask(len(a["ids"]), a["spans"]); assert sum(m) == len(a["ids"]) - sum(l for _, l in a["spans"])
bh = C.behavior(a["txt"]); assert bh["seq"] == ["test", "write", "test"] and bh["test_before_write"] == 1 and bh["test_after_write"] == 1, bh

b = r[1]; st = b["state"]; sc, d = env.score(st, b["txt"])
assert "SyntaxError" in b["txt"] and st["syntax_err"] == 1, b["txt"]
assert st["calls"] == 4 and b["ids"][-1] == 0 and "keep going" not in b["txt"], (st["calls"], b["txt"])     # 第 4 次调用后结束
assert d["correct"] == 0, d                                                                                  # 文件是语法错的版本

c = r[2]; assert c["fake"] and c["state"]["calls"] == 1, (c["fake"], c["state"])

e = r[3]; st = e["state"]; sc, d = env.score(st, e["txt"])
assert st["n_ask"] == 1 and "<reply>" in e["txt"] and d["correct"] == 1 and abs(sc - 0.94) < 1e-9, (sc, d, e["txt"])
assert len(e["spans"]) == 3, e["spans"]

f = r[4]; st = f["state"]; sc, d = env.score(st, f["txt"])
assert "passed 2/2" in f["txt"] and d["vis_pass"] == 2 and d["hidden_pass"] == 0 and d["correct"] == 0 and sc == 0.0, (sc, d)   # ★ 硬编码被隐藏测试抓住

# <write> 的格式噪声：首行一个空格、markdown 围栏 → 都能编译；真语法错照报
assert C.normalize_code(" def f():\n    return 1") == "def f():\n   return 1" and compile(C.normalize_code(" def f():\n    return 1"), "x", "exec")
assert C.normalize_code("```python\ndef f():\n    return 1\n```") == "def f():\n    return 1"
assert C.normalize_code("\n    def f():\n        return 1\n") == "def f():\n    return 1"

# 沙盒：死循环超时、断网、大输出截断、看不到密钥
with tempfile.TemporaryDirectory() as dd:
    n0 = C.N_TIMEOUT
    rc, out, err = C.run_in_sandbox("while True: pass", dd, timeout=1); assert rc == -9 and "Timeout" in err and C.N_TIMEOUT == n0 + 1, (rc, err, C.N_TIMEOUT - n0)
    rc, out, err = C.run_in_sandbox("import socket; socket.socket()", dd); assert rc != 0 and "network disabled" in err, err
    rc, out, err = C.run_in_sandbox("print('x' * 5000)", dd); assert len(out) <= 600, len(out)
    rc, out, err = C.run_in_sandbox("import os; print(os.environ.get('DEEPSEEK_API_KEY'))", dd); assert out.strip() == "None", out
    rc, out, err = C.run_in_sandbox("import os; print(sorted(os.listdir('.')))", dd); assert out.strip() == "[]", out          # ★ 盘上没有脚本：读不到测试里的期望值
    if __import__("platform").system() == "Linux":                                                                             # ★ 内存上限（prelude 里设）：1 GB 申请不到；想把上限改回去也不行
        rc, out, err = C.run_in_sandbox("b = bytearray(1 << 30); print('got')", dd); assert "got" not in out and "MemoryError" in err, (out, err)
        rc, out, err = C.run_in_sandbox("import resource; resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))", dd); assert rc != 0 and "ValueError" in err, (rc, err)
# 作弊：函数在运行时读盘上的脚本找断言 → 读不到，隐藏测试照挂
CHEAT = "def add(a, b):\n    import os\n    return 3 if 'add' in ''.join(open(f).read() for f in os.listdir('.') if f.endswith('.py')) else 0"
st = env.reset(prob); st["code"] = CHEAT; sc, d = env.score(st, ""); assert d["correct"] == 0, d

# 带报错开局：initial_obs 有 traceback；不带：没有。10 个模板都以 <think> 收尾
obs = env.initial_obs(dict(prob)); assert obs.startswith("passed 1/2") and "AssertionError" in obs, obs
for pid in C.SETS["all"]:
    s1 = C.render(pid, dict(prob), with_tb=True, env=env); s0 = C.render(pid, dict(prob), with_tb=False, env=env)
    assert s1.endswith("<think>") and "<answer>" in s1 and "AssertionError" in s1 and "<write>" in s1, pid
    assert s0.endswith("<think>") and "AssertionError" not in s0, pid
print("code_env 假后端 + 真沙盒单测：全部通过（test/write/run/ask、秤、隐藏测试抓硬编码、假结果、语法错、调用上限、mask、行为读数、沙盒六项、10 模板 × 2 开局）")
