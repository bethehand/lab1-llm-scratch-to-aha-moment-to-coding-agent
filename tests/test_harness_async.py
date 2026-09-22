"""级别二异步 harness：假引擎（请求乱序完成）+ 真沙盒，输出必须和同步轮次一字不差。python3 tests/test_harness_async.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness as H
import code_env as C
import guess_env as G
import pkg_env as P


class Tok:
    eos_token_id, pad_token_id = 0, 1
    def encode(self, s, add_special_tokens=False): return [ord(c) for c in s]
    def decode(self, ids, skip_special_tokens=True): return "".join(chr(i) for i in ids if i > 1)


class FakeBackend:
    """脚本化的模型：每次 generate 按顺序吐下一段，碰停止串就停；最后一段末尾补 EOS"""
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


class FakeEngineBackend(FakeBackend):
    """有逐步接口的假引擎：每 step 只完成一个请求（最后加进来的那个，故意乱序），逼 harness 处理交错完成"""
    def __init__(self, scripts, per_step=1):
        super().__init__(scripts); self.reqs = {}; self.per_step = per_step; self.n_steps = 0
    def engine_add(self, rid, ids, budget, temp, stops): self.reqs[rid] = (list(ids), budget, stops)
    def engine_step(self):
        self.n_steps += 1; out = []
        for rid in list(self.reqs)[::-1][: self.per_step]:
            ids, b, stops = self.reqs.pop(rid); out.append((rid, self.generate([ids], [b], 0.0, stops)[0]))
        return out
    def engine_busy(self): return bool(self.reqs)


tok = Tok()


def same(a, b, keys=("ids", "txt", "spans", "fake", "cont", "cut")):
    for x, y in zip(a, b):
        for k in keys:
            assert x[k] == y[k], (k, x[k], y[k])


# ① 修 bug（code_env，真沙盒）：五条轨迹，同步 vs 异步（每步只完成一条、倒序）
BUG = "def add(a, b):\n    return a - b"; FIX = "def add(a, b):\n    return a + b"
prob = dict(task_id=1, split="train", kind="arith_swap", bug_type="wrong", text="add", bug_code=BUG, ref_code=FIX, diff=["-    return a + b", "+    return a - b"],
            visible_tests=["assert add(1, 2) == 3", "assert add(0, 0) == 0"], hidden_test="assert add(5, 7) == 12", setup="")
env = C.CodeEnv(max_calls=4, call_cost=0.02, ask_fn=lambda q, p, code: FIX)
S = {"Q1:": ["Q1:<test></test>", " sign\n<write>" + FIX + "</write>", "\n<test></test>", "\n</think>\n<answer>done</answer>"],
     "Q2:": ["Q2:<write>def add(a, b)\n    return a + b</write>", "<run>print(add(2, 2))</run>", "<test></test>", "<run>print(1)</run>", " keep <answer>done</answer>"],
     "Q3:": ["Q3:<test></test>", "<result>passed 2/2</result> <answer>done</answer>"],
     "Q4:": ["Q4:<ask>how?</ask>", "\n<write>" + FIX + "</write>", "<test></test>", "</think><answer>done</answer>"],
     "Q5:": ["Q5:<write>" + FIX + "</write>", "<test></test>", "<answer>done</answer>"]}
ps = list(S)
r_sync = H.generate_with_env(FakeBackend(S), tok, ps, env, [env.reset(prob) for _ in ps], max_new=2000, temp=0.0, sync=True)
be = FakeEngineBackend(S)
r_async = H.generate_with_env(be, tok, ps, env, [env.reset(prob) for _ in ps], max_new=2000, temp=0.0, step_workers=4)
same(r_sync, r_async)
r_win = H.generate_with_env(FakeEngineBackend(S), tok, ps, env, [env.reset(prob) for _ in ps], max_new=2000, temp=0.0, step_workers=4, inflight=2)   # 滑动窗口 2：一次最多两条在引擎里
same(r_sync, r_win); assert H.generate_with_env.last_timing["inflight"] == 2
assert H.generate_with_env.last_timing["mode"] == "async" and H.generate_with_env.last_timing["env_calls"] == sum(x["state"]["calls"] for x in r_async), H.generate_with_env.last_timing
for x, y in zip(r_sync, r_async):
    assert env.score(x["state"], x["txt"]) == env.score(y["state"], y["txt"])
print(f"① code_env 五条：同步 vs 异步一字不差，假引擎推了 {be.n_steps} 步")

# ② 猜数字（多轮、带假观测、撞轮数上限）
genv = G.GuessEnv(N=100, T=10)
Sg = {"G1:": ["G1:<guess>50</guess>", "<guess>75</guess>", "<guess>62</guess>", "</think><answer>62</answer>"],
      "G2:": ["G2:<guess>50</guess>", "<obs>higher</obs><guess>75</guess>", "<answer>75</answer>"],
      "G3:": ["G3:" + "<guess>1</guess>" * 12 + "<answer>1</answer>"]}
pg = list(Sg)
a = H.generate_with_env(FakeBackend(Sg), tok, pg, genv, [genv.reset(62) for _ in pg], max_new=400, temp=0.0, sync=True)
b = H.generate_with_env(FakeEngineBackend(Sg, per_step=2), tok, pg, genv, [genv.reset(62) for _ in pg], max_new=400, temp=0.0)
same(a, b)
assert all(x["state"]["turns"] == y["state"]["turns"] for x, y in zip(a, b))
print("② guess_env 三条：一致（含假观测、轮数上限）")

# ③ 多文件包 + 预算截断（max_total）+ 关掉异步的开关
R1 = dict(prob, task_id=11, visible_tests=["assert add(1, 2) == 3", "assert add(0, 0) == 0"])
pkg = P.make_pkg([R1], {}); penv = P.PkgEnv(max_calls=8, timeout=2)
Sp = {"P1:": ["P1:<test></test>", "<write file=\"add.py\">" + FIX + "</write>", "<test></test>", "<answer>done</answer>"],
      "P2:": ["P2:<test></test>", "<write file=\"add.py\">" + FIX + "</write>", "<test></test>", "<answer>done</answer>"]}
pp = list(Sp)
a = H.generate_with_env(FakeBackend(Sp), tok, pp, penv, [penv.reset(pkg) for _ in pp], max_new=2000, temp=0.0, max_total=120, sync=True)
b = H.generate_with_env(FakeEngineBackend(Sp), tok, pp, penv, [penv.reset(pkg) for _ in pp], max_new=2000, temp=0.0, max_total=120)
same(a, b); assert a[0]["cut"] and b[0]["cut"], "预算 120 应该截断"
os.environ["HARNESS_ASYNC"] = "0"
c = H.generate_with_env(FakeEngineBackend(Sp), tok, pp, penv, [penv.reset(pkg) for _ in pp], max_new=2000, temp=0.0, max_total=120)
assert H.generate_with_env.last_timing.get("mode") != "async"; same(a, c)
del os.environ["HARNESS_ASYNC"]
# ④ 接口对不上 → 自动退回同步
class Broken(FakeEngineBackend):
    def engine_add(self, *a, **k): raise TypeError("signature changed")
d = H.generate_with_env(Broken(Sp), tok, pp, penv, [penv.reset(pkg) for _ in pp], max_new=2000, temp=0.0, max_total=120)
same(a, d)
print("③④ pkg_env + 预算截断一致；HARNESS_ASYNC=0 和接口不合都退回同步")
print("异步 harness 单测：全部通过")
