"""假 tokenizer + 假后端，离线验 harness.generate_with_env + guess_env：正常二分、invalid、重复、假观测、轮数上限、mask、奖励。python3 tests/test_harness.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness as H
import guess_env as G

class Tok:
    eos_token_id, pad_token_id = 0, 1
    def encode(self, s, add_special_tokens=False): return [ord(c) for c in s]
    def decode(self, ids, skip_special_tokens=True): return "".join(chr(i) for i in ids if i > 1)

class FakeBackend:
    def __init__(self, scripts): self.scripts = scripts
    def generate(self, rows, budgets, temp, stops):
        tok = Tok(); outs = []
        for row, b in zip(rows, budgets):
            cur = tok.decode(row)
            script = next(sc for pr, sc in self.scripts.items() if cur.startswith(pr))
            full = script[len(cur):] if script.startswith(cur) else ""
            if not full:
                outs.append([0]); continue
            cut = min([full.find(s) + len(s) for s in stops if s in full] + [len(full)])
            seg = full[:cut][:b]; ids = tok.encode(seg)
            if cut >= len(full) and len(seg) == cut: ids.append(0)
            outs.append(ids)
        return outs

tok = Tok(); env = G.GuessEnv(N=100, T=4)
P1 = "G1:"; S1 = P1 + "<guess>50</guess><obs>higher</obs> so 51-100\n<guess>75</guess><obs>lower</obs>\n<guess>62</guess><obs>correct</obs>\n</think>\n<answer>62</answer>"
P2 = "G2:"; S2 = P2 + "<guess>abc</guess><obs>invalid</obs><guess>50</guess><obs>higher</obs><guess>50</guess><obs>invalid</obs><guess>70</guess><obs>lower</obs> out of turns"
P3 = "G3:"; S3 = P3 + "<guess>10</guess><obs>higher</obs> <obs>correct</obs> <answer>10</answer>"       # 第二个 obs 是自己编的
P4 = "G4:"; S4 = P4 + "<guess>30</guess><obs>higher</obs><guess>40</guess><obs>lower</obs>\n</think>\n<answer>35</answer>"   # 没猜中就交卷（瞎答）
be = FakeBackend({P1: S1, P2: S2, P3: S3, P4: S4})
states = [env.reset(62), env.reset(60), env.reset(50), env.reset(35)]      # 第 3 条秘密数 50：环境回 higher 与脚本一致，随后模型自己写的 <obs>correct</obs> 才是假的
r = H.generate_with_env(be, tok, [P1, P2, P3, P4], env, states, max_new=300, temp=0.0)

a = r[0]; st = a["state"]; assert a["txt"] == S1[len(P1):], a["txt"]; assert st["found"] and st["turns"] == 3 and not a["fake"] and a["ids"][-1] == 0
assert a["spans"] == [(len("<guess>50</guess>"), len("<obs>higher</obs>")),
                      (len("<guess>50</guess><obs>higher</obs> so 51-100\n<guess>75</guess>"), len("<obs>lower</obs>")),
                      (len("<guess>50</guess><obs>higher</obs> so 51-100\n<guess>75</guess><obs>lower</obs>\n<guess>62</guess>"), len("<obs>correct</obs>"))], a["spans"]
sc, d = env.score(st, a["txt"]); assert abs(sc - 0.90) < 1e-9 and d["correct"] == 1 and d["turns"] == 3, (sc, d)
import calc_tool_vllm as T; m = T.response_mask(len(a["ids"]), a["spans"]); assert sum(m) == len(a["ids"]) - sum(l for _, l in a["spans"])
b = r[1]; st = b["state"]; assert st["turns"] == 4 and st["invalid"] == 1 and st["repeat"] == 1 and not st["found"] and b["ids"][-1] == 0 and "out of turns" not in b["txt"], (st, b["txt"])
sc, d = env.score(st, b["txt"]); assert sc == 0.0 and d["correct"] == 0
c = r[2]; assert c["fake"] and c["state"]["turns"] == 1 and c["ids"][-1] == 0, c
dd = r[3]; sc, d = env.score(dd["state"], dd["txt"]); assert d["correct"] == 0 and not dd["state"]["found"] and sc == 0.0, (sc, d)   # ★ 瞎答撞中不给分
bh = G.behavior(a["txt"]); assert bh["dir_n"] == 2 and bh["dir_ok"] == 2 and bh["in_range_ok"] == 3 and bh["bisect_ok"] == 3, bh
bh2 = G.behavior(S4[len(P4):]); assert bh2["dir_ok"] == 1 and bh2["dir_n"] == 1 and bh2["bisect_ok"] == 0, bh2
r2 = H.generate_with_env(be, tok, [P1], env, [env.reset(62)], max_new=25, temp=0.0)[0]; assert r2["cut"], r2
# ★ 引擎上下文上限：题面 3 个 id，max_total=16 → 这条的预算 = 16−3−8 = 5，不管 max_new 多大都截在 5 个 id，不会把超长的行喂给引擎
r3 = H.generate_with_env(be, tok, [P1], env, [env.reset(62)], max_new=300, temp=0.0, max_total=16)[0]; assert r3["cut"] and len(r3["ids"]) <= 5, (len(r3["ids"]), r3)
# ★ 猜中之后还猜：环境直接结束，轮数不再涨，没 <answer> 得 0
P5 = "G5:"; S5 = P5 + "<guess>62</guess><obs>correct</obs> more <guess>70</guess><obs>lower</obs> <answer>70</answer>"
r5 = H.generate_with_env(FakeBackend({P5: S5}), tok, [P5], env, [env.reset(62)], max_new=200, temp=0.0)[0]
assert r5["state"]["turns"] == 1 and r5["ids"][-1] == 0 and "<answer>" not in r5["txt"] and env.score(r5["state"], r5["txt"])[0] == 0.0, (r5["state"], r5["txt"])
for pid in G.SETS["all"]:
    s = G.render(pid); assert s.endswith("<think>") and "<guess>" in s and "<answer>" in s and "100" in s, pid
print("harness + guess_env 假后端单测：全部通过（二分、invalid、重复、假观测、轮数上限、mask、奖励、行为读数、预算、上下文上限、10 个模板）")
