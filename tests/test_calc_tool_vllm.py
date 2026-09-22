"""假 tokenizer（逐字符）+ 假后端，离线验证停、注入、续、mask、假结果、预算、CONT。python3 tests/test_calc_tool_vllm.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calc_tool_vllm as T

class Tok:
    eos_token_id, pad_token_id = 0, 1
    def encode(self, s, add_special_tokens=False): return [ord(c) for c in s]
    def decode(self, ids, skip_special_tokens=True): return "".join(chr(i) for i in ids if i > 1)
    def __call__(self, s): return type("R", (), {"input_ids": self.encode(s)})()

class FakeBackend:
    """脚本 = 模型「心里」的完整文本（含它以为会看到的结果）。每次续写：从当前文本之后往下吐，碰到停止串就停，吐完就 EOS"""
    def __init__(self, scripts): self.scripts = scripts
    def generate(self, rows, budgets, temp, stops=T.STOPS):
        tok = Tok(); outs = []
        for row, b in zip(rows, budgets):
            cur = tok.decode(row)
            script = next(sc for pr, sc in self.scripts.items() if cur.startswith(pr))
            full = script[len(cur):] if script.startswith(cur) else ""
            if not full:
                outs.append([0]); continue
            cut = min([full.find(s) + len(s) for s in stops if s in full] + [len(full)])
            seg = full[:cut][:b]
            ids = tok.encode(seg)
            if cut >= len(full) and len(seg) == cut: ids.append(0)
            outs.append(ids)
        return outs

tok = Tok()
P1 = "Q1:"; S1 = P1 + "<calc>1 + 2</calc><result>3</result> (too low)\n<calc>3 * 4</calc><result>12</result> (perfect match)\n<answer>3 * 4</answer>"
P2 = "Q2:"; S2 = P2 + "<calc>1 + 2</calc><result>3</result> <result>5</result> (perfect)"       # 第二个 result 是模型自己编的
P3 = "Q3:"; S3 = P3 + "<calc>2 * 2</calc><result>4</result> done\nUser: next question"          # 幻觉出下一轮
P4 = "Q4:"; S4 = P4 + "<calc>7 / 0</calc><result>error</result> hmm"                              # 报错
be = FakeBackend({P1: S1, P2: S2, P3: S3, P4: S4})
r = T.generate_with_tools(be, tok, [P1, P2, P3, P4], max_new=200, temp=0.0)

a = r[0]; assert a["txt"] == S1[len(P1):], a["txt"]; assert a["n_calls"] == 2 and not a["fake"] and a["ids"][-1] == 0
inj1 = len("<calc>1 + 2</calc>"); assert a["spans"] == [(inj1, len("<result>3</result>")), (inj1 + len("<result>3</result> (too low)\n<calc>3 * 4</calc>"), len("<result>12</result>"))], a["spans"]
m = T.response_mask(len(a["ids"]), a["spans"]); assert sum(m) == len(a["ids"]) - sum(l for _, l in a["spans"])
b = r[1]; assert b["fake"] and b["n_calls"] == 1 and b["ids"][-1] == 0, b
c = r[2]; assert c["cont"] and c["txt"].endswith(" done") and c["ids"][-1] == 0 and c["n_calls"] == 1, c
d = r[3]; assert d["n_err"] == 1 and "<result>error</result>" in d["txt"], d
# 预算：max_new 小于第一次注入后的长度 → cut
r2 = T.generate_with_tools(be, tok, [P1], max_new=20, temp=0.0)[0]; assert r2["cut"] and r2["n_calls"] == 0, r2
# 调用上限
r3 = T.generate_with_tools(be, tok, [P1], max_new=200, temp=0.0, max_calls=1)[0]; assert r3["n_calls"] == 1 and not r3["fake"], r3
# ★ 写了 </calc> 却没有 <calc>：注入 error 继续，不崩（SFT-1 在留出模板上撞过 IndexError）
P8 = "Q8:"; S8 = P8 + "no opener</calc><result>error</result> then <calc>1 + 1</calc><result>2</result> ok"
r8 = T.generate_with_tools(FakeBackend({P8: S8}), tok, [P8], max_new=200, temp=0.0)[0]
assert r8["txt"] == S8[len(P8):] and r8["n_calls"] == 2 and r8["n_err"] == 1 and not r8["fake"], r8
# cut_at_stop：多出来的 token 被截掉
ids = tok.encode("<calc>1 + 2</calc> extra"); assert tok.decode(T.cut_at_stop(tok, ids)) == "<calc>1 + 2</calc>"
# ── 求助工具：<ask> 停 → 假专家回答 → 注入 <reply> → 用计算器验证 → answer；上限 2 次；自己编 <reply> = fake ──
P5 = "Q5:"; S5 = P5 + "<calc>1 + 2</calc><result>3</result> (too low)\nI will ask.\n<ask>make 12 from [3, 4]</ask><reply>EQUATION: 3 * 4</reply>\n<calc>3 * 4</calc><result>12</result> (perfect match)\n<answer>3 * 4</answer>"
P6 = "Q6:"; S6 = P6 + "<ask>a</ask><reply>EQUATION: 1</reply> <ask>b</ask><reply>EQUATION: 2</reply> <ask>c</ask> done"
P7 = "Q7:"; S7 = P7 + "<ask>x</ask><reply>EQUATION: 1</reply> <reply>EQUATION: 9</reply> end"
class FakeBackendAsk(FakeBackend):
    def generate(self, rows, budgets, temp, stops=T.STOPS_ASK):
        return super().generate(rows, budgets, temp, stops=stops)
be2 = FakeBackendAsk({P5: S5, P6: S6, P7: S7})
fake_expert = lambda q: {"make 12 from [3, 4]": "EQUATION: 3 * 4", "a": "EQUATION: 1", "b": "EQUATION: 2", "x": "EQUATION: 1"}.get(q.strip(), "error")
# ★ 写了 </ask> 却没有 <ask>：不问专家，注入 <reply>error</reply> 继续
P9 = "Q9:"; S9 = P9 + "</ask><reply>error</reply> then <ask>a</ask><reply>EQUATION: 1</reply> end"
r9 = T.generate_with_tools(FakeBackendAsk({P9: S9}), tok, [P9], max_new=200, temp=0.0, ask=True, max_asks=2, ask_fn=fake_expert)[0]
assert r9["txt"] == S9[len(P9):] and r9["n_asks"] == 2 and r9["n_err"] == 1 and not r9["fake"], r9
r = T.generate_with_tools(be2, tok, [P5, P6, P7], max_new=300, temp=0.0, ask=True, max_asks=2, ask_fn=fake_expert)
a = r[0]; assert a["txt"] == S5[len(P5):], a["txt"]; assert a["n_asks"] == 1 and a["n_calls"] == 2 and not a["fake"]
assert len(a["spans"]) == 3 and tok.decode(a["ids"][a["spans"][1][0]: a["spans"][1][0] + a["spans"][1][1]]) == "<reply>EQUATION: 3 * 4</reply>"
b = r[1]; assert b["n_asks"] == 2, b                       # 第三次 <ask> 超上限：不再注入，序列结束
c = r[2]; assert c["fake"] and c["n_asks"] == 1, c         # 自己写 <reply> = 假回复
r0 = T.generate_with_tools(be, tok, [P1], max_new=200, temp=0.0)[0]; assert r0["n_asks"] == 0 and r0["n_calls"] == 2   # 不开 ask 行为不变
print("calc_tool_vllm 假后端单测：全部通过（停、注入、spans、mask、假结果、CONT、报错、预算、上限、cut_at_stop、求助工具 ×3、无开标签 ×2）")
