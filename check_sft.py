#!/usr/bin/env python3
"""
check_sft.py —— SFT 数据体检：一份 jsonl 里的轨迹长什么样。臂 1（穷举器）和臂 2（自采样）并排看。

    python3 check_sft.py sft_cd3.jsonl sft_cd3_self.jsonl sft_cd4.jsonl sft_cd4_self.jsonl

每份报：条数、token 长度、每条几行标注、标注精度（算对且方向对）、重复形态比例、
写了 perfect/works 的比例、乘除出现率、（穷举器数据）命中层分布。
"""
import re, sys, json
from collections import Counter
import countdown as CD

TOOLCALL = re.compile(r"<calc>(.*?)</calc>\s*<result>(.*?)</result>", re.S)   # 工具版教材：归一成「E = V」再体检
ANNOT = re.compile(r"((?:\(|\d)[\d\s\+\-\*/\(\)]*?)\s*=\s*(-?\d+(?:\.\d+)?)\s*\(?\s*(too (?:high|low|big|small|large))", re.I)
CLAIM = re.compile(r"\b(perfect|works|correct|matches)\b", re.I)
MULDIV = re.compile(r"[\d)]\s*[\*/×÷]\s*[\d(]")
LINE = re.compile(r"^\s*([\d\s\+\-\*/\(\)]+?)\s*=\s*-?\d", re.M)

def canon(expr):
    """纯加减：'25 + 7 - 3' 和 '7 + 25 - 3' 归成同一形态（按项集合）；含 ×÷ 的按原样比（去空格）"""
    e = expr.replace(" ", "")
    if "*" in e or "/" in e or "(" in e:
        return ("md", e)
    return tuple(sorted(re.findall(r"[+-]?\d+", e if e[:1] in "+-" else "+" + e)))

try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    tl = lambda s: len(tok(s).input_ids)
except Exception:
    tok = None
    tl = lambda s: int(len(s) / 3.2)

print(f"{'文件':22} {'条':>5} {'tok中位/P90/最长':>18} {'标注/条':>7} {'精度':>6} {'重复形态':>8} {'说perfect':>9} {'含×÷':>6}  命中层")
for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path)]
    if not rows:
        print(f"{path:22} 空"); continue
    lens = sorted(tl(r["response"]) for r in rows)
    q = lambda f: lens[int(f * (len(lens) - 1))]
    n_an = ok_an = 0; dup = tot_lines = 0; claim = md = 0
    for r in rows:
        txt, tgt = r["response"], r.get("target")
        if "<calc>" in txt:
            txt = TOOLCALL.sub(r"\1 = \2", txt)
        if tgt is not None:
            for m in ANNOT.finditer(txt):
                try:
                    real = CD.safe_eval(CD.clean_expr(m.group(1)))
                except Exception:
                    continue
                n_an += 1
                said_high = any(w in m.group(3).lower() for w in ("high", "big", "large"))
                if abs(real - float(m.group(2))) <= 1e-3 * max(1.0, abs(real)) and said_high == (real > tgt):
                    ok_an += 1
        exprs = [canon(e) for e in LINE.findall(txt)]
        tot_lines += len(exprs); dup += len(exprs) - len(set(exprs))
        claim += bool(CLAIM.search(txt)); md += bool(MULDIV.search(txt))
    n = len(rows)
    lv = Counter(r.get("level") for r in rows if "level" in r)
    lvs = "/".join(f"{lv.get(i,0)}" for i in (0, 1, 2)) if lv else "—"
    prec = f"{ok_an/n_an:.2f}" if n_an else "—"
    print(f"{path:22} {n:>5} {q(.5):>6}/{q(.9)}/{lens[-1]:<6} {n_an/n:>7.1f} {prec:>6} "
          f"{dup/max(1,tot_lines):>8.2f} {claim/n:>9.2f} {md/n:>6.2f}  {lvs}")
if tok is None:
    print("（没连上 tokenizer，token 长度按字符/3.2 估）")
print("\n读法：精度 = 「X = V (too high/low)」里算对且方向对的比例，穷举器该是 1.00；重复形态 = 同一条里同一组项出现第二次的行占比")
