#!/usr/bin/env python3
"""
calc_tool.py —— 计算器 harness：停、注入、续。训练器（grpo_tool_mp.py）和评测脚本（--tool）共用这一份。

协议
    模型写  <calc>42 + 12 - 18 - 6</calc>           → 生成在 </calc> 停
    harness 抠出式子 → countdown.safe_eval → 接上  <result>30</result>  → 模型继续写
    算不了 → <result>error</result>；每条最多 max_calls 次（默认 48：老师的 4 数轨迹一行一次、乘积表 6 次、第 2 层每行 2 次，16 不够）
    模型自己写出 <result>（不是 harness 注入的）= 假结果 → 立刻停，标 fake（训练器罚分）

返回每条：
    ids     响应 token（含注入的 result 段），末尾带 EOS（如果停得干净）
    txt     响应文本
    spans   [(起点, 长度)] 注入段在 ids 里的位置 —— 训练时 mask 掉，不进梯度
    n_calls / n_err / fake / cont / cut
"""
import re
import torch
import countdown as CD

CALC_OPEN, CALC_CLOSE = "<calc>", "</calc>"
RES_OPEN, RES_CLOSE = "<result>", "</result>"
CONT = "\nUser:"
CALC_RE = re.compile(r"<calc>(.*?)</calc>", re.S)

TOOL_HINT = ("You have a calculator: write <calc>EXPRESSION</calc> and the exact value will be returned as "
             "<result>VALUE</result>. Use it for every calculation. ")


def tool_prompt(nums, target):
    """R1 模板 + 工具说明（插在 Show your work 之前）"""
    return CD.make_prompt(nums, target).replace("Show your work in", TOOL_HINT + "Show your work in")


def evaluate(expr):
    try:
        v = CD.safe_eval(CD.clean_expr(expr))
    except Exception:
        return "error"
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.4g}"


def response_mask(n, spans):
    """长度 n 的 1/0 列表，注入段置 0"""
    m = [1] * n
    for s, l in spans:
        for i in range(s, min(n, s + l)):
            m[i] = 0
    return m


@torch.no_grad()
def generate_with_tools(model, tok, prompts, max_new=1024, temp=1.0, max_calls=48, gen_bs=32, dev=None):
    dev = dev or next(model.parameters()).device
    EOS, PAD = tok.eos_token_id, (tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    stop_ids = tok.encode(CONT, add_special_tokens=False)
    seqs = [dict(prompt=tok.encode(p, add_special_tokens=False), gen=[], spans=[], n_calls=0, n_err=0,
                 fake=False, cont=False, cut=False, done=False) for p in prompts]
    rounds = 0
    while True:
        active = [s for s in seqs if not s["done"]]
        if not active:
            break
        rounds += 1
        for c0 in range(0, len(active), gen_bs):
            chunk = active[c0: c0 + gen_bs]
            rows = [s["prompt"] + s["gen"] for s in chunk]
            L = max(len(r) for r in rows)
            ids = torch.full((len(rows), L), PAD, dtype=torch.long)
            att = torch.zeros((len(rows), L), dtype=torch.long)
            for i, r in enumerate(rows):                          # 左填充
                ids[i, L - len(r):] = torch.tensor(r); att[i, L - len(r):] = 1
            ids, att = ids.to(dev), att.to(dev)
            budget = max(1, max_new - min(len(s["gen"]) for s in chunk))
            kw = dict(max_new_tokens=budget, do_sample=temp > 0, top_p=1.0, pad_token_id=PAD)
            if temp > 0:
                kw["temperature"] = temp
            out = model.generate(input_ids=ids, attention_mask=att,
                                 stop_strings=[CALC_CLOSE, RES_OPEN, CONT], tokenizer=tok, **kw)
            for i, s in enumerate(chunk):
                new = out[i, L:].tolist()
                while new and new[-1] == PAD:                     # 批里别人还在写时自己被填的 pad
                    new.pop()
                if EOS in new:
                    new = new[: new.index(EOS) + 1]
                remain = max_new - len(s["gen"])
                if len(new) > remain:
                    new = new[:remain]
                s["gen"] += new
                text_new = tok.decode(new, skip_special_tokens=True)
                full = tok.decode(s["gen"], skip_special_tokens=True)
                if new and new[-1] == EOS:
                    s["done"] = True
                elif CONT in full:                                # 幻觉出下一轮：截掉，补 EOS
                    s["cont"] = True
                    keep = tok.encode(full.split(CONT)[0], add_special_tokens=False)
                    s["gen"] = keep + [EOS]; s["spans"] = [sp for sp in s["spans"] if sp[0] + sp[1] <= len(keep)]
                    s["done"] = True
                elif RES_OPEN in text_new:                         # ★ 自己编结果：立刻停
                    s["fake"] = True; s["gen"].append(EOS); s["done"] = True
                elif text_new.rstrip().endswith(CALC_CLOSE) and s["n_calls"] < max_calls:
                    expr = CALC_RE.findall(full)[-1]
                    v = evaluate(expr)
                    inj = tok.encode(f"{RES_OPEN}{v}{RES_CLOSE}", add_special_tokens=False)
                    if len(s["gen"]) + len(inj) > max_new:
                        s["cut"] = True; s["done"] = True
                    else:
                        s["spans"].append((len(s["gen"]), len(inj)))
                        s["gen"] += inj; s["n_calls"] += 1; s["n_err"] += (v == "error")
                elif len(s["gen"]) >= max_new:
                    s["cut"] = True; s["done"] = True
                else:                                              # 没停在标签上又没 EOS：预算用完或调用满了
                    s["cut"] = len(s["gen"]) >= max_new; s["done"] = True
    return [dict(ids=s["gen"], txt=tok.decode(s["gen"], skip_special_tokens=True), spans=s["spans"],
                 n_calls=s["n_calls"], n_err=s["n_err"], fake=s["fake"], cont=s["cont"], cut=s["cut"],
                 rounds=rounds) for s in seqs]
