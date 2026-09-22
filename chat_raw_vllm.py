#!/usr/bin/env python3
"""
chat_raw_vllm.py —— 零提示词的原始对话：你粘贴什么，模型就从什么后面接着写。看的是模型最原始的行为。vllm_env 里跑。

    python3 chat_raw_vllm.py --hf-dir hf_armS                    # 工具 harness 开着：模型写 <calc>…</calc> 就停、算、注入、续
    python3 chat_raw_vllm.py --hf-dir hf_armS --no-tool          # 不接计算器：看它自己会不会编 <result>
    python3 chat_raw_vllm.py --hf-dir hf_armS --temp 1 -s 4      # 温度 1 采 4 条

用法：粘贴整段提示词（多行也行），在空行上按回车发送；q 退出。
输出两遍：① 原样（模型写的 + harness 注入的拼在一起，就是模型看到的上下文）② 标注版（⟦…⟧ 里是 harness 注入的，其余是模型自己写的）
如果提示词里能认出 Countdown 题（Using the numbers [..] … equals N），末尾顺便判一下 <answer> 对不对。
"""
import re, sys, argparse, select
from transformers import AutoTokenizer
import countdown as CD
import calc_tool_vllm as calc_tool

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B", help="tokenizer 来源")
ap.add_argument("--hf-dir", required=True)
ap.add_argument("--temp", type=float, default=0.0)
ap.add_argument("-s", "--samples", type=int, default=1)
ap.add_argument("--max-new", type=int, default=1400)
ap.add_argument("--no-tool", action="store_true")
ap.add_argument("--gpu-mem", type=float, default=0.6)
ap.add_argument("--no-annot", action="store_true", help="不打第二遍标注版（⟦⟧ 注入段）")
ap.add_argument("--ask", action="store_true", help="★ 开求助工具：模型写 <ask>…</ask> 就停、问 DeepSeek reasoner、注入 <reply>…</reply>（要先 export DEEPSEEK_API_KEY）")
ap.add_argument("--max-asks", type=int, default=3, help="每条最多求助几次（臂 A 训练时是 3）")
ap.add_argument("--ask-model", default="deepseek-reasoner", help="专家；探针用 deepseek-chat")
args = ap.parse_args()
if args.ask:
    import deepseek_tool as _DS; _DS.get_key()                # 没密钥就在这儿停，别等到模型第一次求助才报 error

CDQ = re.compile(r"Using the numbers \[([\d,\s]+)\], create an equation that equals (\d+)")


def read_block(prompt):
    print(prompt, end="", flush=True)
    lines = []
    def pending():
        try:
            return bool(select.select([sys.stdin], [], [], 0.15)[0])
        except (OSError, ValueError):
            return False
    while True:
        try:
            line = sys.stdin.readline()
        except KeyboardInterrupt:
            print("\n  （已清空，重新输入；q 退出）"); lines = []; print(prompt, end="", flush=True); continue
        if line == "":
            return "\n".join(lines) if lines else None
        line = line.rstrip("\n").rstrip("\r")
        if not lines and line.strip().lower() in ("q", "quit", "exit"):
            return None
        more = pending()
        if line.strip() == "" and lines and not more:
            return "\n".join(lines)
        if line.strip() == "" and not lines:
            continue
        lines.append(line)


def annotate(tok, ids, spans):
    """把注入段用 ⟦ ⟧ 括起来"""
    out, i = [], 0
    for s, l in sorted(spans):
        out.append(tok.decode(ids[i:s], skip_special_tokens=True))
        out.append("⟦" + tok.decode(ids[s:s + l], skip_special_tokens=True) + "⟧")
        i = s + l
    out.append(tok.decode(ids[i:], skip_special_tokens=True))
    return "".join(out)


tok = AutoTokenizer.from_pretrained(args.model)
be = calc_tool.make_backend("vllm", tok=tok, hf_dir=args.hf_dir, gpu_mem=args.gpu_mem, max_model_len=args.max_new + 1200)
print(f"\n★ 引擎就绪 ← {args.hf_dir}   温度 {args.temp}   每次 {args.samples} 条   计算器 {'关' if args.no_tool else '开'}   求助 {'开' if args.ask else '关'}   粘贴提示词，空行回车发送，q 退出\n")

while True:
    p = read_block("提示词> ")
    if p is None:
        break
    prompts = [p] * args.samples
    if args.no_tool:
        res = calc_tool.plain_generate(be, tok, prompts, max_new=args.max_new, temp=args.temp)
        outs = [dict(ids=o["ids"], txt=o["txt"], spans=[], n_calls=0, n_err=0, fake=("<result>" in o["txt"]), cont=o["cont"], cut=False) for o in res]
    else:
        outs = calc_tool.generate_with_tools(be, tok, prompts, max_new=args.max_new, temp=args.temp, ask=args.ask, max_asks=args.max_asks,
                                             ask_fn=(lambda q: calc_tool.ask_expert(q, model=args.ask_model)) if args.ask else None)
    m = CDQ.search(p)
    for k, o in enumerate(outs):
        print("─" * 78 + (f"  第 {k+1} 条" if args.samples > 1 else ""))
        print(o["txt"])
        if o["spans"] and not args.no_annot:
            print("─ 标注版（⟦⟧ = harness 注入）" + "─" * 50)
            print(annotate(tok, o["ids"], o["spans"]))
        tail = f"  {len(o['ids'])} token   调用 {o['n_calls']}   求助 {o.get('n_asks', 0)}   报错 {o['n_err']}" + ("   ⚠ 自己写了 <result>/<reply>" if o["fake"] else "") + ("   ⚠ 幻觉出 User:" if o["cont"] else "") + ("   ⚠ 撞顶" if o["cut"] else "")
        if m:
            nums = [int(x) for x in m.group(1).split(",")]; tgt = int(m.group(2))
            r, d = CD.reward(o["txt"], nums, tgt)
            tail += f"   判分 {'✓ 对' if d['correct'] > 0 else '✗ 错'}（{r:.2f}）"
        print(tail)
    print()
