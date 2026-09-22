#!/usr/bin/env python3
"""
deepseek_tool.py —— DeepSeek API 的最小封装：零依赖（urllib）、温度 0、按问题缓存、可当命令行也可 import。

    export DEEPSEEK_API_KEY=sk-...                                   # 密钥只放环境变量，不写进任何文件
    python3 deepseek_tool.py "9.11 和 9.9 哪个大"                      # 随便问
    python3 deepseek_tool.py --countdown 10 13 15 --target 45          # Countdown 题 → 只回 EQUATION: …
    python3 deepseek_tool.py --countdown 10 13 15 --target 45 --model deepseek-reasoner
    python3 deepseek_tool.py --test                                    # 10 道硬题，chat 和 reasoner 各问一遍，判分器验，看谁靠谱

模型名（2026-09 实测）：deepseek-chat = v4-flash 不思考（便宜、快、Countdown 硬题会错）；deepseek-reasoner = v4-flash 带思考（会对，多花一百多 token）；
deepseek-v4-pro 更贵。reasoner 的推理段在 reasoning_content 里，我们只用 content。
缓存：cache/deepseek_cache.json，键 = 模型 + 问题；命中不再调用，环境可复现、省钱。
"""
import os, sys, json, time, hashlib, argparse, threading, urllib.request, urllib.error

URL = "https://api.deepseek.com/chat/completions"


def get_key():
    """只读环境变量 DEEPSEEK_API_KEY，没有就退出。从不打印密钥本身"""
    k = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not k:
        raise SystemExit("没设 DEEPSEEK_API_KEY：先 export DEEPSEEK_API_KEY=sk-...（每个 tmux 窗口各设一次）")
    return k
_LOCK = threading.Lock()                       # 多线程并发调用时缓存读写加锁
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "deepseek_cache.json")
_cache = None


def _load():
    global _cache
    if _cache is None:
        _cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    return _cache


def _save():
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    tmp = CACHE + ".tmp"
    json.dump(_cache, open(tmp, "w"), ensure_ascii=False, indent=0); os.replace(tmp, CACHE)


def ask(question, model="deepseek-chat", system=None, max_tokens=400, temperature=0.0, retries=3, use_cache=True, timeout=180):
    """→ dict(content, reasoning, usage, cached)。出错抛异常（重试 3 次）"""
    key = get_key()
    ck = hashlib.md5(f"{model}|{system}|{question}|{max_tokens}|{temperature}".encode()).hexdigest()
    with _LOCK:
        c = _load()
        if use_cache and ck in c and c[ck].get("content"):        # ★ 空正文（旧版缓存过的截断回复）不算命中，重新问
            return dict(c[ck], cached=True)
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]
    body = json.dumps({"model": model, "messages": msgs, "temperature": temperature, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                j = json.load(r)
            m = j["choices"][0]["message"]
            fin = j["choices"][0].get("finish_reason")
            out = dict(content=(m.get("content") or "").strip(), reasoning=m.get("reasoning_content"),
                       usage=j.get("usage", {}), model=j.get("model"), finish=fin)
            if use_cache and fin == "stop" and out["content"]:      # ★ 被 max_tokens 截断的（finish=length、正文为空）不进缓存
                with _LOCK:
                    c[ck] = out; _save()
            return dict(out, cached=False)
        except urllib.error.HTTPError as e:
            err = e.read().decode()[:300]
            if e.code in (429, 500, 502, 503) and i < retries - 1:
                time.sleep(2 * (i + 1)); continue
            raise SystemExit(f"API 错误 {e.code}: {err}")
        except (urllib.error.URLError, TimeoutError) as e:
            if i < retries - 1:
                time.sleep(2 * (i + 1)); continue
            raise


CD_SYS = "You solve Countdown puzzles. Reply with exactly one line: EQUATION: <equation using each given number exactly once with + - * / and parentheses>. No other text."


def show(r, system, question):
    """★ 把一次调用的输入和输出全部打出来：system、user、推理段（全文）、正文、用量"""
    print("── 输入 system ──\n" + (system or "（无）"))
    print("── 输入 user ──\n" + question)
    if r.get("reasoning"):
        print("── 输出 推理段（reasoning_content，全文）──\n" + r["reasoning"])
    print("── 输出 正文（content）──\n" + (r["content"] or "（空：推理没写完就被 max_tokens 截断，加 --max-tokens）"))
    print("── 用量 ──", r.get("usage"), "  模型", r.get("model"), "  结束原因", r.get("finish"), "  （缓存）" if r.get("cached") else "")


def ask_countdown(nums, target, model="deepseek-chat", max_tokens=None):
    """思考模型的推理段也算在 max_tokens 里，硬题要几百到几千；默认 reasoner/pro 4000，chat 80"""
    q = f"Using the numbers {list(nums)} each exactly once with + - * /, write one equation that equals {target}."
    if max_tokens is None:
        max_tokens = 4000 if ("reason" in model or "pro" in model) else 80
    r = ask(q, model=model, system=CD_SYS, max_tokens=max_tokens)
    r["question"] = q
    eq = r["content"].split("EQUATION:")[-1].strip().split("=")[0].strip() if "EQUATION:" in r["content"] else r["content"].strip()
    return dict(r, equation=eq)


HARD = [([20, 25, 30], 15), ([10, 13, 15], 45), ([11, 19, 28], 99), ([13, 37, 39], 40), ([20, 22, 25, 30], 37),
        ([5, 7, 14, 33], 43), ([1, 2, 11, 26], 28), ([4, 7, 12, 23], 37), ([2, 3, 7, 3], 39), ([2, 1, 3, 4], 24)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--countdown", type=int, nargs="+")
    ap.add_argument("--target", type=int)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=None, help="思考模型的推理段也算在内，硬题给 4000")
    a = ap.parse_args()
    if a.test:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import countdown as CD
        for model in ("deepseek-chat", "deepseek-reasoner"):
            ok = tok_out = 0; t0 = time.time()
            print(f"\n== {model} ==")
            for nums, tgt in HARD:
                r = ask_countdown(nums, tgt, model=model)          # reasoner 默认 4000
                _, d = CD.reward(f"<answer> {r['equation']} </answer>", nums, tgt)
                ok += d["correct"] > 0; tok_out += r["usage"].get("completion_tokens", 0)
                print(f"  {str(nums):18} → {tgt:>3}   {r['equation']:28} {'✓' if d['correct'] > 0 else '✗'}   {r['usage'].get('completion_tokens', 0):>4} tok{'  (缓存)' if r['cached'] else ''}")
            print(f"  对 {ok}/{len(HARD)}   输出 token 合计 {tok_out}   {time.time()-t0:.0f} s")
    elif a.countdown:
        r = ask_countdown(a.countdown, a.target, model=a.model, max_tokens=a.max_tokens)
        show(r, CD_SYS, r["question"])
        print("→ 抠出的式子：", r["equation"])
    elif a.question:
        r = ask(a.question, model=a.model, use_cache=not a.no_cache, max_tokens=a.max_tokens or 400)
        show(r, None, a.question)
    else:
        ap.print_help()
