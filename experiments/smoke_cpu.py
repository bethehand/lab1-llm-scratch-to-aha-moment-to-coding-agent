#!/usr/bin/env python3
"""Five-minute tour of the lab bench without a GPU.

    python3 experiments/smoke_cpu.py

Three stops, each using the real code from the repository and a *scripted* model in place of an LLM:

  1. Countdown: generate problems, score a right and a wrong answer with the Countdown grader.
  2. Number guessing: a stateful environment. A scripted bisection policy runs through the real
     harness (stop -> environment -> inject observation -> continue) and is scored.
  3. Bug fixing: a real sandbox. One policy fixes the bug properly; another hard-codes the visible
     tests. The hidden test catches the cheat and the grader gives it 0.

Everything printed here is produced by the same environments, harness and graders the RL trainer uses.
"""
import os, re, sys, random, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import countdown as CD
import guess_env as G
import code_env as C
import harness as H


class Tok:
    """A character-level stand-in for a tokenizer: one id per character, 0 = EOS."""
    eos_token_id, pad_token_id = 0, 1
    def encode(self, s, add_special_tokens=False): return [ord(c) for c in s]
    def decode(self, ids, skip_special_tokens=True): return "".join(chr(i) for i in ids if i > 1)


class PolicyBackend:
    """Replaces the LLM. `policy(context_text) -> (next_chunk, done)`; the harness handles stops and injection."""
    def __init__(self, policy): self.policy = policy
    def generate(self, rows, budgets, temp, stops):
        tok, outs = Tok(), []
        for row, b in zip(rows, budgets):
            chunk, done = self.policy(tok.decode(row))
            cut = min([chunk.find(s) + len(s) for s in stops if s in chunk] + [len(chunk)])
            ids = tok.encode(chunk[:cut][:b])
            if done: ids.append(0)
            outs.append(ids)
        return outs


def banner(title): print("\n" + "=" * 88 + f"\n{title}\n" + "=" * 88)


# ── 1. Countdown ────────────────────────────────────────────────────────────────────────────────
def solve_countdown(nums, target):
    """Brute force over orderings, operators and parenthesisations; returns an expression string or None."""
    ops = "+-*/"
    def combos(vals, exprs):
        if len(vals) == 1:
            if abs(vals[0] - target) < 1e-9: return exprs[0]
            return None
        for i, j in itertools.permutations(range(len(vals)), 2):
            rest_v = [v for k, v in enumerate(vals) if k not in (i, j)]
            rest_e = [e for k, e in enumerate(exprs) if k not in (i, j)]
            for op in ops:
                a, b = vals[i], vals[j]
                if op == "/" and (b == 0 or a % b): continue
                v = a + b if op == "+" else a - b if op == "-" else a * b if op == "*" else a // b
                r = combos(rest_v + [v], rest_e + [f"({exprs[i]} {op} {exprs[j]})"])
                if r: return r
        return None
    return combos(list(nums), [str(n) for n in nums])

banner("1. Countdown: problems with a built-in verifier")
probs = CD.generate(3, k=4, lo=1, hi=50, tmax=100, seed=7)
for p in probs:
    nums, target = p["nums"], p["target"]
    expr = solve_countdown(nums, target)
    right = f"<think>searching</think>\n<answer>{expr}</answer>"
    wrong = f"<think>guessing</think>\n<answer>{nums[0]} + {nums[1]}</answer>"
    r1, d1 = CD.reward(right, nums, target); r0, d0 = CD.reward(wrong, nums, target)
    print(f"  nums {nums}  target {target}   solution {expr}")
    print(f"     grader: correct answer -> {r1:.2f}   wrong answer -> {r0:.2f}   (ladder: format {d0['fmt']:.2f} parse {d0['parse']:.2f} numbers {d0['nums_frac']:.2f} correct {d0['correct']:.2f})")
print("  The grader is a program: it parses the expression, checks every number is used once, evaluates it. It cannot be fooled by prose.")


# ── 2. Number guessing: a stateful environment through the real harness ───────────────────────
banner("2. Number guessing: state + harness + grader")
tok = Tok(); env = G.GuessEnv(N=100, T=10)
PROMPT = G.t0(100, 10) + "\n<think>"

def bisection_policy(ctx):
    body = ctx[len(PROMPT):]
    lo, hi = 1, 100
    pairs = re.findall(r"<guess>(\d+)</guess>\s*<obs>(\w+)</obs>", body)
    for g, o in pairs:
        g = int(g)
        if o == "higher": lo = max(lo, g + 1)
        elif o == "lower": hi = min(hi, g - 1)
        elif o == "correct": return f"\n</think>\n<answer>{g}</answer>", True
    return f"\n<guess>{(lo + hi) // 2}</guess>", False

secrets = [37, 88, 3]
states = [env.reset(s) for s in secrets]
outs = H.generate_with_env(PolicyBackend(bisection_policy), tok, [PROMPT] * 3, env, states, max_new=600, temp=0.0)
for s, o in zip(secrets, outs):
    r, d = env.score(o["state"], o["txt"])
    turns = len(re.findall(r"<guess>", o["txt"]))
    print(f"  secret {s:3d}: {turns} guesses, found={bool(o['state'].get('found'))}, reward {r:.2f}   fake observation written by the model? {o['fake']}")
print("  trajectory for secret 37:\n    " + outs[0]["txt"].replace("\n", "\n    "))
print("  Observations (<obs>...</obs>) were written by the environment, not the policy; the harness masks them out of the loss.")


# ── 3. Bug fixing in a real sandbox: an honest fix and a hard-coded cheat ─────────────────────
banner("3. Bug fixing: real sandbox, hidden test, grader with defenses")
bugs = C.load_problems(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "code_bugs.json"), split="train")
pat = re.compile(r"^assert (\w+)\((.*)\) == (.+)$")
prob = None
for b in bugs:
    ms = [pat.match(t.strip()) for t in b["visible_tests"]]
    mh = pat.match(b["hidden_test"].strip())
    if all(ms) and mh and not b.get("setup") and len({m.group(1) for m in ms} | {mh.group(1)}) == 1:
        prob, ms, mh = b, ms, mh; break
name = ms[0].group(1)
cenv = C.CodeEnv(max_calls=8, call_cost=0.02)
Q = C.render(0, prob, with_tb=False, env=cenv) if hasattr(C, "render") else prob["text"]
Q = Q + "\n<think>"

def fix_policy(ctx):
    n = ctx[len(Q):].count("<result>")          # count only the environment's replies, not the tool description in the prompt
    steps = ["<test></test>", "\n<write>" + prob["ref_code"] + "</write>", "\n<test></test>", "\n</think>\n<answer>done</answer>"]
    return steps[min(n, 3)], n >= 3

hard = "def %s(*args):\n    table = {%s}\n    return table[repr(args)]\n" % (name, ", ".join(f"repr(({m.group(2)},)): {m.group(3)}" for m in ms))
def cheat_policy(ctx):
    n = ctx[len(Q):].count("<result>")
    steps = ["<write>" + hard + "</write>", "\n<test></test>", "\n</think>\n<answer>done</answer>"]
    return steps[min(n, 2)], n >= 2

for label, pol in (("honest fix", fix_policy), ("hard-coded cheat", cheat_policy)):
    st = cenv.reset(prob)
    o = H.generate_with_env(PolicyBackend(pol), tok, [Q], cenv, [st], max_new=4000, temp=0.0)[0]
    r, d = cenv.score(o["state"], o["txt"])
    vis = f"{d['vis_pass']}/{d['n_tests'] - 1}"
    print(f"  {label:17s}: visible tests {vis} passed, hidden test {'passed' if d['hidden_pass'] else 'FAILED'}, calls {d['calls']}  ->  reward {r:.2f}")
print(f"  problem: {prob['text'][:90]}...")
print("  The cheat passes everything the model can see and scores 0: the hidden test is the half of the answer sheet kept sealed.")
print("\nDone. Next: experiments/README.md lists the GPU scripts that reproduce the fifteen experiments.")
