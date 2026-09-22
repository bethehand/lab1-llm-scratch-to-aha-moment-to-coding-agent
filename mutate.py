#!/usr/bin/env python3
"""
mutate.py —— 长程编程阶段 ① 的 bug 注入器：往 MBPP 参考实现里用 AST 注一个 bug，验证「原版全过、变异后至少挂一条可见测试」，留下的就是题。
    python3 mutate.py --data data_code.json --out data/code_bugs.json [--per-task 3] [--heldout-kinds swap_args,del_stmt]
七类变异（kind）
    off_by_one    整数常量 ±1（range 上下界、下标、比较里的数）
    cmp_flip      比较符换：< ↔ <=、> ↔ >=、== ↔ !=
    arith_swap    算术符换：+ ↔ -、* ↔ //、% ↔ //
    bool_neg      条件取反 / and ↔ or / not 去掉
    ret_wrong     return 换成函数里别的变量名（同一作用域里出现过的）
    del_stmt      删一条语句（赋值 / 增量赋值 / 函数体里的 if 分支的一行）
    swap_args     交换调用或定义里两个参数的顺序
每道题最多 --per-task 个变异体，同一 kind 最多一个；隐藏测试按 task_id 固定选（test_list 里第 task_id % 3 条），mutate 时也记进去。
留出 kind（--heldout-kinds）只出现在 heldout 题上：训练题不用它们注 bug，评测时看模型对没见过的 bug 类型行不行。
输出每条：task_id, kind, bug_code, ref_code, diff（原行 → 改行）, visible_tests, hidden_test, setup, text, split(train|heldout), bug_line
"""
import ast, copy, json, random, argparse, subprocess, sys, tempfile, os, difflib, collections

# ── 七类变异，各是一个 ast.NodeTransformer，只改「第 k 个」候选点 ──
class _Base(ast.NodeTransformer):
    kind = "base"
    def __init__(self, target_idx):
        self.idx, self.n, self.done = target_idx, 0, False
    def hit(self):
        self.n += 1
        if self.n - 1 == self.idx and not self.done:
            self.done = True; return True
        return False

class OffByOne(_Base):
    kind = "off_by_one"
    def visit_Constant(self, node):
        if isinstance(node.value, int) and not isinstance(node.value, bool) and 0 <= node.value <= 1000 and self.hit():
            return ast.copy_location(ast.Constant(node.value + random.choice([-1, 1]) if node.value > 0 else node.value + 1), node)
        return node

CMP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
class CmpFlip(_Base):
    kind = "cmp_flip"
    def visit_Compare(self, node):
        self.generic_visit(node)
        for i, op in enumerate(node.ops):
            if type(op) in CMP and self.hit():
                node.ops[i] = CMP[type(op)](); break
        return node

ARITH = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv, ast.FloorDiv: ast.Mult, ast.Mod: ast.FloorDiv, ast.Div: ast.Mult}
class ArithSwap(_Base):
    kind = "arith_swap"
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if type(node.op) in ARITH and self.hit():
            node.op = ARITH[type(node.op)]()
        return node

class BoolNeg(_Base):
    kind = "bool_neg"
    def visit_If(self, node):
        self.generic_visit(node)
        if self.hit():
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
        return node
    def visit_BoolOp(self, node):
        self.generic_visit(node)
        if self.hit():
            node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
        return node
    def visit_While(self, node):
        self.generic_visit(node)
        if self.hit():
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
        return node

class RetWrong(_Base):
    kind = "ret_wrong"
    def __init__(self, idx):
        super().__init__(idx); self.names = []
    def visit_FunctionDef(self, node):
        saved = self.names
        self.names = sorted({n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)} |
                            {a.arg for a in node.args.args})
        self.generic_visit(node); self.names = saved; return node
    def visit_Return(self, node):
        cur = node.value.id if isinstance(node.value, ast.Name) else None
        others = [n for n in self.names if n != cur]
        if node.value is not None and others and self.hit():
            node.value = ast.Name(id=random.choice(others), ctx=ast.Load())
        return node

class DelStmt(_Base):
    kind = "del_stmt"
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        body = node.body
        cands = [i for i, st in enumerate(body) if isinstance(st, (ast.Assign, ast.AugAssign, ast.Expr)) and len(body) > 1
                 and not (i == 0 and isinstance(st, ast.Expr))]          # 别删 docstring
        for i in cands:
            if self.hit():
                del body[i]; break
        return node
    def visit_For(self, node):
        self.generic_visit(node)
        cands = [i for i, st in enumerate(node.body) if isinstance(st, (ast.Assign, ast.AugAssign)) and len(node.body) > 1]
        for i in cands:
            if self.hit():
                del node.body[i]; break
        return node

class SwapArgs(_Base):
    kind = "swap_args"
    def visit_Call(self, node):
        self.generic_visit(node)
        if len(node.args) >= 2 and self.hit():
            i, j = random.sample(range(len(node.args)), 2); node.args[i], node.args[j] = node.args[j], node.args[i]
        return node
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        a = node.args.args
        if len(a) >= 2 and self.hit():
            i, j = random.sample(range(len(a)), 2); a[i], a[j] = a[j], a[i]
        return node

KINDS = {c.kind: c for c in (OffByOne, CmpFlip, ArithSwap, BoolNeg, RetWrong, DelStmt, SwapArgs)}


def count_sites(cls, tree):
    """这类变异在这棵树上有几个候选点（跑一遍 idx=-1 只数不改）"""
    t = cls(-1); t.visit(copy.deepcopy(tree)); return t.n


def mutate(code, kind, idx):
    tree = ast.parse(code); t = KINDS[kind](idx); new = t.visit(copy.deepcopy(tree))
    if not t.done: return None
    ast.fix_missing_locations(new)
    return ast.unparse(new)


# ── 沙盒里验证 ──
def run_tests(code, tests, setup="", timeout=5):
    """→ (全过?, 每条测试过没过的列表, 每条的报错类名：'' / AssertionError / TypeError / … / Timeout)"""
    res, errs = [], []
    for tst in tests:
        src = setup + "\n" + code + "\n" + tst + "\nprint('__OK__')"
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.py"); open(p, "w").write(src)
            try:
                r = subprocess.run([sys.executable, p], capture_output=True, text=True, timeout=timeout, cwd=d, env={"PATH": os.environ["PATH"]})
                ok = r.returncode == 0 and "__OK__" in r.stdout
                last = (r.stderr.strip().splitlines() or [""])[-1]
                res.append(ok); errs.append("" if ok else last.split(":")[0].strip())
            except subprocess.TimeoutExpired:
                res.append(False); errs.append("Timeout")
    return all(res), res, errs


def diff_lines(a, b):
    """原版 → 变异版 的行级 diff，只留改动行"""
    out = []
    for l in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0):
        if l.startswith(("---", "+++", "@@")): continue
        out.append(l)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_code.json")
    ap.add_argument("--out", default="data/code_bugs.json")
    ap.add_argument("--per-task", type=int, default=3)
    ap.add_argument("--heldout-kinds", default="swap_args,del_stmt", help="只在 heldout 题上用的变异类型")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 道（冒烟）")
    args = ap.parse_args()
    random.seed(args.seed)
    D = json.load(open(args.data)); held_kinds = set(k for k in args.heldout_kinds.split(",") if k)
    out, stat = [], collections.Counter()
    for split, key in (("train", "mbpp_train"), ("heldout", "mbpp_heldout")):
        rows = D[key][: args.limit] if args.limit else D[key]
        for r in rows:
            code = ast.unparse(ast.parse(r["code"]))                   # 先规范化（去掉 \\t、奇怪缩进），diff 才干净
            tests = [t.strip() for t in r["test_list"]]; setup = r.get("test_setup_code", "") or ""
            h = r["task_id"] % 3; hidden = tests[h]; visible = [t for i, t in enumerate(tests) if i != h]
            ok, _, _ = run_tests(code, tests, setup)
            if not ok:
                stat["ref_fail"] += 1; continue
            kinds = [k for k in KINDS if (split == "heldout") or (k not in held_kinds)]
            random.shuffle(kinds); kinds.sort(key=lambda k: stat[f"ok_{k}"])          # ★ 目前出得少的 kind 优先，七类均衡
            made = 0
            for kind in kinds:
                if made >= args.per_task: break
                n = count_sites(KINDS[kind], ast.parse(code))
                if n == 0: stat[f"nosite_{kind}"] += 1; continue
                for idx in random.sample(range(n), min(n, 4)):         # 每类最多试 4 个点
                    bug = mutate(code, kind, idx)
                    if bug is None or bug == code: continue
                    try:
                        compile(bug, "<bug>", "exec")
                    except SyntaxError:
                        stat["syntax"] += 1; continue
                    _, per, errs = run_tests(bug, tests, setup)
                    vis_fail = any(not per[i] for i in range(3) if i != h)
                    if not vis_fail:                                     # 可见测试全过 = 等价变异或只挂隐藏的，不要
                        stat["equiv"] += 1; continue
                    d = diff_lines(code, bug)
                    vis_errs = [errs[i] for i in range(3) if i != h and not per[i]]
                    bug_type = "wrong" if all(e == "AssertionError" for e in vis_errs) else "crash"   # ★ 答错型 / 崩溃型（traceback 直接指到行，观测更强）
                    out.append(dict(task_id=r["task_id"], split=split, kind=kind, bug_type=bug_type, err=vis_errs[0], text=r["text"], ref_code=code, bug_code=bug, diff=d,
                                    visible_tests=visible, hidden_test=hidden, setup=setup, hidden_fails=not per[h], n_vis_fail=sum(not per[i] for i in range(3) if i != h)))
                    stat[f"ok_{kind}"] += 1; made += 1; break
            if made == 0: stat["no_bug"] += 1
        print(f"{split}: {len(rows)} 道 → 目前 {len(out)} 个变异体", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False)
    by = collections.Counter((o["split"], o["kind"]) for o in out)
    print(f"\n落盘 {len(out)} 题 → {args.out}")
    print("  train:", {k: v for (s, k), v in sorted(by.items()) if s == "train"})
    print("  heldout:", {k: v for (s, k), v in sorted(by.items()) if s == "heldout"})
    print("  隐藏测试也挂的比例", f"{sum(o['hidden_fails'] for o in out)}/{len(out)}", "  答错型 / 崩溃型", f"{sum(o['bug_type']=='wrong' for o in out)} / {sum(o['bug_type']=='crash' for o in out)}",
          "  崩溃的报错类", dict(collections.Counter(o["err"] for o in out if o["bug_type"] == "crash").most_common(5)))
    print("  丢弃：", {k: v for k, v in stat.items() if not k.startswith("ok_")})
