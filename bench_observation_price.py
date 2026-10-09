#!/usr/bin/env python3
"""The price of observation, measured on any Hugging Face model.

Same problems, same hidden tests, two settings:
  strong  the model may run the author's visible tests (<test>) while it works
  weak    the tests are taken away; the model can only write its own checks and run them (<run>)
The difference between the two scores is the price the model pays when it is deployed without tests.

    source ~/vllm_env/bin/activate
    python3 bench_observation_price.py --hf-dir Qwen/Qwen2.5-Coder-7B-Instruct --model Qwen/Qwen2.5-Coder-7B-Instruct
    python3 bench_observation_price.py --hf-dir hf_exRL_bin --model hf_coder_1.5b -n 100 -s 8 --pass32
    python3 bench_observation_price.py --hf-dir icedduck/lab1-exRL_bin     # the published checkpoint; expect about .46 / .31

Runs eval_ex.py twice (as subprocesses, so vLLM frees the GPU between settings), then reads the raw records
and prints the table below plus a figure. --from-raw re-analyses existing raw files without a GPU.

Reference points from this repository (Qwen2.5-Coder-1.5B, 100 training problems x 8, PLAN.md 10563):
  SFT       strong .329   weak .240   price -.09
  SFT + RL  strong .446   weak .295   price -.16
"""
import os, sys, json, math, argparse, subprocess, collections

HERE = os.path.dirname(os.path.abspath(__file__))
REF = {"SFT (1.5B, this repo)": (.329, .240), "SFT+RL (1.5B, this repo)": (.446, .295)}


def pass_at_k(n, c, k):
    """Unbiased estimator (Chen et al. 2021): 1 - C(n-c, k) / C(n, k)."""
    if n - c < k: return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def analyse(raw, ks=(1, 8, 32)):
    rec = raw["records"]
    by = collections.defaultdict(list)
    for r in rec: by[r["slug"]].append(r)
    out = {"n_problems": len(by), "samples": raw.get("n_samples"), "obs": raw.get("obs"), "name": raw.get("name")}
    out["pass@1"] = sum(r["d"]["correct"] for r in rec) / len(rec)
    for k in ks:
        if k == 1: continue
        vals = [pass_at_k(len(v), int(sum(r["d"]["correct"] for r in v)), k) for v in by.values() if len(v) >= k]
        out[f"pass@{k}"] = sum(vals) / len(vals) if vals else None
    fv = [r["d"].get("first_ver_all") for r in rec if r["d"].get("first_ver_all") is not None]
    out["first_draft"] = sum(fv) / len(fv) if fv else None
    out["fake_obs"] = sum(1 for r in rec if r.get("fake")) / len(rec)
    out["submitted"] = sum(r["d"].get("has_answer", 0) for r in rec) / len(rec)
    return out


def run_eval(args, obs, out_json):
    cmd = [sys.executable, os.path.join(HERE, "eval_ex.py"), "--hf-dir", args.hf_dir, "--model", args.model, "--data", args.data, "--split", args.split,
           "-n", str(args.n_problems), "-s", str(args.samples), "--seed", str(args.seed), "--obs", obs, "--gpu-mem", str(args.gpu_mem),
           "--max-new", str(args.max_new), "--out", out_json, "--show", "0"]
    print("$ " + " ".join(cmd), flush=True)
    if args.dry: return
    subprocess.run(cmd, check=True, cwd=HERE)


def figure(strong, weak, path, title):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping the figure"); return
    fig, ax = plt.subplots(figsize=(7.2, 4.2)); w = 0.36
    groups = [("with tests\n(strong observation)", strong, "#1F5F8B", "#BFD6E8"), ("tests removed\n(weak observation)", weak, "#B45A1C", "#F0D2B8")]
    for i, (label, m, dark, light) in enumerate(groups):
        if m.get("first_draft") is not None:
            ax.bar(i - w / 2, m["first_draft"], w, color=light, edgecolor=dark, linewidth=0.6)
            ax.text(i - w / 2, m["first_draft"] + .01, f"{m['first_draft']:.2f}", ha="center", fontsize=9.5)
        ax.bar(i + w / 2, m["pass@1"], w, color=dark)
        ax.text(i + w / 2, m["pass@1"] + .01, f"{m['pass@1']:.2f}", ha="center", fontsize=9.5, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels([g[0] for g in groups]); ax.set_ylim(0, max(1.0, strong["pass@1"] + .15))
    ax.set_ylabel(f"pass rate ({strong['n_problems']} problems x {strong['samples']} attempts)")
    ax.yaxis.grid(True, color="#E1E4E8"); ax.set_axisbelow(True); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#BFD6E8", edgecolor="#1F5F8B", label="first draft passes"), Patch(facecolor="#1F5F8B", label="final submission passes")], frameon=False, fontsize=9, loc="upper right")
    price = weak["pass@1"] - strong["pass@1"]
    ax.set_title(f"{title}: price of observation {price:+.2f}", loc="left", fontsize=12)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig); print("figure ->", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-dir", help="model directory or Hugging Face id")
    ap.add_argument("--model", default=None, help="tokenizer source; defaults to --hf-dir (the repo's own checkpoints use hf_coder_1.5b)")
    ap.add_argument("--data", default="data/exercism.json")
    ap.add_argument("--split", default="train", choices=["train", "heldout"], help="train = the 100 problems the repo's models were trained on; heldout = 30 never-trained problems")
    ap.add_argument("-n", "--n-problems", type=int, default=100)
    ap.add_argument("-s", "--samples", type=int, default=8, help="8 for pass@1; use --pass32 for the ceiling (32 samples, 4x the cost)")
    ap.add_argument("--pass32", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.6)
    ap.add_argument("--max-new", type=int, default=4096)
    ap.add_argument("--out", default=None, help="output directory (default: bench_<model name>)")
    ap.add_argument("--from-raw", nargs=2, metavar=("STRONG_RAW", "WEAK_RAW"), help="re-analyse two existing *_raw.json files (no GPU)")
    ap.add_argument("--dry", action="store_true", help="print the eval commands and exit")
    args = ap.parse_args()
    if args.pass32: args.samples = max(args.samples, 32)

    if args.from_raw:
        strong_raw, weak_raw = (json.load(open(f)) for f in args.from_raw); name = strong_raw.get("name", "model"); outdir = args.out or f"bench_{name}"
    else:
        assert args.hf_dir, "--hf-dir is required (or --from-raw)"
        args.model = args.model or args.hf_dir
        name = os.path.basename(args.hf_dir.rstrip("/")); outdir = args.out or f"bench_{name}"
        os.makedirs(outdir, exist_ok=True)
        for obs in ("strong", "weak"):
            run_eval(args, obs, os.path.join(outdir, f"{obs}.json"))
        if args.dry: return
        strong_raw = json.load(open(os.path.join(outdir, "strong_raw.json"))); weak_raw = json.load(open(os.path.join(outdir, "weak_raw.json")))
    os.makedirs(outdir, exist_ok=True)
    S, W = analyse(strong_raw), analyse(weak_raw)

    def cell(v): return "   -  " if v is None else f"{v:6.3f}"
    print(f"\n{name}: {S['n_problems']} problems x {S['samples']} samples, split {strong_raw.get('split')}")
    print(f"{'':28s}{'with tests':>12s}{'tests removed':>15s}{'price':>9s}")
    rows = [("first draft passes", "first_draft"), ("final submission, pass@1", "pass@1"), ("pass@8", "pass@8"), ("pass@32 (ceiling)", "pass@32"),
            ("fabricated observations", "fake_obs"), ("submitted an answer", "submitted")]
    for label, key in rows:
        if S.get(key) is None and W.get(key) is None: continue
        price = (W[key] - S[key]) if (S.get(key) is not None and W.get(key) is not None) else None
        print(f"{label:28s}{cell(S.get(key)):>12s}{cell(W.get(key)):>15s}{cell(price):>9s}")
    print("\nreference (this repo, Qwen2.5-Coder-1.5B):")
    for k, (s, w) in REF.items(): print(f"{k:28s}{s:12.3f}{w:15.3f}{w - s:9.3f}")
    summary = dict(name=name, strong=S, weak=W, price_pass1=W["pass@1"] - S["pass@1"], reference=REF)
    json.dump(summary, open(os.path.join(outdir, "summary.json"), "w"), indent=1)
    print("summary ->", os.path.join(outdir, "summary.json"))
    figure(S, W, os.path.join(outdir, "observation_price.svg"), name)


if __name__ == "__main__":
    main()
