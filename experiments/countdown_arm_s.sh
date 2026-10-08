#!/usr/bin/env bash
# Experiment 9: Countdown Arm S, penalty-free reward + pool screening + KL anchor (report §3.9; PLAN.md 7738–7872).
#
# Starting point: the calculator-tool SFT checkpoint of Arm T (report §3.8). Three ingredients, changed together
# against Arm T's RL, which "learned bad habits" from the same start:
#   grader v3   correct earns 1 - 0.05 * len/max_new; wrong, no answer and loops earn 0; nothing is penalized (--judge v3)
#   screening   keep only problems the SFT model solves sometimes but not always (1 <= c <= 7 of 8 samples)
#   anchor      KL to the SFT model, beta 0.02 (--ref-init, --beta)
#
# Hardware as run: 4x RTX 4090; screening ~20 min, RL 300 steps ~2.5 h with vLLM rollouts.
# The RL command is transcribed verbatim from PLAN.md 7738–7739; the data preparation steps are reconstructed from
# the recorded flags (marked RECONSTRUCTED). Run from the repository root inside the vLLM venv.
#
# Recorded numbers (out-of-pool 100 problems, temperature 1; PLAN.md 7860–7868):
#   CD-4 pass@1   SFT start .710  ->  Arm S .748 (+.04, confirmed twice at 150/300 steps and once at x64)
#   CD-4 pass@64  .900 -> .900   (ceiling unchanged: the same 10 problems stay unsolved)
#   CD-3 pass@64  1.000          (nothing shaped away on the untrained task)
#   GSM8K         R1 template .795, neutral template .635 (both up; no collateral damage)
set -euo pipefail
cd "$(dirname "$0")/.."
NPROC=${NPROC:-4}
df -h . | tail -1

# ── 0. Problems and the tool-version textbook ───────────────────────────────── RECONSTRUCTED from PLAN.md 5978, 7338, 7359, 7436
[ -f data/countdown4.json ] || python3 countdown.py --gen 100000 --k 4 --out data/countdown4.json       # 4-number problems; [95000:95100] is the out-of-pool test slice
[ -f data/countdown.json  ] || python3 countdown.py --gen 100000 --k 3 --out data/countdown.json        # 3-number problems (transfer task)
python3 enum_traces.py --tool --data data/countdown.json  --n 2000 --hi 80000 --out sft_cd3_tool.jsonl  # program teacher, every value as <calc>expr</calc><result>V</result>
python3 enum_traces.py --tool --data data/countdown4.json --n 2000 --hi 80000 --out sft_cd4_tool.jsonl
torchrun --nproc_per_node=$NPROC sft_qwen.py --data sft_cd3_tool.jsonl sft_cd4_tool.jsonl --out out_sft_tool   # val loss 0.71 -> 0.059 recorded
python3 export_hf.py --ckpt out_sft_tool/ckpt.pt --out hf_sft_tool

# ── 1. Pool screening with the SFT model: 8000 problems x 8 samples, one shard per GPU ─ RECONSTRUCTED from PLAN.md 7777–7783 and 11037
for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python3 screen_pool.py --hf-dir hf_sft_tool --data data/countdown4.json --n 8000 --lo 0 --hi 90000 -k 8 \
        --shard $i/4 --out data/countdown4_mixed.json &
done; wait
python3 screen_pool.py --data data/countdown4.json --n 8000 -k 8 --merge --out data/countdown4_mixed.json
# recorded: 4810 all-correct (60%) / 1546 mixed (19%) / 1644 all-wrong (21%) -> 1546 problems enter the pool

# ── 2. RL, 300 steps ────────────────────────────────────────────────────────── VERBATIM PLAN.md 7738–7739
torchrun --nproc_per_node=$NPROC grpo_tool_mp_vllm.py --probs 8 -k 16 --gen-bs 32 --steps 300 --init out_sft_tool/ckpt.pt --data data/countdown4_mixed.json \
    --val-data data/countdown4.json --judge v3 --ref-init out_sft_tool/ckpt.pt --beta 0.02 --max-new 1400 --len-soft 0 --patience 0 --tool --engine vllm --out out_armS
python3 export_hf.py --ckpt out_armS/ckpt_latest.pt --out hf_armS

# ── 3. Evaluation on the out-of-pool slice, calculator on, expert off ───────── baseline_countdown_vllm.py --help
for M in hf_sft_tool hf_armS; do
    python3 baseline_countdown_vllm.py --engine vllm --hf-dir $M --tool --data data/countdown4.json --offset 95000 -n 100 -s 8  --max-new 1400 --out cd4_${M}_s8.json
    python3 baseline_countdown_vllm.py --engine vllm --hf-dir $M --tool --data data/countdown4.json --offset 95000 -n 100 -s 64 --max-new 1400 --out cd4_${M}_s64.json
    python3 baseline_countdown_vllm.py --engine vllm --hf-dir $M --tool --data data/countdown.json  --offset 95000 -n 100 -s 64 --max-new 1400 --out cd3_${M}_s64.json
done
echo "Reconcile against the numbers at the top of this script."
