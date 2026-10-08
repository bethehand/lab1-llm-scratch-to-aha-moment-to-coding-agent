#!/usr/bin/env bash
# Experiment 11: number guessing, the first stateful environment, three arms (report §3.11; PLAN.md 8700–8930).
#
#   E1  pure RL from the Countdown Arm A checkpoint, range 1..100, 150 steps
#   E3  curriculum: 50 steps at range 1..20, then 100 steps at 1..100 (same total compute as E1)
#   E2  seeding: 200 scripted bisection demos (20% deliberately sloppy) -> SFT -> the same RL
#
# Hardware as run: 4x RTX 4090, ~35 min per RL arm. Commands below are transcribed from the lab log where it
# records them literally and reconstructed from the recorded flags elsewhere (marked RECONSTRUCTED); the trainer's
# own --help is the authority. Run from the repository root inside the vLLM venv:  bash experiments/guess_number.sh
#
# Recorded numbers (100 secrets x 8 samples, temperature 1; PLAN.md 8892 and report fig. "number guessing"):
#                       training templates   held-out templates   range 1..1000
#   start (hf_armA)        .043                  -                    -
#   E1 pure RL             .632                 .359                 .142
#   E3 curriculum          .838                 .512                 .171
#   E2 SFT only            .976                 .873                 .621
#   E2 SFT + RL            .996                 .926                 .713
set -euo pipefail
cd "$(dirname "$0")/.."
START=${START:-out_armA/ckpt_latest.pt}          # Countdown Arm A checkpoint (report §3.10); any 1.5B ckpt that already emits <guess> tags works as a start
NPROC=${NPROC:-4}
df -h . | tail -1                         # the lab log's rule: check the disk before every run

# The trainer writes ckpt_latest.pt / ckpt_best.pt; only sft_qwen.py writes ckpt.pt.
# ── E1: pure RL on range 1..100 ──────────────────────────────────────────────── RECONSTRUCTED from PLAN.md 8752 (flags) and README
torchrun --nproc_per_node=$NPROC grpo_tool_mp_vllm.py --task guess --guess-N 100 --guess-T 10 --guess-cost 0.05 \
    --init "$START" --ref-init "$START" --steps 150 -k 16 --probs 8 --beta 0.02 --engine vllm --out out_E1
python3 export_hf.py --ckpt out_E1/ckpt_latest.pt --out hf_E1

# ── E3: curriculum, 1..20 for 50 steps then 1..100 for 100 steps ─────────────── RECONSTRUCTED from PLAN.md 8797–8811
torchrun --nproc_per_node=$NPROC grpo_tool_mp_vllm.py --task guess --guess-N 20 --guess-T 10 --guess-cost 0.05 \
    --init "$START" --ref-init "$START" --steps 50 -k 16 --probs 8 --beta 0.02 --engine vllm --out out_E3a
torchrun --nproc_per_node=$NPROC grpo_tool_mp_vllm.py --task guess --guess-N 100 --guess-T 10 --guess-cost 0.05 \
    --init out_E3a/ckpt_latest.pt --ref-init out_E3a/ckpt_latest.pt --steps 100 -k 16 --probs 8 --beta 0.02 --engine vllm --out out_E3b
python3 export_hf.py --ckpt out_E3b/ckpt_latest.pt --out hf_E3b

# ── E2: 200 scripted demos -> SFT -> RL ──────────────────────────────────────── RECONSTRUCTED from PLAN.md 8874, 8892 and make_guess_demos.py
python3 make_guess_demos.py --n 200 --N 100 --T 10 --sloppy 0.2 --out sft_guess.jsonl
torchrun --nproc_per_node=$NPROC sft_qwen.py --init "$START" --data sft_guess.jsonl --out out_sft_guess        # about 2 minutes
python3 export_hf.py --ckpt out_sft_guess/ckpt.pt --out hf_sft_guess
torchrun --nproc_per_node=$NPROC grpo_tool_mp_vllm.py --task guess --guess-N 100 --guess-T 10 --guess-cost 0.05 \
    --init out_sft_guess/ckpt.pt --ref-init out_sft_guess/ckpt.pt --steps 150 -k 16 --probs 8 --beta 0.02 --engine vllm --out out_E2
python3 export_hf.py --ckpt out_E2/ckpt_latest.pt --out hf_E2

# ── Evaluation: three protocols per model ────────────────────────────────────── eval_guess.py --help for the template sets
for M in hf_E1 hf_E3b hf_sft_guess hf_E2; do
    python3 eval_guess.py --hf-dir $M -n 100 -s 8 --N 100 --T 10 --prompt-set train   --out guess_${M}_train.json
    python3 eval_guess.py --hf-dir $M -n 100 -s 8 --N 100 --T 10 --prompt-set heldout --out guess_${M}_heldout.json
    python3 eval_guess.py --hf-dir $M -n 100 -s 8 --N 1000 --T 12 --prompt-set train  --out guess_${M}_N1000.json
done
echo "Reconcile against the table at the top of this script (noise floor: about ±.03 at 100 x 8)."
