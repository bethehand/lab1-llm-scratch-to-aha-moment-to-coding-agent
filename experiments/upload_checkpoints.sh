#!/usr/bin/env bash
# Upload the trained checkpoints to Hugging Face so others can reproduce the evaluations without training.
#
#   huggingface-cli login                          # once, on the machine that holds the hf_* directories
#   HF_USER=<your hf account> bash experiments/upload_checkpoints.sh         # tier 1: the 7 checkpoints behind the headline figures (~20 GB)
#   HF_USER=<your hf account> ALL=1 bash experiments/upload_checkpoints.sh   # tier 1 + 2: every checkpoint a report figure uses (~56 GB)
#
# Each hf_* directory is an export_hf.py output (config.json, model.safetensors, tokenizer files, export_info.json), 2.9 GB.
# Raw trainer checkpoints (out_*/ckpt_latest.pt, with optimizer state) are not uploaded; they are only needed to resume training.
# Re-exports of Qwen's own models (hf_base, hf_coder_1.5b, hf_qwen3_1.7b_base) are not uploaded; use the Qwen repositories directly.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HF_USER:?set HF_USER to your Hugging Face account name}"

CODER="Qwen/Qwen2.5-Coder-1.5B"; BASE15="Qwen/Qwen2.5-1.5B"
declare -A DESC BASE
# tier 1: the figures on the homepage
DESC[hf_sft_ex]="coding stage ②-B: SFT on the DeepSeek textbook, exercism modules from scratch with tests; pass@1 .329 with tests, .240 without (report §3.14, §3.15)"; BASE[hf_sft_ex]=$CODER
DESC[hf_exRL_bin]="coding stage ②-B: SFT + GRPO, binary reward, strong observation; pass@1 .446 with tests, .295 without, re-measured .459 / .314 (report §3.14, §3.15)"; BASE[hf_exRL_bin]=$CODER
DESC[hf_sft_exw]="coding stage ③: SFT on the weak-observation textbook, the model writes its own assertions; pass@1 .186 (report §3.15)"; BASE[hf_sft_exw]=$CODER
DESC[hf_exwRL_bin]="coding stage ③: SFT + GRPO under weak observation; pass@1 .282, re-measured .296 (report §3.15)"; BASE[hf_exwRL_bin]=$CODER
DESC[hf_sft_tool]="Countdown: SFT on the calculator-tool textbook, the start of Arms T and S; CD-4 pass@1 .71 (report §3.8)"; BASE[hf_sft_tool]=$BASE15
DESC[hf_armS]="Countdown Arm S: penalty-free reward + pool screening + KL anchor; CD-4 pass@1 .748, pass@64 .90 (report §3.9)"; BASE[hf_armS]=$BASE15
DESC[hf_armA]="Countdown Arm A: calculator tool + ask-an-expert with a budget; CD-4 pass@1 .729 alone, .979 with the expert (report §3.10)"; BASE[hf_armA]=$BASE15
TIER1="hf_sft_ex hf_exRL_bin hf_sft_exw hf_exwRL_bin hf_sft_tool hf_armS hf_armA"
# tier 2: the remaining figures
DESC[hf_sft_code_coder]="coding stage ①: SFT on the program teacher's bug-fixing demos; pass@1 .606 (report §3.12)"; BASE[hf_sft_code_coder]=$CODER
DESC[hf_codeRL]="coding stage ①: SFT + GRPO, 150 steps; pass@1 .749 (report §3.12)"; BASE[hf_codeRL]=$CODER
DESC[hf_codeRL_dyn]="coding stage ①: SFT + GRPO with dynamic sampling; pass@1 .718, pass@32 .980 (report §3.12)"; BASE[hf_codeRL_dyn]=$CODER
DESC[hf_sft_pkg_B]="coding stage ②-A: SFT on textbook B, K buggy files per package; K=3 fix rate .069 (report §3.13)"; BASE[hf_sft_pkg_B]=$CODER
DESC[hf_pkgRL_bin]="coding stage ②-A: SFT + GRPO, binary reward, K=3; fix rate .245 (report §3.13)"; BASE[hf_pkgRL_bin]=$CODER
DESC[hf_pkgRL_frac]="coding stage ②-A: SFT + GRPO, proportional reward, K=3; fix rate .195 (report §3.13)"; BASE[hf_pkgRL_frac]=$CODER
DESC[hf_E1]="number guessing E1: pure GRPO from the Countdown Arm A checkpoint, range 1..100; hit rate .632 (report §3.11)"; BASE[hf_E1]=$BASE15
DESC[hf_E3]="number guessing E3: curriculum 1..20 then 1..100; hit rate .838 (report §3.11)"; BASE[hf_E3]=$BASE15
DESC[hf_sft_guess]="number guessing E2: SFT on 200 scripted bisection demos; hit rate .976 (report §3.11)"; BASE[hf_sft_guess]=$BASE15
DESC[hf_E2]="number guessing E2: demo SFT + GRPO; hit rate .996 (report §3.11)"; BASE[hf_E2]=$BASE15
DESC[hf_armT]="Countdown Arm T: calculator-tool SFT + GRPO on the v1 grader; the run that shaped the seed away (report §3.8)"; BASE[hf_armT]=$BASE15
DESC[hf_armS150]="Countdown Arm S at 150 steps (report §3.9)"; BASE[hf_armS150]=$BASE15
DESC[hf_armS_b0]="Countdown Arm S′: the anchor removed, β 0 (report §3.9)"; BASE[hf_armS_b0]=$BASE15
TIER2="hf_sft_code_coder hf_codeRL hf_codeRL_dyn hf_sft_pkg_B hf_pkgRL_bin hf_pkgRL_frac hf_E1 hf_E3 hf_sft_guess hf_E2 hf_armT hf_armS150 hf_armS_b0"

LIST="$TIER1"; [ "${ALL:-0}" = 1 ] && LIST="$TIER1 $TIER2"
for D in $LIST; do
  [ -d "$D" ] || { echo "skip $D (not here)"; continue; }
  REPO="$HF_USER/lab1-${D#hf_}"
  cat > "$D/README.md" <<EOF
---
license: apache-2.0
base_model: ${BASE[$D]}
tags: [lab1, grpo, coding-agent, countdown]
---
# ${REPO}

${DESC[$D]}.

Part of *LLM from scratch, to the aha moment, to a coding agent: fifteen experiments on four RTX 4090s*.
Code, report, the evaluation protocol and the scripts that produced this checkpoint:
https://github.com/bethehand/lab1-llm-scratch-to-aha-moment-to-coding-agent

Evaluate it yourself from the repository root, for example \`python3 bench_observation_price.py --hf-dir ${REPO}\` for the coding-line checkpoints.
Weights are derived from ${BASE[$D]} and remain under the Qwen license.
EOF
  echo "== $D -> $REPO"
  huggingface-cli upload "$REPO" "$D" . --repo-type model --commit-message "upload $D from the lab machine"
done
echo "done; update README.md 'Where --init comes from' with the links"
