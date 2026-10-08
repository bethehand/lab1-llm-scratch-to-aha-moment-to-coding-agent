#!/usr/bin/env bash
# Upload the trained checkpoints to Hugging Face so others can reproduce the evaluations without training.
#
#   huggingface-cli login                 # once, on the machine that holds the hf_* directories
#   HF_USER=<your hf account> bash experiments/upload_checkpoints.sh
#
# Each hf_* directory is an export_hf.py output (config.json, model.safetensors, tokenizer files, export_info.json), 2.9 GB.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HF_USER:?set HF_USER to your Hugging Face account name}"

declare -A DESC=(
  [hf_coder_1.5b]="Qwen2.5-Coder-1.5B exported to the local HF-directory format the evaluators read; the base of the coding line"
  [hf_sft_ex]="coding stage ②-B: SFT on the DeepSeek textbook, exercism modules from scratch with tests (report §3.14)"
  [hf_exRL_bin]="coding stage ②-B: SFT + GRPO, binary reward, strong observation; pass@1 .446 with tests, .295 without (report §3.14, §3.15)"
  [hf_sft_exw]="coding stage ③: SFT on the weak-observation textbook, assertions written by the model itself (report §3.15)"
  [hf_exwRL_bin]="coding stage ③: SFT + GRPO under weak observation; pass@1 .282 (report §3.15)"
  [hf_armA]="Countdown Arm A: calculator tool + ask-an-expert, pass@1 .729 alone / .979 with the expert (report §3.10)"
)
for D in hf_coder_1.5b hf_sft_ex hf_exRL_bin hf_sft_exw hf_exwRL_bin hf_armA; do
  [ -d "$D" ] || { echo "skip $D (not here)"; continue; }
  REPO="$HF_USER/lab1-${D#hf_}"
  BASE="Qwen/Qwen2.5-Coder-1.5B"; [ "$D" = hf_armA ] && BASE="Qwen/Qwen2.5-1.5B"
  cat > "$D/README.md" <<EOF
---
license: apache-2.0
base_model: ${BASE}
tags: [lab1, grpo, coding-agent, countdown]
---
# ${REPO}

${DESC[$D]}.

Part of *LLM from scratch, to the aha moment, to a coding agent: fifteen experiments on four RTX 4090s*.
Code, report and the evaluation protocol: https://github.com/bethehand/lab1-llm-scratch-to-aha-moment-to-coding-agent

Evaluate it yourself: \`python3 bench_observation_price.py --hf-dir <this repo> --model <this repo>\` from the repository root.
Weights are derived from Qwen2.5 and remain under the Qwen license.
EOF
  echo "== $D -> $REPO"
  huggingface-cli upload "$REPO" "$D" . --repo-type model --commit-message "upload $D from the lab machine"
done
echo "done; update README.md 'Where --init comes from' with the links"
