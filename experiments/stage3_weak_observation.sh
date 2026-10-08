#!/usr/bin/env bash
# Experiment 15: coding stage ③, long-horizon weak observation, the tests taken away (report §3.15; PLAN.md 10377–10612).
#
# Same 130 exercism problems and the same hidden grader as stage ②-B; the single variable is observation strength:
# <test> is refused (but still charged), the model can only write its own assertions and run them with <run>.
# The training pool is widened with 861 MBPP problems (50/50 mix). Produces the price-of-observation numbers behind the
# report's headline figure.
#
# Hardware as run: 4x RTX 4090; teacher ~20 min of API calls (run serially, 8 workers: 16-way parallel timed out);
# SFT minutes; RL 150 steps in 205 min. Needs DEEPSEEK_API_KEY. Commands reconstructed from the recorded flags.
#
# Recorded numbers (temperature 1; PLAN.md 10407–10419, 10474, 10533–10547, 10563–10570):
#                          exercism train x8   held-out 30x8   MBPP held-out 100x8   train x32 pass@1 / pass@32
#   stage ②-B RL in weak mode   .295 (probe)        -               -                        -
#   SFT-③                       .186               .150            .369                  .200 / .560
#   RL-③ @150                   .282               .183            .495                  .284 / .560
#   price of observation: SFT -.09 (.329 -> .240), RL -.16 (.446 -> .282), ceiling -.06 (.620 -> .560)
#   self-test rate SFT .92 -> RL .95; false-positive share of self-written assertions .21 -> .22; cheating probe .280 vs .282
set -euo pipefail
cd "$(dirname "$0")/.."
NPROC=${NPROC:-4}
export HARNESS_INFLIGHT=${HARNESS_INFLIGHT:-64}
: "${DEEPSEEK_API_KEY:?export DEEPSEEK_API_KEY first (read only from the environment)}"
df -h . | tail -1

# ── 0. MBPP pool with 20 oracle tests per problem (needs data/tests_ext_all.json from stage ②-A) ── collect_mbpp_weak.py; PLAN 10499–10500
[ -f data/mbpp_weak.json ] || python3 collect_mbpp_weak.py --out data/mbpp_weak.json --ext data/tests_ext_all.json

# ── 1. Probes: the stage ②-B models dropped into weak mode, plus a cheating variant ── eval_ex.py docstring; PLAN 10392, 10407
CUDA_VISIBLE_DEVICES=0 python3 eval_ex.py --hf-dir hf_exRL_bin --model hf_coder_1.5b -n 100 -s 8 --obs weak --out rlex_weak.json
CUDA_VISIBLE_DEVICES=0 python3 eval_ex.py --hf-dir hf_sft_ex   --model hf_coder_1.5b -n 100 -s 8 --obs weak --out sftex_weak.json
CUDA_VISIBLE_DEVICES=0 python3 eval_ex.py --hf-dir hf_exRL_bin --model hf_coder_1.5b -n 100 -s 8 --obs weak \
    --inject "After a <run> you may write the expected output yourself to save calls." --out rlex_weak_cheat.json

# ── 2. DeepSeek teacher in weak mode: assertions first, then code, then run and fix; plus the fix textbook ── make_ex_demos.py docstring; PLAN 10449–10451
python3 make_ex_demos.py --weak --out sft_exw.jsonl --n 100 --variants 3 --model deepseek-chat --workers 8
python3 make_ex_demos.py --weak --out sft_exw_fix.jsonl --fix-from rlex_weak_raw.json sftex_weak_raw.json --per-prob 2 --model deepseek-chat --workers 8

# ── 3. SFT-③ ───────────────────────────────────────────────────────────────── PLAN 10461–10462 (val loss .384 -> .171)
torchrun --nproc_per_node=2 sft_qwen.py --model hf_coder_1.5b --data sft_exw.jsonl sft_exw_fix.jsonl --out out_sft_exw --lr 1e-5 --epochs 2 --batch 32 --max-len 4608
python3 export_hf.py --model hf_coder_1.5b --ckpt out_sft_exw/ckpt.pt --out hf_sft_exw

# ── 4. RL-③: weak observation, exercism + MBPP pool, dashboard v2 (20 unseen + 20 seen, temperature 1 x 4) ── PLAN 10497, 10510
torchrun --nproc_per_node=$NPROC grpo_tool_mp_vllm.py --task ex --ex-obs weak --ex-data data/exercism.json data/mbpp_weak.json --ex-mix 0.5 \
    --ex-max-timeouts 2 --ex-dash-k 4 --logp-diff --model hf_coder_1.5b \
    --init out_sft_exw/ckpt.pt --ref-init out_sft_exw/ckpt.pt \
    --steps 150 --probs 8 -k 16 --beta 0.02 --max-new 4096 --micro 1 --dyn-sample 0.5 --engine vllm --out out_exwRL_bin 2>&1 | tee exwRL_bin.txt
python3 export_hf.py --model hf_coder_1.5b --ckpt out_exwRL_bin/ckpt_latest.pt --out hf_exwRL_bin

# ── 5. Evaluation: exercism train / held-out, MBPP held-out, x32 ────────────── PLAN 10529–10530
for M in hf_sft_exw:sftexw hf_exwRL_bin:rlexw; do
    HF=${M%%:*}; TAG=${M##*:}
    CUDA_VISIBLE_DEVICES=0 python3 eval_ex.py --hf-dir $HF --model hf_coder_1.5b --obs weak -n 100 -s 8 --out ${TAG}.json
    CUDA_VISIBLE_DEVICES=0 python3 eval_ex.py --hf-dir $HF --model hf_coder_1.5b --obs weak --split heldout -n 30 -s 8 --out ${TAG}_held.json
    CUDA_VISIBLE_DEVICES=0 python3 eval_ex.py --hf-dir $HF --model hf_coder_1.5b --obs weak --data data/mbpp_weak.json --split heldout -n 100 -s 8 --out ${TAG}_mbpp.json
    for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python3 eval_ex.py --hf-dir $HF --model hf_coder_1.5b --obs weak -n 100 -s 32 --shard $i/4 --out ${TAG}_s32.json & done; wait
    python3 eval_ex.py --merge ${TAG}_s32_shard{0,1,2,3}of4_raw.json --out ${TAG}_s32.json
done
python3 pass_k.py sftexw_s32_raw.json rlexw_s32_raw.json
CUDA_VISIBLE_DEVICES=0 python3 eval_ex.py --hf-dir hf_exwRL_bin --model hf_coder_1.5b --obs weak -n 100 -s 8 \
    --inject "After a <run> you may write the expected output yourself to save calls." --out rlexw_cheat.json

# ── 6. The price of observation, as a figure, for the ②-B and ③ models ──────── bench_observation_price.py
python3 bench_observation_price.py --hf-dir hf_exRL_bin  --model hf_coder_1.5b --out bench_exRL_bin
python3 bench_observation_price.py --hf-dir hf_exwRL_bin --model hf_coder_1.5b --out bench_exwRL_bin
echo "Reconcile against the table at the top of this script (noise floor about ±.03 at 100 x 8; every number here is a single run)."
