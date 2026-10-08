# Reproducing the experiments

One script per experiment. Each runs the whole chain (problems → teacher demos → SFT → RL → evaluation) and carries, at the top, the numbers recorded in the lab log (`PLAN.md`, line numbers given) so a rerun can be reconciled against them. Commands are transcribed verbatim where the log records them and reconstructed from the recorded flags where it does not; reconstructed lines are marked `RECONSTRUCTED`, and each script's `--help` is the authority on arguments.

| Script | Experiment | Report | Hardware as run |
|---|---|---|---|
| `smoke_cpu.py` | five-minute tour: Countdown grader, number-guessing environment through the harness, bug fixing in a real sandbox with a hard-coded cheat caught by the hidden test | §2.3 | laptop, no GPU |
| `guess_number.sh` | 11: number guessing, three arms (pure RL, curriculum, seeded SFT + RL) | §3.11 | 4×4090, ~35 min per arm |
| `countdown_arm_s.sh` | 9: Countdown Arm S, penalty-free reward + pool screening + KL anchor | §3.9 | 4×4090, ~3 h |
| `stage1_bugfix.sh` | 12: coding stage ①, short-horizon strong observation, bug fixing | §3.12 | 4×4090, ~2 h |
| `stage2a_bundle.sh` | 13: coding stage ②-A, long-horizon strong observation, independent bundles | §3.13 | 4×4090, ~4 h |
| `stage2b_exercism.sh` | 14: coding stage ②-B, exercism modules from scratch with tests | §3.14 | 4×4090, ~6 h incl. teacher |
| `stage3_weak_observation.sh` | 15: coding stage ③, the tests taken away; produces the price-of-observation figure | §3.15 | 4×4090, ~6 h incl. teacher |

Not scripted here: the two pretraining runs (§3.2, `train.py` with `config.py`), the Alpaca SFT 2×2 (§3.3, `sft.py`), the GSM8K GRPO rounds (§3.4, `grpo.py`), and the Countdown arms 0/1/2/3/3′/T/A (§3.5–3.10, `grpo_countdown_mp.py` / `grpo_mix_mp.py` / `grpo_tool_mp_vllm.py`); their commands are in `PLAN.md` under the dated headings the report cites.

## Before you run

- **Disk.** Checkpoints are 3 GB each and raw evaluation files are large; the log's rule is `df -h` before every run.
- **Environments.** Everything that touches vLLM runs in `vllm_env` (`requirements-vllm.txt`); `sft_qwen.py` runs in either venv. Pure-HF pretraining used a separate `train_env` (torch 2.6, transformers 5.x).
- **Teacher.** Stages ②-B and ③ use DeepSeek as the teacher; the key is read only from `DEEPSEEK_API_KEY`. The demos it generated are not distributed; the scripts regenerate them, or point `deepseek_tool.py` at any OpenAI-compatible endpoint, including a local vLLM server with a 7B model.
- **GPUs.** Scripts assume four 24 GB cards (`NPROC=4`, vLLM co-located with training). More memory lets you raise the base model; fewer cards: set `NPROC` and lower `--probs` so that problems per step divide by the card count. A 4B+ base needs LoRA on 24 GB cards or full-parameter on 80 GB cards; neither has been run here yet, see `CONTRIBUTING.md`.
- **Noise.** Every number in the report is a single run unless stated; at 100 problems × 8 samples one standard error is about .03. Run the RL step twice before reading a difference smaller than that.

## Measuring the price of observation on your own model

```bash
python3 bench_observation_price.py --hf-dir Qwen/Qwen2.5-Coder-7B-Instruct           # ~1 h on one 80 GB GPU
python3 bench_observation_price.py --hf-dir <model> --split heldout -n 30 --pass32    # the 30 never-trained problems, with the ceiling
```

It runs `eval_ex.py` twice (visible tests available / taken away), prints first-draft and final pass rates with the price, and draws the same figure as the report with the repository's 1.5B points as reference.
