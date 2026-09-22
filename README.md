# LLM from scratch → the aha moment → a coding agent

Fifteen experiments on four RTX 4090s: one variable at a time, predictions written before each run and reconciled after. Pretraining from scratch (95M / 201M), SFT, GRPO on GSM8K, chasing the R1 "aha moment" on Countdown, a stateful number-guessing environment, and a four-stage coding-agent curriculum (fix bugs → bundle bugs → write real modules → verify without tests).

**Full report** with every setting, table, figure and the prediction ledger: **[English](https://bethehand.github.io/lab1-llm-scratch-to-aha-moment-to-coding-agent/en/) · [中文](https://bethehand.github.io/lab1-llm-scratch-to-aha-moment-to-coding-agent/zh/)**　·　Chinese README: [README.zh-CN.md](README.zh-CN.md)

## Headline results

| Line | What was measured | Result |
|---|---|---|
| Pretraining | 95M on 1.5B tokens, 201M on 10B tokens (FineWeb-Edu) | perplexity 30.2 / 18.1; beats the Chinchilla formula, loses to GPT-2 small by 0.15 nats |
| GSM8K GRPO | pass@1 vs pass@64, Qwen2.5-1.5B-Instruct | pass@1 +.21, pass@64 unchanged at .980: RL reorders, it does not create |
| Countdown (4 numbers) | pass@1 lifted arm by arm on Qwen2.5-1.5B Base | .02 → .39 (RL) → .56 (seeded SFT+RL) → .71 (calculator) → .75 (subtractive scale + anchor) → .98 with an expert on call |
| Number guessing | first stateful environment, three arms | pure RL grows bisection from a 4% seed (.63); 200 seeded demos beat both RL arms (.976); N=1000 generalisation .14 vs .71 |
| Coding, four stages | SFT then RL on Qwen2.5-Coder-1.5B | RL lifts pass@1 by .10 to .18 at every stage; pass@32 ceilings barely move; on held-out problems RL cuts the tail |
| Price of observation | same problems, same hidden tests, only the visible tests removed | SFT −.09, RL −.16, ceiling −.06; the whole price lands in the fix loop after the first version |

![Countdown ladder](docs/assets/en/countdown_ladder.svg)

![GSM8K pass@k](docs/assets/en/gsm8k_passk.svg)

![The price of observation](docs/assets/en/observation_price.svg)

Six regularities recur with their own numbers: RL reweights inside the support set; pretraining gives the parts, SFT the procedure, RL the preference; the shape of the reward decides what gets learned; the price of observation; fixing is a part, not a habit; path-opening = support density × feedback density × sampling budget. All conclusions are limited to the model sizes (95M to 1.5B) and tasks used here.

## Related work

- **DeepSeek-R1** (2025): cold-start SFT followed by GRPO with verifiable rewards, and the "aha moment"; our Countdown line reproduces the cold-start necessity and the functional form of the aha moment at 1.5B.
- **Yue et al. 2025**, *Does RL really incentivize reasoning capacity beyond the base model?*: RL raises pass@1 but not large-k pass@k; we reproduce this independently on five lines at 1.5B and add the "support set is defined by the training-time sample count" refinement and the held-out tail cutting.
- **GRPO / DAPO**: group-relative advantages without a critic; dynamic sampling of mixed-success groups. Our trainer implements both; dynamic sampling bought speed and gradient utilization but no score (Conclusion ⑲).

## Install

```bash
git clone https://github.com/bethehand/lab1-llm-scratch-to-aha-moment-to-coding-agent
cd lab1-llm-scratch-to-aha-moment-to-coding-agent
python3 -m venv ~/vllm_env && source ~/vllm_env/bin/activate
pip install -r requirements-vllm.txt          # vLLM 0.8.5.post1, transformers 4.51.3, torch, bitsandbytes
export DEEPSEEK_API_KEY=...                   # only for the DeepSeek teacher and the ask-expert tool
```

Base models come from Hugging Face: `Qwen/Qwen2.5-1.5B` (Countdown line), `Qwen/Qwen2.5-1.5B-Instruct` (GSM8K), `Qwen/Qwen2.5-Coder-1.5B` (coding line). `download_qwen.py` fetches the Instruct model; `export_hf.py --out hf_base` writes a base model into the local HF-directory format the evaluators read.

**Where `--init <ckpt>` comes from.** Every RL run starts from an SFT checkpoint produced by `sft_qwen.py`, saved as `out_<name>/ckpt.pt`. The number-guessing run in the README started from the Countdown line's Arm A checkpoint. Trained checkpoints are not in this repository (they are 3 GB each); a Hugging Face release is on the to-do list, until then the SFT steps in the commands below regenerate them in minutes.

## What is in this repository

| Layer | Files | Role |
|---|---|---|
| Problem builders | `mutate.py` `collect_exercism.py` `collect_mbpp_weak.py` `countdown.py` `guess_env.py` | problems with ground truth, generated without limit |
| Environments | `code_env.py` `pkg_env.py` `ex_env.py` `guess_env.py` | the four-method interface: stops / fake_tags / step / score |
| Sandbox and scale | inside the environments, `strict_score.py` | subprocess isolation, five defenses, hidden tests, terminal-only reward |
| Harness | `harness.py` (sync and async), `calc_tool_vllm.py` (HF / vLLM backends) | stop → environment → inject → continue; injected spans are masked |
| Trainer | `grpo_tool_mp_vllm.py` | four processes, vLLM co-located on the training GPUs, weights pushed every step, dynamic sampling, chunked log-probs, KL anchor, per-task branches, dashboard |
| SFT | `sft_qwen.py` | loss on the response only; injected observations labelled −100 |
| Teachers | `enum_traces.py` `make_*_demos.py` `deepseek_tool.py` | program teacher and DeepSeek teacher; demonstrations kept only if they pass the scale |
| Evaluation | `eval_*.py` `pass_k.py` `baseline_*.py` `compare_*.py` | pass@1 at ×8, pass@32 at ×32, behavioural readouts, cheating probes |
| Chat | `chat_*_vllm.py` | raw conversations to inspect model behaviour |
| Pretraining | `config.py` `model.py` `train.py` `prepare_data.py` `estimate.py` `evaluate.py` | 95M / 201M from scratch |
| Tests | `tests/` | fake model + real sandbox |
| Lab notebook | `PLAN.md` | the raw record (Chinese, ~11,000 lines); every number in the report can be traced back to it |

Scripts sit flat in the root and import each other by file name. Do not move them.

## Environment

Two virtual environments were used on the training machine (exact versions in `requirements-*.txt`):

- `vllm_env`: vLLM 0.8.5.post1, transformers 4.51.3. Everything that touches vLLM runs here: the trainer, `eval_*`, `chat_*_vllm`, `export_hf`, `tests`, `make_*_demos`.
- `train_env`: torch 2.6, transformers 5.x. Pure HF training (pretraining; `sft_qwen.py` runs in either).

The DeepSeek key is read only from the environment variable `DEEPSEEK_API_KEY`. Never write it into a file.

## Reproducing two small experiments

Number guessing, one of the three arms, about 35 minutes on four 4090s:

```bash
source ~/vllm_env/bin/activate
torchrun --nproc_per_node=4 grpo_tool_mp_vllm.py --task guess --init <starting ckpt.pt> --steps 150 -k 16 --beta 0.02 --engine vllm --out out_guess
python3 export_hf.py --ckpt out_guess/ckpt.pt --out hf_guess      # ckpt.pt → HF directory
python3 eval_guess.py --hf-dir hf_guess -n 100 -s 8 --out guess_eval.json
```

Stage ①, bug fixing, about two hours:

```bash
python3 mutate.py --data data_code.json --out data/code_bugs.json
python3 make_code_demos.py --bugs data/code_bugs.json --out sft_code.jsonl
torchrun --nproc_per_node=4 sft_qwen.py --data sft_code.jsonl --out out_sft_code
torchrun --nproc_per_node=4 grpo_tool_mp_vllm.py --task code --init out_sft_code/ckpt.pt --steps 150 --engine vllm --dyn-sample 0.5 --out out_codeRL
python3 export_hf.py --ckpt out_codeRL/ckpt.pt --out hf_codeRL
python3 eval_code.py --hf-dir hf_codeRL --data data/code_bugs.json -n 100 -s 8 --out code_eval.json
```

The docstring at the top of each script is the authority on its arguments.

## Data and licenses

- Code: MIT.
- Problem sources: FineWeb-Edu (ODC-By), Alpaca (CC BY-NC 4.0), GSM8K (MIT), MBPP (CC-BY-4.0), exercism/python (MIT); Qwen2.5 weights under their own license.
- Teacher trajectories generated with DeepSeek are not distributed with this repository.

## Citation

```
Qirun Li. LLM from scratch, to the aha moment, to a coding agent: fifteen experiments on four RTX 4090s. 2026.
https://github.com/bethehand/lab1-llm-scratch-to-aha-moment-to-coding-agent
```
