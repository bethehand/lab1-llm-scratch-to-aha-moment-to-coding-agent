# 从零预训练到 coding agent

四张 RTX 4090 上的十五个实验：一次只变一个量，跑前写预测，跑完对账。

**报告（含全部设置、数据、图、结论、预测台账）：** https://<你的用户名>.github.io/lab1-llm-scratch-to-aha-moment-to-coding-agent/  ← 开好 Pages 后把用户名填上

## 这个仓库有什么

| 层 | 文件 | 干什么 |
|---|---|---|
| 造题 | `mutate.py` `collect_exercism.py` `collect_mbpp_weak.py` `countdown.py` `guess_env.py` | 题加标准答案，可无限造 |
| 环境 | `code_env.py` `pkg_env.py` `ex_env.py` `guess_env.py` | stops / fake_tags / step / score 四接口 |
| 沙盒和秤 | 环境内部，`strict_score.py` | 子进程隔离、五防线、隐藏测试、只奖终局 |
| harness | `harness.py`（同步和异步）`calc_tool_vllm.py`（HF / vLLM 后端） | 停、环境、注入、续；注入段 mask |
| 训练器 | `grpo_tool_mp_vllm.py` | 四进程、vLLM 同卡共存、每步灌权重、动态采样、分块 logp、KL 锚、任务分支、仪表盘 |
| SFT | `sft_qwen.py` | 只对回答算 loss，注入段 −100 |
| 老师 | `enum_traces.py` `make_*_demos.py` `deepseek_tool.py` | 程序老师和 DeepSeek 老师，秤筛后落盘 |
| 评测 | `eval_*.py` `pass_k.py` `baseline_*.py` `compare_*.py` | ×8 的 pass@1、×32 的 pass@32、行为读数、作弊探针 |
| 对话 | `chat_*_vllm.py` | 原样对话看模型行为 |
| 预训练 | `config.py` `model.py` `train.py` `prepare_data.py` `estimate.py` `evaluate.py` | 从零训 95M / 201M |
| 单测 | `tests/` | 假模型 + 真沙盒 |

脚本平铺在根目录，彼此按文件名 import，不要移动。

## 环境

两个 venv（GS01 上的实际配置，版本见 `requirements-*.txt`）：

- `vllm_env`：vLLM 0.8.5.post1、transformers 4.51.3。凡碰 vLLM 的脚本一律在这里跑：训练器、eval_*、chat_*_vllm、export_hf、tests、make_*_demos。
- `train_env`：torch 2.6、transformers 5.x。纯 HF 训练（预训练、sft_qwen.py 两个环境都能跑）。

DeepSeek 密钥只走环境变量：`export DEEPSEEK_API_KEY=...`，任何文件里都不要写。

## 复现两个最小实验

猜数字，三臂之一，约 35 分钟：

```bash
source ~/vllm_env/bin/activate
torchrun --nproc_per_node=4 grpo_tool_mp_vllm.py --task guess --init <起点 ckpt.pt> --steps 150 -k 16 --beta 0.02 --engine vllm --out out_guess
python3 export_hf.py ...            # ckpt.pt → HF 目录 hf_guess，参数见脚本 docstring
python3 eval_guess.py --hf-dir hf_guess -n 100 -s 8 --out guess_eval.json
```

阶段 ① 修 bug，约两小时：

```bash
python3 mutate.py --data data_code.json --out data/code_bugs.json
python3 make_code_demos.py --bugs data/code_bugs.json --out sft_code.jsonl
torchrun --nproc_per_node=4 sft_qwen.py --data sft_code.jsonl --out out_sft_code
torchrun --nproc_per_node=4 grpo_tool_mp_vllm.py --task code --init out_sft_code/ckpt.pt --steps 150 --engine vllm --dyn-sample 0.5 --out out_codeRL
python3 export_hf.py ...            # 同上
python3 eval_code.py --hf-dir hf_codeRL --data data/code_bugs.json -n 100 -s 8 --out code_eval.json
```

参数以各脚本顶部的 docstring 为准。

## 数据与许可

- 代码：MIT。
- 题库来源：FineWeb-Edu（ODC-By）、Alpaca（CC BY-NC 4.0）、GSM8K（MIT）、MBPP（CC-BY-4.0）、exercism/python（MIT）、Qwen2.5 系列权重按其许可。
- DeepSeek 生成的教材轨迹不随仓库发布。

## 引用

```
Qirun Li. 从零预训练到 coding agent：四张 4090 上的十五个实验. 2026.
```
