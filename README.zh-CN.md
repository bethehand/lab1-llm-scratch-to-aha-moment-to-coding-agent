# 从零预训练到 coding agent

四张 RTX 4090 上的十五个实验：一次只变一个量，跑前写预测，跑完对账。从零预训练（95M、201M）、SFT、GSM8K 上的 GRPO、用 Countdown 追 R1 的「顿悟」、第一个有状态的环境（猜数字）、四级编程 agent 课程（修 bug → 拼包 → 从头写真实模块 → 拿走测试自己验证）。

**完整报告**，含全部设置、表格、图和预测台账：**[中文](https://bethehand.github.io/lab1-llm-scratch-to-aha-moment-to-coding-agent/zh/) · [English](https://bethehand.github.io/lab1-llm-scratch-to-aha-moment-to-coding-agent/en/)**　·　English README: [README.md](README.md)

## 主要结果

| 线 | 量的是什么 | 结果 |
|---|---|---|
| 预训练 | 95M 配 1.5B token，201M 配 10B token，FineWeb-Edu | 困惑度 30.2 / 18.1；跑赢 Chinchilla 公式，输 GPT-2 small 0.15 |
| GSM8K GRPO | pass@1 对 pass@64，Qwen2.5-1.5B-Instruct | pass@1 +.21，pass@64 钉在 .980：RL 只重排不创造 |
| Countdown 4 数 | pass@1 一臂一臂抬，Qwen2.5-1.5B Base | .02 → .39（RL）→ .56（撒种 SFT+RL）→ .71（计算器）→ .75（秤做减法加锚）→ 开专家出口 .98 |
| 猜数字 | 第一个有状态的环境，三臂 | 纯 RL 从 4% 的种子长出二分（.63）；200 条撒种示范超过两条 RL 臂（.976）；N=1000 泛化 .14 对 .71 |
| 编程四级 | Qwen2.5-Coder-1.5B 上先 SFT 再 RL | 每级 RL 抬 pass@1 .10 到 .18；pass@32 天花板几乎不动；留出题上 RL 削尾巴 |
| 观测的价格 | 同题同隐藏测试，只拿走可见测试 | SFT −.09、RL −.16、天花板 −.06；价全落在首版之后的改循环 |

六条规律反复出现，各有自己的数：RL 在支撑集内挪；预训练给零件、SFT 给流程、RL 给偏好；秤的形状决定学到什么；观测的价格；改是零件不是习惯；开路 = 支撑集密度 × 反馈密度 × 采样预算。所有结论限定在这里用到的模型规模（95M 到 1.5B）和任务上。

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
| 单测 | `tests/` | 假模型加真沙盒 |
| 实验日志 | `PLAN.md` | 原始记录，约一万一千行；报告里每个数都能在里面查到 |

脚本平铺在根目录，彼此按文件名 import，不要移动。

## 环境

训练机上用两个 venv，版本见 `requirements-*.txt`：

- `vllm_env`：vLLM 0.8.5.post1、transformers 4.51.3。凡碰 vLLM 的脚本一律在这里跑：训练器、`eval_*`、`chat_*_vllm`、`export_hf`、`tests`、`make_*_demos`。
- `train_env`：torch 2.6、transformers 5.x。纯 HF 训练（预训练；`sft_qwen.py` 两边都能跑）。

DeepSeek 密钥只从环境变量 `DEEPSEEK_API_KEY` 读，任何文件里都不要写。

## 复现两个最小实验

猜数字，三臂之一，四卡约 35 分钟：

```bash
source ~/vllm_env/bin/activate
torchrun --nproc_per_node=4 grpo_tool_mp_vllm.py --task guess --init <起点 ckpt.pt> --steps 150 -k 16 --beta 0.02 --engine vllm --out out_guess
python3 export_hf.py --ckpt out_guess/ckpt.pt --out hf_guess      # ckpt.pt → HF 目录
python3 eval_guess.py --hf-dir hf_guess -n 100 -s 8 --out guess_eval.json
```

阶段 ① 修 bug，约两小时：

```bash
python3 mutate.py --data data_code.json --out data/code_bugs.json
python3 make_code_demos.py --bugs data/code_bugs.json --out sft_code.jsonl
torchrun --nproc_per_node=4 sft_qwen.py --data sft_code.jsonl --out out_sft_code
torchrun --nproc_per_node=4 grpo_tool_mp_vllm.py --task code --init out_sft_code/ckpt.pt --steps 150 --engine vllm --dyn-sample 0.5 --out out_codeRL
python3 export_hf.py --ckpt out_codeRL/ckpt.pt --out hf_codeRL
python3 eval_code.py --hf-dir hf_codeRL --data data/code_bugs.json -n 100 -s 8 --out code_eval.json
```

参数以各脚本顶部的 docstring 为准。

## 数据与许可

- 代码：MIT。
- 题库来源：FineWeb-Edu（ODC-By）、Alpaca（CC BY-NC 4.0）、GSM8K（MIT）、MBPP（CC-BY-4.0）、exercism/python（MIT）；Qwen2.5 权重按其许可。
- DeepSeek 生成的教材轨迹不随仓库发布。

## 引用

```
Qirun Li. LLM from scratch, to the aha moment, to a coding agent: fifteen experiments on four RTX 4090s. 2026.
https://github.com/bethehand/lab1-llm-scratch-to-aha-moment-to-coding-agent
```
