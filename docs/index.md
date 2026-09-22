---
title: LLM from scratch, to the aha moment, to a coding agent
---

# LLM from scratch, to the aha moment, to a coding agent

**Fifteen experiments on four RTX 4090s: one variable at a time, predictions written before each run, reconciled after.**

- **[English report](en/)**
- **[中文报告](zh/)**

Code: [github.com/<user>/lab1-llm-scratch-to-aha-moment-to-coding-agent](https://github.com/<user>/lab1-llm-scratch-to-aha-moment-to-coding-agent)

---

Pretrain two English models from scratch (95M, 201M), SFT, then reinforcement learning: reproduce GRPO on GSM8K, chase DeepSeek-R1's "aha moment" on the Countdown game and lift its pass@1 from 0.02 to 0.75 (0.98 with an expert on call), build the first stateful environment (number guessing), then a four-stage coding-agent curriculum: fix bugs, bundle bugs, write real modules from scratch, and finally remove the tests so the model must verify itself. Six regularities fall out, each with its own numbers: RL reweights inside the support set; pretraining gives the parts, SFT the procedure, RL the preference; the shape of the reward decides what gets learned; the price of observation; fixing is a part, not a habit; path-opening = support density × feedback density × sampling budget.

从零预训练两个英文模型（95M、201M），做 SFT，再转到 RL：在 GSM8K 上复现 GRPO，用 Countdown 追 DeepSeek-R1 的「顿悟」并把 pass@1 从 0.02 抬到 0.75（开专家出口 0.98），做第一个有状态的环境（猜数字），最后进入四级编程 agent 课程：修 bug、拼包、从头写真实模块、拿走测试让模型自己验证。六条规律各有自己的数。
