# 从零训练一个英文 LLM — 决策记录

> 目标：在 4×RTX 4090 上从零训练一个英文语言模型，完整走通
> 数据 → 分词 → 训练 → 评估 → 推理 全流程。

## 流程总览

| 步骤 | 内容 | 状态 |
|---|---|---|
| 0 | 定算力预算 | ✅ 已定稿 |
| 1 | 定分词器 | ✅ 已定稿 |
| 2 | 定模型配置 | ✅ 已定稿 |
| 3 | 准备数据（下载 → 分词 → 打包 .bin） | ✅ 已完成 |
| 4 | 写训练代码 | ✅ model.py + train.py |
| 5 | 三级验证 | ✅ 5a / 5b / 5c 全部通过 |
| 6 | 正式训练 + 监控 | ✅ 已完成 |
| 7 | 评估 | ✅ 已完成 |
| 8 | 部署推理 | 🔄 进行中 |

---

## 第 0 步：算力预算 ✅ 定稿

### 硬件
```
4 × NVIDIA RTX 4090      24 GB × 4 = 96 GB 显存
驱动 550.144.03 / CUDA 12.4
拓扑  全 NODE、完全对称、同 NUMA node 1
PCIe  4.0 x16 满配（负载下实测 gen=4, width=16）
```

### 实测的通信带宽
```
4 卡 all-reduce busbw = 3.7 GB/s
原因：GeForce 卡 P2P 被驱动禁用 + 四卡共享 CPU 根复合体
对策：grad_accum = 16，把通信开销从 40% 压到 2.5%
结论：不影响可行性，不去打 P2P 补丁（只能省 2.4 小时/7天）
```

### 算力预算
```
峰值      165 TFLOPS × 4 = 660 TFLOPS (BF16 稠密)
MFU       0.3   ← 估计值，待第 5b 步实测修正
有效      198 TFLOPS

时间预算   1.5 小时 = 5,400 秒
算力预算   198e12 × 5400 = 1.07 × 10^18 FLOPs
```

### 由预算反推的规模（Chinchilla, D = 20N）
```
N = sqrt(1.07e18 / 120) = 9.4e7  →  约 94M 参数
D = 20 × N              = 1.9e9  →  约 1.9B token
```

参照：这个预算约等于 GPT-2 (124M) 完整训练量的 1/6。

### 待办
- [ ] 第 5b 步实测真实 MFU，回来修正 N 和 D
- [ ] 训练前执行：`sudo nvidia-smi -pm 1` 和 `sudo nvidia-smi -pl 350`
- [ ] 启动命令必须绑核：`numactl --cpunodebind=1 --membind=1 torchrun ...`

---

## 第 1 步：分词器 ✅ 定稿

### 决定：用 GPT-2 现成的 byte-level BPE

```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")
vocab_size = 50257
```

### 实测确认的结构
```
50257 = 256（字节底座） + 50000（BPE 合并） + 1（<|endoftext|>）
id 0~255   全是单字节，覆盖全部 256 个字节值 → 永不出现 <unk>
id 256 起  按合并顺序排列：' t', ' a', 'he', 'in', 're', 'on', ' the', ...
65.9% 的 token 以空格开头（pre-tokenization 把空格粘在词前）
token 长度分布峰值在 4~5 字节
```

### 实测压缩率（同一分词器，文本类型差 4 倍）
```
英文短句      4.40 字节/token
学术散文      6.25
Python 代码   1.75
中文          1.50    ← 不可用，但我们训英文，无影响
```
**✅ 已在第 3 步实测**：FineWeb-Edu 真实语料 = **4.62 字节/token**
（我编的测试句量出 6.07，偏高不可信；4.62 正中 GPT-2 常引的 ~4.4 附近）

### ✅ 在 GS01 上的验收结果（全部通过）
```
✅ tiktoken 加载成功  n_vocab=50257  eot=50256
✅ uint16 存储可行 (50257 < 65536)
✅ 往返一致性 8/8（含制表符、多空格、Unicode、emoji、转义符）
✅ 200 条随机字节文本往返全部一致 → byte-level 无 <unk> 实证
✅ 文档分隔编码正确：[32, 50256, 33]
```
验证脚本：`~/check_tokenizer.py`（在 GS01 上）

### 文档分隔方案（已定）
每篇文档结尾追加 `<|endoftext|>` (50256)，所有文档首尾相连成一条长序列。
- 好处：模型学会「文档结束」→ 推理时能自己停；零 padding 浪费
- 代价：一个窗口可能横跨两篇文档 —— 第一轮不做 document masking
```python
ids = []
for doc in documents:
    ids.extend(enc.encode(doc))
    ids.append(50256)
```

### 数据量估算（供第 3 步）
```
要下载文本    1.9B token × ~4.5 字节  ≈  8~11 GB
分词后 .bin   1.9B token × 2 字节     =  3.8 GB
磁盘准备      约 20 GB
```

### ⚠️ 已知代价（第二轮要改）
```
94M 模型 + C=768 + 绑权重:
  GPT-2 50257  →  词表参数 38.6M = 41% ⚠️  只能堆 8 层
  自训   16000  →  词表参数 12.3M = 13% ✅  能堆 12 层
```
第一轮接受这个代价（换取零工作量、零 bug 风险）。
第二轮自训一个 16k 词表，同预算下深度从 8 层提到 12 层。

### 实测复现的病态案例（L1 讲过的）
```
数字   '1234'→['12','34']  '1235'→['12','35']   切法不一致
缩进   8 个空格 = 8 个 token，代码压缩率只有 1.75
中文   4 个汉字 → 8 个乱码 token
拼写   'strawberry'→['st','raw','berry']  模型看不到单个字母
```

---

## 第 2 步：模型配置 ✅ 定稿

配置文件：`config.py`　　验收工具：`estimate.py`

### 三个拍板的决定
```
T = 1024        （不用 512：只快 10%，但上下文砍半、没法跟 GPT-2 对比）
接受 1.9 小时    （不砍 D：只省 24 分钟，不值得把 D/N 从 20 降到 16）
C=768 / 8 层     （不用 640/13：深宽比 96 更健康，且跟 GPT-2 同宽便于对比）
```

### 最终配置
```
vocab_size = 50257    C = 768        n_layer = 8
n_head = 12           head_dim = 64  d_ff = 2048
block_size = 1024     tie_embeddings = True
→ 95,233,536 参数

batch_size_per_gpu = 16    grad_accum = 8    world_size = 4
grad_checkpointing = False        ← 显存够，不开，省 33% 算力
→ 524,288 token/步，共 3,623 步

lr = 6e-4 → 6e-5 (cosine)   betas = (0.9, 0.95)   wd = 0.1
grad_clip = 1.0             warmup = 200 步
```

### estimate.py 的核算结果
```
【参数量】
  每层 attention (4C²)     2,359,296
  每层 MLP (3·C·d_ff)      4,718,592
  每层 RMSNorm                 1,536
  × 8 层                  56,635,392   59.5%
  词表 (绑权重)            38,597,376   40.5%  ⚠️ 大词表的代价
  总计                    95,233,536   = 95.2M

【单卡显存】
  ① 参数+梯度+Adam         1.42 GB
  ② 激活值                 3.00 GB
  ③ logits + 梯度          3.07 GB   ← B×T×vocab，大词表的第二笔账
  ④ CUDA context/碎片      1.50 GB
  合计                     8.99 GB / 24 GB (37%)  ✅

【算力】
  6ND                      1.086e18
  + attention T/(6C)       +22.2%     ← 6ND 从不包含这一项
  真实总量                 1.327e18
  预计时长                 1.86 小时
```

### ⚠️ 关于 `6ND` 的修正
`6ND` 只数跟参数有关的矩阵乘，**不含 attention 的 QKᵀ 和 AV**。
正确公式：`真实 FLOPs = 6ND × (1 + T/(6C))`
本配置下这一项是 **+22%**，不能忽略（C 越小占比越大）。

### 跑起来后必须对照的四个数
```
★ 初始 loss   ln(50257) = 10.825
★ 显存        约 9.0 GB
★ 每步耗时    约 1.85 秒
★ MFU         实测后回填 config.assumed_mfu，重跑 estimate.py 修正预算
```

---

## 第 3 步：准备数据 ✅ 已完成

脚本：`download_data.py`（只下载）→ `prepare_data.py`（分词+打包+验收）

### 数据源
```
FineWeb-Edu  sample/10BT  共 14 个 parquet，26.6 GB ≈ 100 亿 token
每个 2GB 文件 ≈ 7.5 亿 token
```

### ⚠️ 第一轮精简：3 个文件 → 2 个
下载实测只有 **0.85 MB/s**（国内直连 HF；hf-mirror 镜像不通已验证），
6.5 GB 要 2.2 小时。决定砍掉一个 train 文件：

```
                 原计划      精简版
parquet 文件      3+val      2+val
下载              6.5 GB     4.5 GB      省 40 分钟
train_tokens      1.9B       1.5B
D / N             20.0       15.8        L9：U 形曲线谷底很平，损失极小
总步数            3,623      2,861
训练时长          1.86 h     1.47 h      省 24 分钟
```
第三个文件留到第二轮再下。

### 文件分配（物理隔离，保证零污染）
```
train  ←  sample/10BT/000, 001     取前 15 亿 token  →  train.bin  3.0 GB
val    ←  sample/10BT/013          取前 500 万 token →  val.bin     10 MB
                   ↑
        用完全不同的 parquet 文件做验证集，比事后去重更彻底
```

### 输出格式
```
uint16 的裸二进制，无任何结构。19 亿个整数首尾相连。
文档之间插 <|endoftext|> (50256) 作分隔。
训练时用 np.memmap 随机切窗口，内存开销接近零。
```

### ✅ 实际产出
```
train.bin   1,500,000,000 token   2.79 GB   1,452,000 篇文档
val.bin         5,000,000 token   0.01 GB       6,000 篇文档

★ 真实压缩率  train 4.62 字节/token   （已回填第 1 步）
              val   5.69 字节/token

平均文档长度  1,033 token/篇        （合理区间 600~1500）
分词速度      3.5 M token/秒，7.1 分钟跑完
```

### ✅ 验收结果
```
✅ token 数正好打满 1.5B（两个 parquet 刚好够）
✅ 最大 token id = 50256 < 50257，无越界
✅ 前 500 万 token 内有 4,791 个 <|endoftext|> 分隔符
✅ 随机 5 段样本全是通顺的教育性英文：
     医学文献引用 / 鸟类学 / 德雷福斯事件 / 考古遗址 / 马其顿腓力二世
   → FineWeb-Edu 的教育性分类器确实在起作用
```

### ⚠️ 已知问题（不阻塞，第 7 步再处理）
```
val 压缩率 5.69 明显高于 train 4.62（+23%）
原因：val 只取了 file 013 的「前 6,000 篇」，parquet 里文档按爬取批次分组，
      前 6000 篇可能来自同一来源，代表性不足。

影响：
  监控训练健康（对比不同 step 的 val loss）  → 无影响，两次同一份数据
  第 7 步跟 GPT-2 对比困惑度                  → 有影响，届时另做随机采样的评测集
```

### 踩过的坑
```
hf-mirror.com 镜像不通（LocalEntryNotFoundError，5 秒即失败）
hf_transfer 对 Xet 存储后端无效（日志里的 "reconstructing file" 即 Xet）
2>&1 | tee 会让 Python 缓冲输出 + 吃掉 tqdm 进度条 → 直接跑，别 tee
```

---

## 第 4 步：写训练代码 ✅ 已完成

文件：`model.py`（模型定义）+ `train.py`（训练循环）+ `inspect_bin.py`（数据检查工具）

### model.py — 把 L3 写成代码
```
RMSNorm      不减均值、无 bias；在 fp32 里算平方和避免 bf16 掉精度
SwiGLU       3 个矩阵 × d_ff(8C/3) = 8C²，跟标准 FFN 参数量相等
RoPE         预计算 cos/sin，只作用在 q/k 上，不进残差流
Attention    F.scaled_dot_product_attention(is_causal=True) → 走 FlashAttention
             因果掩码是「并行算 T 个位置」的代价，microgpt 一次一个 token 不需要
pre-norm     x = x + attn(norm(x))，残差通路上零操作
权重绑定      lm_head.weight = wte.weight，省 38.6M
残差出口初始化 wo / w_down 用 std=0.02/sqrt(2·n_layer)，让 block 初始接近恒等
```

### train.py
```
DDP + 梯度累积（只在最后一个 micro-step 同步，压通信开销）
cosine lr + warmup 200 步
bf16 autocast + torch.compile
checkpoint 保存 / 断点续训
定期 eval + 定期采样（肉眼看它在不在学）
MFU 实测（含 attention 的 T/(6C) 项）
--smoke 模式：玩具配置 + 随机数据，不依赖 GPU 和数据
```

### 修过的 bug
```
--steps 与 world_size 的顺序依赖：--steps 换算成 token 时 world_size 还是默认的 4，
导致单卡跑 --steps 20 实际跑了 80 步。修法：先从环境变量读 WORLD_SIZE 再处理 --steps。
```

---

## 第 5 步：三级验证 🔄

### ✅ 5a · 代码通不通（GS01 已验证）
```
python3 model.py
python3 train.py --smoke --steps 50
```
```
✅ 参数量 95,233,536       与 estimate.py 完全一致
✅ 初始 loss 10.9492       ln(50257)=10.8249，偏差 0.12
✅ RoPE 自检 80.060669 = 80.060669   精确相等
✅ 74 个参数张量拿到梯度
✅ 设备 cuda × 1，dtype torch.bfloat16   ← CUDA 和 bf16 都生效
✅ loss 全程 10.85 不降    ← 随机数据，这是正确行为
✅ checkpoint 写出

MFU 2% —— 玩具配置太小（每步仅 1024 token），GPU 跑不满，无参考价值
```

### ✅ 5b · 单卡 + 真配置 + 真数据（100 步）
```
loss   10.9589 → 6.4195     （val 6.3732）
MFU    67.2%                ← 远超估算的 30%
显存   9.7 GB               vs estimate.py 预测 9.0 GB（差 8%）
每步   826 ms
torch.compile ✅ 编译通过（首次约 2 分钟）
```
生成样本（step 100）：已是真实英文单词 + 正确空格标点，但无语法。

### ✅ 5c · 四卡 DDP（200 步）
```
loss   10.9530 → 5.2885     （val 5.2158）
MFU    54.7%                ← 比单卡掉 18.6%，就是通信代价
每步   1017 ms（batch 大 4 倍）
加速比 3.25×                 并行效率 81%
显存   9.9 GB
```

**通信开销的精确来源：**
```
模型参数是 fp32（autocast 只影响前向计算），所以梯度也是 fp32
  95.2M × 4 字节           = 381 MB
  ring all-reduce 搬运      = 2 × 3/4 × 381 = 571 MB
  ÷ 实测 3.7 GB/s          = 154 ms
  154 ÷ 826（计算）        = 18.6%     ← 与 MFU 跌幅完全吻合
```
（我先前按 bf16 梯度估的 5% 是错的。）

**断点续训（连续中断两次）：**
```
step 27  中断 → 续训 loss 从 8.49 接着降  ✅ 不是从 10.82 重来
step 123 中断 → 续训 loss 从 5.83 接着降  ✅

对照：
              step199   final train   final val
一口气跑完     5.2885    5.2431        5.2158
中断两次       5.2683    5.2529        5.2567
              差异全在随机波动内 → 权重/动量/lr 调度全部正确恢复
```

### 🐛 5c 抓到的真 bug（已修）
```
问题：checkpoint 只在 eval 时、且只在 val 改善时才存
      → --eval-every 250 的正式训练，崩在第 240 步会丢掉全部 4 分钟
修复：
  ① --save-every（默认 100）每 N 步无条件存 ckpt_latest.pt
  ② SIGINT 处理器：Ctrl-C 后本步跑完存盘再退，四个 rank 用 all_reduce 对齐停止时机（防死锁）
  ③ --resume 优先载入 ckpt_latest.pt
  ④ 抽出 save_ckpt()，消除三处重复
  ⑤ destroy_process_group() 保证执行

三种 checkpoint：
  ckpt_latest.pt  每 100 步无条件存   ← 崩溃恢复
  ckpt.pt         val 最好的那次      ← 最终要用的
  ckpt_final.pt   跑完全程的最后一个
```

### 代码里避开的两个经典坑
```
DDP 包装后存盘   → key 多 "module." 前缀
compile 后存盘   → key 多 "_orig_mod." 前缀
解法：全程保留未包装的 raw_model 引用，存盘用 raw_model.state_dict()
```

### ★ 预算修正（回填 config.assumed_mfu = 0.547）
```
             第 0 步估算   第 5 步实测
单卡 MFU       0.30         0.672
四卡 MFU       0.30         0.547
每步耗时        1.85 s       1.015 s
总时长         1.47 小时     48.6 分钟

estimate.py 回填后预测每步 1.01 秒，实测 1.015 秒 —— 核算模型已校准
```

### 硬件设置（正式训练前已完成）
```
sudo nvidia-smi -pm 1      持久化模式
sudo nvidia-smi -pl 350    限功耗（4×450W 满载 49 分钟，限功耗更稳）
numactl --cpunodebind=1 --membind=1    绑到 GPU 所在的 NUMA node 1
```

---

## 第 6 步：正式训练 🔄

### 方案 A（已选定）
```
1.5B token / 2,861 步 / 约 48 分钟 / 1 个 epoch / D-N 比 15.8
理由：快反馈优先。跑完看结果再决定要不要加练。
     加练要重开一轮完整的 cosine，不能简单 --resume 拉长 max_steps
     （cosine 已衰减到底，续训等于用极小学习率硬跑）
```

### 启动命令
```
tmux new -s train
numactl --cpunodebind=1 --membind=1 \
  torchrun --nproc_per_node=4 train.py \
    --save-every 100 --eval-every 250 --sample-every 500 --out out_run1
```

### ✅ 训练结果
```
2,861 步跑完，约 48 分钟，全程无异常
最终  train 3.4050   val 3.4024
产出  out_run1/{ckpt.pt, ckpt_latest.pt, ckpt_final.pt}  各 1.1 GB
```

checkpoint 大小反验证了 L2 的显存账：
```
参数 fp32   95,233,536 × 4 = 381 MB
Adam m                     = 381 MB
Adam v                     = 381 MB
                             ───────
                             1.14 GB  ✅（梯度不存，所以是 12 字节/参数）
```

loss 轨迹：
```
step     0   10.83   ← ln(50257)
step   500    4.18
step  1000    3.75
step  2000    3.49
step  2861    3.40   ← 最后 100 步在 3.40±0.04 震荡，已收敛
```

全程实测 vs 第 0 步估算：
```
              估算        实测
MFU(四卡)     0.30       0.547
每步          1.85 s     1.015 s
总时长        1.47 h     48 分钟
显存          9.0 GB     9.9 GB
```

---

## 第 7 步：评估 ✅ 已完成

脚本：`evaluate.py`　　监控：`monitor.py`

### ① 完整验证集（500 万 token 全扫，17 秒）
```
val loss    3.4068
困惑度       30.17
信息量       4.915 bits/token

对照训练时的 6.5% 抽样估计 3.4024 —— 只差 0.004，抽样是准的
```

### ② 对比 L9 的 Chinchilla 公式
```
L = 1.69 + 406.4/N^0.34 + 410.7/D^0.28
  = 1.69 + 0.787 + 1.107 = 3.5845

公式预测  3.5845
实测      3.4068     跑赢 0.178
```
原因：Chinchilla 常数在 MassiveText 上拟合；FineWeb-Edu 经教育性筛选，
信息密度更高。**印证 L11：缩放律的形式通用，常数依赖数据。**

参照：GPT-2 small (124M, 9B token) 在 WebText 上困惑度约 29。
本模型用 3/4 参数、1/6 数据达到同档水平。

### ★ 核心发现：语言能力 ✅ / 事实知识 ❌
```
❌ "The capital of France is Egypt."
❌ "Water boils at a temperature of almost 70 degrees Celsius."
❌ "In 1969, humans first discovered the earliest known human form in the Americas."

✅ 但每一句语法、文体、术语搭配都正确：
   "After a brief stint as a paleontologist at UC Berkeley ... during the Miocene"
   —— 知道 paleontologist/fossils/Miocene 是一伙的，知道学术写作长什么样
```

原因（两个变量都不够）：
```
D  1.5B token 里，某个具体事实只出现几十次 —— 低于记忆阈值
   （Llama 3 的 15T 是这里的 10,000 倍）
N  知识容量约 2 bits/参数（Allen-Zhu & Li 2024）
   95.2M × 2bit = 23.8 MB，扣掉 40.5% 词表后有效约 14 MB
   英文维基百科纯文本约 20 GB → 本模型装得下 0.07%
   Llama 3 70B: 70e9 × 2bit = 17.5 GB ≈ 整个维基百科
```
结论：**1.5B token 足够学会「怎么说」，远不够学会「说什么」。**

### ⑤ 跟 GPT-2 small 正面对比（同一份 val 数据）
```
你的模型     95.2M 参数, 1.5B token   →  loss 3.4068   困惑度 30.17
GPT-2 small 124.4M 参数, ~9B token   →  loss 3.2587   困惑度 26.02
                                        GPT-2 赢 0.148
```

**★ 用 L9 公式拆解这 0.148：**
```
                   公式预测    实测      偏差
  本模型            3.5845    3.4068   −0.178   跑赢公式
  GPT-2            3.0780    3.2587   +0.181   跑输公式
                                       ───────
  预测差距 0.5065 − 0.359 = 0.148 = 实测差距   ✅ 完美对上
```

两个偏差各有明确来源，而且几乎相等：
```
本模型 −0.178   FineWeb-Edu 的数据质量红利（教育性筛选，信息密度高于
                Chinchilla 拟合用的 MassiveText）← 这是自己测出的数字
GPT-2  +0.181   分布偏移惩罚（它在 WebText 上训练，跨到 FineWeb-Edu 上评估）
```

**反转：主场优势没想象中大。** GPT-2 论文报它在自己的 WebText 上困惑度约 29，
这里在 FineWeb-Edu 上是 26.02 —— 更低。说明 FineWeb-Edu 对任何模型都更好预测
（干净的教育性文本 > 原始网页）。所以不是"你占主场便宜"，是"数据更干净，
两边都受益，你受益更多"。

**结论**：GPT-2 在客场（+0.18 惩罚）还赢 0.148，绝对水平确实高一档 —— 
合理，它参数多 31%、数据多 6 倍。差距 0.148 < 0.3 的判据，训练成功。

### ✅ 学会了文档边界
生成中会自己吐出 `<|endoftext|>` 然后开新文档 —— 第 1 步那个
「每篇文档后追加 EOT」的设计生效了，推理时模型能自己停。

### 🐛 中途发现并修复：重复循环
```
现象  step 2000/2500 的样本严重重复（"the person who lives" ×6）
误判  我在 step 500 时说「重复会随训练减少」—— 错了，反而更固化
真因  这是解码问题不是训练问题。模型越强，对「延续已建立模式」
      的置信度越高，越容易锁死（Holtzman 2019 的经典发现）
修复  model.py 的 generate() 加入 top-p（nucleus）+ repetition_penalty
      架构一行没动，checkpoint 完全兼容
```

### 生成质量的演化轨迹
```
step  500   短句正确 + 学会列表格式    "The origin of life is unknown."
step 1000   段落 + 主题连贯 + 从句嵌套 "The soul has no body."
step 1500   抽象论述（开始循环）
step 2000   重复循环严重
step 2500   重复循环
最终        连贯、分段、无重复、会虚构具名引用
```
中段反而比两头差 —— 那是「对局部模式置信度快速上升、但还不会换话题」的阶段。

---

## 第 8 步：部署推理 🔄 进行中

脚本：`chat.py`（交互式续写 + 速度拆解）

### 给 model.py 加了 KV cache
```
Attention.forward(x, cos, sin, cache=None, pos=0)
  cache 给了就走缓存路径；训练时 cache=None，行为完全不变
  RoPE 用绝对位置 pos，带 cache 解码时才转得对
  掩码规则：q 和 k 一样长 = prefill/训练 → 遮未来
            q 比 k 短     = 带 cache 解码 → 不遮

回归测试：参数量 95,233,536 不变、初始 loss 10.9492 不变、冒烟训练正常  ✅
正确性：  带 cache 与不带 cache 逐 token 结果完全一致（贪心和采样都验了）  ✅
```

### ★ 实测结论：GPU 上 KV cache 几乎没用（反直觉）
```
本地 CPU   128 token 生成    3.0× 加速
GS01 GPU   200 token 生成    157.4 vs 153.2 tok/s —— 量不出差别
```

原因：**瓶颈不是计算量，是 kernel 启动开销**
```
带 cache 每步:   2×95.2M×1   = 1.9e8 FLOPs ÷165TFLOPS = 0.001 ms
不带 cache 每步: 2×95.2M×105 = 2.0e10 FLOPs           = 0.12  ms
实测每步                                                 6.35  ms
                                                        ────────
                              真正在计算的比例  0.02% ~ 1.8%
```
时间花在：8 层 × 约 12 个 kernel = 约 100 次 kernel 启动（每次 20~50μs）
        + 对 50257 个 logits 的全排序（top_p）+ Python 循环

**KV cache 要 T > 5,500 才能在这个模型上压过开销，而 block_size 只有 1024。**
它对大模型和长序列才关键。

CPU 上有用是因为 CPU 算力慢 3000 倍、没有 kernel 启动开销 —— **瓶颈换了。**

### 实测速度（RTX 4090, fp32, batch=1）
```
prefill  15 token   6 ms    2,406 tok/s   ← 但只用了 0.3% 算力，prompt 太短
decode  120 token 747 ms      160.6 tok/s ← 显存带宽理论上限 2,646 tok/s，利用率 6%
KV cache 每 token 3,072 字节 → 满 1024 上下文 3.2 MB
```

### 🐛 踩到的坑
```
① 温度设成 2.0 忘了调回来，输出全是乱码 —— 排查了半天发现是解码参数
   再次印证：生成质量差先怀疑解码，再怀疑模型
② repetition_penalty 用 torch.unique：输出大小依赖数据 → 每步强制 GPU↔CPU 同步
   已改成增量维护的布尔掩码（但实测提升不明显，说明它不是主要瓶颈）
③ chat.py 没有预热：第一次前向 286ms（含 cuDNN 算法选择 + kernel JIT），
   后续降到个位数毫秒。测性能必须先跑几次丢掉 —— 待修
```

### 正常温度下的真实输出（T=0.8）
prompt: `Photosynthesis is the process by which`
```
...bacteria in the flowers grow on top of the plants. Flowers become small and
produce more than one plant within a year...
Researchers believe that by changing the way light interacts with the environment
(such as turning off lights), we can encourage more living creatures to be born...
In this article, we will discuss how you can help make such a difference for your
children's garden.
What are some common questions that your kids may ask about growing vegetables?
For starters, it is essential to note that...
```
```
形式  ✅ 三段成文、段间过渡、括号插入语、"In this article we will discuss"
      这类文章结构标记、文体高度一致（亲子园艺博客口吻）、无重复
内容  ❌ "细菌进行光合作用"、"关灯促进植物生长"、主题从光合作用漂到园艺
```

**★ 更值得注意：它在「续写博客」而不是「回答问题」。**
给它一个科普开头，它按网页文章的套路往下写（引入 → 伪引用 → FAQ 小标题）。
这是缺第 ③ 层（行为/格式）的活标本 —— 做完 SFT 它才会变成「问什么答什么」，
但依然会说细菌进行光合作用（知识不会因 SFT 变对）。

---

# 待办清单

## A. 第 8 步收尾（现在）
- [ ] `/set temp 0.8` 后重新生成 —— **还没见过这个模型在正常温度下的真实水平**
- [ ] `python3 chat.py --compile` 测 torch.compile 能否压下 kernel 开销
- [x] ~~跟 GPT-2 small 对比~~ ✅ 已完成，差距 0.148，见第 7 步 ⑤

## B. 推理优化（有兴趣再做）
- [ ] CUDA Graph：把整个解码步录成一张图，彻底消除启动开销
- [ ] 批处理：batch=1 时 GPU 有 99.98% 算力闲置；batch=32 总吞吐可到约 4,800 tok/s
      （这就是 vLLM 的 continuous batching 在做的事）
- [ ] 量化 int8/int4：权重减小 → decode 的带宽上限进一步提高

## C. 第二轮训练
- [ ] **换 16k 自训词表**：词表参数占比 40.5% → 13%，同预算 8 层 → 12 层（收益最大）
- [ ] eval 加 `dist.all_reduce`：现在只用 rank 0 的 20 个 batch，浪费 3/4 算力
- [ ] bf16 梯度压缩：MFU 54.7% → 约 61%
- [ ] 下第三个 parquet，token 从 1.5B → 2.26B
- [ ] 或者超标训练：2 epoch × 1.5B = 3B token-visit（4 遍法则内），D/N 从 15.8 → 31.5

## D. 数据侧
- [ ] 做一个**随机采样**的 val 集（现在的 val 只取了 file 013 的前 6,000 篇，
      代表性不足，跟 GPT-2 对比时不公平）

## E. CS336 课程（还剩 6 讲没碰）
- [ ] L12 评测
- [ ] L15 SFT/RLHF
- [ ] L5 GPU / L6 Kernel / L7-L8 并行（系统四讲，按约定放最后）
- 部分覆盖：L4 MoE、L10 推理、L13-14 数据、L16-17 对齐

---

# 下一步：四个选项（2026-08-28 记）

## A. 给 200M 模型做 SFT　　✅ 已完成 2026-08-30（实际 23.7 分钟，见文末「第三阶段：SFT」）
```
数据    Alpaca / Dolly / OpenHermes 的子集，几万条
成本    6 × 201M × 50M token = 6.0e16 FLOPs ÷ 392T = 2.6 分钟
效果    从「只会续写网页」变成「问什么答什么」
价值    亲手验证第 ③ 层（行为/格式）几千条就能激活
        —— 但它依然会说「法国首都是协和广场」，知识不会因 SFT 变对
```

## B. 跑 GRPO 复现 R1-Zero 的顿悟时刻　　✅ 第一次已完成 2026-09-01（见文末「第四阶段：GRPO / RLVR」）
```
模型    Qwen2.5-0.5B 或 1.5B（下现成的）
任务    Countdown 数字游戏 或 GSM8K
方法    GRPO + 可验证奖励（程序判对错，不需要奖励模型）
参考    TinyZero（Berkeley 复现，约 $30 算力）
成本    每 RL step ≈ 15 分钟（rollout 14 分 + 训练 1.5 分），20 步约 5 小时
价值    亲眼看到模型自己学会「等一下，我需要重新检查」的自我验证行为
        把 L16/L17 从理论变成实物
关键    GRPO 无 value 网络（PPO 要 4 份模型，GRPO 只要 2 份）
        配合 LoRA 的 reference 妙用（关掉 adapter 即参考模型）→ 实际只 1 份
```

## C. 第三轮训练　　时间待定
```
目标    下满 14 个 parquet（26.6 GB），做到 1.0 epoch
理由    把重复数据的 0.12 loss 代价拿回来
待办    还要再下 6 个文件（约 12 GB，按 0.85MB/s 需 4 小时）
其他可选改动：
  - 换自训 16k 词表（词表占比 22.4% → 7%，同预算能堆更多层）
    ⚠️ 但换词表后 loss 跟前两轮不可比，会失去唯一的纵向度量
  - 上更大模型：1B 是 DDP 的硬上限（需梯度检查点 + B=8），约一周
```

## D. 补 CS336 剩下的 6 讲
```
L12 评测        —— 最相关：本项目已多次踩到「跨数据集比困惑度不公平」的坑
L15 SFT/RLHF    —— 做 A 之前该看
L5~L8 系统四讲   —— GPU / Kernel / 并行，按约定放最后
```

## 硬件能力边界（2026-08-28 算清）
```
训练  DDP 硬上限约 1B（开梯度检查点 + B=8），需约一周
      2B 以上要 FSDP，但 PCIe 只有 3.7 GB/s，慢到不能忍
推理  bf16 约 40B / int8 约 80B / int4 约 150B
      ✅ Llama 3 70B (int4, 35GB) 能跑
      ❌ DeepSeek-V3 671B (int4, 336GB) / Kimi K3 2.8T (MXFP4, 1.4TB) 装不下
      注意 MoE 的坑：省的是计算不是显存，全部专家都要装进显存

复现历史模型所需时间：
  GPT-2 small (124M/9B)   4.7 小时
  GPT-2 XL   (1.5B/9B)    2.4 天   ← 一个周末能复现 2019 年的旗舰
  GPT-3 175B              25 年 ❌
  Llama 3 70B             509 年 ❌

后训练可行性：
  ✅ SFT / DPO（200M 几分钟，7B+LoRA 约 1.5 小时）
  ✅ GRPO 小模型短任务（1.5B + 数学/游戏，一晚上）
  ❌ 长轨迹 agent RL —— 瓶颈是 KV cache 显存不是算力：
     8B 模型 + 5 万 token 轨迹 = 6.4 GB/条 → 96GB 只能并发 12 条 → 一个 RL step 要 12 小时
     1.5B 模型 + 1 万 token 轨迹 = 287 MB/条 → 能并发 324 条 ✅ 可行
```



---

# 第二轮：200M 模型 / 12 小时预算

## 配置对比

|  | 第一轮 | 第二轮 |
|---|---|---|
| 参数 | 95,233,536 | **201,035,520** (2.11×) |
| C / n_layer / n_head | 768 / 8 / 12 | **896 / 16 / 14** |
| d_ff | 2048 | 2432 |
| 深宽比 | 96 | 56 |
| 词表占比 | 40.5% | **22.4%**（模型变大摊薄，词表没换）|
| D | 1.5B | **10B** (6.7×) |
| D/N | 15.8 | **49.7** |
| 独立 token | 1.5B (1 epoch) | **6.0B (1.66 epoch)** |
| 总步数 | 2,861 | **19,073** |
| lr | 6e-4 | 5e-4 |
| warmup | 200 (7.0%) | 500 (2.6%) |
| 显存 | 9.0 GB | **14.6 GB** |
| 时长 | 48 分钟 | **约 11 小时**（bf16 压缩后约 10 小时）|
| loss | 3.4068 实测 | **~2.77 预测** |
| 困惑度 | 30.17 | **~16 预测** |

★ 词表保持 50257 不变 —— 换词表会让 loss 跟第一轮不可比。

## 本轮的代码改动

| 文件 | 改了什么 | 对应待办 |
|---|---|---|
| config.py | C=896/16层/14头/d_ff=2432/10B/lr 5e-4/warmup 500 | — |
| train.py | ① eval 跨卡 all_reduce（4× 样本量，免费降噪）| C-2 |
| train.py | ② bf16 梯度压缩（通信 18.5%→9.3%，省约 1 小时）| C-3 |
| train.py | ③ **原子写盘**（先写 .tmp 再 os.replace）| 新发现的单点故障 |
| train.py | ④ 启动时磁盘检查 | 同上 |
| prepare_data.py | 自动扫描本地 parquet，013 留 val | — |
| chat.py | 三条路径预热（第一轮冷启动 286ms 是假数字）| A-2 |
| chat.py / evaluate.py | 默认路径 out_run2 / run2.log | — |
| check_ready.py | **新增**：开跑前逐文件检查特征标记 | — |

## 执行清单

```
0  python3 check_ready.py                          30 秒
     必须看到 10 个文件全 ✅、参数量 201,035,520

1  tmux new -s prep && python3 -u prepare_data.py   29 分钟
     确认：扫描到 8 个训练文件
     验收：约 6.0B 独立 token / 1.66 epoch ✅ / 压缩率 ~4.62 / 5 段样本通顺

2  python3 estimate.py                             10 秒
     确认：201,035,520 参数、14.56 GB、11.05 小时、19,073 步

3  python3 train.py --steps 30 --out out5b2        5 分钟
     验收：磁盘检查通过、初始 loss ≈10.82、nvidia-smi 约 16 GB
     OOM → batch_size_per_gpu 16→12, grad_accum 8→11

4  numactl --cpunodebind=1 --membind=1 \
     torchrun --nproc_per_node=4 train.py --steps 50 --out out5c2    8 分钟
     验收：启动信息有「bf16 梯度压缩 开 / eval 跨卡平均 开」
           ★ 每步应约 1.9 秒（比预测的 2.09 快，因为 bf16 压缩）
     若 1.9 秒 → config.assumed_mfu 0.547→0.60，重跑 estimate.py

5  sudo nvidia-smi -pm 1 && sudo nvidia-smi -pl 350

6  tmux new -s train2
   numactl --cpunodebind=1 --membind=1 \
     torchrun --nproc_per_node=4 train.py \
       --save-every 200 --eval-every 1000 --sample-every 2000 --out out_run2

   ★ 开跑后立刻接日志（第一轮差点忘）：
   tmux capture-pane -p -S -8000 -t train2:0.0 > ~/test001/run2.log
   tmux pipe-pane -o -t train2:0.0 'cat >> ~/test001/run2.log'

7  watch -n 60 'python3 monitor.py --session train2 --plot'
```

## 中断恢复

同样的命令末尾加 `--resume`。最多丢 200 步 = 6.3 分钟。
第一轮已实测两次：中断续训的最终结果跟一口气跑完只差 0.02。

## ✅ 第二轮结果（2026-08-28 完成）

### 训练
```
19,073 步 × 1.965 秒 = 10.4 小时（预测 10.2）
全程 MFU 58~60%，显存 17.3 GB 纹丝不动，零异常
最终 train 2.8708   val 2.9097
```

### 评估（完整 val 集，500 万 token）
```
val loss   2.8961
困惑度      18.10
信息量      4.178 bits/token

压缩视角：原始英文 4.62 字节/token × 8 = 36.96 bits/token
          模型压到 4.178 bits/token  →  压缩比 8.8:1（gzip 约 3:1）
```

### 三方对比
| | 参数 | 数据 | val loss | 困惑度 |
|---|---|---|---|---|
| 第一轮 | 95M | 1.5B | 3.4068 | 30.17 |
| GPT-2 small | 124M | 9B | 3.2587 | 26.02 |
| **第二轮** | **201M** | **10B** | **2.8961** | **18.10** |

**第一轮输 GPT-2 small 0.148，第二轮赢 0.363 —— 摆动 0.51。**
```
算力 ×10.8  →  loss −0.51  →  困惑度 −40%
```

### ★ 实测出「重复数据的代价」
```
              公式预测   实测      跑赢公式
第一轮         3.5845   3.4068   +0.178   （1.0 epoch，全新数据）
第二轮         2.9516   2.8961   +0.056   （1.66 epoch，含重复）
                                  ──────
                  数据质量红利缩水  0.122
```
**重复 1.66 遍的代价 ≈ 0.12 loss。** 这是 L11「数据受限缩放」在自己模型上量到的一个真实数据点。
（第三轮下满 14 个 parquet 做到 1.0 epoch，理论上能把这 0.12 拿回来。）

### ★★ 最重要的发现：事实错误的「质变」

表面看事实仍然全错：
```
❌ "The capital of France is the Place de la Concorde, in Brittany"
❌ "Water boils at about 4,5 degrees Celsius"（第一轮说 70°C）
❌ "a virus is bigger than a bacterium"（反了）
```

**但错的方式完全不同了：**
```
第一轮  "Photosynthesis ... bacteria in the flowers grow on top of the plants"
        然后漂到亲子园艺博客        ← 不知道这个话题在说什么

第二轮  "organisms convert carbon dioxide and other gases into energy,
         usually in the form of sugars"
        carbon dioxide ✅ convert into energy ✅ sugars ✅
        只有「把光说成气体」错了
```

法国那段更明显：
```
"Place de la Concorde ... founded by Louis IX (1254-1270), who bought it from
 Philip II ... to Pope Clement IV (1274)"

Place de la Concorde  ✅ 真的是巴黎地标
Brittany              ✅ 真的是法国地区
Louis IX              ✅ 真的是法国国王（1226-1270）
Philip II             ✅ 真的是法国国王
Pope Clement IV       ✅ 真的是同时代教皇（1265-1268）
```
**每个实体都真实，全在「法国中世纪」这个正确的语义邻域里，只是关系全是编的。**

> **学会了「知识的形状」，还没学会「知识的内容」。**
> 知道谈法国该提哪些人和地方，不知道谁跟谁是什么关系。

### 生成质量演化（9 段样本）
同一个 prompt，九个 step 进入了完全不同的语域：
```
step  2,000  泛泛而谈
step  4,000  心理学式阶段论
step  6,000  引用圣经，出现引文格式
step  8,000  出现编号列表 "1) The term life is..."
step 10,000  ★ 虚构署名 "(Sidney Cavanagh)" + 第一人称叙事
step 12,000  宗教劝诫文体
step 14,000  健康科普（diabetes, cardiovascular disease）
step 16,000  学术定义文体
step 18,000  ★ 完整 SEO 文章收尾 "This article has provided you with..."
```
**它学的不是「一种」写法，是网页文本的整个分布。** 每次采样落进不同的模式。

### 解码对比里再次暴露的检测器缺陷
基线（无重复惩罚）输出 "eternal" 六次、句式反复 —— 明显的语义循环，
但 4-gram 检测报了 ✅。**字面重复检测抓不到语义循环，度量工具仍需改进。**

### 第二轮踩到的坑
```
① 新 venv 的 PyTorch 为更高 CUDA 编译 → 静默回退 CPU
   已加防呆：CUDA 不可用时直接报错退出（--smoke 除外）
   教训：pip install torch 装的是「当时的默认版本」，要用 --index-url 锁 cu124
② 进度条拿 10^12 哨兵值当分母 → 显示 0.0% 和 76 小时的假 ETA
③ capture-pane 抓进了 5b/5c 的 scrollback → monitor 把三次运行混算，步时虚高 2.3 倍
   已修：检测 step 回退，只解析最后一次运行
④ monitor 的显存告警阈值是第一轮设的（15GB）→ 第二轮 17.3GB 误报
   已修：改成看增长趋势（>15%）而非绝对值
⑤ checkpoint 写盘不是原子的 → 已改成先写 .tmp 再 os.replace
```

**三次被「显示」而不是「实质」误导（tee 缓冲、哨兵值分母、日志混算）。**
教训：看到反常数字先问「这个数字怎么算出来的」，尤其当它跟别的指标矛盾时
（步时说慢 2.3 倍，但 MFU 说正常 —— 矛盾本身就是线索）。

---

## 训完之后的对照测试集

拿同一组 prompt 跑第一轮和第二轮，直观对比：
```
常见事实（预期明显改善）
  The capital of France is                  ← 第一轮答 "Egypt"
  Water boils at a temperature of           ← 第一轮答 "70 degrees Celsius"
  The largest planet in our solar system is
  The chemical symbol for gold is

主题保持（预期明显改善）
  Photosynthesis is the process by which    ← 第一轮漂到园艺博客
  The Roman Empire collapsed because

推理（预期两轮都不行，要 1B+ 加 RLVR）
  If a train travels 60 miles per hour for 2 hours, it covers
  2 + 2 =

长尾知识（预期两轮都不行，39 MB 容量只是维基百科的 0.2%）
  The capital of Burkina Faso is
  The author of "One Hundred Years of Solitude" is
```

---
---

# 第三阶段：SFT（2026-08-30 完成）

> **一句话结论：SFT 只花了预训练 0.13% 的算力，就把模型从「续写网页」变成了「回答问题」——
> 但它没有让模型多知道任何一件事，而且这个阶段的好坏，`val loss` 完全测不出来。**

## 执行记录

```
基座      out_run2/ckpt_final.pt   201,035,520 参数   step 19,073   val 2.8933（FineWeb）
数据      Alpaca 52,002 条  →  过滤后 50,297 条（回答 <4 token 的丢掉）
          49,797 训练 / 500 验证
token     总计 4,486,763，其中参与 loss 2,995,937（67%，只算回答部分）
          3 epoch = 1,346 万 token  = 预训练 100 亿的 0.13%

配置      lr 3e-5（预训练 5e-4 的 1/17）  weight_decay 0  cosine + warmup 400
          batch 8   max_len 512   单卡   18,672 步（3 × 6,224）
用时      23.7 分钟

val       微调前          2.2416
          最优  step  1,000   1.9233   → out_sft/ckpt.pt
          最终  step 18,672   2.4116   → out_sft/ckpt_final.pt
```

### ⚠️ 成本预估错了 9 倍（原估 2.6 分钟，实际 23.7 分钟）

原估算 `6 × 201M × 50M ÷ 392T = 2.6 分钟` 三处都错：

| | 预估 | 实际 | 原因 |
|---|---|---|---|
| 卡数 | 4 | **1** | 小 batch 下多卡是负优化（见下） |
| token | 50M | **13.5M** | Alpaca 比想象的小得多 |
| MFU | 0.594 | **约 0.19** | batch 只有 8，矩阵乘法喂不饱 4090 |
| padding | 未计 | **约 2.7 倍浪费** | 变长样本右补齐到批内最长 |

**★ 为什么 SFT 必须单卡：通信量取决于模型大小，计算量取决于 batch 大小。**

```
DDP 每步 all-reduce = 201M × 2 bytes = 402 MB，跟 batch 无关
本机 all-reduce 3.7 GB/s（GeForce P2P 被禁）→ 163 ms/步

预训练  每步计算 1,900 ms，通信 163 ms  →  通信占 8%     ✅ 摊得动
SFT     每步计算   22 ms，通信 163 ms  →  通信占 88%    ❌ 淹没计算

处理 32 条样本：单卡 4 步 × 22 ms = 88 ms   /   4 卡 1 步 = 22+163 = 185 ms
→ 多卡慢 2.1 倍
```

## 踩到的坑

**① `model.forward` 的推理优化撞上了 SFT 的 loss**

```
ValueError: Expected input batch_size (8) to match target batch_size (1408)
```

`forward(idx, targets=None)` 走推理分支，**只返回最后一个位置的 logits**（省 (T−1)/T 的 lm_head 计算）。
`sft.py` 原来只传 idx，拿到 (8,1,50257) 而不是 (8,176,50257)。

修法（比原来还干净）：把不算 loss 的位置的 target 设成 −1，交给 `model.py` 里已有的 `ignore_index=-1`：
```python
tgt = x[:, 1:].clone()
tgt[m[:, 1:] == 0] = -1
_, loss = model(x[:, :-1], tgt)
```
`cross_entropy(ignore_index=-1, reduction='mean')` 只对非 −1 的位置求平均，跟 `(ls*mm).sum()/mm.sum()` 完全等价。

**② 误判「训练卡死」**

日志停在 step 16,200 不动 → `nvidia-smi` 全空 → 以为崩了。
实际是**跑完了**：`ckpt_final.pt` 只在脚本最后一行写，它的时间戳（16:10）就是证据。
教训：**判断进程死活先看它最后该产生的那个文件在不在，比看日志可靠。**

**③ 我根据 val 曲线误判「训坏了」**（详见下面的结论 ②）

## 分布内结果：2×2 对照

```
              原始问题        Alpaca 格式
  base 权重       ①               ②
  sft  权重       ④               ③

  ① → ②  只改提示  →  光靠格式有多大用
  ② → ③  只改权重  →  ★ 训练有多大用
  ③ → ④  只改提示  →  格式依赖有多深
```

7 道分布内问题（事实/解释/列举/生成/建议/推理），上限 90 token：

| | 自己停下 | 平均长度 | 恰当回应（人工判） |
|---|---|---|---|
| ① BASE + 原始问题 | **0/7** | 90.0 | 0/7 |
| ② BASE + Alpaca 格式 | 2/7 | 76.4 | 0/7 |
| ③ SFT + Alpaca 格式 | **6/7** | **34.3** | **5/7** |
| ④ SFT + 原始问题 | **6/7** | 38.3 | **2/7** |

**① ≈ ②：套上 `### Response:` 几乎没用**（有微弱效果 —— 光合作用那题 ② 给出了正确定义，但紧接着写 `--- From Biology Homepage`，说明它还在"抄网页"模式）。
**② ≠ ③：变化全部来自那 1,000 步训练。** 这就是第 ② 列作为对照组的价值 —— 没有它，无法排除"只是提示词起作用"。

## ★ 最重要的结构性发现：「会停」和「会答」是可以分离的

第 ④ 格（SFT 权重 + 不给格式）**停下来的比例跟 ③ 一样（6/7），但恰当回应只有 2/7**：

```
问：Give me three tips for learning a new language.
④ 输出：（0 token，立刻 EOT）           ← 判定"这段文字已经完整了"

问：What is the capital of France?
③ "The capital of France is Paris."      ✅
④ "France is located in Europe and it has a population of over 20 million."  ❌

问：At what temperature does water boil?
④ "2. The boiling point of water is 4.0 degrees Celsius.
    What is the average temperature of a pond at the equator?"   ← 以"2."开头，还反问
```

**解释：**
```
「答完输出 EOT」  是局部统计规律 —— 看到完整句子就停，不需要理解语境
                  → 学得又快又牢，完全烧进权重，不依赖任何标记

「识别这是个待回答的问题」 需要语境判断
                  → 跟 "### Response:" 这个标记绑死了
```

**实用推论：用别人的开源 SFT/Chat 模型时 chat template 不能错。** 用错了模型不会明显报错，而是进入这种"会停但不会答"的半残状态。

## 分布外测试：过拟合的代价

5 类 Alpaca 里几乎没有的问题，欠训（step 1,000）vs 过训（step 18,672）：

| | 欠训 val 1.9233 | 过训 val 2.4116 | 谁好 |
|---|---|---|---|
| **① 恰好五个词**<br>`Answer in exactly five words: what is gravity?` | ❌ 反问了一个问题 | ✅ "Gravity is the force that caused objects to fall towards the Earth."（12 词，超了但答了） | 过训 |
| **② 假前提**<br>`Why is the sky green?` | ❌ "due to photosynthesis" | ❌ "because of carbon dioxide... climate change" | 平（都顺着编） |
| **③ 多轮追问**<br>`2+2=4` 之后问 `Why?` | "2 and 4 are both numbers."（扯到了上文） | ❌ **"2 + 2 = 5."** | ★ 欠训 |
| **④ 长输入总结**<br>110 token 的亚马逊段落 | ✅ "vulnerable to deforestation, as its vast size limits its ability to absorb CO₂" | ❌ "down to 10% of its original area"（原文是砍掉 17%） | 欠训 |
| **⑤ 中文**<br>`法国的首都是哪里？` | ❌ 无关英文 | ❌ 无关英文 | 平（都废） |

**1 : 2，外加两平 —— 没有明显赢家。**

### ★ ③ 那题抓到了「模板压过指令」

```
提示   ### Instruction: What is 2 + 2?
       ### Response: 4
       ### Instruction: Why?              ← 问的是"为什么"
       ### Response:

过训   "2 + 2 = 5."
       ↑ ① 完全无视 Why?，跑回去重答第一题
         ② 套上"把问题改写成等式"的模板后，填了个 5
```

**这就是过拟合的具体形态：模板启动得太强，把指令本身盖掉了。** ④ 那题同理 —— 过训的输出更短更"像回答"，但离原文更远，模板赢了内容。

### 额外发现：SFT 抹掉了中文能力（温和的灾难性遗忘）

```
BASE   "资会读件 发沒有为不可想和一下共同义。或要增加，但是不得到中…"   ← 乱码，但在输出中文字符
欠训   "The first sentence of the paragraph is..."                      ← 转成英文
过训   "The title of the book is 'The Art of One'smine'."               ← 也是英文
```

用 100% 英文数据微调，把预训练里那点微弱的多语言痕迹压了下去。
**推论：做多语言模型时 SFT 数据必须按语言配比，不能只用英文。**

---

# ★★ SFT 阶段的评价指标：**不能用 val loss**

这是这一阶段最值钱的教训，也是我在过程中判断错的地方。

## 我错在哪

看到 val 从 1.9233（step 1,000）一路升到 2.4116（step 18,672），我判定"过拟合了、训坏了、后面 96% 的训练是浪费"。
**实际跑输出对比，过训的模型在多数题上更好：**

| 题 | 欠训 val 1.92 | 过训 val 2.41 |
|---|---|---|
| List **three** healthy breakfast foods | 给了 **6 个** | **"Three healthy breakfast foods are oatmeal, eggs, and milk."** ✅ |
| Explain photosynthesis | "chlorophyll a, b, c, d, e, f and g in leaves, **roots, flowers**"（退化列表） | "transfer of electrons and protons... exchange of oxygen and carbon dioxide" ✅ |
| Write a short poem | "I'm like an island on a reef" | "The sun sets over the horizon, / Filling the night with a gentle light" ✅ |

**val 更差的模型，「给三个」这个约束反而遵守对了。**

## 为什么 val loss 在 SFT 里会反向

**val loss 测的不是「答得好不好」，是「跟参考答案逐字有多像」。**

```
问题        List three healthy breakfast foods.
Alpaca 参考  "1. Oatmeal\n2. Greek yogurt\n3. Whole grain toast"
模型输出     "Three healthy breakfast foods are oatmeal, eggs, and milk."
             ↑ 更好的回答，但跟参考答案一个字都不一样  →  loss 很高
```

**机制：越自信的模型，答偏了罚得越狠。**

```
第一个 token，参考答案是 "1"

step  1,000 的模型   还很"松"，什么都可能说
                     p("1") ≈ 0.10  →  loss = −ln 0.10 = 2.30

step 18,672 的模型   已形成自己的答题风格，笃定要说 "Three"
                     p("1") ≈ 0.01  →  loss = −ln 0.01 = 4.60   ← 罚了两倍
```

**训练让模型从「什么都可能说」变成「知道自己要说什么」。这个过程必然推高在参考答案上的 loss —— 而"知道自己要说什么"正是让输出可读的东西。**

## 根本区别

```
预训练    目标 = 压缩文本        →  val loss 就是目标本身      ✅ 好指标
                                    （Chinchilla、scaling law 全建立在它上面）

SFT       目标 = 学会一种行为    →  val loss 只是一个不相关的代理指标   ❌ 坏指标
                                    参考答案只是"一种"正确答法，不是唯一
```

## 那该用什么

| 方法 | 说明 | 我们这次用的 |
|---|---|---|
| **人工盲评** | 直接看输出，成对比较 | ✅ 2×2 网格 + 分布外测试 |
| **可量化的行为指标** | 自己停下的比例、平均长度、约束遵循率 | ✅ 上面那张表 |
| **LLM-as-judge** | 用更强的模型打分 | ❌ 没做（可以用 GPT-4 类模型做） |
| **下游 benchmark** | MMLU / GSM8K / AlpacaEval / IFEval | ❌ 没做，值得补 |
| val loss | — | **⚠️ 只用来确认"训练在跑"，不用来选 checkpoint** |

**★ 一句话：预训练看 loss，后训练看行为。**

---

# SFT 三条结论

```
① 「会停」≠「会答」
   EOT 是局部统计规律，学得牢、不依赖格式（6/7 停）
   「识别这是个问题」依赖格式标记（恰当回应从 5/7 掉到 2/7）
   → chat template 用错不会报错，只会让模型半残

② SFT 的 val loss 不能预测行为质量
   val 1.9233 和 val 2.4116 的两个模型，分布内分布外各有胜负
   → 预训练看 loss，后训练看行为

③ 3 epoch 的过拟合代价：有，但可控
   代价：模板压过指令（多轮那题："Why?" → "2 + 2 = 5."）
   收益：句式更利落、约束遵循更好（"给三个"答对了）
   → 净效果接近持平
```

## 三层理论的现场验证（这次实验的总账）

| 层 | 靠什么获得 | 花了多久 | 现场证据 |
|---|---|---|---|
| **① 语言** | 十亿级 token | **10.4 小时** | 每句都语法正确、通顺、词汇得体 |
| **② 知识** | 万亿级 token + 大 N | **10.4 小时** | Paris ✅ / 沸点 40°C ❌ / 叶绿素半对半编 |
| **③ 行为** | **几千条样本** | **★ 77 秒**（1,000 步 × 77 ms） | 格式完美、会停、会分点 |
| **④ 推理** | RL | **没有** | 60 × 2 → "78.9 km/h" |

**第 ③ 层用了总算力的 0.13%，77 秒。这是「SFT 不创造能力，只是重新分配概率」最直接的证明。**

## ★★ 翻转分析：RL 到底改变了什么（2026-09-02）

`find_flips.py`：贪心解码跑 TEST[200:500] 共 300 题，逐题对比训练前后。
★ 贪心 = 确定性，任何翻转必然来自权重变化，不是采样运气。

## 交叉表

```
                    训练后 ✅   训练后 ❌
    对照 ✅            190         16     ← ★ 弄坏
    对照 ❌             40         54
                       ↑ 修好

  对照 0.6867  →  训练后 0.7667     净 +0.0800 = (40−16)/300
  McNemar  χ² = 24²/56 = 10.29   z = 3.21   ★ p ≈ 0.0013  显著
```

**★ 更有信息量的口径：**
```
原始答错 94 道  →  修好 40 道   ★ 修复率 42.6%
原始答对 206 道 →  弄坏 16 道   ★ 破坏率  7.8%
                                 ★ 修复率是破坏率的 5.5 倍

★ 56 道翻转（18.7%），净赚只有 24 道
   → RL 不是"稳步提升"，是"大规模重新洗牌，净赚一点"
```

## ★★★ 修好的 4/4 案例：**全是「读题更完整」**

| 题 | 对照漏掉了什么 | 训练后 |
|---|---|---|
| 210 Sara 买夹克 | 漏掉「已存 $10」→ 13 | ✅ 算进去 → 10 |
| 226 宝可梦卡 | 漏掉 30 张火系（总数算成 66 而非 96）→ 50 | ✅ 三类都算 → 33 |
| 230 Mark 买车 | 漏掉「每辆」都要 $1000 注册费 → 264,000 | ✅ 12×1000 → 276,000 |
| 234 Steve 番茄 | 漏掉女朋友也吃 → 14 | ✅ 6+3=9/天 → 21 |

> **★★ 四道全是「漏条件」，不是「算错」。RL 教会的不是「算得更准」，是「读得更全」。**

**★ 这跟 pass@64 = 0.0000 完全一致：**
```
「仔细读题不漏条件」是模型本来就会的行为
  → 温度 1.0 下 64 次里总有几次读全了
  → ★ RL 只是把它的概率从"偶尔"推到"默认"
支撑集没变，改变的是「哪种阅读策略成为主流」
★ 这是「重排概率」在语义层面的样子 —— 重排的不是 token，是【阅读策略】
```

**⚠️ 我预测错了：我说「按 pass@64=0 的结论应该主要是少犯算术错」，实际全是漏条件。**

## 弄坏的 4 例：没有单一模式

| 题 | 训练后错在哪 | 类型 |
|---|---|---|
| 214 泳池 | `B = 2S−16+16`（应为 +32）| 纯代数滑手 |
| 249 饼干 | 凭空写 `13×5=65` + **★ 漏掉最后乘 200 卡路里** | 读题错 + 漏步骤 |
| 293 三人游戏 | 说「21 不大于 20」 | 比较错误 |
| 299 开车 | `$150/月 × 50 个月`（把 50 周当 50 月） | 数字混淆 |

**★ 题 249 是熵坍缩代价的标本：** 对照 337 token 算完卡路里 ✅；
训练后 269 token 算完饼干数就停了 ❌ —— 更短、跳步骤，跟「长度 265→217」一致。

## ★★ 题 293 的陷阱：「答对」≠「推理对」

```
正确      Mike 21>20 +1=22   Jim 18 无   Tony 42>20 +1=43   总 83

对照 ✅83  Mike 22 ✓
          ★ Jim 给了 +1（说"18 大于 20"）→ 19    ❌ 错
          ★ Tony 没给（说"已经超过 20"）→ 42     ❌ 错
          22+19+42 = 83   ★★ 两个错误正好抵消，蒙对

训练后 ❌82 只错一个（说"21 不大于 20"）→ 21+18+43 = 82
```

**★ 训练后犯错更【少】却答错；对照犯错更多却蒙对。**
**→ 净提升 +0.080 可能【低估】了真实的推理改善（对照那 206 道"对"里有一部分是蒙的）。**

## 被推翻的假设：「RL 学会了硬凑答案」

手动测那道 50 学生题时，训练后写了
`"there seems to be an inconsistency... however, assuming... V = 22"`，
我猜二元奖励惩罚「承认不知道」→ 模型学会硬凑。**扫全部翻转样本后推翻：**

```
          对照命中   ★ 训练后命中
  fixed     3/20        1/20
  broken    2/16        1/16
  ────────────────────────────
  合计   5/36 = 13.9%   ★ 2/36 = 5.6%
```

**★ 方向是【反的】—— 训练后用犹豫措辞的频率只有对照的 40%。**
（Fisher p≈0.43，不显著，但方向跟另外两个指标一致。）

**★ 那次观察为什么会出现：** 读题错 → M = 11.67 → 一个【明显荒谬】的中间值 → 模型察觉。
而 16 道弄坏的错误都不产生荒谬中间值（算错一个数，结果看着正常）→ 察觉不到 → 不会硬凑。

**★★ 三个独立维度同时收窄，指向同一个熵坍缩：**
```
logP        −0.78 → −0.22      更自信（熵掉 72%，有效选择数 2.18 → 1.25）
长度         265  →  217        更简洁
★ 犹豫措辞   13.9% →  5.6%      更果断
```

---

# ★★★ 理论框架：采样半径（2026-09-02，从「多重时间线」比喻推出）

## 先换掉一个不准的比喻

```
❌ 「预训练修路，RL 装路标」
   ★ 「路」暗示路径已经存在于某处 —— 实际是每一步现算出来的

✅ ★ 「多重未来时间线，RL 调整我们倾向于前往哪一条」
   ① 生成是逐 token 展开，每一步都在分岔
   ② 分支是指数级的：T 个 token 有 151,936^T 条可能
   ③ ★ 模型持有的就是对这些时间线的【概率分配】
```

## ★ 精确化：没有「零概率」的时间线

```
softmax 恒为正 → 每一条时间线的概率都 > 0

  "44 + 19 + 35"      p ≈ 0.20
  "44 × 19 × 35"      p ≈ 0.02
  "紫色的大象跳舞"     p ≈ 1e-40    ← 不是 0，只是永远碰不到

★ 「有效可达范围」由采样次数决定，不是由模型决定
```

## ★★ 核心机制：低概率时间线会**自锁**

```
某条正确的时间线 p = 0.001
  → 采样 8 次碰不到（碰到概率仅 0.8%）
  → ★ 从没出现在组里 → 拿不到奖励 → 优势为 0
  → ★ 梯度为 0 → 概率不变，还是 0.001
  → 下次还是碰不到 → ★★ 永远锁死

★★ RL 只能强化「至少被采样到过」的时间线。
```

## ★★★ 采样半径公式

```
概率为 p 的行为，在 k 次采样里至少出现一次的概率 = 1 − (1−p)^k
★ 要有一半以上机会出现：  p > 0.693 / k
```

| 训练时组大小 k | **RL 能触及的最低概率** |
|---|---|
| **k = 8**（我们用的） | **★ p > 0.087** |
| k = 64 | p > 0.011 |
| k = 256 | p > 0.0027 |

## ★★ 这解释了为什么 Δpass@64 = 0.0000 是**必然的**

```
用 k=8  训练  →  只能改动 p > 0.087 的时间线
用 k=64 测量  →  测的是 p > 0.011 的范围

★★ 中间那段（0.011 < p < 0.087）—— RL 根本够不着
   → Δpass@64 = 0.0000 不是巧合，是采样半径决定的
```

## ★★★ 由此得到一个可证伪的新预测

```
★ 用 k=64 训练（而不是 k=8）→ pass@64 应该会动
  因为 RL 这次能触及 p > 0.011 的时间线
★ 而 pass@256 仍然不动

→ ★ 这是个干净的实验：同样的配置只改组大小 k，看 pass@k 曲线的哪一段被抬起来
```

## ★★ 补充：一个行为被 RL 强化，需要**两个**条件（2026-09-02）

采样半径只管第一条，第二条一直被忽略：

```
① 它出现在采样里              ← ★ 采样半径管这个（0.7/k）
★★ ② 带着它的那条轨迹【答对率更高】  ← ★ 这一条同样是必要条件
```

**★ 第二条的实测证据（Countdown 基线，Qwen2.5-1.5B Base）：**

```
纠错措辞频率   ★ 0.166    ← 它【会】写 "wait / hmm / let me try"
正确率         ★ 0.0138   ← ★ 但说了也白说

→ 对 1.5B 来说，「说了 wait」和「答对」几乎不相关
→ 带着它的轨迹拿不到更高奖励 → 优势不为正 → ★ 这个行为不会被强化
```

**★★ 所以「换更大的模型」的真正原因，不是「它更常说 wait」，
而是「在它身上，说 wait 之后【真的能算对】」—— 行为才跟奖励挂上钩。**

### ★ 三个条件缺一不可

```
① 预训练    ★ 决定有哪些「零件」，以及每个零件的初始概率
② 组大小 k  ★ 决定你能"看见"多低概率的组合（半径 0.7/k）
③ 模型大小  ★ 决定那个行为【有没有用】—— 用了能不能真的答对
            ← 只有"有用"，它才跟奖励相关，才会被强化

★ 1.5B 卡在第三条：零件有（0.166 会说 wait），k=8 勉强看得见，
  但★ 说了不管用（正确率 0.0138）→ 所以强化不起来
```

## ★★★ 组合涌现：RL 能组装「零件散落各处」的行为

**严格版：预训练里完全没有的行为，RL 放大不出来。**

```
softmax 恒为正 → 任何 token 序列概率都 > 0，不是绝对的零
但如果概率是 1e-30：
  采样半径 0.7/k，k=1000 时也只到 0.0007
  ★ 要够到 1e-30 得采样 1e30 次 —— 宇宙年龄都不够
→ ★ 实践上：放大不出来
```

**★ 但有一个精妙的例外 —— 零件散落时，组合是可达的：**

```
假设模型从没见过完整的「发现错误 → 回头 → 换路 → 答对」

但它见过零件：
  "wait"                  p ≈ 0.20   （日常对话里到处都是）
  "let me try X instead"  p ≈ 0.10
  重新计算的动作           p ≈ 0.30

★ 完整组合的概率 ≈ 0.20 × 0.10 × 0.30 = 0.006
   —— 低，但★ 不是天文数字，采样几百次能碰到几次
   → RL 把各零件的概率各推高一点 → 组合从 0.006 → 0.05 → 0.3
```

> **★★ 更准的说法：**
> **RL 放大不出「零件都没有」的行为，**
> **★ 但能把「零件散落各处」的行为【组装起来】并放大。**

**★★★ R1-Zero 的「顿悟时刻」很可能就是这种组合涌现 ——
那些词模型都会说，只是从没在正确的位置、以正确的顺序连起来过。**

## ★ 熵坍缩 = 采样半径在收缩

```
熵下降 → 概率越来越集中在少数几条时间线
        → ★ 其他线的 p 掉下去 → 掉出采样半径
        → 可触及的时间线越来越少

★ 直接观测证据（第四阶段实测）：
    有信号率      96% → 82%（k=64 下）
    64/64 全对    0 道 → 8 道   ← 这 8 道再也提供不了梯度
    训练时有信号率 78% → 53%
```

**★ 这就是「燃料耗尽」的精确含义：不是熵本身有价值，是熵撑开了采样半径。**

---

# ★★★ 第四阶段最终七条结论

```
① 学习率是决定性的
   1e-6 完全不动（KL 0.0016）  5e-6 立刻起效（KL 0.18，★ ×115）

② ★ RL 有效且显著
   贪心 0.6867 → 0.7667（+0.080）  McNemar z=3.21, p≈0.001
   修好 40 / 弄坏 16，★ 修复率是破坏率的 5.5 倍

③ ★★ 只重排，不创造
   pass@64 三个模型完全相同 = 0.9800（50 道里 49 道，漏的是同一道）
   Δ 从 +0.21 单调收缩到 ★ 0.0000，没有余数

④ ★★ 提升的机制是「读题更完整」
   4/4 修好案例都是"没漏掉条件"，不是"算得更准"
   ★ 与 ③ 一致：重排的不是 token，是【阅读策略】

⑤ ★★ 代价是熵坍缩
   logP −0.78→−0.22 · 长度 265→217 · 犹豫措辞 13.9%→5.6%
   贪心与 pass@1 的差距缩小 81%（0.1605 → 0.0298）

⑥ ★ 自发涌现
   run2 在没有格式奖励的情况下，把 #### 输出率从 56% 提到 97%
   —— RL 会强化任何跟奖励【相关】的东西，不管你有没有打算奖励它

⑦ ★★ 方法论
   「见顶回落」是 200 题验证集的噪声，不是真实退化
   在同一份数据上又选又评，虚高约 0.02
   ★「答对」≠「推理对」—— 题 293 是两个错误抵消蒙对的
```

**★ ④ 是最独特的一条：它把 ③ 那个抽象的「重排概率」落到了语义层面。**

---

# 下次改什么

```
--epochs 3  →  1        参数/数据比 = 201M ÷ 3M = 67 参数每 loss-token，极度过参数化
--eval-every 400 → 100  前期采样密一点（但记住 val 只用来看"在不在跑"）
评估              加 IFEval 类的约束遵循测试，比 val loss 有意义得多
packing           把多条短样本拼满一个序列，省掉 2.7 倍的 padding 浪费
                  （23.7 分钟的任务不值得，但真做起来 SFT 会快 2 倍以上）
```

## 产出的文件

```
sft.py            Alpaca SFT，单卡，loss mask 用 ignore_index=-1
inspect_sft.py    训练前看数据：原始 JSON → 拼接结果 → token 级 mask → 长度统计
compare_sft.py    2×2 网格（分布内）+ 5 类分布外测试，三模型同屏
out_sft/ckpt.pt         step  1,000  val 1.9233   欠训
out_sft/ckpt_final.pt   step 18,672  val 2.4116   过训
```


---

# 待办：RL 采样加速（2026-09-01 记）

第一次 GRPO 用 **A 方案**（纯 HF transformers、单卡）跑通，先保证算法正确、容易调试。
以下两条等 A 跑通后再上。

## B. HF + 4 卡数据并行采样　　预期约 4×

```
背景   RL 有约 80% 的时间花在采样上，而采样是吞吐问题不是延迟问题
做法   ★ 4 张卡各放一个完整模型副本（数据并行），不是 device_map="auto"（模型并行）
       一步采 8 题 × 8 条 = 64 条轨迹 → 每卡 16 条
注意   device_map="auto" 是把模型切开分到 4 卡，只在放不下时才用
       Qwen 1.5B 只有 3.08 GB，用它纯属负优化（本机 all-reduce 只有 3.7 GB/s）
```

## ★★ 为什么 RL 的工程重心全在采样上（2026-09-02 实测总结）

```
① ★ RL 有 85% 的时间在采样
   GSM8K 实测：31 秒采样 + 5 秒训练
   Countdown 实测：7 秒采样 + 2 秒训练

② ★ 采样天生比训练难加速
   · 小矩阵     预训练一次 forward 吃 52 万 token，采样一次只算 batch×1 个
   · 访存瓶颈   ★ 不管算 52 万还是 32 个 token，都要把 3.08 GB 权重读一遍
                预训练 MFU 0.658（吃饱）  vs  chat.py 利用率 12%（88% 在等）
   · ★ 自回归串行  要算第 n+1 个 token 必须先有第 n 个 —— 因果关系，物理上没法并行

③ ★★ 所以 RL 的工程重心全在采样上
   vLLM / PagedAttention / 连续批处理 —— 全是为了解决这一件事
   唯一的解法是「横向加宽」：同时生成更多序列，摊薄权重读取
```

## ⚠️ 实测负面结果：**多线程并行采样反而慢 4.5 倍**（2026-09-02）

```
grpo_countdown.py 最初用 ThreadPoolExecutor 让 4 张卡并行采样
理由：GPU 算子释放 GIL，所以线程应该是真并行

★ 实测（同样 16 条轨迹，max_new 256）
    4 卡（4 线程）   采样 ★ 35 秒
    1 卡             采样 ★  7 秒
                     ────────────
                     ★★ 单卡快 4.5 倍
```

**两个原因叠加：**
```
① ★★ GIL 没释放
   HF 的 generate() 每生成一个 token 都要跑一大段 Python
   （采样逻辑、logits processor、停止条件判断）—— ★ 那段持有 GIL
   → 4 个线程互相抢锁 + 上下文切换开销
   ★ 「GPU 算子释放 GIL」对纯 CUDA 调用成立，对 HF 的 generate 不成立

② batch 被切碎   16 条 / 4 卡 = 每卡 4 条，GPU 更吃不饱
```

**★ 对比：预训练 DDP 4 卡加速 3.25×。同样 4 卡，一个 ×3.25，一个 ×0.22。**

### ★ 根因：DDP 是多进程，线程方案是单进程

```
预训练 DDP    torchrun --nproc_per_node=4  → ★ 4 个独立进程，各有各的 GIL
线程采样      1 个进程 4 个线程            → ★ 共享 GIL
```

### ★★ 而且 PyTorch 只提供「梯度同步」，不提供「权重同步」

```
DDP 同步的是【梯度】     ← 框架自动（backward 时 all-reduce）
                          ★ 各卡自己应用梯度 → 权重自动保持一致

采样并行要同步的是【权重】 ← ★ 框架不管，因为采样不是训练操作（没有梯度）
                          GPU 0 训练完，得把 3.08 GB 权重【发】给另外 3 张卡
                          不发的话它们用的还是旧权重 → ★ 破坏 on-policy
```

**★ verl / OpenRLHF / TRL 这些 RL 框架把这套封装好了。我们自己写是为了搞懂机制 ——
用框架的话根本碰不到这个坑，也就学不到「为什么线程不行」。**

## ★★ RL 的吞吐 = 单位时间能验证多少个「想法」

```
一条轨迹  = 模型对这道题的一次尝试 = ★ 一个待验证的想法
判分      = 验证
★ 吞吐    = 每秒能验证几个想法

实测   442 tok/s ÷ 250 token/条 = ★ 1.8 个想法/秒
```

**★★ 换算成信息量，对比预训练：**

```
RL      1.8 轨迹/秒 × 约 2 bit（阶梯奖励）= ★ 3.6 bit/秒

预训练  100 亿 token / 10.4 小时 = 26.7 万 token/秒
        × 约 4 bit 的"惊讶度" = ★ 100 万 bit/秒

                                 ★★ 相差 30 万倍
```

**★ 这就是「RL 的信息带宽是预训练的百万分之一」的精确版本。
也是为什么采样吞吐 = 学习速率本身 —— 提高采样吞吐，字面意义上就是提高学习速度。**

## ★ C. vLLM 采样（待办，2026-09-02 重新评估）

```
为什么值得做   ★ RL 有 85% 时间在采样，而采样是【学习速率本身】
                （吞吐 = 每秒能验证几个想法 = 每秒获得多少 bit）

vLLM 解决的核心问题   ★★ HF 的静态批处理：一批 32 条一起进一起出
                          短的写完了干等长的 → 大量算 padding
                      ★ vLLM 连续批处理：谁写完谁下岗，立刻补新的
                          → 批永远是满的，零 padding → 5~10 倍

另外两个   PagedAttention   KV cache 像操作系统分页，碎片率 <5%
           前缀复用          ★ GRPO 同一道题采 k 次，prompt 完全相同
                             → 那段 KV 只算一次，k 条共享

★ 难点：权重同步
   训练进程每步更新权重，vLLM 进程还拿着旧的
   必须每步推 3.08 GB 过去，否则破坏 on-policy
   解法：NCCL broadcast / vLLM 的 load_weights / 共享显存
   ★ verl、OpenRLHF 都封装好了，可以直接读它们的实现

时机   ★ 等多进程版（待办 B）跑通、确认要跑 2000 步全量时再上
```

## B-旧. vLLM 采样 + HF 训练　　预期约 15~20×

```
收益   采样加速 5~10×（PagedAttention + 连续批处理 + 前缀复用）
       采样占 80% → 总训练时间约缩到 1/3
★ 难点  权重同步：训练进程每步更新权重，vLLM 进程还拿着旧的
        必须每步把新权重推过去（NCCL broadcast / vLLM 的 load_weights）
        这是 C 方案复杂度高的唯一原因
时机   留到第二次实验（Base 版 R1-Zero 复现），那次采样量更大，值得
参考   verl / OpenRLHF 已经把这套集成好了，可以直接读它们的实现
```

## 相关实测数字（chat_qwen.py 量的）

```
Qwen2.5-1.5B  权重 bf16 3.08 GB
              decode 理论上限 1008 GB/s ÷ 3.08 GB = 327 tok/s
              HF transformers 实测约 40~60 tok/s  →  带宽利用率约 15%
              （对比自己的 201M：理论 2507，实测 30~40，利用率仅 1.5%
                ★ 模型越大固定开销占比越小，利用率越高）
KV cache      28 层 × 2 个 KV 头(GQA) × 128 × 2 × 2 = 28 KB/token
```


## D. LoRA vs 全参微调的对照实验（2026-09-01 记）

第一次 GRPO 决定用 **8-bit Adam 全参微调**。LoRA 版留作对照。

```
为什么值得对比   没人给过这个问题的答案，而两次跑加起来只要 9 小时
                 同样的数据、步数、随机种子 → 直接量出「LoRA 损失了多少」

LoRA 侧的理由（当时讨论的）
  · RL 的 lr=1e-6，比 SFT 低 30 倍，300 步权重挪动极小 → 低秩够用
  · ★ pass@8=0.855 说明能力早在权重里，RL 只重排概率
    —— 正是低秩改动最擅长表达的那种变化
  · 自带正则化，RL 最常见的失败是"跑飞"，参数少 = 跑不远 = 更稳
  · 反向传播省约 20%（冻结的权重不用算 dW）
  · ★ 参考模型免费：关掉 adapter 就是原模型，省 3.1 GB

配置          r=16, alpha=32, 加在 28 层的 q/k/v/o 上
              可训练 4.4M / 1540M = 0.28%
显存          约 6 GB（对比全参 8-bit Adam 的 15~16 GB）
```

### LoRA 初始化的细节（讨论时确认的）

```
A 随机初始化，B 全零

为什么 B 必须零   第 0 步 B·(A·x)=0 → 输出跟原模型一模一样
                  否则一挂上就被随机噪声搞坏
为什么不都置零    ∂L/∂B 需要 A≠0，∂L/∂A 需要 B≠0
                  ★ 都是零 → 两边梯度都是 0 → 永远动不了
```

---
---

# 第四阶段：GRPO / RLVR（2026-09-01，部分完成）

> **一句话结论：RL 主要不是在改进「最好的那条路」，而是在【消灭差的路】——
> 贪心正确率只涨 0.030，温度 1.0 下却涨了 0.218。这个差距本身就是熵坍缩的量化。**

## 选型与基线

```
模型      Qwen2.5-1.5B-Instruct   1.54B 参数，28 层，C=1536
          12 注意力头 / ★ 2 个 KV 头（GQA）  词表 151,936  上下文 32,768
          ★ 为什么不用自己的 201M：GSM8K 正确率≈0 → 8 条全错 → 优势全 0 → 训练根本不启动
          （RL 是放大器不是发电机，需要非零的起始成功率）

数据      GSM8K   train 7,473 / test 1,319
环境      ★ 判分器 3 行正则（_reward.py）—— 所有 RL 环境里最简单的一种

基线（200 题 × 8 次，温度 1.0）
  pass@1        0.456      ★ 落在 0.20~0.60 黄金区间
  pass@8        0.855      ← 当时以为是"能力天花板"
  有信号        77.5%      全对 8.0% / 全错 14.5%
  格式率        57.1%
  截断            0%       max_new_tokens=1024 够用
  速度       500 tok/s（batch 32；batch 8 时只有 230 —— ★ 批量翻倍效率翻倍）

★ 贪心（温度 0）基线   0.7050
  ← 跟温度 1.0 的 0.456 不可比，但它才是训练过程中评估用的起点
```

## 配置

```
微调          全参 + 8-bit Adam（bitsandbytes，★ pip install 必须加 --no-deps）
每步          8 题 × 8 条 = 64 轨迹
★ 微批        4 条 × 梯度累积 16 次
              原因：词表 151,936，64 条 × 300 位置的 logits = 5.83 GB，比模型本身还大
              cross_entropy 里还有个 fp32 副本 → 微批 4 才安全
max_new       1024      温度 1.0      奖励 纯 0/1 正确性（★ 不加格式奖励，见下）
lr            1e-6 → ★ 5e-6（关键修正）
KL β          0.001     clip 省略（见下）    显存峰值 15~20 GB
时长          300 步 = 171 分钟   ★ 采样占 85~89%
```

### ★ 为什么省掉了 clip / 重要性采样比

```
每批 rollout 只做一次 opt.step()（16 个微批之间不更新）
→ 采样时和算 loss 时是同一份权重 → ratio = P_new/P_old 恒等于 1
→ clip(1, 0.8, 1.2) = 1，空操作
只有一批 rollout 要重复用几次（PPO 的多 epoch）时才需要它
```

### ★ 为什么最终没加格式奖励

手动测 5 轮时，乌龟那题出现「有 #### 的 3 条全对，无 #### 的 5 条全抽出 40（题目里的"40 秒"）」，
我据此推断抽取器有系统性假阴性，主张加格式奖励。
**200 题 1600 条的正式基线推翻了它：有 #### 正确率 0.465，无 #### 0.443，差 1.0 倍。**
→ 乌龟那题是巧合（答案后面正好跟着 "in 40 seconds"）。省掉一个超参。

**教训：5 个样本的推断不可信，尤其当它符合你的预期时。**

## ★★ 两次实验：学习率是决定性的

| | **第一次** | **第二次** |
|---|---|---|
| lr | 1e-6 | **5e-6** |
| 步数 | 150（中止） | 300 |
| **KL（末）** | **0.0016** | **0.18**（★ ×115） |
| **logP** | −0.78 → −0.785（平） | −0.78 → **−0.22** |
| 训练正确率 | 0.586 → 0.647 | 0.586 → **0.812** |
| 有信号率 | 0.76 → 0.79（不变） | 0.78 → **0.53** |
| **贪心 test** | 0.705 → **0.670**（−0.035） | 0.705 → **★ 0.805**(step 224) → 0.735 |

**第一次的诊断：KL 只有 0.0016，模型几乎没动。**
每步参数位移 ≈ lr = 1e-6（Adam 归一化后步长约等于 lr），150 步累积 1.5e-4，
而权重典型量级 0.02 → 相对变化仅 0.75%。**代码没错（|g| 非零、KL 在涨、符号复查过），就是 lr 太低。**

## ★★★ 核心发现：熵坍缩的精确量化

```
                    温度 1.0      贪心        差距
  训练前             0.456       0.705      ★ 0.249
  训练后             0.674       0.735      ★ 0.061
                                            ──────────
                                            缩小 75%
```

**含义：**
```
训练前   贪心路径已经很好，但随机采样经常偏离它 → 0.705 掉到 0.456
         概率质量分散在很多条路上

训练后   概率质量高度集中在贪心路径附近
         → 温度 1.0 采样也基本走那条路 → 0.674，逼近贪心的 0.735
```

**★ RL 主要没在改进「最好的那条路」（贪心只涨 0.030），而是在【消灭差的路】（温度 1.0 涨 0.218）。**

### 这一条解释了另外三个观察

```
① pass@1 涨 0.218 而贪心只涨 0.030    → 优化的是分布，不是最优路径
② 有信号率 0.775 → 0.555              → 8 条采样越来越像
③ 全对（8/8）从 8% 飙到 38%           → 同题 8 次几乎走同一条路
   同期 logP −0.78 → −0.22（e^-0.22 = 80% 把握，起点 46%）
```

## ★ 训练会见顶回落

```
step 124   0.7500
step 149   0.7600
step 174   0.7400
step 199   0.7500
★ step 224 0.8050   峰值（比起点 +0.100，吃掉可落差的 66.7%）
step 249   0.7800
step 274   0.7450
step 299   0.7350
```

**同期训练正确率还在涨（0.586 → 0.812）→ ★ 泛化崩塌。**
后 75 步在做的事：把已经会的题练得更熟，代价是没见过的题做得更差。

### ⚠️ 踩的坑：峰值 checkpoint 被覆盖了

原脚本只存 `ckpt_latest.pt`，每 25 步覆盖一次 → step 225 存的 0.805 被 250/275/300 依次覆盖。
**已修复：新增 `ckpt_best.pt`（★ 只存模型不存优化器，3 GB 而非 7 GB）+ `--patience` 早停 + 熵坍缩预警。**

## ★ 长度没变长 —— 跟 R1-Zero 相反

```
平均长度   273 → 247   ★ 略降，不是变长
```

```
Qwen2.5-Instruct 已有成熟思维链（SFT 教的）
  → RL 不需要"发明"推理，只需把已有的推理变可靠
  → 表现为「更自信、更简洁」

R1-Zero 用 Base 模型，得从零长出推理
  → 才会看到思维链暴涨
```

**这是「用 Instruct 还是 Base」最直观的差别，实测到了。**

## pass@k 对比（k=8，200 题，温度 1.0）

| 指标 | 训练前 | 训练后(step 300) | 变化 |
|---|---|---|---|
| pass@1 | 0.456 | **0.674** | **+0.218** |
| **★ pass@8** | 0.855 | **0.935** | **★ +0.080** |
| 有信号 | 0.775 | 0.555 | −0.220 |
| 全对 8/8 | 8.0% | **38.0%** | +30 |
| 全错 0/8 | 14.5% | 6.5% | −8 |
| 平均长度 | 273 | 247 | −26 |

### ⚠️ 我预测 pass@8 会略降，实际涨了 0.080 —— 但有个测量陷阱

```
★ pass@8 不是"能力天花板"，是"用 8 次采样能探测到的能力"

  某题训练前 p = 0.02   pass@8 = 1 − 0.98⁸ = 14.9%  → 八成被测成"全错"
  训练后    p = 0.10   pass@8 = 1 − 0.90⁸ = 57%    → 多半被测成"有对"
  ★ 能力一点没变，测出来的 pass@8 却涨了
```

**→ 用 k=64 重测才有区分力（正在跑）。**

---

# ⚠️ 方法论：数据切片的使用记录与偏差分析（2026-09-02 补）

## TEST[0:200] 在哪些地方用过

```
① grpo.py 的训练中评估    evaluate() 用 TEST[:eval_n]，eval_n=200
                          每 25 步一次，一次训练共约 12 次
   ★★ ckpt_best 就是「这 12 次里分数最高的那次」存下来的  → 选择发生在这里
   ★  早停（--patience 3）也是看这个数

② baseline_gsm8k.py 基线   默认 offset=0 → TEST[0:200]
                          pass@1 0.456 / pass@8 0.855（训练前）
                          pass@1 0.674 / pass@8 0.935（step 300）

③ pass@64 测量            -n 50 offset=0 → TEST[0:50]（★ 是 ② 的子集）
```

## ★ 哪些数字真的有偏（逐个模型判定）

选择偏差的判据只有一条：**这批题参与过「选出这个模型」的决策吗？**

| 模型 | 怎么来的 | 判定 |
|---|---|---|
| 原始 Qwen | 下载的，没选过 | ✅ 干净 |
| lr 1e-6 末值 | 跑到 150 步中止 | ✅ 干净 |
| **lr 5e-6 run1 step 300** | **固定跑满 300 步** | ✅ **干净** |
| **★ ckpt_best (run2 step 100)** | **从 12 次评估里挑最高分** | ❌ **有偏** |
| run2 末值 (step 174) | 早停触发，早停看的是同一批题 | ⚠️ 弱相关 |

**★ 结论：「原始 0.7050 → run1 末值 0.7350，+0.030」这个对比本身干净。
只有「峰值 0.8050 / 0.7650」被选择偏差污染。**

## 但还有第二个理由要统一口径 —— 可比性（不是偏差）

```
TEST[0:200] 和 TEST[200:500] 是两批不同的题，难度可能差 2~3 个点
→ ★ 跨切片算 Δ 无效：分不清"模型强了"还是"这批题简单"

holdout_eval.py 的第一行（原始模型在两切片上的差）专门分离这两者：
  差 ≈ 0      → 切片难度相当，ckpt_best 的差就是纯选择偏差
  差 = 0.03   → ★ 那是题目难度差异，要从 ckpt_best 的差里扣掉
```

## ★★ 两个统计陷阱，同一个实验里连撞两次

```
陷阱一：测量灵敏度不够
  k=8 时 pass@8 涨了 0.080，看起来"天花板变高了"
  ★ 实际：某题 p 从 0.02 涨到 0.10，pass@8 从 15% 变 57%
    能力没变，只是被"探测到"的概率涨了
  → 解法：加大 k。k=64 时 Δ 只剩 +0.020（一道题）

陷阱二：在同一份数据上又选又评
  从 12 次评估里挑最高分 = 同时挑"模型好"和"这次运气好"
  ★ 两次实验 12 个观测：均值 0.7525，标准差 0.0223
    0.805 是均值 +2.35 SD —— 完全在"12 次取最大值"的正常范围内
  → 解法：换一份从没参与过决策的切片重评
```

## ★ 教训：探索性实验 vs 确认性实验

```
探索性   不知道会发生什么，边做边发现问题
         ★ 我们前五步都是这样：贪心 → pass@8 → pass@64 → 重跑 → holdout
         每一步都在质疑上一步的数字
         合理，但结论零散、口径不统一，而且补测成本高（多跑了三轮）

确认性   ★ 假设先写死，对比表先画出来，只负责填数字
         口径统一，跑完直接下结论
```

**★ 下一个实验（Base 版 R1-Zero）开工前，先把空表写出来再动手。**

## 最终对比表的设计（全部在 TEST[200:500] 上，口径统一）

| 指标 | 怎么测 | 对应的假设 |
|---|---|---|
| **★ 贪心正确率** | T=0，300 题 | RL 的真实提升，预期 +0.03~+0.05 |
| pass@1 | T=1，k=64，50 题 | 分布的平均质量 |
| **★ pass@64** | 同上 | 能力天花板，预期 \|Δ\| < 0.03 |
| **★ 温度差距** | 贪心 − pass@1 | 熵坍缩，预期 0.25 → 0.06 |
| logP（熵） | 训练日志读，不另测 | −0.78 → 约 −0.25 |
| 平均长度 | | 略降（Instruct 起点，不是 Base） |

---

# ★★ 最终结果（2026-09-02 完成，全部在无偏切片上测）

所有测量都在 **TEST[200:500]（贪心）/ TEST[200:250]（pass@k）** 上做 ——
这些题从未参与过 checkpoint 选择、早停或任何决策。

## ① 贪心正确率（温度 0，300 道新题）

| 模型 | 用过的 200 题 | ★ 没用过 300 题 | 偏差 | ★ 相对原始 |
|---|---|---|---|---|
| 原始 Qwen | 0.7050 | **0.6867** | +0.0183 | — |
| out_grpo_best step 100 | 0.7650 | 0.7267 | **+0.0383** | +0.0400 |
| out_grpo_best step 175 | 0.7350 | 0.7433 | −0.0083 | +0.0567 |
| **★ out_grpo_lr5e6 step 300** | 0.7350 | **0.7667** | −0.0317 | **★ +0.0800** |

```
★ 原始模型的偏差 +0.0183 = 纯题目难度差异（它没被选过）
★ ckpt_best 的偏差 +0.0383 − 0.0183 = 选择偏差约 +0.020
   → 0.7650 里有 0.020 是「12 次评估里挑中了运气好的那次」

统计显著性（n=300，配对检验）
  原始 → step300  +0.080   z≈3.1   ✅ 显著
  step100→step300 +0.040   z≈1.7   ⚠️ 不显著
  题目难度差异     +0.018   z≈0.5   ❌ 不显著
```

## ★★★ ② pass@k 完整曲线（温度 1.0，n=64，50 道新题）

| k | 原始 Qwen | step 100 | **★ step 300** | Δ(100) | **Δ(300)** |
|---|---|---|---|---|---|
| **1** | 0.5262 | 0.6994 | **0.7369** | +0.1732 | **★ +0.2107** |
| 2 | 0.6829 | 0.8122 | 0.8454 | +0.1293 | +0.1625 |
| 4 | 0.7978 | 0.8809 | 0.9077 | +0.0831 | +0.1099 |
| 8 | 0.8767 | 0.9269 | 0.9445 | +0.0502 | +0.0678 |
| 16 | 0.9247 | 0.9568 | 0.9619 | +0.0321 | +0.0372 |
| 32 | 0.9536 | 0.9697 | 0.9699 | +0.0161 | +0.0163 |
| **★ 64** | **0.9800** | **0.9800** | **0.9800** | **★ 0.0000** | **★ 0.0000** |

> **★★★ 三个模型的 pass@64 完全相同 = 0.9800（50 道里 49 道，且漏掉的是同一道）。
> Δ 从 +0.21 单调收缩到 0.0000，没有余数。**
>
> **这是「RL 只重排概率，不创造能力」能拿到的最干净的证据。**

## ★★ ③ 熵坍缩（无偏切片上的量化）

| | 贪心 | pass@1 | ★ 差距 |
|---|---|---|---|
| 原始 Qwen | 0.6867 | 0.5262 | **0.1605** |
| step 100 | 0.7267 | 0.6994 | **0.0273** |
| step 300 | 0.7667 | 0.7369 | **0.0298** |

**★ 「随手一答」与「最有把握的答案」之间的鸿沟缩小 81%。**
RL 消灭差路，不改进最优路 —— 贪心只涨 0.080，pass@1 涨 0.211。

## ★★ ④ 自发涌现：run 2 自己长出了 `####` 格式

```
                 输出 #### 的比例
  原始 Qwen          0.563
  ★ step 100 (run2)  0.972     ← 97.2%
  step 300  (run1)   0.597     ← 跟原始差不多

★ 奖励里只有 0/1 正确性，没有任何格式奖励
★ 同种子同配置，只因 GPU 非确定性走了不同轨迹 —— 一次长出来，一次没有
```

**机制（run 2 的数据支持）：**
```
带 #### 的回答  3111 条  正确率 0.705
不带的            89 条  正确率 0.494    ★ 差 1.4 倍

→ 判分器对带 #### 的抽取 100% 可靠，不带的偶尔抽错
→ 带 #### 的拿正奖励概率略高 → 被强化 → 熵坍缩固化成 97%
```

**★ 这是 R1「顿悟时刻」的同类现象：没人要求的行为，因为能提高奖励而被自发放大。
R1 长出的是"停下来重新检查"，这里长出的是"输出可解析的格式"。**

## ⑤ 推翻的结论：「训练 224 步后退化」不成立

```
在 TEST[0:200]（选 checkpoint 用的那批）
  step 224   0.8050  ← 峰值
  step 300   0.7350  ← 看起来跌了 0.07

★ 在 TEST[200:500]（从没用过）
  step 100   0.7267
  step 300   0.7667  ← 反而涨 0.04

三个独立指标（贪心 / pass@1 / pass@8）全都说训练更久更好
→ ★ 那个"见顶回落"是 200 道题上的噪声，不是真实退化
```

**⚠️ 连带后果：早停的判据也建立在那 200 道题上 → 它可能过早停了训练（run 2 在 174 步被停掉）。**

## ⑥ 该用哪个 checkpoint

```
★ out_grpo_lr5e6/ckpt_latest.pt（run 1，step 300）
  贪心 0.7667 / pass@1 0.7369 / pass@8 0.9445 —— 每一项都最好

❌ 不是 ckpt_best —— 它的高分含约 0.02 的选择偏差
```

---

# 下次改什么

```
lr              ★ 直接从 3e-6~5e-6 起步，1e-6 是浪费时间
每步题数        8 → 16 或 32   ★ 8 题的梯度噪声太大（步与步方向互相抵消）
★★ 组大小 k     8 → 64        ← 见「采样半径」那节
                这不只是降噪：★ k 决定了 RL 能触及的概率下限（0.7/k）
                k=8  只能改动 p>0.087 的时间线
                k=64 能改动 p>0.011 的 → ★ 预测 pass@64 会开始动
                ★ 这是个可证伪的实验：只改 k，看 pass@k 曲线哪一段被抬起来
熵正则          加一项，防止 logP 冲到 −0.2
                （或者用动态 KL：logP 涨太快时自动加大 β）
题目筛选        跳过历史正确率 >0.9 和 <0.05 的题
                ★ 基线里 8% 全对 + 14.5% 全错 = 22.5% 的算力白花
存最优 + 早停    ✅ 已加
采样加速        ★ 待办 B（4 卡数据并行）/ C（vLLM），见前文
```

# 产出的文件

```
_reward.py            GSM8K 判分器（唯一定义处）
download_qwen.py      下模型 + 跟自己的 201M 架构并排对比
download_gsm8k.py     下数据 + 判分器的 6 个测试用例（含奖励作弊的例子）
chat_qwen.py          聊天 + ★ /grpo 现场演示 GRPO 前 4 步 + /show 看完整 token
baseline_gsm8k.py     ★ pass@k 测量，支持 --ckpt 加载训练后权重，按 k 分组对账
grpo.py               ★ 训练主脚本，333 行，七步跟教学顺序一一对应
compare_runs.py       多次训练的六条曲线 ASCII 对比 + 自动诊断
compare_passk.py      多个模型的 pass@k 汇总表

out_grpo/             lr 1e-6，150 步，失败的对照
out_grpo_lr5e6/       lr 5e-6，300 步，末值 0.7350
out_grpo_best/        重跑中，将含 ckpt_best.pt
```

---

# ★★★ 方法论：环境（Environment）—— 新信息的唯一来源（2026-09-02）

## 一、修正一个我讲错的论证

**我原本的说法（错的）：**

```
一条 rollout 是模型自己按现有分布采出来的
  → 模型预测它交叉熵 ≈ 0
  → 零新信息，只有奖励那 1 bit
  → ★ 所以 RL 数据回不到预训练
```

**★★ 错在哪：把「信息量」定义成了对【生成它的那个模型】的惊讶度。**

**训练数据的价值，是对【读它的那个模型】的惊讶度：**

```
一条【被环境验证为正确】的 5000 token 轨迹
  ★ 对生成它的模型         →  0 bit（自己写的，当然不惊讶）
  ★★ 对下一代 / 学生模型    →  5000 × 2~3 bit = 完整的高质量训练数据

★★ 环境盖的那个「正确」的章，把 5000 个廉价 token 变成了 5000 个昂贵 token
```

## 二、所以回流确实存在，而且有正式名字

```
① rejection sampling（STaR 的工业版）
   ★ RL 模型生成海量样本 → 规则奖励 + 奖励模型筛选 → 高质量推理语料
   ★★ DeepSeek-R1 的关键环节

② ★★ mid-training —— 2026 年已是正式的训练阶段
   ICML 2026 的研究把语料显式切成三份不相交的 split：
   pre-training / ★ mid-training / post-training
```

**★ 量级：DeepSeek-R1 那批约 60 万条 × 5000 token ≈ 30 亿 token，
对比预训练 15 万亿 token → ★★ 占比 0.02%。**

**★ 所以它不是「积累到替代人类语料」，是「少量但浓度极高，在预训练末期定向注入」——
这正是它叫 mid-training 而不是 pretraining 的原因。**

## 三、AlphaZero 是「环境能产出人类没有的知识」最硬的证据

```
★ 完全不用人类棋谱，纯自我对弈
★ 人类的国际象棋概念【自发出现】在它的网络表示里
★★ 研究者从它身上【提取出人类没有的新概念】—— machine-unique knowledge
★★★ 这些概念【可以教给人类大师】
★ 自我对弈数据里「可教时刻」比人类棋谱还多
```

（Bridging the human–AI knowledge gap through concept discovery and transfer in AlphaZero, PNAS）

## 四、★★ 修正版的飞轮图

```
    ┌──────────────────── 人类文本（★ 2026~2032 见底）
    │
预训练（人类数据 + ★ 改写型合成数据）
    │
    ├──★★ mid-training ◀────────┐
    │   （★ 高浓度验证数据）      │
RL 训练 ──────────────────────────┤
    │   ★ 轨迹 → 筛选 → ★★ 留下  │
专家模型（★ 分领域，不是分环境）
    │
★ on-policy 蒸馏成一个学生 → 下一代
```

## 五、★★★ 增长速度由什么决定

```
❌ 不是算力     —— 有算力可以采无限多轨迹，但都在同一个环境里绕
★★ 是环境的多样性 —— 新信息只能从新环境里来

★ AutoEnv 已把「造一个环境」压到 $4
★ OpenReward：330+ 环境、450 万任务、托管 API
★ Prime Intellect Environments Hub：环境发布一次，任何训练框架都能用
★ Open Reward Standard (ORS)：在 MCP 之上扩展出 RL 原语
★★ 但 AgentScaler 的结论：多样性比数量重要，而【多样性还造不出来】

★★★ 积累速度 = 人（或 AI）能造出多少种【真正不同】的环境
     —— 整条链上唯一还没被自动化的一环
```

**★ 另一句值得记的：「评测和训练正在合并 —— 如果评测框架产出结构化奖励并记录完整轨迹，
那些输出本身就是训练数据。」→ ★★ 写一个好评测 = 造一份训练数据。**

---

# ★★★ 怎么造环境（从 Countdown 实战提炼）

## 一、环境的五个组件

```
Tasks     题目 —— ★ 必须能程序化无限生成
Harness   提示词模板 + max_new + 停止条件 + 工具 + 沙箱
Verifier  ★★ 判分函数：completion → 奖励 [0,1]
State     多轮时的状态（单轮环境为空）
Config    难度旋钮（Countdown 里是「几个数」「目标范围」）
```

**★ countdown.py 就是一个完整的环境。回头看，五个组件都在里面。**

## 二、★★ 可验证性是一个连续谱，不是有/无

| 档 | 判分方式 | 例子 | 单次成本 | 单次耗时 | ★ 能不能直接跑 GRPO |
|---|---|---|---|---|---|
| ① **规则可验证（RLVR）** | 一个函数 | 数学答案、代码测试、Countdown | **$0** | ~0.1 ms | **★★ 能，最强的一档** |
| ② 模型可验证 | LLM-as-judge | 写作质量、翻译、总结 | ~$0.001 | ~1 s | ★ 能，但判分器会被 hack |
| ③ 人类反馈 | 人 | 偏好、安全、风格 | ~$0.5 | ~60 s | ★★ 不能直接跑，见下 |
| ④ 无法验证 | 无 | 「写一首好诗」「给我人生建议」 | — | — | ❌ 只能靠预训练 + SFT |

**★★ 算笔账：一次 RL 训练要几百万次判分**

```
规则判分     100 万次 × $0      = ★ 免费
LLM judge    100 万次 × $0.001  = ★ $1,000（可接受）
★★ 人类      100 万次 × $0.5    = ★★★ $50 万（不可能）

→ ★ 所以 RLHF 的做法：人标 10 万条 → 训一个【奖励模型】→ 用模型判 100 万次
  ★★ 奖励模型的本质，就是把「贵而慢的验证」压缩成「便宜而快的验证」
```

## 三、★★ 造环境检查清单（每一条都是踩过的坑）

```
① ★ 起点非零、非满
   基线正确率是 0 → 没有梯度信号（★ 4 个数的 Countdown 就是，0.0175）
   基线正确率是 1 → 没有提升空间
   ★★ 甜区大致在 0.05 ~ 0.5

② ★★ 组内必须有方差
   同一题采 k 条，得分要有差异 —— ★ 全同则优势全 0，梯度为 0
   ★★ 可测指标：组内总分标准差（我们实测 0.1009，100% 的题有信号）
   ★ 注意：要算【总分】的方差，不是【正确率】的方差
      （只算正确率时 86% 的题「无信号」，算总分时 100% 有信号）

③ ★★ 稀疏 → 阶梯
   只奖励最终答案，起点低的时候训不动
   ★ Countdown 的四级：fmt 0.05 / parse 0.05 / nums 0.15 / correct 0.75
   ★★ 中间级要给【连续】的部分分，不是 0/1
      nums_frac = max(0, (匹配数 − 多余数) / 总数)

④ ★★★ 判准只能有一份定义
   同一个判分器出现在 3 个文件里 → 改了一个忘了另两个 → 结论全错
   ★ 抽成一个模块（_reward.py），训练和评测共用

⑤ ★★ 想清楚「最懒的通过方式」是什么
   GSM8K 判分器取【第一个】数字 → 模型直接复述题干里的数就能骗过
   ★ 改成取【最后一个】数字才对
   ★ 写完判分器，先自己扮演「作弊的模型」攻一遍

⑥ ★★ 提示词和奖励不能矛盾
   提示说「每个数字至多用一次」，奖励却要求「恰好一次」→ 模型无所适从

⑦ ★★ 必须处理模型的真实输出，不是你想象的输出
   实测踩到的：`44 + 19 + 35 = 98`（带等号）、LaTeX \times、markdown **、Unicode ×、÷
   ★ 先跑 100 条基线，把真实输出打印出来看，再写解析

⑧ ★ eval 要安全
   用 AST 白名单，不能用裸 eval（★ 测试用例里放一条 __import__('os') 确认被拒）

⑨ ★★★ 最隐蔽的一条：有没有一个退化策略能通吃
   ★ 3 个数的 Countdown 搜索空间只有 192 种 → 不需要搜索
   → ★★ RL 找到的最优解是「闭嘴，直接给答案」
   → 长度 249 → 61，纠错措辞 0.16 → 0.00（★ 推理被训没了）
   ★★ 单环境训练的经典失败模式：不是 reward hacking（模型没作弊，
      正确率真从 3.5% 涨到 25%），是【环境过拟合】
   ★★★ 解法不是修奖励，是【多环境混训】—— 让任何单一退化策略都拿不到高分

   ★★★ 补（2026-09-02，反向迁移测试后）：这一条同时决定了任务是不是【好的迁移源】
      问「在这个任务上拿高分最省事的方法是什么」
        答案是一种通用技能  →  好源，训它别的相近任务会跟着涨（GSM8K → Countdown pass@8 2.9×）
        答案是一条捷径      →  坏源，训它别的任务会掉（Countdown-3 → GSM8K −0.13 / −0.36）
      Countdown-3：192 种组合，最省事是模式匹配 → 坏源
      GSM8K：必须读全条件逐步算 → 好源
      ★ 「有没有退化策略能通吃」和「是不是好的迁移源」是同一个问题的两面。详见「反向测试」一节。
```

## 四、★★ reward hacking vs 环境过拟合（容易混）

| | 定义 | 表现 | 怎么发现 |
|---|---|---|---|
| **Reward hacking** | ★ 骗过判分器，没真解决问题 | 奖励涨、真实能力不涨 | ★ 单独测真实指标 |
| **★★ 环境过拟合** | ★ 真解决了这个环境，但策略不迁移 | ★★ 奖励涨、真实能力也涨、但换环境就崩 | ★★★ 必须换环境测 |

**★ 后者更隐蔽：所有曲线都健康，只有换环境才露馅。**

## 五、★ 环境的难度旋钮要留出来

```
★ Countdown 的教训：
  4 个数 → 是真搜索，但正确率 0.0175 → ★ 训不动
  3 个数 → 能训（0.035 → 0.25），但不需要搜索 → ★★ 长不出推理

★★ 中间那个「够难所以需要搜索、又够简单所以有非零起点」的窗口，我在降难度时跨过去了

★ 所以：难度要做成【可调参数】，而且要先扫一遍难度曲线再定
```


---

# ★★★ 第五阶段：R1-Zero 复现（Base + Countdown）—— 完成（2026-09-02）

## 配置

```
模型      Qwen/Qwen2.5-1.5B（★ Base，不是 Instruct）
任务      Countdown 3 个数
每步      8 题 × k=16 = 128 轨迹（★ 每 rank 2 题 × 16 = 32 条）
lr 5e-6   KL β 0.0005   max_new 512   micro 2   AdamW8bit
硬件      ★ 4×4090 多进程（torchrun），手动 all_reduce 梯度
用时      ★ 94.2 分钟 / 300 步 = 约 19 秒/步
```

## 一、★★ 结果：RL 极其成功（在分数维度上）

| 指标（eval 贪心） | 起点 | 峰值 @275 | 倍数 |
|---|---|---|---|
| 总分 | 0.2205 | **0.5537** | 2.5× |
| 格式 | 0.900 | **1.000** | 饱和 |
| 解析 | 0.900 | **1.000** | 饱和 |
| 数字 | 0.695 | **1.000** | 饱和 |
| **★★ 答案正确** | **0.0350** | **★ 0.4050** | **★★★ 11.6×** |

**★ 对比 GSM8K 那次（贪心 +0.080），这次的提升量级完全不同。**

## 二、★★★ 三个假设的裁决

```
★ H1 长度增长          ❌ 彻底失败，且【方向相反】
   eval 贪心   186 → 48   （★★ 0.26×，预期是 2~3×）
   训练温度1   232 → 71

★★ H2 纠错措辞涌现     ❌ 彻底失败，且【被消灭】
   宽松（有没有）  0.146 → 0.005   （★★ 减少 96.6%）
   严格（几次/条） 0.24  → 0.01
   ★ 因果诊断从 −0.029 变成 nan —— 不是没算，是【绝迹到无法统计】

★★ H4 四级阶梯自下而上饱和  ✅ 成立
   格式/解析/数字在 step ~110 全部饱和
   ★ 正确率一路爬到最后（0.035 → 0.405）
```

## 三、★★★★ 最重要的：**15 步的诊断，正确预测了 300 步的结果**

```
★ step 15 时测到
   「★差」（有纠错措辞的轨迹 − 没有的，平均奖励差）
   均值 −0.0295，15 步里 12 负 3 正
   ★ 符号检验 p = 0.018

★ 当时的判断
   「说 wait 的轨迹得分更低 → 优势为负
     → ★★ RL 不但不会放大它，还会把它训没」

★★★ 300 步后的实测
   纠错措辞 0.146 → 0.005（★ 减少 96.6%）
   ★★ 预测完全兑现
```

**★★ 方法论价值：一个 5 分钟、15 步的诊断，能预测 1.5 小时训练的定性结果。
这个方法可迁移到任何 RL 实验 —— 想知道某个行为会不会被强化，
就测「带这个行为的轨迹 vs 不带的，平均奖励差」。**

**★ 通用配方：**
```
① 定义一个可正则匹配的行为标记（如 wait/hmm/actually）
② 每步统计：有标记的轨迹平均奖励 − 无标记的平均奖励
③ ★ 跑 15~20 步，做符号检验
④ ★★ 显著为负 → 该行为会被消灭；显著为正 → 会被放大；不显著 → 无关
```

## 四、★★ 为什么 H1/H2 反向：**任务不需要搜索**

```
3 个数的 Countdown 搜索空间
  排列 3! = 6 × 运算符 4² = 16 × 括号 2 = ★ 192 种

★ 192 种对 1.5B 来说不是「搜索」，是「模式匹配」
★★ 奖励结构里没有任何一项奖励思考
   （fmt 0.05 + parse 0.05 + nums 0.15 + correct 0.75，全部能用短答案拿到）
→ ★★★ RL 正确地发现：闭嘴直接答，正确率更高（0.035 → 0.405）
```

**★ 这不是 reward hacking —— 模型没作弊，它真的答对得更多了。
是【环境过拟合】：真解决了这个环境的问题，但学到的策略不迁移。**

**★★ 我掉进的窄窗口：**
```
4 个数 → 是真搜索，但基线正确率 0.0175 → ★ 训不动
3 个数 → 能训（0.035 → 0.405），但不需要搜索 → ★★ 长不出推理
★★★ 中间那个窗口，在降难度时被跨过去了
```

## 五、★ 其他观察

```
① ★★ 组内标准差在收缩 —— 熵坍缩在 Countdown 上的表现
   step 0   sd 0.204
   step 113 sd 0.199
   step 223 sd 0.080
   step 299 sd 0.063
   ★ 梯度信号在变弱，训练接近饱和

② ★ 梯度范数后期剧烈波动
   step 283 |g| 0.08  ←→  step 293 |g| 2.23（★ 差 28 倍）
   ★★ 原因：每步只有 8 道题，题目难度差异主导

③ ★ 相邻步分数可差 2.7 倍
   step 286 分 0.302 C0.070  ←→  step 287 分 0.824 C0.766
   ★★ 同样是 8 题太少 → 下次应加大每步题数

④ ★ KL 稳定在 0.05~0.08，没有失控

⑤ ★★ 采样时间后期大幅下降（13~15s → 常见 3~5s）
   ★ 因为输出从 232 token 缩到 71 token
   → ★★ 总用时 94.2 分钟，比预估的 100 分钟还快
```

## 六、★★ 多进程 4 卡的实测收益

```
             采样    训练   通信   合计
单卡估算     78 s   15 s    0    93 s
★ 多进程     13.5s   5 s   1.2s  ★ 19.7 s
                                  ─────
                            ★★ 加速 4.7×

★ 节省来源拆解
   采样并行（解决 GIL）  64.5 / 73 = ★ 88%
   训练并行（1卡→4卡）   10.0 / 73 =   14%
   通信开销             −1.2 / 73 =   −2%
★★ 主要功劳是 GIL，不是训练并行，更不是权重同步
```

## 七、★★★ 待验证：环境过拟合假说

```
★ 假说：ckpt_best 学到的「闭嘴直接答」只在 Countdown 上成立
★★ 决定性实验：拿 ckpt_best 去测 GSM8K（需要中间步骤才能算对）

   模型 A  Qwen2.5-1.5B-Base 原始
   模型 B  ckpt_best（Countdown RL 后）
   ★ 用同一个 prompt 模板测两者，200 题

   ★★ 预测：B 显著低于 A → 环境过拟合被证实
   ★ 如果 B ≈ A → 说明只是格式偏好变了，能力没被破坏
```

**★ 第二轮的设计（不是「3 个数改 4 个数」）：**
```
★★ 多环境混训 —— Countdown（短答案最优）+ GSM8K（★ 必须写中间步骤）
★ 预测：混训后长度不会崩、纠错措辞不会归零
★★ 若成立 → 证明「捷径是环境的单一性喂出来的，不是模型的能力上限」
```

## 产出

```
grpo_countdown_mp.py   ★ 多进程 4 卡 GRPO，423 行
countdown.py           ★ Countdown 环境（任务生成 + 提示 + 四级判分）
baseline_countdown.py  ★ 基线测量 + 信号统计
out_cd_mp/ckpt_best.pt ★★ 峰值 0.5537 @ step 275
cd_mp_run1.log         完整 300 步日志
```


---

# ★★★ 跨任务测试：Countdown RL 后的模型去做 GSM8K（2026-09-02）

## 设计：2×2，把「权重变量」和「模板变量」分开

```
                  R1 模板（★ 训练过的）   中性模板（★ 没训过）
   ① Base              A                       B
   ② RL 后             C                       D
```

## ★★ 结果（200 题，贪心，GSM8K test[0:200]）

| | R1 模板 | 中性模板 |
|---|---|---|
| **正确率** Base | 0.5200 | 0.4800 |
| **正确率** RL 后 | **0.1600** | **0.3500** |
| ★ Δ | **−0.3600** | **−0.1300** |
| **长度** Base | 183 | 257 |
| **长度** RL 后 | **71** | **272** |
| ★ Δ | **−112** | **+15** |

```
★ McNemar（配对）
  R1 模板    训坏 84 / 训好 12   z = +7.35   ★★ 极显著
  中性模板   训坏 36 / 训好 10   z = +3.83   ★★ 显著
```

## ★★★ 两个机制同时存在，可以用【长度】完美分离

```
★ 机制 A —— 灾难性遗忘（通用能力损伤）
   证据：中性模板下 −0.13，★★ 但长度没掉（257 → 272，反而涨了）
   ★★ 解读：还是那样写，但【算错了更多】—— 纯粹的能力下降

★★ 机制 B —— 条件性策略绑定（模板触发的退化）
   证据：R1 模板下额外的 −0.23，★★ 伴随长度 −112（掉 61%）
   ★★ 解读：一看到 <think> 这个「暗号」就不写过程了

★★★ 分解
   R1 模板的 −0.36 = 通用损伤 −0.13 + ★ 模板特异的额外损伤 −0.23
```

**★★ 判别规则（可迁移到任何跨任务测试）：**

| 长度 | 正确率 | 结论 |
|---|---|---|
| **不变** | 跌 | ★ 能力损伤（灾难性遗忘） |
| **暴跌** | 跌 | ★★ 策略切换（条件性策略绑定） |
| 不变 | 不变 | ★ 无副作用 |

## ⚠️ 教训：**同一个统计错误犯了第二次**

```
★ 20 题的初跑结果
   中性模板 Base 0.25  vs  RL 0.25   → Δ = 0.00
   ★ 我据此下结论「能力没坏，只是条件性策略绑定」

★★ 200 题
   中性模板 Base 0.48  vs  RL 0.35   → ★ Δ = −0.13，z = 3.83 显著

★★★ 20 题时两个格子都碰巧偏低（0.25/0.25），差值看起来是 0
```

**★★ 这跟第四阶段那条「5 个样本的推断不可信，尤其当它符合你的预期时」是同一个错误。
★★★ 第二次犯 → 定成硬规则：**

```
★ 任何「Δ ≈ 0 所以某效应不存在」的结论，
  ★★ 在样本 < 100 时一律不下 —— 零效应比非零效应更需要样本量
   （非零效应可以靠效应量大来弥补样本小；零效应不行）
```

## ★ 对下一轮的意义

```
★ 单环境 RL 有两种代价，都得治：
  ① 灾难性遗忘  → ★ 用 KL 约束（β 调大）/ LoRA / 混入原分布数据
  ② ★★ 策略绑定 → ★ 多环境混训（同一模板下不存在通吃的退化策略）

★★ 第二轮设计：Countdown + GSM8K 混训，同一个 R1 模板
   ★ 可证伪预测：R1 模板下的长度不会崩（−112 → 接近 0）
   ★ 附带检查：中性模板下的正确率损伤会不会也变小（若变小，说明混训也缓解了遗忘）
```

## 产出

```
crosstask_eval.py   ★ 2×2 跨任务评测，210 行
crosstask.json      ★ 原始数据（每题的 ok / pred / gold / 长度 / 输出文本）
```


---

# ★★★ 路线图（2026-09-02 定）

```
① ★ 验证「条件性策略绑定」   —— 混训，主判据是【长度分化】
② ★★ 追顿悟                 —— ★ 用 ★差 做配置扫描，不盲试
③ ★★★ 多轮环境（Agent RL）  —— 猜数字，credit assignment
```

---

## 阶段一：验证「条件性策略绑定」

### ★★ 先把这个概念定义精确

```
★ 定义
  RL 改变的不是 P(短答案)
  ★★ 而是 P(短答案 | 训练时的那个上下文)

  → ★ 策略被【绑定】在一个上下文上，那个上下文成了开关
```

**★ 已有证据（相关性，不是因果）：**

| | R1 模板（训练过） | 中性模板（没训过） |
|---|---|---|
| 长度 | 183 → **71**（−61%） | 257 → **272**（+6%） |

**★★ 同一组权重，行为差异完全由上下文决定。**

### ★★★ 机制声称 + 可证伪推论

```
★ 声称：退化策略之所以学得成，是因为【所有训练轨迹共享同一个上下文】
        → ★★ 模型可以在【上下文】这一层解决问题，不用读题

★★ 推论：打破「同一上下文 → 同一最优策略」的对应，退化策略就学不成
        → ★ 混训：同一个 R1 模板下，Countdown 要短答案，GSM8K 要长推理
```

### ⚠️ 我原来说的判据有漏洞

```
❌ 原判据「混训后 R1 模板下长度不崩」
   ★ 漏洞：长度不崩可能只是因为 GSM8K 把平均值拉高了，
          ★★ 不能证明绑定被打破

✅ ★★★ 正确的主判据：【长度分化】
   ★ 训练日志里【分任务】打长度：
      step 0    Countdown 题 ~230   GSM8K 题 ~250   → 差 20（★ 没分化）
      step 300  Countdown 题 ~70    GSM8K 题 ~250   → ★★ 差 180（分化了）

   ★★ 分化 = 模型不再用「看到模板就闭嘴」这个开关，而是【在读题】
   ★★★ 这才是「绑定被打破」的直接证据
```

### 次判据

```
★ 跨任务测第三个任务（不能再用 GSM8K —— 混训里训过它，会污染）
  候选：简单常识问答 / MMLU 子集
  ★★ 看通用能力（灾难性遗忘）有没有被缓解
```

### 实现要点

```
① ★ 任务模块化：countdown.py 旁边加 gsm8k_task.py（判分器复用 _reward.py）
② ★ 取题按比例混采（默认 50/50）
③ ★★ 两个任务的奖励必须归一到同一尺度
      —— 否则梯度被高分任务主导
      Countdown 满分 1.0，GSM8K 也要归到 1.0
④ ★★★ 日志【分任务】打：长度、正确率、★差
```

---

## 阶段二：追顿悟 —— ★★ 用 ★差 做配置扫描，不盲试

### ★ 从已有教训推出顿悟需要的三个条件

```
① ★ 任务必须【真的需要搜索】
   否则 RL 会主动消灭推理（★ 3 个数的 Countdown 已证）

② ★ 起点正确率非零
   4 个数是 0.0175 —— ★ 太低，训不动

③ ★★★ ★差 必须【不显著为负】
   ★ 这是唯一能在 15~20 步内测出来的条件
```

### ★★★ 所以方法是：**先扫描，后长跑**

```
★ 每个配置只跑 20 步（约 7 分钟），只看两个数：
   ① ★差 的符号（符号检验）
   ② 组内标准差（有没有梯度信号）

★★ 扫 6~8 个配置 ≈ 1 小时
★★★ 找到 ★差 ≥ 0 的配置，再投入几小时长跑
```

**★ 这把「盲目试参数」变成「有指标的搜索」—— 是 ★差 这个工具最有价值的用法。**

### 待扫的配置维度

```
数字个数     3 / 4 / 5        ★ 4 个数才是真搜索
目标范围     容易 / 难
模型         1.5B / 3B(+LoRA) ★ 3B 显存装不下全参
是否混训     单 / 混
```

### ★ 判读

```
★★ 找到 ★差 ≥ 0 的配置  → 投入长跑，顿悟有机会
★  全部配置 ★差 都为负  → ★★★ 这本身是个硬结论：
   「1.5~3B 规模上，纠错措辞与奖励系统性负相关」
   —— 给「模型多大才配得上顿悟」这个问题定了一个实测下界
```

---

## 阶段三：多轮环境（Agent RL）

### ★ 为什么是猜数字

```
★ 环境想一个 1~1000 的数
★ 模型猜 → 环境回「大了/小了」→ 再猜 → …
★ 奖励：猜中给分，★ 步数越少分越高
```

| | 猜数字 | Countdown |
|---|---|---|
| 状态 | **★ 有**（历史反馈） | 无 |
| 轮数 | **★★ 多轮** | 单轮 |
| **★★★ 最优策略** | **★ 已知（二分，10 步内必中）** | 未知 |

**★★ 「最优策略已知」是关键 —— 能精确量化模型离最优多远，而不是只看正确率。**

### ★★ 它能测的核心问题：**模型会不会【用反馈】**

```
瞎猜的模型         ★ 忽略「大了/小了」→ 平均要很多步
★★ 会用反馈的模型   ★ 每次缩小一半 → 10 步内必中

★★★ 如果 RL 让模型从瞎猜学到二分 —— 那是一次可精确验证的策略涌现
     （比 wait/hmm 硬得多，因为二分能被程序判定）
```

### ★ 会撞上的新技术问题（单轮 RL 里不存在的）

```
① ★ 跨轮的 logP 怎么算（环境反馈那些 token 不能算进 loss）
② ★ 状态怎么进 prompt（历史全拼进去？还是压缩？）
③ ★★★ 中间轮次的优势怎么分配 —— credit assignment
      最终只有一个奖励，但有 10 轮动作
```


---

# ★★★ RL 阶段总览（2026-09-02 归档）—— 到目前为止做了什么

细节散在前面各节，这一节是索引 + 两处之前没写清楚的读法。

## 一、时间线

| 日期 | 做了什么 | 关键数字 | 细节在 |
|---|---|---|---|
| 09-01 | RL 理论：GRPO / RLVR / KL / 环境 / Agent RL | 选定 Qwen2.5-1.5B-Instruct + GSM8K | — |
| 09-01~02 | GSM8K GRPO：两轮训练 + 四轮验证 | 贪心 0.687 → 0.767（p≈0.001），pass@64 不变 | 第四阶段、最终结果、采样半径 |
| 09-02 | Countdown 环境：任务生成 + 判分 + 阶梯奖励 | 三轮修 bug，100% 的题有信号 | 怎么造环境 |
| 09-02 | 4 卡并行：线程版失败，多进程版成功 | 线程慢 4.5×，多进程快 4.7×（19.7 秒/步） | 待办 RL 采样加速 |
| 09-02 | R1-Zero 复现：Base + Countdown，300 步 | 正确率 0.035 → 0.405，长度 186 → 48，反思 0.146 → 0.005 | 第五阶段 |
| 09-02 | 跨任务 2×2：RL 后的模型去做 GSM8K | R1 模板 0.52 → 0.16，中性模板 0.48 → 0.35 | 跨任务测试 |
| 09-02 | 混训 Countdown + GSM8K（进行中） | 50 步：捷径没形成，CD 变长，两个正确率都涨 | 路线图·阶段一 |

## 二、GSM8K：两轮训练 + 四轮验证，每轮回答一个问题

### 两轮训练

| | lr | 步数 | 结果 | 目录 |
|---|---|---|---|---|
| 第一轮 | 1e-6 | 150 | 曲线不动，失败，留作对照 | out_grpo/ |
| 第二轮 | 5e-6 | 300 | TEST[0:200] 从 0.705 涨到峰值 0.765 | out_grpo_lr5e6/ · out_grpo_best/ |

第二轮重跑过一次（加存最优 checkpoint），两次曲线不同 —— GPU 浮点加法不满足结合律，
1e-7 的差异 300 步后走成不同轨迹。这是「GPU 非确定性」的来源。

### 四轮验证

**① 真的变好了吗？** —— holdout
```
训练时看的是 TEST[0:200]，那条曲线 224 步后往下走，一度判定「训过头」
★ 拿从没碰过的 TEST[200:500] 共 300 题重测：
   0.6867 → 0.7667，+0.080，McNemar z=3.21，p≈0.001
   ★ step 300 (0.7667) > step 100 (0.7267)  →「224 步后退化」是 200 题的噪声
```

**② 变好的是什么？** —— pass@k 曲线（每题 64 条，50 道无偏题）
```
pass@1   0.5262 → 0.7369   +0.21
pass@64  0.9800 → 0.9800   ★ Δ = 0.0000
温度差距（贪心 − pass@1）  0.1605 → 0.0298，缩 81%
logP     −0.78 → −0.22（熵坍缩）
★ 天花板没抬高，是把偶尔能对的变成稳定能对 → 推出采样半径理论
```

**③ 怎么变好的？** —— 翻转分析
```
逐题比训前训后：40 题错→对，16 题对→错
★ 修好的案例 4/4 都是同一种：原来漏掉题干里某个条件，训后读全了
   → RL 提升的机制是「读题更完整」，不是「算得更准」
```

**④ 测量本身可信吗？** —— 修工具
```
★ 判分器取【第一个】数字 → 乌龟题五条回答全抓到题干的「40 秒」→ 改取【最后一个】
★ 判分器在 3 个文件各有一份 → 统一到 _reward.py（判准只能一份定义）
★ 从 5 道题推断判分器系统性漏判 7 倍 → 200 题实测 1.0 倍 → 推断错了
   （「小样本 + 符合预期」那条教训的第一次）
★ 选择偏差：从 N 次评估里挑最大值，虚高约 +0.02
```

## 三、跨任务 2×2 的读法 —— 同一个模型，只换提示词

四个格子 = **2 个模型 × 2 段提示词**，题目是同一批 GSM8K test[0:200]，贪心解码。

|  | 训练时用的模板（R1，`<think>` 结尾） | 没见过的模板（`Question: … Answer:`） |
|---|---|---|
| **Base**（没训过） | 正确率 0.52 · 长度 183 | 正确率 0.48 · 长度 257 |
| **RL 后**（Countdown ckpt_best，step 275） | **正确率 0.16 · 长度 71** | **正确率 0.35 · 长度 272** |

```
上下两行 = 不同的模型（权重不同）
左右两列 = ★ 同一个模型，换开头那几行文字
```

**看 RL 后那一行：同一组权重、同样 200 题，只因开头文字不同，
正确率 0.16 对 0.35，长度 71 对 272 —— 这就是「绑定在模板上」的字面证据。**

**两列各自的下降幅度：**
```
左列掉 0.36  =  ★ 右列也躲不掉的 0.13（能力受损，灾难性遗忘）
              +  ★ 只在左列出现的 0.23（模板触发，策略绑定）

用长度分开：右列长度没变（257 → 272），左列暴跌（183 → 71）
McNemar：左列 z=+7.35（训坏 84 / 训好 12），右列 z=+3.83（训坏 36 / 训好 10）
```

## 四、RL 阶段到目前为止的收获（一句话版）

```
理论   采样半径 p > 0.7/k · 强化两条件 · 组合涌现 · RL 不加知识只重分配 ·
       新信息只从环境来 · 验证过的轨迹对下一代是数据 · 环境是瓶颈 ·
       单环境 → 模板反射 · 多环境 = 堵捷径
诊断   ★差 符号检验 15 步预测 300 步 · 长度分离「遗忘」和「绑定」
工程   GIL 让线程版更慢 · 多进程数据并行不需要同步权重 · 采样占七成 · 8-bit Adam
方法   判准一份 · 贵的先存 · 小样本不可信（犯了两次）· 零效应更需要样本 ·
       探索/确认分切片 · 目标表先定 · 长跑前 15 步诊断 · 一次改一个变量
```


---

# ★★★ RL 阶段叙述版（2026-09-02）—— 每一步是怎么被上一步逼出来的

## 一、GSM8K：先证明 RL 有效，顺便撞上它的边界

补完理论后选 Qwen2.5-1.5B-Instruct + GSM8K：够小能全参、有程序判分、基线 0.7 在黄金区间。
lr 1e-6 跑 150 步不动；换 5e-6 跑 300 步，验证集 0.705 → 0.765，第一次看到 RL 在动。

**修工具。** 乌龟题五条回答判分器全判对但答案明显错 —— 取的是回答里【第一个】数字，
抓到题干的「40 秒」。改取最后一个。接着发现判分器在三个文件各一份，统一成 _reward.py。

**四轮验证，每次结论都被下一个实验推翻：**
- TEST[0:200] 曲线 224 步后往下，判定「训过头」→ 拿从没碰过的 TEST[200:500] 重测，
  step 300 比 step 100 还好，0.6867 → 0.7667，p≈0.001。「退化」是 200 题的噪声。
- pass@k：pass@1 0.526 → 0.737，★ pass@64 0.980 → 0.980 一点没动。
  ★★ 这是 RL 阶段最重要的一个数：RL 没让模型会做任何原来不会的题，只是把偶尔对变成稳定对。
- 翻转：40 错→对，16 对→错。修好的 4/4 都是「漏了题干一个条件，训后读全了」。
  RL 改的不是算术，是读题完整度。
- 自发涌现：没有格式奖励，`####` 出现率 56% → 97%，因为带它更容易被判对。

**从 Δpass@64 = 0 推出采样半径**：k 条里只有概率 > 0.7/k 的行为会被采到，采不到就没梯度。
RL 只能在半径内重新分配概率。熵坍缩 = 半径收缩。
一个行为被强化要两个条件：① 出现在采样里 ② 带着它的轨迹得分更高。

## 二、Countdown：造环境、上 4 卡、复现 R1-Zero

**为什么换。** GSM8K 上有效但没看到推理变长、没看到 wait。想复现 R1-Zero「Base 纯 RL 长出思维链」。
要换 Base（否则说不清是 SFT 还是 RL 的功劳），换需要搜索的任务。

**造环境是第一次。** 判分器要处理真实写法（带等号、LaTeX、Unicode），AST 白名单防注入，
提示词「至多一次」vs 奖励「恰好一次」的矛盾要统一。4 个数基线 0.0175 训不动，降到 3 个数。
稀疏 0/1 让 99% 的组全错、优势全零，改四级阶梯，数字那级给连续部分分。
★ 测信号时先算正确率的方差，86% 的题「无信号」；改算总分的方差，100% 有 —— 统计量要跟奖励一致。

**上 4 卡走了弯路。** 一进程四线程比单卡【慢 4.5 倍】—— GIL，HF generate 每个 token 回 Python 一趟。
换 torchrun 四进程，各采各算，all_reduce 求和。梯度相同 → 权重天然一致 → 不需要同步权重。
93 秒/步 → 19.7 秒，4.7 倍。省的 73 秒里 88% 来自采样并行。

**300 步 94 分钟。** 正确率 0.035 → 0.405（11.6 倍），阶梯自下而上饱和 —— RL 极成功。
但想看的两件事全反了：长度 186 → 48，反思 0.146 → 0.005，被消灭。

**★★ 第 15 步就预测到了。** ★差 列（有反思措辞的轨迹均分 − 没有的）前 15 步 12 负，p=0.018。
判断：说 wait 的分更低 → 优势为负 → RL 会压它。300 步后反思减少 96.6%。
5 分钟诊断预测了 90 分钟训练。

**为什么反了。** 3 个数只有 192 种组合，不需要搜索。奖励没有一项奖励思考，最优策略是闭嘴直接答。
不是作弊（真的更对了），是对这一个环境的过拟合。

## 三、跨任务和混训：RL 到底改了什么

**2×2。** 两模型 × 两模板，同一批 GSM8K test[0:200]，贪心。
R1 模板 0.52 → 0.16，中性模板 0.48 → 0.35。长度：R1 下 183 → 71，中性下 257 → 272 没动。
同一组权重只换开头几行，行为完全两样 → 「变短」挂在模板上 → 条件性策略绑定。
中性模板那 0.13 是换提示词也躲不掉的 → 能力本身受损 → 灾难性遗忘。用长度分开。
★ 第二次犯同样的错：20 题中性模板 0.25 对 0.25 说「能力没坏」，200 题掉 0.13。零效应更需要样本。

**混训进行中。** 50 步：捷径没形成，但 CD 变长到 315 而不是变短，GSM 梯度话语权 3~7 倍。
两个正确率都涨。要看 CD 能不能靠长输出继续爬。

**一句话：GSM8K 证明 RL 能把偶尔对变成稳定对但碰不到天花板；
Countdown 证明 RL 会为了奖励主动杀掉没用的行为、而且 15 步就能预测；
跨任务证明它学到的可能只是一个绑在提示词上的反射。**

## 四、附：2×2 的两段模板原文，以及「为什么两个模板下都变差」

### R1 模板（训练时那段，38,400 条轨迹都以它开头）

```
A conversation between User and Assistant. The user asks a question, and the Assistant
solves it. The Assistant first thinks about the reasoning process in the mind and then
provides the user with the answer. The reasoning process and answer are enclosed within
<think> </think> and <answer> </answer> tags, respectively.
User: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and
bakes muffins for her friends every day with four. She sells the remainder at the
farmers' market daily for $2 per fresh duck egg. How much in dollars does she make
every day at the farmers' market?
Show your work in <think> </think> tags. Return the final numeric answer in
<answer> </answer> tags, for example <answer> 42 </answer>.
Assistant: Let me solve this step by step.
<think>
```

### 中性模板（训练时从没出现过）

```
Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning
and bakes muffins for her friends every day with four. She sells the remainder at the
farmers' market daily for $2 per fresh duck egg. How much in dollars does she make
every day at the farmers' market?
Answer: Let's solve this step by step.
```

### 为什么两个模板下 RL 后都比 Base 差 —— 两个原因，各占一个模板

```
★ R1 模板下（0.52 → 0.16，长度 183 → 71）
   ① 灾难性遗忘（跟中性模板共享的那 0.13）
   ② ★★ 策略绑定（额外的 0.23）：这段文字在训练里出现了 38,400 次，
      每次后面跟的都是「写 70 个 token 交卷」。模型学到「这段文字 → 交卷」。
      GSM8K 套上同一段文字，反射触发，步骤不写，答案错。

★ 中性模板下（0.48 → 0.35，长度 257 → 272）
   只有 ① 灾难性遗忘。反射没触发（长度没变、还在列步骤），但列的步骤错得更多。
   ★ 原因：全参训练 300 步，1.54B 个权重全在动。训练信号里没有任何一项说
     「保住 GSM8K 的能力」—— GSM8K 从没出现在训练里。KL β 0.0005 几乎不约束，
     而且参考模型只在 Countdown 的提示上算 KL，对 GSM8K 的行为没有任何保护。
     权重朝 Countdown 需要的模式漂了 300 步，顺带把 GSM8K 需要的部分冲掉了一些。

★★ 两个效应叠着，长度是分离它们的钥匙：
   长度不变 + 正确率跌 = 遗忘        长度暴跌 + 正确率跌 = 绑定
★ 治法不同：遗忘靠 KL β 调大 / LoRA / 混入原分布；绑定靠多环境混训。
```


---

# ★★★ 遗忘、迁移、以及 2026 年工业界的做法（2026-09-02，查证后归档）

## 一、灾难性遗忘是什么

所有任务共用同一套权重，没有分区。梯度只看当前任务的 loss：某个权重挪一下能让 Countdown 涨，
它就挪，不管这个权重是不是 GSM8K 依赖的。没有任何信号说「别忘了 GSM8K」，因为它不在训练集里。

```
实测：Countdown 训 300 步 → GSM8K 中性模板下 0.48 → 0.35
      模型还在写 272 token 的步骤，只是步骤错得更多 —— 能力不是消失，是被冲淡
```

**lr 是第一道防线。** 预训练 5e-4 / SFT 3e-5（1/17）/ RL 5e-6（1/100）。
预训练从随机开始没东西要保护；SFT 和 RL 从会很多的模型开始，每步都可能覆盖已有知识。
RL 的 lr 已经比预训练低 100 倍，300 步仍掉 0.13 —— lr 不够，还要 KL。

**KL β 是狗绳。** `(pg + β·kl).backward()`。参考模型站原地，β 是绳子松紧。
我们的 0.0005：训练后期 KL 0.06 × 0.0005 = 0.00003，比策略梯度项小一千倍，等于没拴。
GRPO 原论文 0.04，DeepSeek-R1 0.001，DAPO 直接删掉。
★ KL 只在训练提示上算 → 只保护 Countdown 上下文下的行为，对 GSM8K 零直接保护 → 治绑定多于治遗忘。

**换任务不能消灭遗忘，只能选择遗忘什么。** 用 GSM8K 训就不会忘 GSM8K，但会忘别的。
真正的办法只有三种：① 在乎的任务全放进训练集（混训）② 限制漂移（KL / 小 lr / LoRA）③ 混入原分布复习。

## 二、理想情况下 RL 应该忘掉什么

概率总和是 1，抬高某些路径必然压低别的。理想答案一句话：**忘掉这个任务上的错误答案，其他什么都不动。**

```
该忘的     ① 同一任务的错误路径（RL 的本意，零附带）
           ② 纯粹的废动作（空洞啰嗦、重复、无信息的犹豫）
不该动的   别的任务的正确行为、通用知识、「需要长推理时能写长」的能力
```

```
GSM8K 那轮接近理想   熵坍缩 = 忘掉「多样地答错」换成「稳定地答对」，pass@64 不动 → 任务内没丢东西
Countdown 那轮不理想  任务内做到了，但附带三处：压掉「这个模板下写步骤」（GSM8K 致命）、
                      压掉 wait（堵死顿悟）、权重漂移让 GSM8K 无关模板下也掉 0.13
```

**理想的 RL 是收缩而不是漂移**：在任务自己的输出分布里把熵收紧，不把整个模型挪走。
用采样半径说：从半径里低分那半边拿走的质量应全部落到高分那半边，半径外不动。
混训、KL、LoRA 全是在往这个理想靠。另有两个直接手段：奖励里加熵奖励；判分器接受多种正确形式。

## 三、2026 年的主线：一个共识 + 一个反驳

**共识：on-policy RL 天然比 SFT 忘得少。**

- **RL's Razor（ICLR 2026）**：完成新任务的方法很多，on-policy RL 自动挑离原模型 KL 最小的那个，
  因为每步都从当前模型自己的分布采样、按奖励重加权。★ 前向 KL 与遗忘量二次相关，R² = 0.96。
  ★ 但要在【新任务的输入】上测 KL。建议：用 KL 预算 + 早停。
- **Retaining by Doing（2510.18874）**：Llama 3（1B/8B）+ Qwen 2.5（1.5B/7B），
  目标任务含 IFEval、MMLU、★ Countdown。
  ```
  Llama-3.1-8B 在 IFEval 上
    RL    +18%   其他任务 −3.4%
    SFT   +25%   其他任务 −27.8%
  ```
  ★ 关键：RL 忘得少不是因为 KL 正则，是因为数据一直从当前策略采。
  三档配方：全 on-policy RL > 每 epoch 重新生成数据的迭代 SFT > 用自己正确样本的自蒸馏。
  「SFT on RL traces」能以更低成本拿回大部分好处。

**反驳：RL Forgets!（2026-07-05）**：Qwen3-VL（2B/4B/8B），用 2025 年中后的新数据集避免污染，
RL 在真实连续训练条件下照样严重遗忘。提出 CPO：不用复习数据，追踪参数移动、对关键参数稀疏正则。
8B 上比 GSPO 高 13.7 点，11 个外部基准 52.5 → 66.7。

**我的读法**：没有定论。一派说 on-policy 天然安全，一派说真实条件下照样忘。
我们的小规模数据（1.5B，Countdown → GSM8K 掉 0.13）站在第二派。

## 四、工业界的具体做法

```
① 推理 RL 直接去掉 KL     DAPO（ByteDance）删 KL，理由：长思维链本来就要离初始模型远
                          配套：非对称裁剪保熵、token 级 loss、动态采样滤全对全错的组、超长软惩罚
                          Qwen2.5-32B 上 AIME 2024 = 50。2026 年 GRPO 是中心，DAPO/GSPO 是主要后继
② 级联而不是混合          Nemotron-Cascade（NVIDIA）：先 RLHF，再按领域【一个接一个】做 RLVR
                          「后续领域很少损害前面的，有时还提高」。14B 在 LiveCodeBench 超过老师 DeepSeek-R1-0528
                          选级联的理由是工程：不同领域回答长度和验证延迟差太多，混一批难调度
③ 有针对性的合成复习      用模型自己生成想保住的能力（对话/指令/推理）的样本混进去
④ 蒸馏合并                分领域训专家，on-policy 蒸馏进一个学生
⑤ LoRA                    碰不到的忘不了
⑥ 接受一部分              几十个基准同时测，目标涨、别的掉一点在容忍内就接受。遗忘是用数字管理的
```

**跟我们的实验对上号：**
- ★ 我们的 β ≈ 0 就是 DAPO 的设置 → 观察到的遗忘不是配置错误，是这条路线公认的代价
- ★ Retaining by Doing 说 RL 只掉 3.4%，我们掉 13%：他们 8B 我们 1.5B；
  他们目标/保留任务差异大（IFEval vs MMLU），我们两个都是算术，权重重叠多、互踩狠
- ★ Nemotron 说级联很少损害 vs 我们单任务伤了 GSM8K：他们从各领域都强的 RLHF 模型出发、
  14B 冗余多、各领域模板一致；我们从 Base、1.5B、模板只在一个任务出现过 → 0.23 的绑定在他们那可能不会形成
- ★ 我们的混训是 Nemotron 刻意避开的路（工程难调度），两条路都在用。
  「GSM 梯度话语权 3~7 倍」正是「异构混合难调」的具体表现
- ★★ 能直接补的测量：拿 ckpt_best 在【GSM8K 提示】上算对 Base 的前向 KL —— 那才是遗忘的预测量。
  我们的 KL 0.06 是在 Countdown 提示上算的，不算数

## 五、coding RL 带动其他任务 —— 真的，但有边界

**Breaking Barriers（ICLR 2026）**：DeepSeek-R1-Distill-Qwen-1.5B，数学/代码/知识密集三类各 4 万条 GRPO/DAPO，16 个基准交叉测：

```
数学 RL → 代码变好   ✓          数学/代码 RL → 医学/法律/金融/表格   ✗ 不动
代码 RL → 数学变好   ✓          知识领域 RL → 数学/代码              ✗ 不动
```

**迁移只在「结构化推理」这一簇内发生。** 代码和数学共享分解、精确、自检这类通用技能；
医学法律要领域知识，RL 不加知识，过不去。

其他证据：代码 RL 能迁到自然语言反事实推理（Executable Counterfactuals）；
TAC（2026-06）用梯度方向对齐度量化领域间迁移量，做成课程调度，1% 开销。
为什么 RL 迁移比 SFT 好：接 RL's Razor —— on-policy 是「在已有能力上收紧」不是「改写」，抬高的是跨任务共用的路径。

**跟我们对照：Countdown → GSM8K 是负迁移。** 三个差异：他们从已会推理的蒸馏模型出发，我们从 Base；
他们每领域 4 万条，我们 300 步；★ 最关键：3 个数的 Countdown 没教任何可迁移技能，它教的是「别想直接答」—— 反技能。
代码是好迁移源，因为它逼模型学的东西（拆解、验证、修错）本身通用。

**相关论文标题**：*When RL Suppresses Its Own Vocabulary: Recovering Reasoning Diversity in Puzzle-to-Math Transfer*（2026-05）
—— RL 在一个任务上压掉某些词汇、迁移到另一个任务时多样性丢失。跟我们的「反思措辞被消灭」是同一现象。

## 六、Mac 当环境（新闻 2026-08-31 ~ 09-01）

**属实**：OpenAI 买了数万台 Mac mini / Mac Studio；Anthropic 通过 AWS 按小时租 Mac mini。用途：训练操作电脑的 agent。

**为什么非得是 Mac**：macOS 许可证只允许在 Apple 硬件上跑，没法像 Linux 开一万个虚拟机。
要让 agent 操作真实 macOS 桌面就得有真实 Mac。AWS 的 Mac 实例 = 机架里的物理 Mac mini。
★ **这些 Mac 是环境，不是算力。** 一台 Mac = 一个 agent 能点击、打字、截屏的沙箱。
新闻里「统一内存打败 H100」要打折扣 —— 说的是跑环境本身（可能顺带本地推理），不是梯度计算。

**规模的含义**：数万台 Mac = 数万个并行环境实例。RL 七成时间在采样，到操作电脑这个场景，
瓶颈从「GPU 生成 token 多快」变成「能同时开多少个桌面」。买 Mac = 买采样并行度 = 「人建环境的速度」的物理形态。

**三个推断逐条：**
```
产生新数据          ✓  「在 (340,200) 点击然后输入…」这种轨迹在人类文本里几乎不存在，纯粹从环境来
复杂度比 coding 高   一半对  步数更长、部分可观测、界面会变 —— 都更难
                           ★ 但验证更难：代码有单元测试，「把邮件整理好」没有
                           奖励要么查最终状态，要么 LLM 当裁判 → 可验证性谱系中间档，被 hack 风险比 coding 大
                           环境有了，难的是每个任务写出可靠判分器
带动其他任务         按 Breaking Barriers 逻辑，迁移在技能相近的簇内
                     能带动 agent 类任务（工具调用、浏览网页、终端）；带不带动纯数学无证据；知识类大概率不行
```

跟 AgentScaler「多样性比数量重要」对上：一台 Mac 几百个应用，每个应用一族任务 —— 可能是买 Mac 而非自己写几千个网页环境的真正理由。

## 来源

- RL's Razor (ICLR 2026): https://proceedings.iclr.cc/paper_files/paper/2026/file/618c95f4557c15b253fb0e6f548ea0c0-Paper-Conference.pdf
- Retaining by Doing: https://arxiv.org/html/2510.18874
- RL Forgets! Towards Continual Policy Optimization (2026-07): https://arxiv.org/html/2607.04364v1
- Nemotron-Cascade: https://arxiv.org/abs/2512.13607
- DAPO: https://arxiv.org/pdf/2503.14476
- Breaking Barriers (ICLR 2026): https://github.com/uiuc-kang-lab/rlvr-generalizability
- Transferability for General Reasoning / TAC (2026-06): https://arxiv.org/abs/2606.25178
- Executable Counterfactuals: https://arxiv.org/pdf/2510.01539
- When RL Suppresses Its Own Vocabulary (2026-05): https://arxiv.org/html/2605.29190v1
- OpenAI Mac mini (2026-09-01): https://www.martincid.com/technology-sv/openai-mac-mini-rl-training-apple-silicon-computer-agents/
- OpenAI & Anthropic Mac purchases: https://mlq.ai/news/openai-reportedly-bought-tens-of-thousands-of-macs-for-computer-use-training/
- Reasoning RL in 2026 (Turing Post): https://www.turingpost.com/p/reasoning-rl-in-2026


---

# ★★★ 阶段一结果：混训 Countdown + GSM8K，300 步（2026-09-02 完成）

## 配置

```
grpo_mix_mp.py   Qwen2.5-1.5B Base   每步 4 CD + 4 GSM × k=16 = 128 轨迹
lr 5e-6  KL β 0.0005  max_new 512  优势 r−μ（不除 σ，跟单任务同公式）
4×4090 多进程，109.6 分钟，约 22 秒/步
跟单任务那轮（out_cd_mp）唯一的差别：每步 8 道 CD → 4 CD + 4 GSM
```

## eval 曲线（贪心，每任务 100 题）

| step | CD 正 | CD 长 | GSM 正 | GSM 长 | 分化 | 总分 |
|---|---|---|---|---|---|---|
| 0 | 0.010 | 188 | 0.460 | 202 | +13 | 0.351 |
| 50 | 0.090 | 288 | 0.680 | 177 | −111 | 0.473 |
| 100 | 0.140 | 383 | 0.780 | 180 | −203 | 0.548 |
| 150 | 0.090 | 507 | 0.830 | 198 | −309 | 0.567 |
| 175 | 0.210 | 441 | 0.790 | 183 | −258 | 0.592 |
| 250 | 0.190 | 501 | 0.790 | 218 | −283 | 0.598 |
| **275** | **0.270** | 461 | **0.800** | 176 | −285 | **0.622 ★ 峰值** |
| 300 | 0.250 | 280 | 0.790 | 159 | −121 | 0.597 |

## 训练时（温度 1）前 20 步 → 后 20 步

```
CD   正确率 ~0.03 → ~0.22    长度 263 → 335    |A| 0.085 → 0.233    ★差 −0.042 → −0.029
GSM  正确率 ~0.40 → ~0.75    长度 206 → 268    |A| 0.353 → 0.198    ★差 −0.192 → −0.282
分化（GSM − CD）  −57 → −67   ★ 温度 1 下基本没动
KL   0 → 0.0117（单任务 step 300 是 0.06，低 5 倍）
```

## 一、主要结论

### ★★ 捷径没有形成 —— 这是确定的

```
单任务   CD 长度 249 → 71（温度 1），eval 贪心 186 → 48
★ 混训   CD 长度 263 → 335（温度 1），eval 贪心 188 → 280~500
```
「看到模板就闭嘴」这条路 300 步下来一次都没走。

### ★★ 两个任务都涨了，代价是 CD 比单任务低

```
CD    0.010 → 0.270    （单任务 0.035 → 0.405，混训只到它的 2/3）
GSM   0.460 → 0.800    （单任务训完后 GSM8K R1 模板下只有 0.16）
```
混训在 CD 上让出 0.135，在 GSM8K 上多拿 0.64。合计远胜。

### ❌ 我预测的「分化 > +100」完全错了，方向都反了

```
预测   CD 变短、GSM 站住 → 分化 +100~+150
实际   温度 1 下两个任务【一起变长】各 +70 token，分化 −57 → −67，没分化
       贪心下 CD 比 GSM 长得多（分化 −121~−309），但 CD 贪心长度 461~507 ≈ max_new 512，
       ★ 大概率是贪心解码掉进重复循环，不是模型「决定」写长（step 300 一下掉到 280 也说明贪心长度不稳）
```

**模型没有学成「按题目内容分配长度」。它学成了一种对两种题都用的「长推理」风格。**

## 二、诚实的科学状态：假说只验证了一半，而且有一个没排除的替代解释

**原假说**：捷径学得成是因为所有轨迹共享同一上下文；打破均一性，捷径就学不成。

**支持的部分**：打破均一性后捷径确实没形成。✓

**没支持的部分**：模型没有因此学会内容条件化的策略。它没走捷径，但也没按任务分化，
而是滚进了另一个「所有题都长推理」的坑。

**★ 没排除的替代解释：GSM 主导。**
前 50 步 GSM 占 88% 的梯度，方向是「写步骤」。CD 可能根本没机会探索短答案，
就被 GSM 的梯度拖进了推理风格。这个解释和「绑定被打破」在这轮数据上【无法区分】——两者预测一样。

**怎么区分**：跑一轮 CD 主导的混训（--gsm-frac 0.25，或开 --adv-std 让两边平衡）。
  ★ 捷径仍不形成 → 绑定解释成立
  ★ 捷径形成了 → GSM 主导解释成立，「混训堵捷径」只在混得够重时有效

## 三、比原假说更准的说法

**RL 放大的是早期赢的那个行为。**

```
单任务   Base 偶尔答对的那几条是短的模式匹配 → 前 50 步放大它们 → 滚进「短答案」的坑
混训     GSM 占 88% 推力、方向是推理 → 前 50 步放大推理 → CD 跟着滚进「长推理」的坑
```
同一个任务、同一个判分器、同一个起点，只因为前 50 步「谁的推力大」不同，终点完全不同。
**捷径不是 CD 任务的必然，是前 50 步动力学的产物。** 这比「上下文均一性」更根本，也更容易操作。

## 四、其他确认

```
① |A| 自平衡    CD 0.085 → 0.233，GSM 0.353 → 0.198，后期【交叉】—— 不需要 --adv-std
                 机制：GSM 饱和到 0.8 离开 p(1−p) 顶点；CD 进入 0.25 的甜区
② KL 低 5 倍     0.0117 vs 单任务 0.06。混训让模型离 Base 近得多（RL's Razor 一致）
③ 反思仍被压    CD ★差 −0.03、GSM −0.28，方向没变。混训不救顿悟，只是压得慢
④ 贪心 CD 循环   eval CD 长度 461~507 顶着 max_new；温度 1 只有 330 → 贪心掉循环
                 ★ 要抽几条贪心输出确认；贪心 eval 低估了 CD 真实能力
⑤ 每步 22 秒    比单任务慢 2 秒，GSM 提示和输出都更长
```

## 五、待做

```
① crosstask_eval.py --ckpt out_mix_mp/ckpt_best.pt --out crosstask_mix.json
   跟单任务（R1 模板 0.16 / 71 token）在同一批 test[0:200] 上对照
   预测：R1 模板 0.75~0.80；中性模板 ≥ 0.48（GSM 在训练集里，遗忘应消失，甚至正迁移到 0.5~0.6）
② 抽 5 条贪心 CD 输出，确认循环
③ 在 GSM8K 提示上算 ckpt_best 对 Base 的前向 KL（RL's Razor 的遗忘预测量）
④ ★ 区分实验：--gsm-frac 0.25 再跑一轮，看捷径形不形成
⑤ GSM-RL(Instruct) → Countdown 的反向遗忘测试（已启动）
```

## 产出

```
grpo_mix_mp.py           混训脚本，533 行
out_mix_mp/ckpt_best.pt  step 275，总分 0.6217（CD 0.27 / GSM 0.80）
out_mix_mp/hist.json     完整曲线
mix_run1.log             300 步日志
```


## 六、混训模型的跨模板测试（crosstask_mix.json，2026-09-02）

同一批 GSM8K test[0:200]、贪心，三个模型 × 两段提示词：

| 模型 | 训练内容 | R1 模板（`<think>` 结尾） | 中性模板（`Question:…Answer:`） |
|---|---|---|---|
| Qwen2.5-1.5B Base | 无 | 0.52 · 183 tok | 0.48 · 257 tok |
| out_cd_mp/ckpt_best | 只训 Countdown | 0.16 · 71 tok | 0.35 · 272 tok |
| **out_mix_mp/ckpt_best** | Countdown + GSM8K 混训 | **0.71 · 164 tok** | **0.47 · 241 tok** |

```
McNemar（混训 vs Base）
  R1 模板    训好 47 / 训坏 9    z = −5.08   显著变好
  中性模板   训好 20 / 训坏 22   z = +0.31   无差异
```

### 三条读法

**① R1 模板列：0.16 → 0.71，两个 RL 模型差 0.55。**
同一个 Base、同一个 Countdown 判分器、同样 300 步，唯一差别是混没混 GSM8K。
注意 test[0:200] 没进过训练（训练用 train 切分，训练时 eval 用 test[200:300]），0.71 是干净的 held-out 数。

**② 中性模板列：0.35 → 0.47，遗忘从 −0.13 变成 −0.01。**
单任务在没见过的模板下掉 0.13（z=3.83），混训只掉 0.01（z=0.31，噪声）。
GSM8K 在训练集里之后，它在【另一个模板】下的能力也没受损。遗忘消失。

**③ ★★ 但混训在中性模板下 0.47 ≈ Base 0.48 —— GSM8K 的提升【没有】迁移到另一个模板。**
R1 模板下 +0.19，中性模板下 0.00。我预测过「可能正迁移到 0.5~0.6」，错了。
★ 提升跟模板绑在一起，就像单任务那轮的损伤跟模板绑在一起。
★★ 绑定是对称的：RL 学到的东西（好的坏的）默认都锁在训练时的上下文里。
   KL 只有 0.0117 —— 模型动得极少，动的那点全在「R1 模板之后该写什么」上，
   「Question:…Answer:」那条路径的权重根本没被碰过。

### 对部署的含义

```
★ 用模板 X 做 RL，就用模板 X 部署。换提示风格，RL 的收益归零，回到 Base 水平。
★ 这是工业界训练和部署严格用同一套 chat template 的原因之一。
```

### 脚本判读的一处失效

crosstask_eval.py 打的「RL 没有跨任务副作用」是按 Countdown-only 那轮写的判读逻辑。
对混训模型，GSM8K 不是跨任务而是训练任务，正确读法是：
「训练模板下有提升；换模板后既没提升也没损伤」。


## 七、反向测试：GSM8K-RL 后的 Instruct 模型去做 Countdown（2026-09-02）

`baseline_countdown.py`，100 题 × 8 条，温度 1，R1 模板（GSM8K 那轮没见过这个模板），max_new 1024。
对照是 Instruct 原始模型，同一批题同一协议。

| 指标 | Instruct 原始 | GSM-RL 后 | Δ |
|---|---|---|---|
| 有 `<answer>` 标签 | 0.866 | 0.913 | +0.05 |
| 能解析 | 0.695 | 0.801 | +0.11 |
| 数字用对的比例 | 0.267 | 0.386 | +0.12 |
| 数字恰好各一次 | 0.129 | 0.198 | +0.07 |
| **答案正确 pass@1** | 0.0150 | **0.0375** | **2.5×** |
| **pass@8** | 0.090 | **0.260** | **2.9×** |
| 有对有错的题 | 9 | 26 | 2.9× |
| 组内标准差 | 0.093 | 0.131 | +0.04 |
| 反思措辞（宽松/严格） | 0.46 / 0.70 | 0.38 / 0.60 | 略降 |
| 长度中位 / 均值 | 524 / 610 | 375 / 484 | −149 / −126 |
| 撞 1024 截断 | 33.9% | 18.8% | −15 pp |

```
800 条里正确数   12 → 30
100 题里至少对一次   9 → 26
```

### ★★★ 遗忘是不对称的

```
训 Countdown → 测 GSM8K   −0.13（没见过的模板）/ −0.36（训练模板）   ★ 负
训 GSM8K → 测 Countdown   pass@1 2.5×，pass@8 2.9×，阶梯每一级都涨      ★ 正
```

**两个方向都是「在没见过的模板下测」**（Countdown 测试用 R1 模板，GSM8K 那轮用的是 chat 模板）。
一个方向掉，一个方向涨。所以「单任务 RL 会伤别的任务」不成立，**要看源任务教了什么**。

### 机制：GSM8K 教的是通用技能，Countdown-3 教的是反技能

```
GSM8K RL 学到的（翻转分析）：把题读完整、别漏条件、算术收紧
  → 到 Countdown 上体现为：能解析 +0.11、数字用对 +0.12 —— 正是「把约束读全」
  → 顺带把 Instruct 的啰嗦压下来：中位 524 → 375，截断 34% → 19%
     少啰嗦 → 更常写到 <answer> → 格式分和解析分跟着涨

Countdown-3 RL 学到的：别想，直接答
  → 到 GSM8K 上是灾难
```

**这跟 Breaking Barriers（数学 ↔ 代码正迁移）和 Nemotron-Cascade（先通用后专门，级联很少损害）都对得上。
也跟前面那句一致：好的迁移源逼模型学的东西本身就通用；Countdown-3 不是。**

### 样例里看到的

```
GSM-RL 模型在 Countdown 上会写：
  「(20 − 26) × 31 = −186. However, this does not equal 100. Let's try another combination」
  「1326 is too high」
  ★ 带反馈的试错措辞（SEARCH 正则能抓到），虽然这条最后还是错了
  ★ 也有 chat 模板泄漏：「Thank you! Is there anything else I can help with?」
  ★ 也有幻觉数字：<answer>(31 − 2) × 2</answer>，2 不在给定数里
```

### 注意

```
① 这次跑的是改 baseline_countdown.py 之前的版本 —— 没有「有效长度」列，
   长度数含幻觉续写；但两个模型同一协议，方向可比
② 跟 CD→GSM 那个测试协议不同（那边贪心 200 题两模板，这边温度 1 100×8 单模板）
   效应方向都很明确（12 vs 30 条对；9 vs 26 题），不影响结论
③ ckpt 用的是 out_grpo_lr5e6/ckpt_latest.pt（step 300）
```

### 对阶段二的一个选项

GSM-RL 后的 Instruct 在 Countdown 上 pass@8 = 0.26，比 Base 高一倍以上，而且已经带搜索措辞。
★ 从它出发做 Countdown RL，起点 p 更高、每条采样信息量更大、学得更快。
代价是不再是「Base 纯 RL」的 R1-Zero 设定 —— 看你要的是复现设定还是看到现象。


### ★★★ 提炼：遗忘的方向由源任务教了什么决定

```
「单任务 RL 会伤别的任务」不成立。两个方向都在没见过的模板下测：
  训 Countdown-3 → GSM8K 掉     源任务教的是「别想直接答」（反技能）
  训 GSM8K → Countdown 涨       源任务教的是「读全约束、验算、少啰嗦」（通用技能）
```

**准确的说法：**
```
RL 抬高的是「在源任务上赢得奖励的那些路径」。
  这些路径是通用技能  →  搬到相近任务上是正迁移
  这些路径是任务捷径  →  搬到相近任务上是负迁移
被压下去的是「跟这些路径竞争的东西」—— 不是按好坏选的，是被挤掉的。
  GSM8K 上啰嗦输给了简洁 → Instruct 的啰嗦被压 → 在 Countdown 上恰好也是好事（截断 34% → 19%）
  ★ 这是巧合式的对齐，不是 RL「知道」啰嗦对 Countdown 不好
```

**两个边界：**
```
① 迁移有半径。数学 ↔ 代码 ↔ Countdown 互通；到医学法律那种知识型任务不通（Breaking Barriers）
② 只测了一个方向的一个任务。GSM8K RL 对别的任务（知识问答、写作）有没有损伤，没测
```

**能提前判断一个任务是不是好的迁移源：**
```
问「在这个任务上拿高分最省事的方法是什么」
  答案是一种通用技能  →  好源
  答案是一条捷径      →  坏源（检查清单第 ⑨ 条）
Countdown-3：192 种组合，最省事是模式匹配 → 坏源
GSM8K：必须读全条件、逐步算 → 好源
```


## 八、搜索措辞的 ★差 —— 预测错了，以及长度数据的更正（2026-09-02）

### 800 条，温度 1，R1 模板，新脚本（带 SEARCH 正则、`User:` 停止串、有效长度）

| | Base | 混训 ckpt_best | 变化 |
|---|---|---|---|
| pass@1 | 0.019 | **0.219** | 11.7× |
| pass@8 | 0.130 | **0.670** | 5.2× |
| wait/hmm 出现率 / 每条几次 | 0.181 / 0.31 | 0.089 / 0.14 | 减半 |
| wait/hmm ★差 | −0.020 | **−0.066** | 更负 → 被压 |
| 搜索措辞 出现率 / 每条几次 | 0.449 / 1.38 | 0.518 / **2.23** | 次数 +62% |
| **搜索措辞 ★差** | −0.009 | **−0.014** | ★ 两边都是零 |
| 有效长度中位 | 192 | **110** | −42% |
| 幻觉出 `User:` 续写 | 8.1% | **45%** | 5.5× |

### ★ 三个结论

**① 搜索措辞跟奖励无关，训练前后都无关。**
★差 −0.014，414 条带 vs 386 条不带，标准误约 0.025 → 零。
5 条贪心样例里「试错的对、断言的错」的规律，800 条上不存在。
★★★ 第三次栽在「小样本 + 符合预期」上。前两次是 Δ≈0 的误判，这次是 Δ 看着很大的误判。
    硬规则补一句：**5 条样例只能提出假设，不能支持任何方向的结论，包括「效应很大」。**

**② RL 让模型说得更多，没让它说得更准。**
搜索措辞每条 1.4 → 2.2 次，但带它的轨迹不比不带的更容易对 → 「表演试错」。
wait/hmm 那套是真的被压：频率减半，★差 −0.02 → −0.07。

**③ ★★ 之前混训分析里的长度数全部要改。**
```
训练日志的 CD 长度 335  =  45% 撞 512（幻觉续写）+ 55% 写 110  的均值
真实推理长度中位 110，比单任务的 71 长 55%，★ 不是三倍
「CD 变长到 335」和整个「分化」的长度分析都被续写污染
GSM 那边的长度同样不可信（续写率没测）
```
★ 单任务模型是真的学会了停（贪心 48 / 温度 1 71，没有续写）；混训模型 45% 不停。
   原因：判分器取第一个 <answer>，续写不扣分，无梯度压它。下一轮已加 `User:` 停止串 + 补 EOS。

### 待做

```
★ analyze_search.py：组内 ★差（控难度）、剂量响应、「perfect match」精度、方向标注精度
  —— 池化 ★差 有难度混淆，这几个视角能分清「表演」和「真验算」
```


## 九、★★★ 拆开看：搜索措辞 ★差 = 0 是两个相反效应抵消的结果（analyze_search.py，2026-09-02）

### 四个视角，Base vs 混训，各 800 条

| 视角 | Base | 混训 |
|---|---|---|
| ① 池化 ★差 | −0.009 | −0.014 |
| ① 组内 ★差（控难度） | −0.008（SE 0.011） | −0.018（SE 0.023） |
| ② 试错 0 次 正确率 | 0.016 | 0.210 |
| ② 试错 1~2 次 | 0.031 | 0.211 |
| ② **试错 3~5 次** | 0.018 | **0.342** |
| ② 试错 6+ 次 | 0.000 | 0.184 |
| ③ 写了「perfect/correct」的真对率 | 0.017（全体 0.019，1.0×） | **0.362**（全体 0.219，**1.65×**） |
| ③b 「X = V (too high/low)」标注数 | 9 | **87**（10×） |
| ③b 标注精度（算对 + 方向对） | 0.667 | **0.816** |

```
显著性
  3~5 次 vs 0~2 次   0.342 vs 0.210，z = 2.56，p ≈ 0.01
  声明精度           0.362 vs 0.219，z ≈ 3.9
  6+ 次 vs 0~2 次    0.184 vs 0.210，z = 0.65，不显著（不更差，但不更好）
```

### ★★★ 三个发现

**① 验证是真的，不是表演。**
Base 说「perfect」时真对率 1.7%，等于瞎猜。混训模型说「perfect」时 36%，比基线高 1.65 倍 —— 这句话携带信息了。
混训模型写「X = V (too high)」的 87 处里，82% 算式算对、方向也标对。Base 只写 9 处。
**RL 把「说 too high」从装饰变成了真的算一遍再比。**

**② 试错分两种，效应相反。**
```
收敛的试错（3~5 次后命中）   正确率 0.342，比不试高 0.13，显著
跑飞的试错（6+ 次不收敛）    正确率 0.184，跟不试差不多，长度 386
```
池化 ★差 把这两种混在一起 → 正负抵消 → 零。
**「搜索」作为二值变量是错的分析单位，正确的单位是「搜索有没有收敛」。**

**③ RL 放大的是「算对的标注」，不是「搜索这个词」。**
标注数 9 → 87，精度 0.67 → 0.82。RL 没有让模型更常说 let's try（那是装饰），
是让它更常写出「算式 = 值 (方向)」这种可验证的中间步骤，而且写得更准。

### 为什么 ★差 看不到

SEARCH 正则抓的是 too high / too low / perfect / let's try / close / another … 一大串。
其中 let's try、close、another 是装饰词，谁都会说，跟对错无关。
真正有信号的是「算式 = 值 (too high/low)」这一种形态，被装饰词淹没了。
★ 训练时的 ★差 也是同一个正则，所以它一直报零 —— 不是行为没变，是探针太钝。

### 两种失败模式（混训模型，800 条）

```
不试直接断言且错   386 × 0.79 ≈ 305 条   38%   ★ 最大的一块
试错跑飞           114 条                 14%
```

### 对顿悟的含义

混训模型已经有了顿悟的功能形态：算一个候选 → 报值 → 跟目标比 → 标方向 → 调整 → 命中。
只是词表不是 wait / hmm，是 too high / too low / perfect match。
★ 它不是「没涌现」，是我们的探针一直在测错的词。

### 方法论：换探针

```
弃用   宽泛的 SEARCH 二值 ★差
改用   ① 「算式 = 值 (方向)」标注的数量和精度（③b）
       ② 剂量响应的 3~5 桶 vs 0 桶
       ③ 「说对了」的精度 vs 基线（③）
★ 这三个指标能分清「表演」和「真验算」，二值 ★差 不能
```


## 十、⑤ 窄行为的 ★差：Base 测不了，混训方向为正但不显著（2026-09-03）

| | Base | 混训 |
|---|---|---|
| 带「算式 = 值 (方向)」的轨迹 | **4 / 800（0.5%）** | **21 / 800（2.6%）** |
| 其中正确率 | 0.000 | 0.333（基线 0.216） |
| 池化 ★差 | −0.021 | +0.033 |
| 组内 ★差（SE） | −0.005（0.052） | +0.014（0.079） |

```
Base   4 条样本，什么都说明不了。★ 0.5% 在采样半径 0.7/16 ≈ 4.4% 之下，理论上 RL 不能直接放大它
混训   21 条里对 7，预期 4.5，z ≈ 1.3，不显著。方向对，样本不够
```

### ★ 对「能力从哪来」的修正

之前说「9 → 87 处标注」，按轨迹是 4 → 21 条。Base 的频率在半径下面，「放大已有相关行为」说不通。
**更可能是组装**：GSM8K 推「算准」（早期 88% 推力）+ Countdown 推「多试候选」，两个零件各在半径内、各被放大，
交集「准确地算多个候选并比较」作为副产品从 0.5% 长到 2.6%。组合涌现的实例。

### 四个视角并排：⑤ 最严所以最小，其他三个显著

| 视角 | 定义 | n | 效应 | 显著性 |
|---|---|---|---|---|
| ⑤ | 严格标注形态 | 21 | 0.333 vs 0.216 | z≈1.3 |
| ② | 试错 3~5 次 | 73 | 0.342 vs 0.210 | **z=2.56** |
| ③ | 说「perfect/correct」 | 127 | 0.362 vs 0.219 | **z≈3.9** |
| ③b | 标注精度 | 87 | 0.82 | SE 0.04 |

效应量全在 +0.12~0.14。功能形态存在但**稀有**：严格 2.6%，宽泛产出型 9%；48% 不试，其中 79% 错。

### 续训要看的
标注轨迹比例从 2.6% 往上走 + 精度守住 0.8 → 在被放大；不动 → 1.5B 到此为止。


---

# ★★★ 复盘：从「Countdown 变短」到「混训」再到「顿悟其实在」—— 决策链（2026-09-03 记）

这一段记的是**为什么走了这条路**，不是结果。结果散在第五阶段到第十节里。

## 链条

```
① Countdown 单训 300 步
   正确率 0.035 → 0.405，但长度 186 → 48，wait 0.146 → 0.005
   第 15 步 ★差 就预测到了：说 wait 的轨迹分低 → RL 压它
   解释：3 个数 192 种组合太简单，RL 找到「不想直接答」的捷径，思考成了负担

② 跨任务 2×2（想知道捷径是能力受损还是别的）
   R1 模板    0.52 → 0.16   长度 183 → 71
   中性模板   0.48 → 0.35   长度 257 → 272
   同一组权重、换开头几行、行为两样 → 变短只在训练模板下触发
   命名：条件性策略绑定 —— 模型学的是「看到这段文字 → 交卷」

③ 从绑定推出可证伪的机制
   捷径学得成，是因为 38,400 条轨迹开头一样，模型在上下文层解决问题、不读题
   推论：同一模板下混两种题（一短一长），「看到模板就闭嘴」不再最优 → 捷径学不成
   ★ 混训 = 这个推论的检验。预测：分化 > 100

④ 用户定的顺序
   四条路：验证机制 / 追顿悟 / 补 LoRA / 整理输出
   ★ 选的是「1. 验证绑定  2. 追顿悟  3. 多轮」—— 混训排在追顿悟【前面】，作为地基
```

## 当时混训的目的

```
主目的    验证「捷径是单一上下文造成的」。解释错了，阶段二「先堵捷径再找任务」就建在沙上
隐含目的  捷径是杀反思的凶手，堵住它是反思活下来的前提 —— 通往顿悟的一步，但是地基不是实验本身
★ 不是    「造新环境太难」。阶段二的计划本来就是在 Countdown 上扫配置（4 个数 / 5 个数 / 3B），
          那就是重新构造环境，是计划好的，没被回避，只是排在后面
```

## 后来发生的事，改变了混训的角色

```
混训原始预测（分化 > 100）★ 失败：两个任务一起变长，没分化
但事后换探针查，混训模型身上长出了顿悟的功能形态：
   算候选 → 报值 → 跟目标比 → 标方向 → 换路 → 命中
   而且是真的：87 处标注 82% 算对方向对，说「perfect」时真对率高 1.65 倍
   只是词不是 wait / hmm，是 too high / too low / perfect match

★ 所以现在继续训混训模型追顿悟，理由跟当初做混训的理由【不是一个】：
   当初 —— 验证机制
   现在 —— 事后分析发现它碰巧已满足强化的两个条件（行为在采样里 + 带它的轨迹得分高）
   阶段二的配置扫描因此没必要了 —— 要找的东西已经在手上
```

## 一句话

混训是为了回答「反思为什么被杀」，回答到一半，发现反思没被杀，只是换了个词。

## 这段经历里的方法论

```
① 一个失败的预测（分化 > 100）不等于一个失败的实验 —— 数据留着，换探针再看
② 「没有」的结论要问一句：是行为没有，还是探针没测到
③ 决策链要记下来 —— 三天后自己都会忘了当初为什么做这一步
```


---

# ★★★ RL 阶段复盘（用户自述 + 评语，2026-09-03）

## 用户自述（原话，略作分段）

> **1.** 补了下 RL 理论，然后用 GSM 作为训练语料，通过 GSM 训练出来的模型，确实在 GSM 的验证数据上变好了，
> pass@1（稳定好）pass@64（天花板），后面还用这个 RL-GSM 模型验证了 countdown 数据，也比原模型更好。
> 到这一步已经说明了 RL 训练是有效的，能稳定提高「奖励觉得好」的概率。
>
> **2.** 接下来想要复现「顿悟」思维链，做了一个 countdown 的训练环境，用 qwen 的 base 模型，预计经过 cd 数据的 rl 训练后
> token 会变长，结果却出现 token 变短，而且 ★差 从一开始就是负值，结果不出所料，token 变短，cd 模型走了捷径，
> 在 cd 的验证数据上是变好了，但是是以走捷径的方式（有点像模式匹配），而且用 cd 模型验证 gsm 数据，不管 r1 提示词
> 还是常规提示词，都变差了。所以总结下来就是 cd 模型走了捷径，从一开始「反思的行为」就不被鼓励，唯有堵住捷径，
> 才有可能出现长的思维链。反思下来就是 cd 数据的问题（会诱导答案短，背答案，捷径可能由单一上下文造成），
> 所以用一半 cd 数据一半 gsm 数据。
> （ps：用 cd 训的模型跑 gsm 有灾难性遗忘，用 gsm 模型跑 cd 反而变好，说明灾难性遗忘不一定出现，
> 重要的是你的奖励机制和数据环境真正教会了模型什么）
>
> **3.** 这时候就有了混训模型，混训模型训完后，把捷径堵住了，在 cd 和 gsm 验证数据上表现都很好。
> 然后在 2×2 跨任务对比中，对比 base 模型，在 gsm 数据下，中性提示词不变（没提高没损害），但 r1 提示词好很多，
> 所以 rl 学到的东西只是一个绑在提示词上的反射。奖励机制和与什么数据一起训练（训练环境）都很重要。
> RL 改的不是模型的能力，而是模型在某个上下文下的行为的概率分布。
> （题外话：我觉得 RL 是能提高模型能力的，通过创造新数据的方式，或者 rl 训练出通用策略的方式）
>
> **4.** 发现混训模型的 token 长度是幻觉续写出来的。发现模型确实有反思的词语出现，换了关键字后，
> base 是假的反思，混训是真的反思。研究了 ★差 的影响，发现可能是组合涌现。发现混训模型有了初步的顿悟形态。
>
> **5.** 对混训模型进行续训，追着看「顿悟」的复现。现在已经出现了「做搜索」，但「顿悟」还是稀有，
> 不知道会不会涨，机械非逻辑，所以继续训练。

## 评语：90% 准确，三处要修

```
① 「成功验证单一环境会让模型走捷径，多样环境可以堵住捷径」—— 过了
   能说：混训后捷径没形成。不能说：这是因为「多样环境」。
   替代解释「GSM 占 88% 梯度、CD 没机会探索短答案」在这轮数据上分不开。
   区分实验 --gsm-frac 0.25 没跑。★ 结果符合预期就把假说当结论，跟 5 条样例的教训同类。

② 「长度不详」—— 不是。有效长度中位 110，800 条测的干净数。不详的只是训练日志里被续写污染的 335。

③ 「唯有堵住捷径才有可能出现长思维链」—— 当时的假设，结果没出现。
   混训 110 比 Base 192 还短。出现的是「9% 的轨迹里做 3~5 次产出型试错」，是功能不是长度。
```

## 两个嵌入的问题

### 「RL 学的只是绑在提示词上的反射」对吗？不好吗？不是智能？

```
对。R1 模板 +0.19，中性 0，KL 0.012。
好不好看用途：训练和部署同一套模板，绑定就是部署上下文，不成问题；想要「怎么问都行」，它是限制。
★ 反例：GSM-RL 的 Instruct 去做 Countdown，任务换了模板也换了，pass@8 涨 2.9×。
   同样是 RL，一次锁在模板上，一次跨任务迁移了。「只是反射」不是普遍规律。
算不算智能：算、比、回头这些能力预训练就有，RL 没加。RL 让模型在训练过的上下文里【更稳定地用出来】。
   会检查作业但只在考试时检查 —— 有技能没习惯，RL 给了习惯，在一个场景里。
   硬指标 pass@64：0.98 → 0.98，天花板没动。按这个标准能力没变。
```

### 「RL 能提高能力」—— 两条路各对一半

```
创造新数据   RL 采样 → 验证器筛 → 训下一代（rejection sampling / mid-training）。真实存在。
             ★ 但提高能力的是第二步（下一代吸收），不是 RL 本身。新信息来自验证器，RL 只生成候选。
训通用策略   有证据（GSM-RL → CD 涨；Breaking Barriers 数学↔代码）。
             ★ 三个限定：只在相近任务簇内；策略得在采样半径内（组装不发明）；训练任务自己的天花板不动。
★ 没测的空缺：Countdown 的 pass@64。pass@8 从 0.13 到 0.26 可能只是把偶尔对的变稳定。
```

## 追问三条

### ① 长度变小了，为什么还有顿悟？还有思维链吗？续训会怎样？

```
110 是全部轨迹的中位。但长度是双峰的：
   48% 不试         105 token
    9% 试 3~5 次    268 token   ← 顿悟在这里，这就是思维链
   14% 试 6+ 次     386 token   ← 跑飞
「变短」= 「大多数轨迹不搜索」，不是「思考变少」。
Base 的 192 里一大半是装饰性的 let's try（1.4 次/条，★差 0），混训把装饰砍了。短 ≠ 想得少，短 = 假想得少。

续训两种可能：
   (a) 那 9% 跟奖励正相关（+0.13）→ 被放大 → 更多轨迹搜索 → 中位长度跟着涨。长度是结果不是目标。
   (b) ★ 48% 的「不试直接答」在 21% 的题上也是对的，短且对，优势也为正。
       短对 vs 长搜索对，谁赢看每道题上谁分高。Countdown-3 多数题简单，短答案赢 → 顿悟停在 10~15%。
★ 这是「任务太简单」的风险，第三次出现。
```

### ② 同时要顿悟功能和长思维链，怎么做？

```
R1 的长度涨是因为 AIME 100 token 解不了。Countdown-3 20 token 就能模式匹配出来，长度不会涨，无论怎么训。

要两个都有，任务必须【要求】多步搜索：
   ① 换 Countdown-4 / 5（搜索空间 10K+），或 GSM8K-hard，或竞赛题
   ② 那时候只有产出型搜索的轨迹能对 → 正优势全归它 → 搜索被放大 → 长度作为结果涨
   ③ 起点要够得着：Countdown-4 在 Base 上 0.0175，k=16 的半径 0.044 之下。
      对策：k 加到 64（半径 0.011）；或从混训模型起（已有搜索技能）；或换 3B
   ④ 课程：3 → 4 → 5 个数，模型在哪饱和就往上加，让 p 停在 0.5 附近
★ 续训完先测混训模型在 Countdown-4 上的 pass@8。到 5% 以上就能开训。
```

### ③ 「绑定就是部署上下文」= harness？

```
对，而且 harness 比「提示词模板」更宽：
   环境五组件里的 Harness = 提示模板 + 系统说明 + 工具 + 停止条件 + 上下文管理
   我们的绑定是绑在模板（文本前缀）上；agent 场景里会绑在工具格式、观测格式上。同一原理。
★ 含义：harness 不是管道，是模型学到的东西的一部分。
   训练用 H 部署用 H' → RL 的收益归零。这是「harness engineering」为什么是工程而不是配置的原因。
```


---

# ★★★ 五模型 × 四格 总表（2026-09-03，用户手绘表格，补齐数据）

## 表

| 模型 | GSM8K · R1 提示词 | GSM8K · 常规提示词 | CD · R1 提示词 | CD · 常规提示词 |
|---|---|---|---|---|
| Qwen-Instruct | 未测 | 0.687 † | 0.015（pass@8 0.09） | 未测 |
| Qwen-Base | 0.52 | 0.48 | 0.019（pass@8 0.13） | 未测 |
| GSM-RL（Instruct + GSM8K） | 未测 | 0.767 † | 0.038（pass@8 0.26） | 未测 |
| CD-RL（Base + Countdown） | 0.16 | 0.35 | 贪心 0.405 ‡ | 未测 |
| 混训（Base + GSM8K + CD） | 0.71 | 0.47 | 0.219（pass@8 0.67） | 未测 |
| **mix4-300**（混训 600 + CD-4 混训 300） | **0.74** | **0.46** | CD-4 池外 0.393（pass@8 0.67）§ | 未测 |

**口径**
```
GSM8K 两列    贪心，test[0:200]，crosstask_eval.py
CD · R1       温度 1，pass@1，100 题 × 8，countdown[0:100]，baseline_countdown.py
†             Instruct 系的 GSM8K 数是 chat 模板 + "#### 数字"（第三种模板），holdout TEST[200:500]
              放「常规」列是因为那是它们的常规模板，但跟 Base 系的 Question:/Answer: 不是同一段文字
‡             CD-RL 只有贪心（countdown[90000:90200]，step 275），温度 1 的 800 条没跑
§             mix4-300 的 CD 格是 countdown4 池外 [95000:95100]，跟上面几行的 countdown[0:100] 不是同一片、不是同一个数据集
CD 列的题     countdown[0:100] 在 90000 题训练池里，300 步抽 2400 题，每题被抽到过约 3%，可忽略非零
```

## 缺的七格怎么补（约 1.5 小时 GPU，训练完再跑或 --bs 8 挤着跑）

```bash
# 第一组：Instruct + GSM-RL 在 GSM8K 两模板（一条出两行）
python3 crosstask_eval.py --model Qwen/Qwen2.5-1.5B-Instruct --ckpt out_grpo_lr5e6/ckpt_latest.pt --out crosstask_gsmrl.json
# 第二组：CD-RL 温度 1
python3 baseline_countdown.py --ckpt out_cd_mp/ckpt_best.pt -n 100 -s 8 --max-new 512
# 第三组：整列 CD · 常规（--plain 新加的：去系统说明 / User:Assistant / <think>，只留题目和 <answer> 要求）
python3 baseline_countdown.py --plain -n 100 -s 8 --max-new 512
python3 baseline_countdown.py --plain --model Qwen/Qwen2.5-1.5B-Instruct -n 100 -s 8 --max-new 1024
python3 baseline_countdown.py --plain --model Qwen/Qwen2.5-1.5B-Instruct --ckpt out_grpo_lr5e6/ckpt_latest.pt -n 100 -s 8 --max-new 1024
python3 baseline_countdown.py --plain --ckpt out_cd_mp/ckpt_best.pt -n 100 -s 8 --max-new 512
python3 baseline_countdown.py --plain --ckpt out_mix_mp/ckpt_best.pt -n 100 -s 8 --max-new 512
```
脚本改动：`baseline_countdown.py` 加 `--plain`；`crosstask_eval.py` 的 `--ckpt` 不传则只测原始模型两格。

## 读法

**竖着读一列 = 不同训练对同一测试做了什么**
```
GSM8K · R1    Base 0.52 → 只训 CD 0.16（−0.36） / 混训 0.71（+0.19）
              同一 Base、同样 300 步、同一判分器，差别只是每步 8 题里有没有 4 道 GSM8K
CD · R1       Instruct 0.015 → GSM-RL 0.038（2.5×，没碰过 Countdown）
              Base 0.019 → 混训 0.219（11.7×） / 只训 CD 贪心 0.405
              单训在自己任务上最高，混训让出 1/3 换 GSM8K 不掉
```

**横着读一行 = 能力离了训练模板还在不在**
```
Base     0.52 vs 0.48   没训过的模型对模板不敏感 —— ★ 对照，说明下面的模板效应全是 RL 造的
CD-RL    0.16 vs 0.35   训练模板比常规【差】0.19 —— <think> 开头成了触发「闭嘴交卷」的坑
混训     0.71 vs 0.47   训练模板比常规【好】0.24 —— 同一段开头成了助推
★★ 模板效应的符号翻了。同一模板、同一 Base，训练内容决定它触发好行为还是坏行为
```

## 五条结论

```
① RL 在训练任务上一定有效 —— 每一行，训练过的那个任务的列都涨
② 对另一个任务的影响看源任务教了什么 —— GSM8K 教「读全、算准」搬到 CD 是正的；CD-3 教「别想直接答」搬到 GSM8K 是负的
③ 混训两个都涨、一个都不掉，代价是单项让出 1/3
④ RL 学到的东西绑在训练模板上，好的坏的都绑 —— Base 行不敏感是证据
⑤ ★ CD · 常规 整列空着是最大缺口：它回答「Countdown 上的收益是不是也绑模板」
   预测：CD-RL 的 0.405 在常规模板下塌回 Base 水平（没 <think> 不触发反射）；混训的 0.219 掉但掉得少
   若两个都塌回 0.02 → 这个项目里【所有】RL 收益都是模板绑定的，④ 从「GSM8K 上成立」变成「两任务都成立」
```


---

# ★★★ 续训 300 → 425（早停），2026-09-03

`--resume --steps 600`，停止串 + 补 EOS 生效，新探针全程记录。46.8 分钟，step 425 因「总分连续 5 次没超过 0.6217」早停。

## eval（贪心，每任务 100 题）

| step | CD 正 | CD 长 | CD 标注@精度 | GSM 正 | GSM 长 | 总分 |
|---|---|---|---|---|---|---|
| 275（上轮峰值） | 0.27 | 461（含续写） | — | 0.80 | 176 | 0.6217 |
| 300 | 0.25 | 280 | — | 0.79 | 159 | 0.5968 |
| 325 | 0.35 | 279 | 0.9@0.74 | 0.80 | 160 | 0.6162 |
| 350 | 0.31 | 326 | 0.3@0.94 | 0.77 | 157 | 0.5785 |
| 375 | 0.30 | 338 | 0.3@0.88 | 0.76 | 155 | 0.5655 |
| 400 | **0.37** | 340 | 0.7@0.87 | **0.81** | 169 | 0.6178 |
| 425 | 0.36 | 344 | **1.0@0.87** | 0.79 | 175 | 0.5980 |

## ★★★ 顿悟探针（本段前 20 步 → 后 20 步，温度 1，只看 CD）

| | 前 20 步 | 后 20 步 | |
|---|---|---|---|
| 标注数/条 | 0.16 | **0.43** | **2.7×** |
| 标注精度 | 0.737 | 0.707 | 略降，要盯 |
| 剂量响应（3~5 次 − 0 次） | +0.097 | +0.088 | 守住为正 |
| 3~5 次桶正确率 | 0.353 | **0.532** | |
| 0 次桶正确率 | 0.256 | **0.444** | ★ 直答也在变好 |
| 6+ 次桶 | 0.214 | 0.259 | |
| 写完不停 | 0.276 | **0.102** | 停止串生效，−63% |

```
CD 温度 1 正确率   前 25 步均值 0.27 → 后 25 步均值 0.47
CD 长度（温度 1）  202 → 230        eval 贪心 279 → 344（★ 续写已截，是干净的）
GSM              稳在 0.76~0.81，长度 155~175
KL               0.012 → 0.02（step 305 一次 0.21 的尖刺，单批异常）
|A|              CD 0.27 / GSM 0.20 —— CD 现在梯度份额更大
wait/hmm ★差     CD −0.15 → −0.20，仍在被压
```

## 判读

### ★★ 早停是误判
总分 = 两任务阶梯分的平均，100 题 eval 的噪声 ±0.03。它在 0.57~0.62 之间抖，没超过 275 步那个 0.6217。
但底下每一个指标都在涨：CD 贪心 0.25 → 0.37，温度 1 0.27 → 0.47，标注 2.7×，长度干净地从 279 到 344，续写从 28% 到 10%。
**跟 GSM8K 那轮「224 步后退化」是同一类错：拿噪声大的复合指标做停止判据。**

### ★★★ 按我们定的三选一判据：顿悟在被放大
```
数量涨（2.7×）+ 精度稳（0.74 → 0.71）+ 剂量为正（+0.09）→ 第一种结局
```
唯一要盯的是精度：0.71 低于我定的 0.8 线，且略降。训练时每步只有 10~30 处标注，噪声大；
需要 800 条的干净测量（step 275 那次是 0.82）来判断是真降还是噪声。

### ★ 两个桶都在涨 —— 不是搜索替代直答
0 次桶 0.26 → 0.44，3~5 次桶 0.35 → 0.53，差距守在 +0.09。模型在两种模式上都变好，
而且在选择什么时候搜（难题）什么时候直答（简单题）。这比「所有题都搜」更像真的能力。

### ★ 长度终于干净地涨了
上轮的长度全被续写污染。这轮续写压到 10%，eval 贪心 CD 从 279 涨到 344，标注从 0.3 涨到 1.0/条。
R1 的 H1（长度涨）在混训 + 干净测量下出现了，虽然幅度远小于 R1（+23%，不是 2~3 倍）。

### 跟单任务对比
```
单任务 CD-3   温度 1 0.59 / 贪心 0.405，靠 70 token 直答，GSM8K 崩到 0.16
混训续训      温度 1 0.47 / 贪心 0.37，靠 344 token 带验证的搜索，GSM8K 0.79
★ 差 0.1 的 CD 正确率，换来 GSM8K +0.63 和一个非退化的策略
```

## 下一步

```
① cp out_mix_mp/ckpt_latest.pt out_mix_mp/ckpt_step425.pt   —— ckpt_best 还是 275，425 在探针上更好
② baseline_countdown.py --ckpt ckpt_step425 -n 100 -s 8 + analyze_search.py
   拿干净的：严格标注轨迹比例（275 时 2.6%）、精度（0.82）、剂量（+0.13）
③ --resume --steps 600 --patience 0   跑完剩下 175 步，不让总分噪声再停它
④ 长期：改 best 判据。总分噪声太大，改成 CD 正确率或标注数
```


---

# ★★★ 「too high / too low」这些词从哪来（2026-09-03）

## 一、Countdown 数据里没有任何文字
```
data/countdown.json 每道题：{"nums": [4, 22, 28], "target": 46}
提示词模板：任务说明 + <think>/<answer> 要求，没有 too high、没有 try another
GSM8K：只用最终数字当答案，解题文字模型从没见过
★ RL 全程模型看到的唯一文本是它自己写的。采样、判分、调权重，没有一步喂过别人的话
```

## 二、词是预训练时就有的
```
Base 在零训练下用同一个提示词采 800 条：
  宽泛搜索词（too high / too low / try another …）   45% 的轨迹，每条 1.4 次
  严格「算式 = 值 (too high)」形态                     4 条 / 800
★ 来源：预训练 18T token 里跟这个提示词长得像的文本 —— 数学辅导、解谜论坛、猜数字游戏、
   教科书试错例题、Countdown 电视节目的讨论。预训练把「一堆数 + 目标 + 要求写过程」这种上下文
   跟「试一个、说太大/太小、再试」这种叙述绑在了一起。我们的提示词恰好长成那样，把它引出来了。
★ 换 AIME 式提示词 → 先验是奥数解答 → 引出的是「wait, let me reconsider」。R1-Zero 的 wait 同理。
```

## 三、RL 做的是 45% → 87%
```
                Base    275 步   425 步
宽泛搜索词      45%     52%      87%
严格标注形态    0.5%    2.6%     26.6%
```
**RL 一个新词都没发明。它只能在模型已经会说的话里挑。**

## 四、一句话（用户的复述，对的）
```
★ 用提示词（上下文）把预训练里的词引出来，再用 RL 放大
   前提：真算了的轨迹更容易答对（条件 ②）
```
三段各有各的角色：
```
预训练   提供词汇和它跟某类上下文的关联    —— 材料
提示词   选择激活哪一类关联                —— 开关
RL       按「带它的轨迹是否更容易对」放大   —— 筛子
```
缺任何一段都不成：没有预训练的材料，RL 无米下锅；提示词不对，材料引不出来（AIME 式提示词引不出 too high）；
没有 RL，45% 停在 45%。

## 验证命令
```bash
python3 -c "
import json
d = json.load(open('cd_base_s8_off0_raw.json'))
hits = [r['txt'] for r in d['records'] if 'too high' in r['txt'].lower() or 'too low' in r['txt'].lower()]
print(len(hits), '条 Base 输出里有 too high/too low'); print(hits[0][:600])"
```


---

# ★★ 口径更正：Countdown 测试改用池外切片 [95000:95100]（2026-09-03）

## 发现
`baseline_countdown.py` 默认 `--offset 0`，之前所有 800 条测试用的是 [0:100]，**在训练池 [0:90000] 里**。
425 模型换到池外 [95000:95100]：pass@1 0.530 → 0.361，pass@8 0.840 → 0.650。差 0.17。

## 排查
```
污染      425 步 × 4 题 = 1700 次抽样 / 90000，每题被抽到过约 1.9%，两道题，最多 +0.02 —— 不是
生成器    读了 countdown.py：随机抽数、随机打乱、随机运算符、落在范围就收、去重。
          第 0 道和第 95000 道 i.i.d. —— 顺序跟难度无关
Base      两片都 0.02，是地板，看不出难度
结论      ★ 100 道题的抽样波动。Countdown 难度双峰（一加就对 vs 必须用乘除），
          训练过的模型是唯一能分辨的仪器，它说两片「够得着的题」差 30%
```

## 池外三点（以后全用这片）

| | Base | 275 | 425 |
|---|---|---|---|
| pass@1 | 0.0225 | 0.158 | **0.361** |
| pass@8 | 0.180 | 0.560 | **0.650** |
| 全错 0/8 | 82 | 44 | 35 |
| 搜索措辞 出现率 / 每条 | 0.41 / 1.4 | 0.52 / 2.2 | 0.88 / 6.9 |
| wait ★差 | −0.02 | −0.09 | −0.16 |
| 长度中位 | 187 | 108 | 239 |
| 续写 | 9.6% | 44.5% | 7.8% |

## 要改的说法
```
「pass@1 0.22 → 0.53」      → 0.16 → 0.36，倍数仍 2.3
「追上单任务」              → 收回。单任务贪心 0.405，混训 425 贪心 0.36 / 池外温度 1 0.36，低 0.05
四格表 CD 列               → 换 95000 的数；Instruct、GSM-RL 也要在 95000 上重跑
```
趋势不变：搜索措辞 41% → 88%，长度 187 → 108 → 239（275 是真变短），全错 82 → 35。

## 教训
测试切片要在训练池外，这条在 GSM8K 那轮（TEST[0:200] vs [200:500]）已经踩过一次。
Countdown 是程序生成的、10 万道，我以为「反正 i.i.d.」就没管，结果一样被抽样波动咬了。
★ 规则：任何评测切片先确认不在训练池里，再确认跟之前的评测同一片。


---

# ★★★ 池外 [95000:95100] 的四视角三点对比 —— 精度警报撤回，真问题是跑飞（2026-09-03）

## 三个模型，各 800 条，温度 1

| | Base | 275 | 425 |
|---|---|---|---|
| pass@1 | 0.022 | 0.158 | **0.361** |
| ⑤ 带严格标注的轨迹 | 3（0.4%） | 23（2.9%） | **217（27%）** |
| ⑤ 带 vs 不带 正确率 | — | 0.391 vs 0.151（**+0.24**） | 0.369 vs 0.358（≈0） |
| ③b 标注数 / 精度 | 10 / 0.20 | 70 / **0.600** | 661 / **0.643** |
| ③ 说「对了」真对率 / 基线 | 1.1× | 1.57× | 1.35× |
| ② 0 次 | 471 条 0.028 | 387 条 0.155 | 96 条 0.323 |
| ② 1~2 次 | 181 条 0.017 | 232 条 0.172 | 186 条 **0.548** |
| ② 3~5 次 | 84 条 0.012 | 66 条 0.212 | 172 条 **0.552** |
| ② 6+ 次 | 64 条 0.016 | 115 条（14%）0.104 | **346 条（43%）0.176** |
| ② 剂量（3~5 − 0） | −0.02 | +0.06 | **+0.23** |

## ★★ 撤回「精度在滑」

```
[0:100]  上   275: 0.816（n=87）  →  425: 0.713（n=640）   看着在掉
[95000] 上    275: 0.600（n=70）  →  425: 0.643（n=661）   ★ 反而略涨
两片合并      275: 113/157 = 0.72  →  425: 881/1301 = 0.68   差 0.04，约 1.5σ，不显著
```
**精度约 0.7，平的。** 之前的「0.82 → 0.71」是一片题上的读数，275 那个 0.82 只有 87 处标注、标准误 0.05。
精度的【水位】跟题的难度有关（简单题算术简单、精度高），两片差 0.2 是这个原因。**跨片比同一模型是错的。**
★ 又一次：单一切片、小 n 的趋势当成了结论。

## ★★★ 真问题：跑飞

```
425 在池外：43% 的轨迹试了 6 次以上，正确率 0.176
            1~5 次的轨迹正确率 0.55，0 次的 0.32
★ 如果那 346 条能收敛到 1~5 次的水平：多对 346 × (0.55 − 0.176) ≈ 129 条 → pass@1 从 0.36 到 0.52
★ 跑飞在吃掉 0.16 的 pass@1。这是现在最大的一块损失。
```

**搜索本身是有用的，而且在难题上更有用：** 剂量响应池内 +0.09，池外 +0.23。1~2 次和 3~5 次都是 0.55，说明「搜」比「不搜」强很多，「搜多少」在 1~5 之间不太要紧。**超过 5 次就崩。**

## 为什么跑飞在涨（14% → 43%）
```
RL 学到了「搜比不搜强」（1~5 次桶 0.55 vs 0 次桶 0.32）→ 推高搜索频率（词 52% → 88%）
但「什么时候停」的信号被稀释：
  难题上 0 次几乎必错，6+ 次偶尔能对 → 在难题的组内，6+ 是正优势
  → RL 在难题上奖励了穷举 → 泛化成「到处穷举」
样例里的跑飞：在 +/− 里绕十几次，一次 × 没试，还重复用数、编造不存在的数
★ 跑飞 = 搜错子空间 + 不守约束 + 不知道停。三个都是「搜索策略」的问题，不是「算术」的问题。
```

## ⑤ ★差 归零的正确读法
275 时严格形态在 2.9% 的轨迹里，带的 0.39 不带的 0.15，差 0.24 —— 稀有且高度预测。
425 时在 27% 里，带的 0.37 不带的 0.36 —— 普及之后不再能区分。
**这是「行为变常见」的必然结果，不是「行为失效」。** 剂量响应 +0.23 证明它还在起作用。

## 两条路治跑飞
```
A. 长度软惩罚（DAPO 的 overlong shaping）
   300 token 内不罚，300 → 512 线性罚到 −0.5
   直接打 6+ 桶（均值 375），放过 1~5 桶（183~221）
   ★ 一个变量、100 步、40 分钟。是过程塑形，但 max_new=512 本来就是硬版本
B. Countdown-4
   穷举在上万种组合里没戏，只有聪明的搜索能成功 → RL 被迫选策略
   ★ 但难题更多 → 跑飞可能先涨后降；起点 p 未知
```
**建议：先 A。** 它直接对着诊断出来的问题，一个变量。100 步后看 6+ 桶从 43% 掉到多少、pass@1 涨到多少。
同时测 425 在 CD-4 上的基线，为 B 做准备。


---

# ★★ 目标对齐：Countdown-3 能教什么、教不了什么（2026-09-03）

## 目标（用户定）
```
复现两样：① 顿悟式的搜索能力   ② 长思维链
```

## Countdown-3 的边界（用户的总结，准确）
```
★ 能教：验证的【形态】—— 试一个、算出来、跟目标比、标方向
        已教会：精度 0.7、剂量 +0.23、27% 轨迹有严格标注，真的在算
        + 正在跑的 425→600（长度软惩罚）如果起效，再教会【停】→ 形态就全了

★ 教不了：验证的【策略】—— 目标 100、手上 20 26 31，该先试乘还是加
        192 种组合，加减里穷举也能撞对一半，奖励分不清「会判断」和「撞对了」
        ★ 根本原因：空间小到策略不重要 ——「部分靠撞对」
★ 教不了：【长度】—— 200 token 就搜完，没有理由写 500
        R1 的 H1 在这个任务上不会出现，无论怎么训
```

## 所以
```
CD-3 到 600 步的判据（池外 800 条）：
   6+ 桶 < 15%、0 次桶不回涨、pass@1 ≥ 0.50、pass@8 ≥ 0.80、精度 ≥ 0.65、剂量为正、长度 180~250
   全达 → CD-3 教完了（不是终点，是没东西可教了）

CD-4 回答另外两个问题：
   策略 —— 七千多种组合，穷举必死，只有会判断的轨迹活下来
   长度 —— 真需要五六百 token，会自己长
```

## 一句话
**CD-3 教形态，CD-4 教策略和长度。先把「停」学了再换。**


---

# ★★★ RL 的四段论 + 「不喂模仿文本」的精确含义 + R1 的 SFT/RL 交替（2026-09-03）

## 四段论：一个行为要被 RL 长出来，四段都得对

```
预训练      提供零件和零件跟上下文的关联         材料
上下文      激活哪一类关联                        开关
环境+奖励   要求某个序列、并且能分辨它            筛子
RL          把零件排成序列、让序列变成默认         组装
```
```
缺预训练      无米（p = 0 是硬底，采不到就领不到奖励，无例外）
上下文不对    引不出（AIME 式提示词引出 wait，Countdown 式引出 too high）
奖励分不清    选不到（CD-3 分不清「会判断该用乘法」和「加减穷举撞对」→ 教不出策略）
有捷径        选错了（单训 CD-3 →「不想直接答」→ 反思被压）
```
**「教」的精确含义：RL 不加东西，它组装。** 零件各自都浅（会算、会比、会说 too high），
RL 把零件排成序列（试→算→比→标→调→停）并让序列变成默认。序列在预训练里 0.4%，组装完 27%。
★ 「深」= 过程的深度，不是知识的深度。天花板 pass@64 不动。

**「预训练要见过」= 见过零件就够，不需要见过整体。**
R1 的完整顿悟在预训练里不是常见整体，wait / 重推 / 换路各是碎片。
★ 「预训练」可换成「RL 之前的任何训练」—— Instruct 的 step-by-step 是 SFT 放的，RL 一样放大。

## 「RL 不喂任何用来模仿的文本」—— 精确版

```
SFT   输入：问题    目标：人写的答案    loss = 交叉熵(输出, 人写的)
      ★ 梯度把模型往「别人写的那串 token」推

RL    输入：问题    目标：无
      自己写 → 判分器给一个数 → loss = −A × logP(自己写的)
      ★ 梯度里的每一个 token 都是模型自己生成的
```

**GSM8K 的 answer 字段有人写的完整解题过程，我们扔了：**
```
读数据    gold = r["answer"].split("####")[-1]    → 只留 "18"，解题文字在这一行就没了
拼提示    R1 模板 + 题目                           → 提示词里只有题目
判分      gsm_reward(txt, "18")                    → 判分器只拿 "18"
loss      rm 掩码：提示词位置 0，生成位置 1         → 连题目都不进 loss
```
训练和判分都没有那两行。题目在提示词里是输入不是目标。
★ 如果把 answer 全文当目标算交叉熵 → 那是 SFT。

## R1 的顺序：SFT → RL → SFT → RL，SFT 数据是模型自己生成的

```
R1-Zero    Base → 纯 RL                      会顿悟，可读性差
R1 四步
  ① 冷启动 SFT       几千条长思维链 → Base    ★ 数据主要是 R1-Zero 输出，人工清洗
  ② 推理 RL          GRPO，规则奖励 + 语言一致性
  ③ 拒绝采样 + SFT   从 ② 采样，筛对的且可读的 60 万条 + 20 万通用 → ★ 重新 SFT 到 Base
  ④ 全场景 RL        GRPO，规则奖励 + 偏好奖励模型
```
**为什么这个顺序：** SFT 需要目标文本，一万 token 的推理链让人写又贵又不自然。
RL 是发现者（在一个模型里把行为长出来），SFT 是整理者（压成干净数据、灌进新起点），再 RL 往前推。
★ R1-Zero 的轨迹对它自己零信息，对从 Base 重起的 R1 是 60 万条数据 —— 「验证过的轨迹对下一代是完整数据」的实例。

我们的路线同形：Alpaca SFT → GSM8K RL 是标准顺序；Base 上的 CD / 混训是 R1-Zero 式纯 RL。
★ 下一步可选：把混训模型的高分轨迹筛出来 SFT 回 Base = R1 的第 ③ 步。


---

# ★★ 为什么不把解题思路给模型做 RL（2026-09-03）

## 三种「给」法，没有一种还是 RL
```
放进提示词      模型看着答案写答案，判分器满分，学到的是抄
当目标算交叉熵  这是 SFT。学「像不像」不是「对不对」—— Alpaca 那轮 val loss 升、输出反而好
逐步打分        过程监督。参考只有一条路（16−3−4=9 / 16−7=9 / 2×(16−3−4) 全对，罚了两条）；
                步骤匹配器难写、易 hack
```

## 为什么让它自己碰
```
① 碰的不是瞎碰      Base 在 GSM8K 46%，零件全在。「自己碰」= 从预训练压缩的人类知识里采样、用判分器筛。
                    人类思路已经通过预训练进了模型，再喂是重复。RL 加的是判分器不是文本
② 判分器认「对」    参考文本认「像」。优化代理就向代理靠拢（SFT 出来会写 <<16-3-4=9>> 计算器标记）
③ on-policy 忘得少  Retaining by Doing：SFT +25%/−27.8%，RL +18%/−3.4%。我们：GSM-RL → CD 反涨 2.9×
④ 参考里没有的      R1 的 wait、AlphaZero 第 37 手、我们的 too high 序列，没有参考解答写着这些。
                    给参考 = 材料换成人写的那一份，只能到那份的水平。R1 超过人写 CoT 靠的就是没给
⑤ 多数任务没参考    Countdown 只有数字；代码有单元测试没标准实现；操作电脑有「完成没」没标准操作序列。
                    判分器比示范便宜得多
```

## 参考确实在用 —— 在 RL 之前
```
R1 第 ① 步冷启动 SFT、我们的 Alpaca SFT：参考把模型带到正确格式和区域，再交给 RL 推
★ 顺序不能反：RL 完再 SFT 人写文本 → 拽回人写分布 → RL 找到的丢掉
   R1 第 ③ 步的 SFT 用的是模型自己的高分轨迹，就是为了不往回拽
```
**一句话：参考解答是一条路的样子，判分器是终点的定义。给样子学到样子，给终点自己找路。
预训练已经给了所有样子，RL 缺的只是终点。**

## 阶梯奖励不是过程奖励
```
过程奖励    读 <think> 每一步，逐步打分。要参考解答或步骤验证器
阶梯奖励    只读 <answer> 那一个表达式：有标签 0.05 / 能解析 0.05 / 数字用对 0.15 / 等于目标 0.75
            ★ 四级全是最终输出的性质，一眼没看 <think>
            <think> 里写胡话 + <answer> 对 = 1.0；推理漂亮 + 式子错 = 0.25
```
**结果奖励 + 部分分。** 部分分给结果的部分性质（格式、合法、约束），不给中间步骤。
★ 所以 too high / too low 搜索是模型自己长的，没有任何一级奖励直接奖它。奖励不管路只管终点。

边界：
```
③b 标注精度   看了 <think>，但是【测量】不是【奖励】。奖了就变过程监督，模型会糊弄验证器
长度惩罚      约束输出长度，不看步骤内容。格式层面，不算过程奖励
若做过程奖励  每行「X = V (too high)」算对加 0.05 —— 没做，参考单一 + 验证器可 hack
```


---

# ★★★ 续训 425 → 600，长度软惩罚（--len-soft 300），2026-09-03

62 分钟，175 步。唯一改动：300 token 以上线性扣分到 512 时 −0.5。

## eval（贪心，CD [90000:90100]，GSM test[200:300]，各 100 题）

| step | CD 正 | CD 长 | 标注@精度 | GSM 正 | GSM 长 |
|---|---|---|---|---|---|
| 425 起点 | 0.36 | 344 | 1.0@0.87 | 0.79 | 175 |
| 450 | 0.48 | 268 | 0.6@0.79 | 0.81 | 165 |
| 475 | 0.54 | 234 | 0.7@0.80 | 0.83 | 164 |
| 500 | 0.48 | 179 | 0.7@0.77 | 0.78 | 165 |
| 550 | 0.51 | 223 | 1.3@0.74 | 0.83 | 166 |
| 575 | 0.52 | 190 | 0.8@0.72 | 0.84 | 168 |
| **600** | **0.61** | **178** | 1.0@0.73 | 0.78 | 161 |

★ 总分列不再跟之前可比 —— eval 的分现在含长度惩罚，短了分就高。只看「正」。

## 训练时探针（温度 1，本段前 20 步 → 后 20 步，只看 CD）

| | 425~444 | 580~599 |
|---|---|---|
| 正确率（25 步均值） | 0.47 | **0.59** |
| 长度 | 240 | **172** |
| 标注数/条 | 0.69 | **0.99** |
| 标注精度 | 0.729 | 0.736 |
| 剂量响应（3~5 − 0） | +0.109 | +0.052 |
| 0 次桶正确率 | 0.464 | **0.708** |
| 3~5 次桶正确率 | 0.574 | 0.684 |
| 6+ 次桶正确率 | 0.325 | 0.215 |
| 写完不停 | 0.088 | **0.016** |
| CD \|A\| | 0.331 | 0.160 |

```
GSM   全程 0.78~0.84，长度 161~175，没受惩罚影响（轨迹本来就不到 300）
KL    0.017 → 0.035，step 597 一次 1.0 的尖刺（|g| 2.34，裁剪兜住，下一步恢复）
wait ★差   −0.22 → −0.28，压得更狠
宽泛搜索词  出现率 0.86 → 0.97，几乎每条都有
```

## 判读

### ★★ 长度惩罚做到了它该做的
```
长度      温度 1 240 → 172，贪心 344 → 178，砍掉近半
续写      8.8% → 1.6%，补 EOS 生效
正确率    贪心 0.36 → 0.61，温度 1 0.47 → 0.59  ← ★ 惩罚没有伤搜索，砍掉跑飞反而涨了
精度      0.73 → 0.74，平的
标注数    0.69 → 0.99，还在涨
```
「惩罚太狠压没搜索」那个担心没发生：标注在涨、精度在守、正确率在涨。

### ★ 一个要重新理解的信号：剂量响应 +0.11 → +0.05
0 次桶从 0.46 涨到 0.71，跟 3~5 次桶（0.68）持平了。
两种读法：
```
(a) 模型学会了【选择】：简单题直接答（0 次桶 = 简单题，0.71），难题才搜（3~5 桶 = 难题，0.68）
    桶按难度分了层，「搜 vs 不搜」的比较变成了「简单题 vs 难题」，剂量响应失去意义
(b) 搜索变装饰了
```
精度 0.74 说明标注还是真算的，倾向 (a)。★ 要 800 条的组内 ★差（同一题里比）才能定。

### ★ CD |A| 0.33 → 0.16：CD-3 在饱和
温度 1 正确率 0.59，越来越多的组全对，优势归零。这是「CD-3 教完了」的信号。

### 跟单任务比
```
单任务 CD-3    贪心 0.405（[90000:200]），长度 48，GSM8K 0.16
混训 600       贪心 0.61（[90000:100]），长度 178，GSM8K 0.78
```
★ 看起来超过了，但两片只有前 100 道重叠，要同片确认。差 0.2，抽样噪声 ±0.05，大概率真的。
**如果确认：混训 + 长度惩罚在 Countdown 上超过单任务，同时保住 GSM8K，同时输出里有可见的验证过程。**

## 待做
```
① ckpt_best 现在 = 600。cp 一份 ckpt_step600.pt
② 池外 800 条 + analyze_search：6+ 桶占比（425 时 43%）、组内 ★差、精度
③ CD-4 基线（还没报结果）
④ 单任务模型在 [90000:90100] 贪心一次，跟 0.61 同片比
```


## 池外分桶：425 → 600（analyze_search.py，2026-09-03）

| 桶 | 425 | 600 |
|---|---|---|
| 0 次 | 96 条（12%）0.323，133 tok | 15 条（2%）0.333，91 tok |
| 1~2 次 | 186 条（23%）0.548，183 tok | 246 条（31%）**0.797**，131 tok |
| 3~5 次 | 172 条（22%）0.552，221 tok | 198 条（25%）0.601，156 tok |
| **6+ 次** | **346 条（43%）0.176，375 tok** | **341 条（43%）0.117，263 tok** |
| ③b 标注数 / 精度 | 661 / 0.643 | 1032 / **0.733** |
| ③ 说「对了」真对率 / 基线 | 0.488 / 0.361（1.35×） | 0.648 / 0.450（1.44×） |
| ⑤ 带严格标注的轨迹 | 217（27%） | 493（62%） |
| ⑤ 带 vs 不带 | 0.369 vs 0.358 | 0.404 vs **0.524** |
| ⑤ 组内 ★差 | −0.005 | −0.000 |

### ★★★ 「跑飞」是两件事，惩罚只治了一件
```
① 长度跑飞（撞 512）        治了：6+ 桶 375 → 263 tok，P95 512 → 369，撞顶 14% → 2%
② 策略失败（搜错子空间）    没治：6+ 桶 346 → 341 条，43% → 43%，正确率 0.176 → 0.117 反而更低
```
6+ 桶现在 = 「硬题上快速失败」：32 道全错的题 × 8 ≈ 256 条，占了这桶的大头。
惩罚把失败从 375 token 压到 263 token，没把失败变成成功。
★ 净效果：硬桶少对 21 条（61 → 40），易桶多对 94 条（102 → 196）。用硬题的一点点换了易题的一大块。

### ★ 精度涨了，但精度不是瓶颈
标注精度 0.64 → 0.73，说「对了」的可信度 1.35× → 1.44×。验证在变准。
⑤ 组内 ★差 = 0：同一道题里，写不写标注跟对不对无关。
★ 意思是：模型算得对、标得对，但在硬题上算的是错的候选 —— 加减里绕，一次乘法不试。
   验证是准的，被验证的东西是错的。瓶颈在【候选生成】不在【验证】。

### ⑤ 带 0.404 < 不带 0.524 —— 选择效应，不是标注有害
易题一次命中，写的是「(Yes, this works)」，不匹配 too high/low 正则 → 算「不带」
硬题多次试错，写 too high / too low → 算「带」
「带」= 硬题，「不带」= 易题。组内 0 才是干净读数。

### 结论
```
形态全了：试 → 算 → 比 → 标 → 停，精度 0.73，续写 1.5%，长度 168
CD-3 上能教的教完了。剩下 43% 的轨迹 / 32% 的题卡在【搜索广度】—— 从不试乘除
这不是长度问题，不是精度问题，是策略问题。长度惩罚治不了，再训 CD-3 也治不了（全错组零梯度，半径之下）
```

### 下一步的判断依据
```
先看硬题的原始输出里【有没有 ×】：
   有（哪怕 2~5%）→ k 加到 32~64 能抽到 → 可以在 CD-3 上继续
   一次都没有     → p = 0，RL 无能为力 → 要么 CD-4 逼出来，要么冷启动 SFT 几条 × 的样例
CD-4 基线还没跑，那个数决定路线
```


---

# ★★ CD-4 基线：600 模型零训练直接做 4 个数（2026-09-03）

`countdown4.json`（--k 4，tmax 100），池外 [95000:95100]，800 条，温度 1。

| | Base（CD-4，早期测） | **600 模型（CD-4）** | 600 模型（CD-3，对照） |
|---|---|---|---|
| pass@1 | 0.0175 | **0.208** | 0.450 |
| pass@8 | — | **0.610** | 0.680 |
| 全对 / 全错 / 有对有错 | — | 0 / 39 / 61 | 16 / 32 / 52 |
| 数字恰好各用一次 | — | 0.675 | 0.934 |
| 组内有差异 | — | **100%**，sd 0.219 | 72%，sd 0.174 |
| 长度中位 / P95 | — | **242 / 477** | 168 / 369 |
| 搜索词每条 | — | 8.1 | 5.9 |
| 续写 / 撞 512 | — | 1.2% / 4.2% | 1.5% / 2.1% |
| (a±b)×c 形态 | — | 1.5% | 未测（文件被覆盖） |

## 判读
```
① ★ 可训。pass@1 0.21 在甜区，100% 的组有信号，sd 0.219 是历史高位。
   我之前担心「起点在半径下面」错了：CD-3 练的东西搬过去 12 倍。
② ★ 空间大。pass@8 0.61 但 pass@1 只有 0.21 —— 0.40 的差距是「偶尔对 → 稳定对」的空间，RL 最擅长的那种
③ ★ 长度自己涨了。168 → 242，任务要求的，不是跑飞（续写 1.2%，撞顶 4.2%）
④ 新失败模式：
   · 数字用不全：nums_ok 0.93 → 0.68，4 个数经常只用 3 个（样例二里一堆 2~3 个数的候选）
   · 链式算术出错：「6+12+18−42 = 0+0 = 0」（是 −6）、「42−6 = 36 (too low)」（方向标反）
   · 放弃后瞎交：「it is not possible」→ 交一个错的
⑤ (a±b)×c 仍是 1.5%，硬尾巴的钥匙还没在手上
```

## 文件名撞车事故
`baseline_countdown.py` 输出名只带 ckpt，CD-4 那次把 CD-3 的 600 原始文件覆盖了，扫描读成 CD-4 轨迹配 CD-3 题目。
已修：文件名带数据集名、raw 记 data 路径、分析脚本按路径读题、暴力求解支持 4 个数。

## 下一步：GSM8K + CD-4，从 600 起
```
--data data/countdown4.json  --resume（从 out_mix_mp/ckpt_latest = 600）
--len-soft 350（CD-4 产出型轨迹 250~300，跑飞 450+）
其余不变。CD-3 拿掉：它饱和了，CD-4 包含它
```


---

# ★★★ 阶段小结：从 step 300 到 CD-4 起跑（2026-09-03）

## 做了什么

| # | 做了什么 | 关键数 |
|---|---|---|
| 1 | 续训 300→425，换新探针 | CD 贪心 0.25→0.37，标注 2.7×，续写 28%→10%。总分噪声触发早停 |
| 2 | 发现测试片在训练池里，换到 [95000:95100] | 425 的 pass@1 0.53（污染）→ 0.36（干净） |
| 3 | 四视角分析 | ★差=0 是两效应抵消：验证是真的（精度 0.82、说对了 1.65×），剂量 +0.13，但 43% 跑飞 |
| 4 | 精度警报，然后撤回 | [0:100] 0.82→0.71，[95000] 0.60→0.64。切片依赖，实际 ~0.7 平的 |
| 5 | 长度软惩罚，425→600 | CD 贪心 0.36→0.61，长度 240→172，续写 1.6%，精度守住，标注 0.69→0.99/条 |
| 6 | 600 池外测 | pass@1 0.45，pass@8 0.68，格式解析数字 0.95+，撞顶 2.1%，有信号的组 72% |
| 7 | 分桶 425→600 | 6+ 桶 43%→43% 一条没少，长度 375→263，正确率 0.18→0.12 |
| 8 | 扫描硬题 | 全错 32 道里 30 道必须乘除；75% 的轨迹试过 ×；(a±b)×c 形态只有 1.5% |
| 9 | 算分布 | CD-3 72.6% 加减可解，CD-4 67%。几个数不是杠杆，tmax 才是 |
| 10 | 600 模型零训练做 CD-4 | pass@1 0.21（Base 0.0175），pass@8 0.61，100% 有信号，长度 242 |

踩的坑：文件名撞车（CD-4 覆盖 CD-3 原始文件）、第四次小样本推断（「只在加减里绕」被 240 条推翻）、
第二次用复合指标早停、剂量响应在模型学会「选择何时搜」之后失去意义。

## 结论，对着目标

**目标一：反思搜索的智能 —— 形态全了，是真的，在放大；策略没长。**
```
试 → 算 → 比 → 标方向 → 调整 → 停
严格标注形态      Base 0.4% → 275 2.6% → 425 27% → 600 62%
标注精度          0.73        说「对了」可信度 1.44×
学会停            续写 45% → 1.5%，跑飞长度 375 → 263
★ 策略窄：先加减、乘法只试直接乘、(a±b)×c 1.5%。CD-3 里 30% 的题要乘除，全错。
   原因：73% 的 CD-3 题加减就够，RL 学到基础率，乘法没有净正优势。
   ★★ CD-3 能教的边界：形态教会了，策略教不了。
```

**目标二：长思维链 —— CD-3 上没有，CD-4 上刚开始。**
```
CD-3   Base 187 → 600 168，任务 200 token 就搜完，反而缩
CD-4   零训练 242，任务自己要求的
★ R1 的「越训越长」还没看到，CD-4 训练是它该出现的地方
```

**混训本身：** 两任务都涨、没一个掉、无捷径、验证可见。CD-3 贪心 0.61 vs 单任务 0.405，GSM8K 0.78 vs 单任务 0.16。

## 接下来：GSM8K + CD-4，从 600 起，300 步

四个理由：
```
① CD-3 饱和：有信号的组 72%，阶梯三级 0.95+，硬尾巴在半径之下
② CD-4 有 0.40 空间：pass@8 0.61 vs pass@1 0.21，全是「偶尔对 → 稳定对」
③ CD-4 的长度是任务要求的 —— 看 H1 的地方
④ CD-4 新失败模式（数字用不全 0.68、链式算术错、放弃瞎交）都在阶梯射程内
```
盯五个数：长度趋势、标注数和精度、(a±b)×c 比例、6+ 桶占比、数字用全比例。

两种结局：
```
(a±b)×c 从 1.5% 往上  → 策略在长，顿悟最后一块拼上，续到饱和
停在 1.5%             → 形态不在分布里。两条路：tmax 拉到 500（53% 必须乘）；
                        或冷启动 SFT 十几条「先组合再乘」样例把 p 抬到 0.05 再 RL
```
再往后：阶段三，猜数字多轮环境，credit assignment。

**一句话：混训在 CD-3 上把顿悟的【形态】长出来了，CD-4 要看能不能长出【策略】和【长度】。两个目标各差一半，指向同一个实验。**


---

# ★★★ RL 不是老师，是园丁（2026-09-03，用户的困惑 → 一个比喻收住）

## 困惑
「RL 只核对答案，没告诉模型怎么做，感觉不是在教，像是采样有可能对的然后组装放大。」
—— 这个描述完全正确。乱的感觉来自「教」这个词，它不合适。

## 园丁
```
老师   讲「这一步该这么做」
园丁   不讲。只做一件事：看哪株结果了，浇它；哪株不结，不管它。
       园丁一粒种子都没种。花园里所有植物都是原来就在的（预训练）。
       几百天后花园完全变了，但没有一株是园丁带来的。
```

## 三句话
```
预训练放零件     会算、会比、会说 too high、会写 <answer>。每样都会，散着。
采样出组合       温度 1 采 16 条，每条是零件的一种随机拼法。模型不知道哪种好，只是在分布里抽。
判分器选         命中 1.0，没命中 0.25。抬 1.0 那条的每个 token，压 0.25 那条的。
                 它不知道「为什么」对 —— 不需要知道，只需要知道「这条对了」。
600 步 × 128 条 = 77,000 次 1 bit，够把「默认拼法」换掉。
```

## 数据里园丁的手
```
Base    「算式 = 值 (方向)」形态 0.4%，★差 = 0（说 too high 跟没说一样）
600 步  同一形态 62%，精度 0.73
★ 没有任何一步告诉过模型「先算再比」。判分器只做了：算了再比的恰好更常命中，命中的拿 1.0。
   0.4% 被浇了 600 次，长成 62%。
```

## 边界（把「乱」收住的那一条）
**园丁不能让花园长出它没有的植物。**
(a±b)×c 1.5%。若它在必须乘除的题上从没出现，那些题 16 条全 0.25，没有 1.0，园丁没东西可浇。
★ 「RL 能不能教会 X」永远先问：X 或 X 的零件，现在在不在采样里？
   在 → 能从 0.4% 浇到 62%。不在 → 换地（换任务让它冒出来）或撒种（冷启动 SFT）。

## 为什么这比「教」强
给它看解法，学到的是那个解法的样子。让它自己试、只说对没对，长出来的可能是人没写过的
（R1 的 wait、我们的「算一个、标方向、换一个」序列，没有哪份参考解答里有）。
而且长出来的东西从它自己的分布里来，离自己近，忘得少。

## 一句定义
**RL = 在某个环境（提示词 + 判分器）下，用奖励放大模型已有行为中「恰好得分高」的那些的概率。**
不是教，是选。选的材料是预训练给的，选的标准是环境给的，选的动作是梯度做的。


### 补两个字，定义就完整了
```
RL = 在某个环境（提示词 + 判分器）下，用奖励放大模型【已有】行为里恰好得分高的那些的概率
     「已有」是关键 —— 材料是预训练给的，标准是环境给的，动作是梯度做的。不在采样里的，奖励再高也放大不了

另一半：同时【压低】得分低的。放大和压低是同一个梯度的两面
     wait 那套措辞被压没、续写被压没，都是这一半
     「强化」这个词说的就是：强化用过且管用的，弱化用过但没用的
```

### 往后看一眼：单轮 vs 多轮
```
我们做的是单轮：一条轨迹、一个分、一次更新
强化学习更大的一块是多轮：十步动作只有最后一个分，是第三步的功劳还是第七步的？
   → credit assignment，阶段三猜数字要撞的
核心一样，还是园丁。只是花园里的植物开花要十天，园丁得想办法知道该浇第几天
```


## CD-3 硬尾巴的解剖（scan_ops.py，600 模型，池外，2026-09-03）

```
全错 31 道，30 道必须乘除 —— 硬尾巴 = 乘除题，没有别的
硬题轨迹 77% 有 × 或 ÷ —— 运算符不是问题
```

### 试的是什么
```
纯直接乘 a×b      47%    12 × 48 = 554 (too high)     两个大数相乘，几乎必爆
组合形态          ~4%    48 / (12 − 2) = 4.8           有形态，分错组
★ 对的式子        有     48 / 2 + 12 = 30 (too low)    ← 找到了，算成 30（是 36），标 too low，走了
```

### ★★ 两个瓶颈叠着
```
搜索广度   乘除子空间里主要试「两个给定数直接乘」，目标几十的题上必爆。
           有用的形态（先除再加、先减再乘、先加再除）只占 ~4%，在 k=16 半径边上
算术精度   标注精度 0.73，27% 算错。算错一个候选损失一次尝试；算错的恰好是答案，损失整道题
联合       生成对的形态 4% × 算对 73% ≈ 3%，8 条里碰一次 22%。30 道一道没碰上 → 那 4% 里大半还是分错组的
```

### 对 CD-4 的含义
CD-4 只有 33% 的题必须乘，基础率跟 CD-3 差不多，**解决不了硬尾巴** —— 加减先试还是多数策略。
CD-4 照跑（有 0.40 的空间），但硬尾巴要另外的压力：
```
① tmax 拉到 500 / 筛必须乘除的题混进去 —— 让加减先试在一半的题上失败
② k 加到 64 —— 把 4% 的组合形态抽出来喂 RL
③ 冷启动 SFT 二十条组合形态样例 —— 4% 抬到 20% 再 RL
```
精度 0.73 是天花板的一部分。算对的比例不上去，搜到了也会丢。

### 工具修正
`(a±b)×c` 正则太窄，抓不到 `48/2+12`、`37+39/13`。改成「同一行既有乘除又有加减」= 混合形态。


---

# ★★ 三个追问：多步奖励、没判分器的题、RL 给 coding 能力（2026-09-03）

## 多步题的奖励
```
单轮多步（Countdown 试十次、GSM8K 五步）
   最后核对答案，整条轨迹同一个优势。看着粗（九对一错 = 十错），但统计上没偏：
   对的轨迹里对的步骤多，几万条平均下来对的步骤被抬得多。R1 只用这个。代价：样本效率低

多轮多步（猜数字、agent 调十次工具）—— credit assignment 真问题，从便宜到贵：
   ① 全给       十步都拿最后那个分。步数少、样本够就行
   ② 步数惩罚   每步扣 0.05。不是过程监督，「更少步猜到」本来就是目标。顺手给了密集信号
   ③ 价值函数   再训一个模型估「这个状态多好」，每步优势 = 状态变好了多少。PPO。多一倍算力
   ④ 中间往后采 走到第五步停，从这采八条往后，看成功率 = 第五步的分。树搜索，贵但准
猜数字打算 ① + ②
```

## 「如何赚到一百万」—— 没有判分器
```
真去做：几年、真金白银、一个样本。模拟经济：保真度是天花板、没人有。规则检查：太弱。
★ 人类偏好（RLHF）：两个回答人选一个，训奖励模型替人打分 —— 工业界对这类的实际做法
   教的是「人觉得好」不是「真的行」。副作用：谄媚
★★ 根本原因：奖励必须比任务便宜、能批量。Countdown 0.1ms，代码跑测试；「赚一百万」验证一次几年
   唯一真实的环境是真实经济，没人拿它训。模型的回答 = 预训练里人类建议的压缩 + RLHF 抛光
   ★ 不会超过训练数据里最好的那个人。Countdown 上能超人是因为判分器比人便宜
能做的：拆成子问题，可查的做准，综合还是没法验
★ 这就是工业界 RL 全堆在代码/数学/agent 上的原因 —— 不是更重要，是有判分器
```

## RL 给 coding 能力 —— 用户的表述对，三处收紧
```
「先有」   有零件就行不需要有整体。coding = 语法 + 库 + 算法模式 + 读报错 + 写测试，各有各的 p
「概率小」 不一定小。Base 写代码 pass@1 可能已 30~50%，RL 做的是 30 → 80 不是 0 → 80
「放大」   pass@1 往 pass@k 靠。GSM8K：pass@1 0.53→0.74，pass@64 0.98→0.98
           天花板唯一能动的方式是【组装】—— 零件拼成采样里没出现过的整体
做不了的   没见过的语言、训练截止后的库。零件不在，园丁没东西浇 → 要继续预训练 / mid-training
工业界堆 coding 的原因：Base 已会（零件全）+ 单元测试免费（筛子便宜）+ 几千题跨领域（多样性够）
```


## 没见过的语料一定不会？—— 「一定」太强，问题是 p 是不是恰好零
```
领域和零件都没见过    不会。园丁没东西浇
零件见过组合没见过    可能会一点，p 小非零，RL 能放大（(a±b)×c：整体 0.4%，零件全有）
恰好见过但极少        p 0.01%，技术上会但 k=16 抽不到 → k 加到几千，或冷启动撒种
★ 18T token 的模型绝大多数东西 p 不是零只是小。「不会」通常是「抽不到」
另一种：题目里直接给（提示词放新库文档）→ 当场能用，但那是「在读」不是「有能力」
```

## Agent 任务 pass@k = 0 怎么训 —— 不在零的地方训
```
原则只有一条：pass@k = 0 的任务上采样，16 条全 0 分，没优势，梯度零。花多少算力都零。
五条路找 p > 0 的地方，工业界全在用：
   拆         「做网站」→ 写函数 / 修 bug / 调 API，先训子任务再合。我们 CD-3 → CD-4 就是
   部分分     整体 0 分但「测试过三个」「文件建对」各给一点。四级阶梯是小版本
   冷启动 SFT 人做几百条示范或强模型做，p 从 0 到 5% 再 RL。每个 agent RL 管线第一步
   难度阶梯   五步 → 十步 → 五十步，每级起点在上级终点
   蒸馏       大模型 p > 0，采它的轨迹训小的。小 agent 模型基本这么来
★ 先让地里有苗，再浇。Agent 的全部工程都在回答「怎么让 pass@k 从零变非零」
```


---

# ★★★ 五个旋钮：RL 之外能动的全部（2026-09-03）

RL 的梯度那一步自己没有旋钮 —— 它只按尺子浇水。环境设计者手上的旋钮有且只有五个，对回四段论的五个位置：

```
撒种          材料        预训练 / SFT / 蒸馏 / 拒绝采样  往分布里放零件，让 p 从 0 变非零
换开关        开关        提示词决定激活哪片先验          ★ 零成本，先试
换地换数据    筛子的题    哪些题、什么配比、拆不拆、难度阶梯
换奖励        筛子的尺    对错 / 阶梯 / 每步 / 增量 / 偏好 / 长度惩罚 / 熵奖励
多抽          组装的宽    k、温度                         决定采样半径多大
```

## 撒种的三个来源
```
人做示范      SFT
强模型做      蒸馏
自己做完筛    拒绝采样（R1 三步 SFT 全是这种，没有人写的）
```
共同点：往分布里放一个 p > 0 的东西，RL 才有得浇。

## 判分器能给什么（换奖励那一格展开）
```
对错          0/1
阶梯 / 多维   几个分量加权（格式 0.05 + 解析 0.05 + 数字 0.15 + 正确 0.75）
连续量        数字用对的比例、跟目标的距离、通过的测试数
每步一个      猜数字每步 −0.05；Lean 每一步接受 / 拒绝
增量          这步新通过 3 个测试 +3，新挂 1 个 −1
偏好 / 排序   「A 比 B 好」，Bradley-Terry 转数；DPO 直接用对子
可验证性质    编译过 / 类型检查过 / 没 lint 错，各给一点
文本 → 数     LLM 裁判先写评语再打分，进梯度的还是那个数
一致性        采 8 条 6 条一样当对（TTRL）
人的动作      点赞、修改的 diff
★ 硬约束：进梯度的必须是一个标量。来源随便，形状随便，最后是一个数
```

## 观测 ≠ 奖励
```
猜数字环境回「大了」→ 观测，进提示词，改下一步怎么猜
猜中 +1、每步 −0.05    → 奖励，进梯度，改这条路值不值
Countdown 里模型自己写的 (too high) 是【自己生成的观测】，环境没给 —— 所以会算错。环境给的观测不会错
```

## 密集奖励的代价
每加一项就多一个能被 game 的地方：「每通过一个测试 +0.1」→ 改测试让它永远过；「每步 −0.05」→ 一步瞎交。
★ 每一项都问：最懒的拿分方式是什么（检查清单第 ⑤ 条，多步任务里问 N 遍）。

## 一句话
「让模型有 X 能力，除了 RL 还能做什么」—— 就这五样。这五样就是环境设计的全部，也是工业界 RL 团队全部的工作内容。


---

# ★★ 形态正则改宽后的扫描 + 标准实验流程（2026-09-03）

## 扫描结果（正则：同一行既有乘除又有加减 = 混合形态）

| | CD-3 600 | CD-4 600（零训练） |
|---|---|---|
| 全错的题 | 31（30 必须乘除，1 加减可解） | 39（**27 必须乘除，12 加减可解**） |
| 硬题轨迹试过 ×÷ | 77% | 72% |
| 硬题轨迹有混合形态 | **62%**，1.55 条/轨迹 | **50%**，1.82 条/轨迹 |
| 全部 800 条有混合形态 | **42%** | **43%** |

### ★★★ 上一轮的 3.6% 是正则太窄；混合形态实际 42%
```
模型不是「不试组合」—— 它每条硬题轨迹平均试 1.5 个混合形态：
   2×12+48、48−12×2、2×12−48、48−2×12、48/(12−2)  ← 五个，全错。对的是 48/2+12
   第二条试到了 48/2+12，算成 30，标 too low，走了
```
**真正的瓶颈：在一个大空间里稀疏地随机采。**
```
3 个数的加减形态约 24 种，模型一条轨迹试 6 个，覆盖够
3 个数的混合形态约 96 种，模型一条轨迹试 1.5 个，覆盖 1.6%
→ p(对的形态出现) ≈ 1.6%，× 算对 73% ≈ 1.2%，240 条里期望 3 条对，实测 0（泊松 P(0)=5%，不矛盾）
★ 1.2% 在 k=16 半径 4.4% 之下。k=64 半径 1.1%，边上。
```
不是「不会乘法」，不是「不试组合」，是「在 96 种里随机试 1.5 种」。**无引导的搜索。**

### CD-4 额外的两个问题
```
12 道加减可解的题也全错   4 个数的加减形态 192 种，模型试 8 个，覆盖 4%；再加链式算术出错
候选只用 3 个数           样例里 15 个候选没有一个用全 4 个数。nums_ok 0.68 的来源。阶梯的数字分会推
```

### 能做什么
```
① 让它别在加减里浪费预算   目标 36、和是 62、加减全试完没有 36 → 该跳过。这是「策略」，要 RL 自己长
② 每条轨迹多试几个混合形态  --len-soft 提到 400 给预算；或让它跳过加减直接试混合
③ k=64                    半径压到 1.1%，够到 1.2%
④ 精度                    27% 算错里有一条是答案
```

## ★★★ 标准流程：每个 checkpoint 训完必做

```
0. 训前（换任务时）
   baseline_countdown.py --data 新任务 --offset 95000        → pass@1 在 0.05~0.5？组内有差异 > 60%？
   不在 → 别训，先换地 / 撒种 / 拆

1. 备份
   cp out_X/ckpt_latest.pt out_X/ckpt_stepN.pt              → ckpt_best 会被后面的覆盖

2. 训练日志总结（脚本自动打）
   记：eval 曲线、探针（标注数/精度/剂量/不停）、|A| 平衡、KL

3. 池外 800 条                                                 ★ 必须 --offset 95000，必须跟上次同片
   baseline_countdown.py --ckpt ckpt_stepN [--data ...] -n 100 -s 8 --max-new 512 --offset 95000
   记：pass@1 / pass@8 / 阶梯四级 / 全对全错 / 长度中位 P95 / 续写 / 撞顶

4. 四视角 + 分桶
   analyze_search.py 上一个_raw.json 这一个_raw.json
   记：剂量四桶 / ③b 精度 / ⑤ 组内 ★差 / 说对了精度

5. 硬题解剖
   scan_ops.py 这一个_raw.json
   记：全错里几道必须乘除 / 混合形态比例 / 两条完整轨迹

6. 跨模板（GSM8K，查遗忘和绑定）
   crosstask_eval.py --ckpt ckpt_stepN --out crosstask_N.json
   记：R1 模板 / 中性模板 各自的正确率和长度

7. 对照
   同片、同脚本、同协议 → 跟上一个 ckpt 比、跟 Base 比
   ★ 脚本改了 → Base 重跑一遍重新锚定

8. 归档 PLAN.md

9. 定下一轮配置
   把第 7 步的结论翻译成【改哪一个旋钮】：撒种 / 换开关 / 换地 / 换奖励 / 多抽
   ★ 一次改一个。例：扫描说「形态在、覆盖不够」→ 多抽（k），不是撒种
```
```
命名   cd_{ckpt}_s8_off95000[_plain][_countdown4]_raw.json   数据集不同名字不同，不会撞
口径   温度 1 采样 = p；贪心 = 部署表现。两个都要，别混
铁律   评测片在训练池外；跨片不比同一模型的绝对值；改脚本先重锚 Base
```


---

# 待办（2026-09-03 记）：CD-4 之后的两条路

## E. R1 第 ③ 步：拒绝采样 → SFT → RL
```
采    从 CD-4 训完的模型采几千条（100 题 × 64 或 1000 题 × 8）
筛    答对 + 无续写 + 标注精度高（③b 全对）+ 试错 ≤ 5 —— 留 10~15%
SFT   回 Base（R1 的做法，甩掉杂质）或回训完的模型（自蒸馏）
再 RL 从 SFT 后的模型起，CD-4 或更难
★ 能一次清掉：续写习惯、加减先试的惯性、模板绑定 —— RL 一路积累的杂质
★ 比 k=64 便宜（SFT 一步 = RL 的 1/10），工业界验证过的路
★ 代价：SFT 收窄分布，熵降，下一轮 RL 半径缩 —— 所以 SFT 完立刻 RL
```

## F. Agent RL：猜数字，多轮
```
环境    1~1000 猜一个数，环境回「大了 / 小了」，最优策略二分 10 步内
奖励    猜中 +1，每步 −0.05，超 15 步 0
新问题  跨轮 logP（环境的话不进 loss）、历史怎么进 prompt、★ credit assignment（十步一个分）
先测    Base 是瞎猜还是已经会二分 —— 决定有没有空间
★ 这是阶段三，单轮 → 多轮是真正的下一个台阶
```

## p 小非零、RL 够不着时的四条路（补进五旋钮）
```
多抽      k 64 / 温度 1.1                       半径压到 p 以下
换地      课程 —— 同技能更易的题先训            火柴人逻辑，永远在「稍微好一点」的台阶上
换奖励    梯子搭细 —— 加「跟目标的距离」一级     先查数字用全再给距离分，防凑
撒种      SFT 几十条                            p 从 1% 到 20%
拆        「该用哪个运算符」单独成子任务
★ 判断「真上不来」：撒了不长（SFT 一百条 p 只到 3%）→ 容量问题，换模型
```


---

# ★★ R1 的流程、RLVR vs RLHF、RL 为什么能超过示例（2026-09-03）

## DeepSeek-R1 四阶段（+ R1-Zero）
```
R1-Zero   V3-Base 纯 GRPO，规则奖励（答案 + 格式）。AIME 15.6% → 71.0%。顿悟在这看到。脏：语言混用
① 冷启动 SFT      几千条长思维链（few-shot 引 V3-Base / R1-Zero 生成 / 人工整理）→ SFT V3-Base。给格式和起点
② 推理 RL         GRPO，数学代码科学逻辑，规则奖励 + 语言一致性奖励（略降性能换可读）
③ 拒绝采样 + SFT  从 ② 采，筛「判对 + 可读」约 60 万条 + 20 万通用 → ★ SFT 到【新的】V3-Base，2 epoch
④ 全场景 RL       ③ 的模型上 GRPO：推理用规则，通用用偏好奖励模型
蒸馏              ③ 那 80 万条直接 SFT 小模型，不做 RL。Qwen-32B：蒸馏 72.6% vs 自己 RL 47.0%
放弃的            PRM（步骤定义不清、被 hack）、MCTS（token 空间太大）。最后就是结果奖励 + GRPO
```
```
对到我们   R1-Zero ↔ Base+CD / 混训（做了，脏的形式不同）
           ① 跳过了   ② ↔ 混训 + 阶梯 + 长度惩罚（长度惩罚 ≈ 语言一致性奖励）
           ③ ↔ 待办 E   ④ 不适用   蒸馏 ↔ 拿混训轨迹 SFT 更小的模型
★ 我们从 Base 直接 RL 没冷启动，格式和续写靠 RL 自己磨（300 步压续写）。R1 几千条 SFT 一开始搞定。
   待办 E 不只是收尾，也是下一轮的冷启动
```

## 偏好奖励模型
```
没有规则判分器的任务（有用性、语气、安全）→ 人在两个回答里选一个 → 几万到几十万对
→ 训一个模型出标量分，P(A>B) = σ(r(A)−r(B))，Bradley-Terry → RL 时当判分器
★ RLHF 最脆的一环：模型学会讨好判分器（啰嗦、自信、顺着说）。我们全是规则判分器，躲开了
```

## RLVR vs RLHF —— 不是二选一，是分工
| | RLVR | RLHF |
|---|---|---|
| 判分器 | 规则函数 | 偏好模型 |
| 适用 | 数学 / 代码 / 逻辑 / Countdown | 有用性 / 语气 / 安全 / 写作 |
| 信号 | 精确、免费、难 hack | 代理、贵、会被讨好 |
| 产出 | 能力（pass@1 涨） | 行为对齐 |
```
2026 两个都用，同一轮 RL 混着（R1 第 ④ 步）。趋势：RLVR 份额涨（能写判分器的任务变多），RLHF 收缩到聊天质量和安全。RLAIF 用模型替人打偏好标签
★ 能用 RLVR 就用 RLVR。用不了才 RLHF —— 它涨的是「像不像人想要的」，不是能力
```

## 蒸馏赢 RL 说明什么 —— 「谁来做 RL」，不是「RL 不如输入知识」
```
32B 在 AIME 硬题上 p 极小，自己的 RL 看不见（半径）。671B 的 p 高，能采到。
蒸馏 = 把 671B 采到并筛过的解当 SFT 目标给 32B。★ 那 80 万条本身是 RL 的产物，没有 R1 的 RL 就没有它们
分工   RL = 发现（找到没有任何数据集里有的行为），做一次，在 p 最高的模型上
       SFT = 复制（行为一旦变成数据，抄比找便宜，抄的模型 p 再低都行）
规则   没有示例数据、且模型 p 够高 → RL；有示范数据 → SFT。示范可以是人写的，也可以是别的模型 RL 出来的
       蒸馏天花板 = 老师；蒸馏完再 RL 一点更好（SFT 收窄，RL 重新收紧）
```

## 没有示例 → RL → 拒绝采样 → SFT → 再 RL —— 这是个循环
```
每一轮 SFT 的数据都比上一轮的模型好，因为筛过。飞轮 / mid-training / on-policy 蒸馏，同一个圈
★ 第一步的 RL 也要 p > 0。真零示例又 p ≈ 0 → 换地撒种。R1 冷启动前也用 few-shot 引出长思维链（换开关）
★ 「没有示例」实践里很少见，真问题是「够不够 SFT 到目标」。R1 的洞察：RL 能超过示例
```

## ★★ RL 为什么能超过示例：SFT 的天花板是示例，RL 的天花板是判分器
```
SFT 学「像示例」，示例多好它最多多好。人写长思维链不自然（没人在证明里写五十次 wait），封顶低
RL 学「判分器认」，采出人没写过的路、判分器说对、就放大。R1-Zero 零示例 71%；我们的「算一个、标方向、换一个」没人写过

RL 超得过的条件（四个同时）
   判分器是规则的（偏好模型封顶在「人喜欢」）
   模型 p 够高
   示例少或质量差
   任务有多条对的路（SFT 只奖示例那条，RL 奖所有对的）
SFT 赢的情况
   小模型（半径太小）；示例又多又好；格式风格（RL 学这个慢，我们 300 步压续写，SFT 几千条一步到位）

★ 词要收紧：「能力更强」—— RL 不加零件，pass@k 不动。它超过 SFT 的是任务表现（pass@1）和解的形态（找到示例里没有的路）
★ R1 配方两个都用各干各的：SFT 管格式和起点（快），RL 管发现（能超），拒绝采样把发现变回 SFT 数据（干净）
```


---

# ★★ 飞轮的精确版 + 撒种与收紧的区别（2026-09-03）

## 飞轮
```
p > 0 → RL 抬 pass@1 → 拒绝采样 → SFT → RL 抬 pass@1 → …
   燃料   每圈的新信息只有两种：环境的裁决（新样本对/错）、组合涌现（稀有）
          环境固定、够得着的题解完 → 裁决全「对」→ 信息零 → 要换地（更难的题）
   停的地方
      熵螺旋向内   SFT 收窄 → 下圈 RL 半径缩 → 采得少 → 筛得窄 → 再收。要维持探索（温度、熵奖励、留不完美样本）
      收益递减     R1 约两圈。「自我改进」论文普遍两三圈后趋零。无人展示无限循环
      零件封顶     转不出预训练没有的。混合算术 1.5B 算不稳 → 精度 0.73 转多少圈都不动 → 换模型
```

## ★★ 「SFT 是不是撒种」看数据从哪来
| | 撒种 SFT | 拒绝采样 SFT（自蒸馏） |
|---|---|---|
| 数据来源 | 模型外面：人写、更大模型采 | 模型自己采、筛过 |
| 数据里有 | 目标模型产不出的 | 目标模型本来能产的（在它 pass@k 里） |
| 对分布 | 放入新质量，p≈0 → p>0 | 集中已有质量 |
| pass@k | 涨 | 直接不涨，靠程序泛化间接涨 |
| pass@1 | 涨 | 涨 |
```
判断：SFT 数据里有没有目标模型自己采不到的东西。有 = 撒种；没有 = 收紧
例    7B R1-Distill 的 Countdown 轨迹（混合形态 30% 命中）SFT 到 1.5B → 撒种，硬题 pass@k 涨
      600 模型自己的轨迹（硬题 0 条对）SFT 回去 → 收紧，硬题 pass@k 不动
```

## R1 第 ③ 步是哪种 —— 看从谁的角度
```
从新 Base 看   ② 的输出是它采不到的 → 撒种 → 新 Base 的 pass@k 涨
从 ② 的模型看  数据全是自己的 → 收紧
★ 「SFT 到新 Base」不只是甩杂质 —— 换个更弱的接收方，同一份数据从收紧变撒种
   蒸馏到 32B 是同一道理走到底：接收方越弱，撒种效应越大
飞轮里 SFT 要推 pass@k，接收方得比数据来源弱。R1 每圈都这么做
```

## 程序泛化（自蒸馏抬 pass@k 的唯一路径）
```
SFT 完默认行为变成「搜、验、换」→ 一道 Base 上 p=0.001 的题，用对程序后 p=0.05 → 进 pass@64
涨的是「原来不用对的程序、现在用了」的题，不是「缺零件」的题
我们的数：严格标注形态 0.4% → 62% 是程序泛化；混合形态覆盖 1.6% 没动是零件层面
```


---

# ★★ 程序泛化是什么 + 没有更好模型时怎么推 pass@k（2026-09-03）

## 程序泛化
```
SFT 自己的轨迹，教的是【程序】不是【答案】
   拿 275 步那 21 条带严格标注的轨迹 SFT 回去 → 模型学到「先算一个、跟目标比、标方向、换一个」这套动作
   → 在【所有题】上默认走这套
   原来不搜直接猜的题 p ≈ 0.001（pass@64 算不会）→ 搜了 p 变 0.05 → 进 pass@64
   ★ 这道题的解从没在 SFT 数据里出现过。涨是因为程序搬过去了
什么时候不泛化：失败原因是「缺零件」不是「没用对程序」
   硬尾巴 30 道，模型已经在搜，缺的是具体混合形态 + 算对。SFT 易题轨迹，程序早有，零件还是没有
```

## 没有更好的模型，怎么推 pass@k
pass@k 涨 = 更多题从 p≈0 变 p>0。新质量只能来自：已有零件的新组合，或放进新零件。

**组合类 —— 不加零件**
```
换地 + 课程   出「必须用某个组合」的题，RL 只能选它。从组合概率稍高的题起，1% → 10%，再上难的
探索          k=64 / 温度 1.1 / 熵奖励。不造新质量，把 1% 的组合暴露给 RL
程序泛化      自蒸馏，对「缺程序」的题有效
```
**加零件类 —— 不换模型**
```
继续预训练    同一模型多吃十亿 token 算术题解。零件是这么进去的，RL 放不进去
★ 工具        给计算器。算术零件外包，精度 0.73 → 1.0。硬题成功率 1.6%×73% → 1.6%×100%，配 k=64 进半径
              ★★ 我们没用过的最大杠杆，工业界标配（代码执行 / 计算器 / 搜索）。不换模型，换 harness
```
**天花板**：前三条重排已有零件，零件缺了转多少圈不动（精度 0.73 就是例子）。后两条加零件，但靠预训练和工具，不靠 RL。
★ 先看卡的是程序还是零件。程序 → 前三条；零件 → 后两条；都不行 → 这个模型的边界。


## SFT 长推理材料推 pass@k —— 看材料从哪来
```
来自模型自己（自蒸馏）  程序泛化：把「偶尔用」变「总是用」，不加零件
来自外面（人写/强模型）  撒种：能放进模型产不出的形态，加零件。R1 冷启动就是这种
共同限制   教的是【形态】不是【执行】。精度 0.73 = 形态学会了，执行 27% 错。执行靠零件或工具
```

---

# ★★★ 修正一条核心结论：「RL 不动 pass@k」说过头了（2026-09-03，用户追问推出来的）

## 用户的推理
自蒸馏能泛化程序 → 泛化能提高 pass@k → RL 也放大程序（0.4% → 62%）→ RL 也应该能推高 pass@k？

## 数据说他是对的
```
GSM8K Instruct RL     pass@64 0.98 → 0.98    不动   ← 之前的结论来自这里
★ Countdown 混训       pass@8  Base 0.18 → 275 0.56 → 425 0.65 → 600 0.68    ★ 涨了 3.8 倍
```
Countdown 上 pass@8 从 0.18 到 0.68，50 道题从「8 条全错」变成「至少一条对」。这就是 pass@k 在涨。

## 为什么两个任务不一样
```
GSM8K Instruct   起点已经默认「一步步算」（SFT 教的），没有程序缺口 → RL 只能收紧 → pass@k 不动
                 剩下那 2% 是缺零件的题，谁都推不动
Countdown Base   起点不搜索、直接猜，有程序缺口 → RL 把「搜、验、换」放大成默认 → 原来不搜的题现在搜了
                 → 一批题从 p≈0 变 p>0 → pass@8 涨
```

## ★★ 精确版
```
pass@k 涨的条件：某个程序变成默认，解锁了原来「没用对程序」而失败的题
   RL 和自蒸馏 SFT 都能做这件事 —— 机制相同：让程序变默认，默认就搬到没见过的题上
   RL 慢（程序要被采到又被奖励，一步步来），但按结果选（只放大管用的）
   自蒸馏快（一次 SFT 集中到筛过的轨迹），但筛子得选对
pass@k 不涨的情况：瓶颈是缺零件
   RL 和 SFT 都推不动。零件只能预训练 / 工具放进去
```
**「RL 不动 pass@k」要改成「RL 不动 pass@k 的【零件天花板】」。天花板之下，RL 能通过让程序变默认来推高 pass@k。**

## 采样半径那条不变
RL 只能放大采到的东西 —— 仍然成立。放大之后的程序会泛化到没采过的题 —— 这是新加的一句。两句合起来才完整。

## 我们的数据里没测的一个空
Countdown 上 Base 的 pass@64 没测过。如果它是 0.6（很多题 p 在 0.01~0.1，64 次够得着 8 次不够），那 600 模型的 pass@8 0.68 主要是收紧；如果它是 0.3，那是真的扩了。
★ 补测：baseline_countdown.py -s 64 --offset 95000，Base 和 600 各一次，看 pass@64 动没动


---

# ★★ CD-4 训练：从 600 起 300 步，len-soft 400（2026-09-03）

104.9 分钟。`--init out_mix_mp/ckpt_step600.pt --data countdown4.json`，GSM8K 照旧一半。

## eval（贪心，CD-4 [90000:90100]，GSM test[200:300]）

| step | CD 正 | CD 长 | 标注@精度 | GSM 正 | GSM 长 |
|---|---|---|---|---|---|
| 起点 | 0.32 | 293 | 1.5@0.77 | 0.78 | 161 |
| 75 | 0.41 | 223 | 1.0@0.86 | 0.80 | 165 |
| 175 | 0.43 | 243 | 1.0@0.66 | 0.81 | 150 |
| 225 | 0.46 | 263 | 1.2@0.73 | 0.82 | 152 |
| **300** | **0.45** | **209** | 1.0@0.75 | **0.84** | 157 |

## 训练探针（温度 1，前 20 步 → 后 20 步，CD）

| | 前 20 | 后 20 |
|---|---|---|
| 正确率（均值） | ~0.30 | ~0.38 |
| 长度 | 242 | **207** |
| 标注数/条 | 1.46 | 1.06 |
| 标注精度 | 0.602 | **0.700** |
| 0 次桶 | 45.6%…0.456 | **0.100**（桶变小、变成「直接放弃」） |
| 3~5 次桶 | 0.349 | 0.544 |
| 6+ 次桶 | 0.150 | 0.201 |
| CD \|A\| | 0.224 | 0.155 |
| KL | ~0.03 | ~0.04 |

## 判读

### ★ 涨了，但温和，而且 175 步后平了
贪心 0.32 → 0.45，温度 1 ~0.30 → ~0.38。175 步起 eval 在 0.43~0.46 抖。CD |A| 0.22 → 0.16，k=16 在饱和。
KL 几乎没动（0.033 → 0.04），权重改得很少。

### ★★★ 长度没涨，反而缩了：293 → 209
H1 在 CD-4 上也没出现。len-soft 400 只罚 P95 那 5~10%，主要原因是：
**结果奖励下，对的短答案和对的长答案同分，RL 收敛到「够用的最短」。** CD-3 需要 ~170，CD-4 需要 ~210，就停在那。
★ R1 的长度涨是因为 AIME 真需要一万 token、他们没有长度惩罚、上下文 32K。
★★ 结论：长思维链是【任务要求】出来的，不是 RL 长出来的。Countdown 任何规模都不会长。要看 H1 得换一个真需要一千 token 的任务。

### ★ 精度涨、标注减、效率升
精度 0.60 → 0.70，标注 1.46 → 1.06/条，正确率涨。少写、写准、更常对 —— 搜索在变高效，不是变少。

### 0 次桶的意思又变了
前 20 步 0 次桶 0.456 = 简单题直接答对；后 20 步 0.100 = 模型几乎总是搜，剩下不搜的是「直接放弃交白卷」。
剂量响应 +0.459 读的是「放弃 vs 搜」不是「搜 vs 不搜」。这个指标在模型学会「总是搜」之后就该退役。

### 对三个目标
```
反思搜索   形态在、精度 0.70~0.83、高效。✓ 已到 Countdown 能给的极限
长思维链   ✗ 任务不要求，不会长。这是任务的边界，不是训练的失败
策略（硬尾巴）  待 800 条 + scan 看混合形态和 nums_ok
```

## 待做（标准流程 1~9）
```
cp out_mix4/ckpt_best.pt out_mix4/ckpt_step300.pt
baseline_countdown.py --ckpt out_mix4/ckpt_step300.pt --data data/countdown4.json -n 100 -s 8 --offset 95000
analyze_search.py cd_..._step600_..._countdown4_raw.json cd_out_mix4_ckpt_step300_..._countdown4_raw.json
scan_ops.py cd_out_mix4_ckpt_step300_..._countdown4_raw.json
crosstask_eval.py --ckpt out_mix4/ckpt_step300.pt --out crosstask_mix4.json
+ pass@64 补测（Base 和 600，各 45 分钟）
```


## CD-4 池外 800 条：mix4-300 vs 600 零训练（2026-09-04）

| | 600 零训练 | **mix4-300** | Δ |
|---|---|---|---|
| pass@1 | 0.208 | **0.393** | +0.185（1.9×） |
| pass@8 | 0.610 | **0.670** | +0.06 |
| **数字恰好各用一次** | **0.675** | **0.950** | **+0.275** ← 最大的单项 |
| 格式 / 解析 / 数字比例 | 0.96 / 0.96 / 0.87 | 0.99 / 0.98 / 0.97 | 阶梯前三级饱和 |
| 全对 / 全错 / 有对有错 | 0 / 39 / 61 | 10 / 33 / 57 | 硬尾巴 39 → 33，只动 6 道 |
| 组内有差异 | 100% | 71% | 饱和中 |
| 搜索词每条 | 8.06 | 6.73 | 少试、更准 |
| 长度中位 / P95 | 242 / 477 | 222 / 375 | 缩 |
| 续写 / 撞顶 | 1.2% / 4.2% | 0.8% / 0.9% | 干净 |

### 判读
```
① 预测中的第一课兑现：nums_ok 0.68 → 0.95。「4 个数用全」是 CD-4 训练最大的单项收益
② pass@1 翻倍，pass@8 只 +0.06 → 主要是收紧，扩了一点（10 道全对，6 道从全错出来）
③ 硬尾巴 33 道基本没动
④ ★ 样例那道 [6,12,18,42]→30 其实有纯加减解：42 + 12 − 6 − 18 = 30
   四个数的 ± 只有 8 种符号形态，模型三条样例各试 5~7 行，却有重复（6+12+42−18 和 6+12−18+42 是同一形态）
   和算错（6+18+42−12 写成 48，实为 54）。差一步没搜到。
   ★ 搜索不系统：不记账、会重试同一形态、算术偶错把方向标反。这不是 ×÷ 覆盖问题，是搜索质量问题
⑤ 长度缩，搜索次数减，正确率涨 → 效率
```

### ★ 样例里两种精度失败，都是「想要它对」
```
样例一   6 + 12 + 18 + 42 − 0 = 76      多写了一个 0（幻觉第五个数），而且 78 算成 76
样例二   6 + 42 − 12 − 18 = 30 (perfect match)   ← 实际是 18。★ 幻觉验证：想要 30 就写了 30
样例三   6 + 42 − 12 − 18 = 18 (too low)          ← 同一个式子，这次算对了
```
判分器没被骗（样例二得 0.25），所以没有奖励流向假 perfect match。但模型还是会写 —— 因为在别的题上「X = 目标 (perfect match)」是对的且被奖励，模式学会了，算术偶尔跟不上。精度 0.70 的 30% 错里有一部分是这种。
★ 这是精度天花板的样子：形态对了，执行会「向目标靠」。工具（计算器）是唯一能根治的。

### CD-4 训练完成的判断
```
目标：收紧够得着的 + 修 nums_ok + 看长度      前两个达成，第三个任务不给
CD-4 现在 0.39 / 0.67，CD-3 是 0.45 / 0.68 —— 差不多了
硬尾巴要嵌套乘除，k=64 + 硬题池也许再捞 5 道，代价 8 小时。★ 收益递减
```


## ★★ mix4-300 全面评价（2026-09-04）

**一句话：Countdown 这条线的终点产品，不是半成品。**
能做的都做到了；做不到的两样，一样是任务不给的（长度），一样是 1.5B 不给的（算术精度）。

### 身世
Base → 混训 600 步（GSM + CD-3）→ 再 300 步（GSM + CD-4）。三代 RL，累计 900 步，KL 只有 0.04。权重改得极少，行为改得很多。

### 会什么（已量化）
| 任务 | 切片 | 上一代 600 | mix4-300 |
|---|---|---|---|
| CD-4 pass@1 / pass@8 | 池外 95000 | 0.21 / 0.61 | **0.39 / 0.67** |
| CD-4 四个数用全 | 池外 95000 | 0.68 | **0.95** |
| CD-4 贪心 | 训练 eval | 0.32 | 0.45 |
| GSM8K 贪心，R1 模板 | test[200:300] | 0.78 | 0.84（100 题噪声内，至少没丢） |
| CD-3 pass@1 / pass@8 | 池外 95000 | 0.45 / 0.68 | 0.48 / 0.66（补测 09-04，保住） |

### 行为长什么样
- 总是搜：不搜的只剩 10%，而且是交白卷不是直接答对
- 搜得省：每条 6.7 个搜索词、1 条严格标注，比上一代少，对得更多
- 精度 0.70~0.75：四条标注三条算对且方向对
- 长度 210~220，任务定的：结果奖励下对的短答案和对的长答案同分 → 收敛到够用的最短
- 干净：续写 0.8%，撞顶 0.9%

### 两个缺陷，都是精度不是形态
- 幻觉验证：式子算出 18，写「= 30 (perfect match)」。判分器没被骗，但句式在别的题上真对过被奖过，它照写
- 搜索不系统：4 数加减只有 8 种符号形态，试 5~7 行却有重复、有算错，然后放弃。差一步

### 三个盲区（流程后三条填）
- 中性模板下技能在不在 → crosstask
- CD-3 有没有保持 → 大概率保持，没数
- 33 道全错里多少「差一步」、多少真需要嵌套乘除 → scan_ops。决定硬尾巴值不值得再打

### 对三个目标
```
反思搜索   ✓  形态、精度、效率都到了 Countdown 能给的极限
长思维链   ✗  任务不要求。Countdown 任何规模都不会长，换任务才有
策略      半  简单题稳了，难题的搜索质量没上去，pass@8 只动 0.06
```

### 五个旋钮
已拧：换地（CD-3→CD-4）、换奖励（len-soft、数字比例分）、换开关（R1 模板）
没拧：多抽（k=64，硬尾巴大概再捞 5 道，8 小时）、撒种（拒绝采样 SFT = 待办 E，也是下一条线的起点）

### ★ 「任务不给的」和「1.5B 不给的」具体指
```
任务不给的 = 长度
  奖励只看结果，对的短答案和对的长答案同分，多写一个 token 都没有回报。
  RL 只会往回报的方向推，所以停在够用的最短（CD-3 170，CD-4 210）。
  不是训练没到位，是这个环境里【根本没有】变长的梯度。换任务（真要一千 token 才解得出的）才有。

1.5B 不给的 = 算术精度
  「6 + 42 − 12 − 18」写成 30 又写成 18，「6 + 18 + 42 − 12」写成 48（实为 54）。
  同一个式子不同轨迹算出不同数，说明是模型算力本身的噪声，不是没学会形态。
  RL 能放大形态、能压低错的概率，但把 1.5B 的心算从 0.75 推到 0.99 不是奖励能做的事
  —— 材料（预训练）决定的零件上限。要根治只有两条：换大模型，或给计算器（工具）。
```


## analyze_search：600 vs mix4-300，CD-4 池外 800 条（2026-09-04）

### ② 剂量响应 —— 涨的 0.185 从哪来
| 桶 | 600：条 / 正确率 / 长 | mix4-300：条 / 正确率 / 长 | 多对的条数 |
|---|---|---|---|
| 0 次 | 11 / 0.18 / 129 | 8 / 0.13 / 81 | −1 |
| 1~2 次 | 132 / 0.41 / 181 | 145 / **0.69** / 144 | +46（31%） |
| 3~5 次 | 127 / 0.28 / 201 | 181 / **0.66** / 166 | **+83（56%）** |
| 6+ 次 | 530 / 0.14 / 298 | 466 / 0.20 / 274 | +20（14%） |
| 合计 | 166 对 / 800 | 314 对 / 800 | +148 |

```
★ 一半多的涨幅来自 3~5 次桶：它变大（127→181）又变准（0.28→0.66）
   = 以前要试 6 次以上还失败的题，现在 3~5 次就解了。搜索变高效，不是变多
★ 6+ 桶还占 58% 的轨迹，正确率只 0.20 —— 33 道全错题住在这里，基本没动
★ 每桶长度都缩了 30~40 token：同样的试错次数，废话少了
```

### ③b 标注精度：0.636 → 0.691
标注数 1317 → 964（每条 1.65 → 1.2）。少写、写准。跟训练日志 0.60 → 0.70 对上。仍有 30% 算错或方向标反。

### ★ ③ 「说对了」的可信度 —— 幻觉验证坐实
```
              P(对 | 说 perfect)   P(对 | 没说)   差
600             0.296              0.170        +0.126
mix4-300        0.443              0.373        +0.070
```
「perfect match」写了 221 条，56% 是假的。相对信息量还从 +0.13 掉到 +0.07。
★ 样例二那种「算出 18 写 = 30 (perfect match)」不是个案。判分器不认，所以没奖励流过去，
但句式已经跟「找到了」绑定，算术跟不上就照写。★ 这是精度天花板最直观的一张脸。

### ⑤ 严格标注 ★差：−0.086 → +0.054（标准误 0.053，≈ 0）
带标注的 720 / 800 = 90%。行为普及后对照组只剩 80 条，多半是交白卷。
★ 和 0 次桶一样，这个探针在「模型总是做」之后退役。① 宽泛 ★差同理（不带的只有 8 条）。

### 探针状态
```
还有用：② 分桶正确率、③b 标注精度、③ 说对了的可信度
已退役：① 宽泛 ★差、⑤ 严格 ★差、0 次桶  —— 行为普及，对照组消失
```

### 对硬尾巴的含义
6+ 桶 466 条里 372 条错。样例说明这些长搜索里有重复形态、有算错。
scan_ops 要回答：33 道全错里几道有纯加减解 → 「搜索质量」能捞回多少。

## scan_ops：mix4-300 硬尾巴解剖，CD-4 池外（2026-09-04）

### ① 33 道全错的解需要什么
```
只用 +− 就有解     7 道（21%）  ← 搜索质量问题：差一步、重复、算错
必须用 × 或 ÷     26 道（79%）  ← 覆盖问题：形态不在采样里
```
★ 上一条「搜索不系统」的判断只解释五分之一。硬尾巴主体还是 ×÷。

### ★ ②③ 读数作废：0.97 是复述污染
MUL/DIV 正则匹配任何 `*` `/`，而 CD-4 模型几乎每条都复述「with the operations +, -, *, /」。
0.966 是「复述了题面」的比例，不是「试过乘除」的比例。⑤ 的两条完整轨迹一个 ×÷ 都没试，却被算成「含 ×」。
已改：要求运算符两边有数字或括号。同步后重跑 scan_ops（不用 GPU，秒出）。

### ④ 才是干净的
| 类别 | 轨迹 | 纯直接乘 | 混合形态 | 混合/条 |
|---|---|---|---|---|
| 全错 | 264 | 0.072 | 0.432 | 0.74 |
| 有对有错 | 456 | 0.013 | 0.311 | 0.39 |
| 全对 | 80 | 0.000 | 0.263 | 0.31 |

全错题的轨迹里约一半试过带乘除的算式，另一半全程 +−。试了的通常只有一行，不是对的那行。
800 条里 34.6% 出现过混合形态（600 时是 42%），比例没涨。RL 没有把 ×÷ 的尝试推多，因为它们很少得分。

### ⑤ [5, 7, 14, 33] → 43 的两条轨迹
解：33 + 14 × 5 / 7 = 43，三种运算符嵌套。8 种 ± 形态穷举后无解，必须上 ×÷。
两条轨迹各试 6~8 行，全是 +−，没有一次乘除。而且：
```
重复     14 + 33 − 5 − 7 和 33 + 14 − 5 − 7 各算一次，没发现是同一个
算错     5+7+14+33 一条写 59 一条写 61；(33+14) 写成 49；14+33+5−7 写成 49（实为 45）
        8 行里 3 行算错
放弃     「may not be possible」，然后交一个等于 49 的式子当答案
```
★ 同一条轨迹里同时看到三种失败：不记账、心算噪声、不换运算符。这就是 6+ 桶 0.20 的长相。

### 对硬尾巴的决定
```
7 道 +− 题    系统搜索能捞（不重复、算对）。但「系统」不是 RL 放大得出来的，它自己没有这个程序
26 道 ×÷ 题   需要 3 运算符嵌套形态，34.6% 的轨迹碰过混合形态但几乎没碰对过。k=64 也难
★ 两类都指向同一件事：模型缺一个【程序】—— 穷举、记账、算准、换运算符
   这个程序自己采样不出来（采样半径外），RL 放大不了 → 只能撒种
```

### ★ 待办 E 的修正
原计划：拒绝采样（模型自己的对轨迹）→ SFT → RL。
问题：自己的轨迹也不系统，SFT 只会收紧「碰运气式搜索」。
修正：★ 写一个 Python 穷举器，按模型现在的格式输出【系统搜索轨迹】（不重复、算对、+− 穷尽后换 ×÷），
SFT 到 mix4-300（或直接到 Base），再 RL。这才是真正的撒种：数据里有模型自己产生不了的东西。
这也是 R1 阶段 ①（冷启动）的做法，只是老师是程序不是人。

## crosstask：mix4-300 在 GSM8K 两模板（test[0:200]，贪心，2026-09-04）

| 模型 | R1 模板 | 中性模板 | McNemar vs Base |
|---|---|---|---|
| Base | 0.52 · 183 | 0.48 · 257 | — |
| mix-300（2026-09-02） | 0.71 · 164 | 0.47 · 241 | R1 z=−5.08 · 中性 z=+0.31 |
| **mix4-300** | **0.74 · 158** | **0.46 · 237** | R1 训好 57 / 训坏 13，z=−5.26 · 中性 26 / 30，z=+0.53 |

### 判读
```
① GSM 没丢也没再涨：0.71 → 0.74，200 题 ±0.03 噪声内。再训 600 步（CD-3 续 + CD-4），GSM 在训练模板下原地
② 条件性绑定原样：中性模板 0.46 ≈ Base 0.48。训练模板下 +0.22 一分都没漏到中性模板，也一分都没伤到
③ 900 步没有累积遗忘：中性列 0.48 → 0.47 → 0.46，每步都在噪声里。混训的保护效果稳定
④ 长度：两个模板各缩 20~25，跟 CD 那边「缩」的趋势一致，是唯一漏过模板的东西（但正确率没动）
⑤ 脚本那句「RL 没有跨任务副作用」仍是旧判读逻辑（见「六」）。正确读法：训练模板下有提升，换模板不升不降
```

## ★★ 标准流程第 9 步：mix4-300 的决定（2026-09-04）

四条评测全部跑完，一张表：
```
① CD-4 池外 800   pass@1 0.39 / pass@8 0.67 / nums_ok 0.95        目标达成
② analyze_search  涨幅 56% 来自 3~5 次桶；精度 0.69；perfect 句 56% 假   高效但精度封顶
③ scan_ops        33 道全错 = 7 道 +− + 26 道 ×÷；缺穷举/记账/算准的程序   采样半径外，RL 放大不了
④ crosstask       GSM R1 0.74 持平、中性 0.46 = Base；无遗忘             绑定不变，无副作用
```

### 决定：Countdown 这条线收工。下一步走待办 E（修正版）
```
不走 k=64 + 硬题池：26 道 ×÷ 题要的三运算符嵌套形态，34.6% 轨迹碰过混合形态但几乎没碰对过；
                     7 道 +− 题差的是「不重复、算对」，多抽也不会让它记账。8 小时换 ≤5 道，不值
走 E（程序老师撒种）：Python 穷举器按现在的格式写系统搜索轨迹 → SFT → RL
   ★ 它一次回答三个问题：
   a. 撒种能不能推高 pass@k（理论上的判断第一次上实验）
   b. 系统搜索是不是「程序」，能不能被 SFT 教会、能不能泛化到没见过的题
   c. ★★ 长思维链的来源：CD-4 系统穷举 8 种 ± 再换 ×÷ 要 30+ 行，600~900 token。
      如果 RL 之后模型在难题上保留长搜索、在简单题上短 —— 那就是 R1 式的「按需变长」，
      来自「难题非长不可」，正好验证「长度是任务要求出来的」
F（Agent RL 猜数字）排在 E 后面
```

### 预测（跑之前写下，跑完对照）
```
SFT 后（RL 前）  CD-4 pass@1 ≥ 0.45，pass@8 ≥ 0.75，6+ 桶正确率 > 0.35，标注精度 > 0.85（老师算的都对）
                长度中位 > 400（系统穷举本身就长）
RL 后           pass@1 ≥ 0.55；长度分叉：简单题 < 250，难题 > 500
                33 道全错里至少 10 道出来（7 道 +− 全部 + 3 道以上 ×÷）
若 SFT 后长度回落到 250 以下且精度回到 0.7 → 撒的种被 RL 冲掉了，说明 SFT 剂量或 lr 不够
```

## ★★ pass@64：Base vs 600，CD-3 池外 [95000:95100]，100 题 × 64（2026-09-04）

补的那个空。问题：600 的 pass@8 0.69 是收紧（Base 多抽几次也能到）还是扩（Base 多抽也到不了）。

| k | Base | 600 | 差 | Base 每翻一倍涨 | 600 每翻一倍涨 |
|---|---|---|---|---|---|
| 1 | 0.016 | 0.456 | 0.44 | | |
| 2 | 0.032 | 0.558 | 0.53 | +0.016 | +0.10 |
| 4 | 0.063 | 0.633 | 0.57 | +0.031 | +0.08 |
| 8 | 0.121 | 0.694 | **0.57** | +0.058 | +0.06 |
| 16 | 0.224 | 0.748 | 0.52 | +0.103 | +0.05 |
| 32 | 0.387 | 0.795 | 0.41 | +0.162 | +0.05 |
| 64 | **0.590** | **0.830** | **0.24** | **+0.203** | +0.035 |

全错（64 次一次没对）：Base 41 道，600 17 道。
Base 的样例：`</thunder`、写 Python、用题目里没有的数「(1+2)/3」—— 格式 0.76、数字用全 0.14、撞顶 10%。

### 判读
```
① 按跑前定的判据（Base pass@64 ≈ 0.6 → 收紧为主）：0.59，收紧为主。
   ★ 但那个判据比错了 k：拿 Base 的 64 次比 600 的 8 次。同 k 比才对。
② 同 k=64：差 0.24，600 每 100 题多解 24 道。这是问题层面的实打实扩张
③ 差距在 k=8 最大（0.57），到 64 缩到 0.24，而且 Base 还在陡涨（每翻倍 +0.20），600 已经平了（+0.035）
   外推：Base k=128 ≈ 0.74，256 ≈ 0.82，512 ≈ 0.86；600 k=512 ≈ 0.87 → 两条线在 k ≈ 500 附近汇合
④ ★★ 所以 600 做的事是：把 Base 要抽 500 次才碰一次的轨迹，压成 8 次就有。
   文献意义上这叫收紧（大 k 极限不变）；任何你实际用的 k 上这都是扩张
```

### ★ 「RL 动不动 pass@k」的最终版
```
RL 抬高所有【有限】k 的 pass@k —— k=64 时 +0.24，k=8 时 +0.57
RL 不抬 pass@∞ —— 零件天花板 ≈ Base 的渐近线 ≈ 0.87，600 的 0.83 已经贴着它
之前两句话各对一半：「RL 不动 pass@k」说的是 k→∞；「Countdown pass@8 0.18→0.68」说的是 k=8
★ 为什么能压 60 倍：Base 的对轨迹要同时满足 格式×解析×数字用全×搜到 = 0.76×0.63×0.14×… ≈ 0.016，
   RL 把每一项都推到 0.95+，乘积变 0.46。每一项都在 Base 采样半径内（收紧），乘起来就是新解出的题（扩）
```

### 对待办 E 的含义
600 在 k=64 还全错的 17 道，p < 1/64。多抽到 500 次也许再捞几道，但那已经是 Base 渐近线的位置。
★ 要过 0.87 这条线，只有撒种。E 的目标不是 pass@1，是 pass@64 过 0.87 —— 这是「撒种推高天花板」唯一干净的证据。
预测补一条：E 之后 CD-3 pass@64 ≥ 0.92，否则撒的种没有新零件。

### ★ 复述纠正：「RL 抬 pass@k」和「RL 不抬天花板」不矛盾（2026-09-04）
用户追问：这个实验里 RL 明明推高了 pass@k。对，测的每个 k 都涨，不打折。「各对一半」说绕了，正确说法是两句话说的是两样东西。

| 题 | Base p | Base @8 | Base @64 | RL p | RL @8 | RL @64 |
|---|---|---|---|---|---|---|
| A | 0.5 | 1.00 | 1.00 | 0.95 | 1.00 | 1.00 |
| B | 0.05 | 0.34 | 0.96 | 0.6 | 1.00 | 1.00 |
| C | 0.005 | 0.04 | 0.27 | 0.1 | 0.57 | 1.00 |
| D | 0 | 0 | 0 | 0 | 0 | 0 |
| 平均 | | 0.35 | 0.56 | | 0.64 | 0.75 |

```
RL 抬 pass@k     = 每道 p > 0 的题，p 变大 → 任何有限 k 都涨（A B C）
RL 不抬天花板    = p = 0 的题还是 0（D）。pass@∞ 只数「有几道题 p > 0」
对到数据：Base 全错 41 → 600 全错 17，中间 24 道是 C 类被抬起来的；剩 17 道 p < 1/64，是 D 还是更小的 C 分不出
天花板为什么要在意：它是「该停 RL 换种」的唯一读数。600 的 0.83 贴着 0.87，再灌 RL 只剩 0.04 可涨
```

# ★★★ 决策点：RL 阶段总结 + 下一步选项（2026-09-04）

## 做了什么（RL 阶段）
| 轮 | 模型 + 任务 | 步数 | 结果 |
|---|---|---|---|
| GSM8K GRPO | Instruct + GSM8K | 两轮 + 四轮验证 | 0.687 → 0.767 holdout |
| CD-3 单任务 | Base + Countdown-3 | 300 | 贪心 0.405；GSM R1 0.52→0.16、中性 0.48→0.35 → 遗忘 + 条件性绑定 |
| 混训 | Base + GSM + CD-3 | 0→300→600 | CD-3 池外 pass@1 0.45 / @8 0.68 / @64 0.83；GSM R1 0.71~0.74、中性 0.47 = Base |
| CD-4 续训 | 600 + CD-4 混 GSM | 300 | CD-4 池外 0.21→0.39 / @8 0.61→0.67；数字用全 0.68→0.95；GSM 持平 |
| pass@64 | Base vs 600，CD-3 | — | 0.59 vs 0.83；Base 每翻倍 +0.20 还在涨，600 已平，约 k=500 汇合于 0.87 |

工具：grpo_mix_mp / baseline_countdown / analyze_search / scan_ops / crosstask_eval / countdown 六个脚本；PLAN.md 5400 行。

## 效果对三个目标
```
反思搜索（顿悟）  ✓  形态真（四视角）、精度 0.70、高效（3~5 次桶 0.28→0.66）。到 Countdown 极限
长思维链          ✗  任务定长：CD-3 停 170，CD-4 停 210。结果奖励下长短同分，没有变长的梯度
策略（硬尾巴）    半  简单题稳；33 道全错 = 7 道 +−（不记账、算错）+ 26 道 ×÷（形态不在采样里）。缺一个程序
副产品           混训防遗忘（三代中性列 0.48/0.47/0.46）；绑定对称；pass@k 有限 k 涨、pass@∞ 不涨
```
方法论账：切片来源错两次；小样本过度解读四次；总分早停一次；测量 bug 四个（续写、词表、正则×2）。跑前写预测这习惯救了好几次。

## 选项
| | 做什么 | 回答什么 | 代价 |
|---|---|---|---|
| A | E 修正版：Python 穷举器写系统搜索轨迹 → SFT → RL | 撒种抬不抬天花板（CD-3 pass@64 > 0.87）；程序可不可教；长度会不会按需分叉 | 穷举器 1 天，SFT 30 分钟，RL 2 小时 |
| A′ | A 的对照臂：自采样拒绝采样 SFT，同预算 | 分离「新零件」效应 vs 单纯 SFT 收紧 | 加 1 小时 |
| B | 计算器工具：`<calc>` 标签，harness 回值再续生成 | 精度 0.70 → 0.99？硬尾巴几道因算准而解？工具 RL 的机制 | 中断-注入 rollout 1 天，RL 2 小时 |
| C | F：Agent RL 猜数字，多轮，观测进 prompt，按轮 mask | 多轮信用分配；长度随轮数涨；harness 长什么样 | 2~3 天 |
| D | 换真需要长链的任务（CD-6 / MATH 子集 / 多跳合成） | 直接测 H1「越训越长」 | 环境 + 判分 1 天 |
| E | 7B Base 跑一次 baseline_countdown，不训 | 精度和系统搜索是不是尺寸问题 | 30 分钟 GPU |
| F | 补五模型表七格 | 完整性 | 1.5 小时 GPU，低价值 |

## 推荐顺序：A（带 A′）→ B → C，E 随时插
```
A 先：最便宜；整条线的理论（撒种 vs 收紧）第一次上实验；若「按需变长」兑现，目标二不用换任务就有了（D 可省）
B 次：修 RL 修不了的那个天花板（1.5B 心算）；中断-注入的 rollout 就是 C 要的基础设施
C 后：目标三。有了 B 的 harness，C 只多「多轮 + mask」
★ 另一条路：更在意 agent 而不是理论 → 直接 B → C，跳过 A
```

## ★ 第五个测量 bug：标注计数只数到每段的第一条（2026-09-04，写部署脚本时撞出来的）

ANNOT 正则的算式组允许以任意空白或 `)` 开头。连续两行「… = 59 (too high)\n5 + 7 + 33 − 14 = 31 (too low)」，
第二条的算式组从上一行的 `)\n` 开始 → `")\n5 + 7 + 33 - 14"` → safe_eval 语法错 → 被 `continue` 跳过。
实测 5 行连续标注只数到 1 条。

### 波及
```
训练日志「标 1.0~1.5 @ 精度」   标注数是「标注段数」不是条数；精度只算了每段第一条（通常是全加那行，最好算的）
analyze_search ③b「1317 条 / 964 条」   同样只是首条。真实条数大概 4~6 倍
剂量响应 ②、搜索词计数              不受影响（SEARCH 正则直接数 too high/low 这些词）
```
★ 精度 0.60 → 0.70 的方向大概率还对（首条精度也在涨），但绝对值要重测：后面几行减法多，可能更低。

### 修法
算式组改为必须以数字或 `(` 开头：`((?:\(|\d)[\d\s\+\-\*/\(\)]*?)`。三个文件同步改：grpo_mix_mp.py / analyze_search.py / chat_countdown.py。
重跑（不用 GPU）：
```
python3 analyze_search.py cd_out_mix_mp_ckpt_step600_s8_off95000_countdown4_raw.json cd_out_mix4_ckpt_step300_countdown4_s8_off95000_raw.json
```
看 ③b 的「共 N 条标注」和精度怎么变。

## 部署推理脚本 chat_countdown.py（2026-09-04）
内置 8 题，从「Base 也能碰到」到「mix4 也解不了」；穷举器给参考解和题型；每条输出后一行判读 + 三面红旗
（幻觉验证 / 幻觉续写 / 撞顶）；`--both` 把 Base 和 ckpt 并排；末尾汇总表。默认贪心，`--temp 1 -s 4` 看多样性。

## ★★ 「Base 也有顿悟？」—— 一道题上看到园丁比喻（2026-09-04）

chat_countdown 上 [3, 7, 25] → 46，同一段 R1 模板、贪心：
```
Base   3+7+25=35 → "That's not quite there, so I'll try another" → 3*7+25=46 → "That's it!"   三个 <answer>，第一个是错的
mix4   3+7+25=35 (too low) → 3+25−7=21 (too low) → 7+25−3=29 (too low) → 3*7+25=46      一个 <answer>，在最后
```
用户：那感觉两个都有顿悟？

### 答：对，零件本来就在 Base 里。RL 没有造出「反思」，它把反思从装饰变成程序。四个差别，都有数
| | Base（R1 模板，温度 1，CD-3 池外） | mix4 / 600 |
|---|---|---|
| 形态 | 有：「not quite there」「try another」 | 有：「too low」「perfect match」 |
| 频率 | 44% 的轨迹有搜索词，1.4 个/条 | 97~99%，5.7~6.7 个/条 |
| 方向性 | 「不对」= 只有判决 | 「too low」= 判决 + 往哪边改。下一步能用 |
| 有效性 ★差 | −0.02：说了反思的轨迹【不】更常对 | +0.20：说了的更常对 |
| 落地 | 每次尝试都塞进 <answer>，判分器取第一个 → 0 分 | 试错在 <think> 里，找到才写唯一的 <answer> |
| pass@1 | 0.016 | 0.39~0.46 |

```
★ 这道题 Base 贪心也解出来了 —— 贪心是分布的众数，简单题的众数可以对，分布本身还是散的（温度 1 只有 0.016）
★ mix4 把分布压到了这条路径上：8 条抽样会长得几乎一样，Base 的 8 条会八个样
★ 2025 年有人专门测过 Qwen2.5 Base：不训 RL 就会写 wait / recheck，而且「表面反思」跟答对没关系。
   跟我们 Base 的 ★差 −0.02 是同一个发现
```

### 「顿悟」这个词该怎么用
```
R1 论文里的 aha moment：一条轨迹里模型停下来重新评估。论文把它写成「涌现」
我们看到的：形态在 Base 里就有（预训练材料）；模板把它引出来（开关）；RL 让它 ① 几乎每条都出现 ② 带方向 ③ 跟答对挂钩 ④ 格式落地
★ 「复现顿悟」复现的不是一个句子，是「反思从装饰变成有用的程序」。这个复现了。
★ 程序还浅：不系统、精度 0.7、不换运算符 —— 这是下一条线（撒种）的事
```
一屏看差别：`chat_countdown.py --both --ckpt out_mix4/ckpt_step300.pt --nums 3 7 25 --target 46 --temp 1 -s 8`

### 用户复述：「mix4 有『有用的』顿悟，Base 只是『装饰的』顿悟」—— 对，加一条修正
```
对的部分   用「强化两条件」判：① 出现在采样里  ② 带着它的轨迹更常对
           Base  ① 有（44%）② 无（★差 −0.02）→ 装饰
           mix4  ① 有（99%）② 有（★差 +0.20）→ 有用
修正       「有用 / 装饰」是分布上的平均，不是每条轨迹的标签。
           Base 偶尔一条反思真管用（贪心那条就是）；mix4 也有假 perfect match、30% 标错的方向、6+ 桶里乱试的
           所以准确说法：Base 的反思【大多】是装饰，mix4 的【大多】有用。RL 移的是这个比例
```

## [2, 3, 5, 7] → 31：一条轨迹把「程序还浅」全暴露了（2026-09-04，raw 模式，用户去掉了系统说明和例子）

```
Base   立刻 </think>，一句「用最大的数 7」，<answer> 7*5−2−3 = 31 </answer>   ← 实际 30。重复三次「It equals 31」「checked all possible」
mix4   13 行加减，全算对、方向全对（13/13），5 种不同符号形态，「= 13」重复 6 次，
       「Let's try another approach」之后还是加减，最后「not possible」，<answer>None of the combinations worked</answer>
```
参考解 3×7 + 5×2 = 31，两个乘积相加。

### 说明什么
```
① 精度是真的：正则修好后 13/13。这条轨迹算术零错
② ★★ 搜索没有记忆、没有策略：13 行只有 5 种形态，同一个 13 算了 6 遍。加减的上限是全加 = 17，
   第一行就该知道要上乘法，它试到第 13 行。「another approach」说了没做 —— mix4 里的装饰性反思
③ 需要的形态「两个乘积相加」不在采样半径里。它会换乘法（[3,7,25] 那题第 4 行就换了），但只会「一个乘积 ± 其余」。
   这就是硬尾巴 26 道的样子
④ Base 的失败是反面：不搜、算错（35−5 = 30 写成 31）、三次断言「等于 31」还说「检查过所有可能」。
   mix4 的失败是诚实放弃。两种失败判分器给 Base 0.25、mix4 0.05 ——
⑤ ★ 奖励塑造了放弃的方式：交一个用全数字的错式子 0.25，交「None」0.05。所以 800 条里放弃都长成 <answer>(6+12+18)−42</answer>。
   这次交 None 大概是模板残缺（少了系统说明和例子）让分布偏了一点
⑥ ★ 模板消融的意外收获：去掉系统说明后 Base 立刻关 </think>（那句「first thinks in the mind」是撑住 Base 的），
   mix4 不受影响 —— 它绑的是「Assistant: Let me solve this step by step.\n<think>」这一段，不是系统说明
```

### 对 E 的意义
这条就是「训前」样本。程序老师会写成：8 种符号形态各一次、不重复、第一行后标「+− 最大 17 < 31 → 需乘法」、再列乘积形态。
SFT 之后同一题应该 ≤ 10 行解出。这是 E 最直观的验收题。

# ★★★ 训过的模型总表（2026-09-04）

## 从零预训练（自己的架构，4×4090）
| 模型 | 数据 | 结果 | 目录 |
|---|---|---|---|
| 95M | 1.5B token 英文 | val 3.41，困惑度 30.2 | 第一轮 |
| 201M | 10B token，10.4 小时 | val 2.90，困惑度 18.1，≈ 有效 619M GPT-2 | 第二轮 |
| 201M + SFT | Alpaca 52K，23.7 分钟 | 会答指令；val loss 升但输出更好（后训练不看 val） | out_sft/ |

## RL（全部 Qwen2.5-1.5B，GRPO）
| 模型 | 起点 → 训练 | 步 | GSM R1 | GSM 中性 | CD-3 池外 @1 / @8 | CD-4 池外 @1 / @8 | CD 长度 | 一句话 |
|---|---|---|---|---|---|---|---|---|
| Base（参照） | — | — | 0.52 | 0.48 | 0.02 / 0.18（@64 0.59） | 0.02 / — | 187 | 零件都有，散着 |
| Instruct（参照） | — | — | 0.687† | — | 0.015 / 0.09‡ | — | — | |
| GSM-RL | Instruct + GSM8K | ~200 | 0.767† | — | 0.038 / 0.26‡ | — | — | 第一次 RL 有效；pass@64 不动；反向迁移到 CD 涨 2.9× |
| CD-RL | Base + CD-3 | 275 | **0.16** | **0.35** | 贪心 0.405 | — | 71（GSM 下） | CD 学会了，GSM 两模板都塌 → 遗忘 + 绑定 |
| mix-275 | Base + GSM + CD-3 | 275 | 0.71 | 0.47 | 0.16 / 0.56 | — | 108 | 混训止住遗忘；续写 44% 的测量 bug 在这发现 |
| mix-425 | mix-300 续 | 425 | 训练 eval ~0.8 | 未测 | 0.36 / 0.65 | — | 239 | 顿悟探针转正；测试片污染在这发现 |
| mix-600 | mix-300 续，len-soft | 600 | 训练 eval 0.78 | 未测 | **0.46 / 0.69（@64 0.83）** | 0.21 / 0.61 | 170 | CD-3 饱和；长度回落；上一代主力 |
| **mix4-300** | mix-600 + GSM + CD-4 | 300 | **0.74** | **0.46** | **0.48 / 0.66** | **0.39 / 0.67** | 166 / 222 | 数字用全 0.68→0.95；CD-3 保住；当前主力 |

```
口径   GSM 两列：贪心，test[0:200]，crosstask_eval.py（R1 = 训练模板；中性 = Question:/Answer:）
       CD 两列：温度 1，100 题 × 8，池外 [95000:95100]，baseline_countdown.py
       † Instruct 系用 chat 模板 + "#### 数字"，holdout TEST[200:500]，跟 Base 系不是同一段文字
       ‡ 池内 [0:100]，没在 95000 上重跑
       CD-RL 的 0.405 是贪心、countdown[90000:90200]，温度 1 的 800 条没跑
       CD 长度：CD-3 池外中位；mix4 那格是 CD-4
```

## 读法
```
GSM R1 列    0.52 → 0.16（单任务）→ 0.71 → 0.74（混训三代）        混训把遗忘变成 +0.22
GSM 中性列   0.48 → 0.35 → 0.47 → 0.46                            混训后中性 = Base，不升不降，绑定
CD-3 列      0.02 → 0.16 → 0.36 → 0.46（@8 0.18 → 0.69）           四代把 500 次抽样压成 8 次
CD-4 列      0.21（零训练）→ 0.39                                   一代，主要修数字用全
长度         187 → 108 → 239 → 170 / 222                            任务定长，不是 RL 长出来的
```

## 补测：mix4-300 在 CD-3 池外（2026-09-04）
| | mix-600 | mix4-300 |
|---|---|---|
| pass@1 / pass@8 | 0.456 / 0.694 | **0.476 / 0.660** |
| 数字用全 | 0.944 | 0.951 |
| 全对 / 全错 / 有对有错 | — / ~31 / — | 28 / 34 / 38 |
| 组内有差异 | 72% | **50%** |
| 搜索词/条 | 5.7 | 5.6 |
| 长度中位 / P95 / 撞顶 | 170 / 362 / 1.1% | 166 / 293 / 0% |

```
① CD-3 保住了：pass@1 +0.02、pass@8 −0.03，都在 100 题的噪声里。CD-4 训练没伤 CD-3
② 分布更尖：28 道八发全中、34 道八发全空，只剩 50% 的题组内有差异。pass@1 涨 pass@8 平 = 质量往已经会的题上堆
③ 50% 有信号 = 再在 CD-3 上训一半算力空转。Countdown 收工的又一个证据
④ ★差 +0.36 是因为不带搜索词的只剩 5 条、全是交白卷。探针已退役，别读
```

## 复述纠正：「三代续训每代加一样，前面的都保住了」（2026-09-04）

用户复述：保持了多样性，加一个东西，在那上面更好，其他没退化。
```
换一个词   保住的不是多样性，是旧任务的【能力】。多样性每代都在掉：组内有差异 72% → 50%，pass@1 涨 pass@8 平
每代加一样  mix-275 搜索形态 + 格式 → 600 精度 + 长度控制 → mix4 四个数用全
没退化      只对测过的三样成立：GSM 训练模板、GSM 中性、CD-3
靠什么保    ★ 旧任务留在训练池。GSM 三代都混；CD-3 没混进 CD-4 那轮，靠形态包含顺带保住
主语        「能保持住」是训练配方的性质不是模型的。单任务那轮同一 Base 同样 300 步，GSM 不在池里就塌到 0.16
```

用户再复述：这是收紧的代价；或者说保持了多能力的多样性，收缩掉不好的或没用的；是比较好的结果。
```
对，两种多样性分开说
  跨任务的广度    保住了（GSM + CD-3 + CD-4 三样都在）
  任务内的分布    变尖了（8 条抽样越来越像）
收掉的大多没用   格式错、数字漏、幻觉续写、长度跑飞 —— 全是该收的
但不全是         ×÷ 尝试 34.6% 没涨；pass@8 0.69 → 0.66；难题上的乱试收成了「放弃」而不是「更好的搜索」；假 perfect match 没收
「比较好」的边界  对当前目标好：训过的分布上可靠、不忘。对下一步是代价：尖分布 = 采样半径小 = 撒种的起点更差
                 ★ R1 阶段 ③ 回到干净 Base 做 SFT，就是为了躲这个代价。E 该不该 SFT 到 mix4 还是 Base，跑前要想
```

## 六个选项：各拧哪个旋钮、服务哪个目标、怎么验收（2026-09-04）
| # | 选项 | 旋钮 | 服务的目标 | 回答什么 | 验收 | 代价 |
|---|---|---|---|---|---|---|
| 1 | 穷举器写系统搜索轨迹 → SFT → RL | 撒种 | 策略（硬尾巴）；顺带长思维链 | 撒种抬不抬天花板；程序可不可教、可不可泛化；难题上长度会不会自然变长 | [2,3,5,7]→31 十行内解出；CD-3 pass@64 ≥ 0.92；6+ 桶正确率 > 0.35 | 穷举器 1 天 + 训练 3 小时 |
| 2 | 自采样拒绝采样 SFT（1 的对照臂） | 撒种的假版本 = 收紧 | 科学对照 | 1 的涨幅里多少是「SFT 本身的收紧」、多少是「新零件」。1 ≈ 2 → 没新零件；1 > 2 → 撒种真有用 | 同 1 的指标并排 | 加 1 小时 |
| 3 | 计算器工具：写 `<calc>` 标签，harness 回值再续写 | 换环境（给观测） | 精度天花板；为 4 建基础设施 | 心算 0.70 → 0.99 后硬尾巴动几道；模型会不会学「该算时调工具」 | 标注精度 > 0.95；假 perfect match 归零 | 中断-注入 rollout 1 天 + 训练 2 小时 |
| 4 | Agent RL 猜数字：多轮，观测进 prompt | 换环境（多轮） | 目标三（agent RL）；长思维链另一条路 | 多轮信用分配；观测进 prompt / 奖励进梯度；harness 长什么样 | 二分搜索涌现：平均轮数逼近 log₂N | 2~3 天 |
| 5 | 换真需要长链的任务 | 换地 | 目标二（长思维链），直接测 H1 | 长度是不是任务要求出来的 | 长度中位随训练涨且正确率同涨 | 环境 + 判分 1 天 |
| 6 | 7B Base 跑一次基线，不训 | 换材料（只看） | 诊断：哪些天花板是尺寸问题 | 7B 天生有没有系统搜索、心算精度、×÷ 形态 | 看 baseline_countdown 的精度和 scan_ops 的形态 | 30 分钟 |

```
关系   1+2 是一个实验两条臂，一起跑
       3 → 4 共用「生成中断、注入、续写」的 rollout，3 是 4 的脚手架
       5 只在 1 没给出「按需变长」时才需要
       6 不改变做不做 1/3，只改变怎么解读结果（4×4090 训不动 7B 全参）
顺序   6（今天，30 分钟）→ 1+2 → 3 → 4；5 看情况
```

## 选项 3 和 4 的机制说明（2026-09-04）
```
计算器 = 最小的 tool use
  模型写 <calc>3 * 7 + 25</calc> → 生成在 </calc> 停 → harness 用 safe_eval 算 → 把 <result>46</result> 接回上下文 → 继续生成
  模型看不到 Python，只看到标签和结果。生产环境的 function calling 就是这个结构，只是工具从四则运算换成任意函数
  训练上多两件事：① rollout 分段（停、注入、续）② <result> 里的 token 不是模型写的，loss 要 mask 掉
  修什么：心算精度 0.70 → 1.0（送进工具的式子）。不修：搜索策略（重复、不换运算符）
  ★ 采样半径同样管工具：Base 没见过 <calc>，p ≈ 0，RL 放大不了 → 要么提示词里给示范（开关），要么 SFT 几条（撒种）

猜数字 = 计算器的多轮版，跟现在的模型关系比看上去大
  ★ mix4 内部已经在用「too high / too low → 调整」这套程序，只是反馈是它自己算的。猜数字把反馈交给环境说
    → 假设：mix4 零训练玩猜数字就比 Base 强（程序泛化到多轮）。上手先测这个
  新东西：多轮（猜 → 环境答 → 再猜）、环境 token mask、回合终止、整条轨迹一个奖励（轮数越少越高）
  为什么选它：最优策略已知（二分，log₂N 轮），能精确量「离最优多远」；搜索空间一维，把多轮机制和搜索难度分开
  之后：同一套 harness 把 Countdown 也变成多轮 + 计算器 = 3 和 4 合体
```

## ★ 目标清单（2026-09-04 统一口径 —— 之前「三个目标」有两套编号，以这份为准）

北极星：用 RL 复现「顿悟」的反思搜索智能 + 长思维链（2026-09-02 定）

| # | 目标 | 状态 | 证据 / 差什么 | 哪些选项服务它 |
|---|---|---|---|---|
| 1 | 复现顿悟：反思搜索从装饰变成有用的程序 | ✓ 完成 | 99% 轨迹有、带方向、★差 +0.2、精度 0.7；Base 44%、★差 −0.02 | — |
| 2 | 长思维链：越训越长、难题上长 | ✗ 未达 | Countdown 任务定长 170/210。三条路：按需变长（1）、多轮累加（4）、换任务（5） | 1、4、5 |
| 3 | Agent RL：多轮环境，观测进 prompt | 未开始 | 缺「停、注入、续、mask」的 rollout | 3 搭脚手架、4 |
| 4 | 验证条件性策略绑定 | ✓ 完成 | 三代 GSM 中性 = Base；去掉系统说明 Base 塌 mix4 不塌 | — |
| 5 | 策略（硬尾巴）：搜索成为真程序 —— 穷举、记账、算准、换运算符 | 半 | 33 道全错 = 7 道 +− + 26 道 ×÷；搜索无记忆 | 1、3 |
| 6 | 理论：撒种能不能抬天花板（RL 动有限 k 不动 pass@∞） | 半 | pass@64 测了 RL 那半；撒种那半没测 | 1 + 2 |
| 7 | 理论：精度天花板是不是 1.5B 心算 | 诊断了 | 同一式子不同轨迹算出不同数 | 3（给工具）、6（看 7B） |

```
1 和 4 已收；2 是北极星里还没兑现的那半；3 是用户自己定的第三步；5 6 7 是路上长出来的
选项 → 目标：1 服务 2 5 6 ｜ 2 服务 6 ｜ 3 服务 3 5 7 ｜ 4 服务 2 3 ｜ 5 服务 2 ｜ 6 服务 7
```

## 术语表：选项表「为了什么」那一列（2026-09-04）
| 词 | 对应目标 | 是什么 | 定义它的数 |
|---|---|---|---|
| 硬尾巴 | 5 | 8 次抽样全错的那些题。难度分布的尾巴，模型现在够不着 | CD-4 池外 33 道 = 7 道纯加减可解 + 26 道必须乘除 |
| 长思维链 | 2 | 输出因为题目需要而变长，难题写得长、简单题写得短。R1 的标志 | 我们停在 CD-3 170、CD-4 210，从没涨过 |
| 给 1 当对照 | 6 | 1 涨了不知道是「SFT 把分布收紧」还是「数据里有新零件」。2 用模型自己的轨迹 SFT，只有收紧没有新零件 | 1 − 2 的差 = 新零件的贡献 |
| 精度天花板 | 7 | 标注「X = V (too high)」里 V 算错或方向标反。1.5B 心算噪声，RL 压不掉 | 精度 0.70；同一式子一条写 30 一条写 18 |
| 给 4 搭脚手架 | 3 | 「生成到标签停 → 外部算 → 结果接回 → 续写」这套 rollout 加 mask。计算器一轮，猜数字 N 轮，代码同一套 | — |
| 目标三 / Agent RL | 3 | 多轮环境：模型说一句、环境答一句、再说。观测进 prompt，奖励进梯度 | 猜数字：最优 log₂N 轮 |
| 长思维链的另一条路 | 2 经由 3 | 多轮任务里上下文随轮数累加，「长」不靠一口气写，靠回合堆 | — |
| 目标二 | 2 | 同「长思维链」 | — |
| 诊断 | 7 | 不训，只看 7B Base 天生有没有：系统搜索、心算精度、乘除形态。有 → 是尺寸问题；没有 → 撒种和工具在 7B 上也需要 | 30 分钟 |

## 六个选项 × 目标编号（2026-09-04，「为了什么」列改成统一清单的编号）
| # | 选项 | 为了哪些目标 |
|---|---|---|
| 1 | 穷举器写系统搜索轨迹 → SFT → RL | **5 策略**（搜索成真程序，捞硬尾巴）＋ **6 理论**（撒种能不能抬天花板）＋ **2 长思维链**（顺带：系统搜索让难题非长不可） |
| 2 | 自采样拒绝采样 SFT | **6 理论**（1 的对照臂：分离收紧和新零件） |
| 3 | 计算器工具 | **7 理论**（精度天花板是不是心算）＋ **5 策略**（算准后硬尾巴动几道）＋ **3 Agent RL**（脚手架：停、注入、续、mask） |
| 4 | Agent RL 猜数字 | **3 Agent RL**（主）＋ **2 长思维链**（另一条路：回合累加） |
| 5 | 换真需要长链的任务 | **2 长思维链**（直接测「越训越长」） |
| 6 | 7B Base 跑基线 | **7 理论**（哪些天花板是尺寸问题） |

目标 1（顿悟）和 4（绑定）已完成，没有选项再服务它们。

# ★★★ 四臂实验计划：撒种 vs 自蒸馏 vs 只 RL vs 换任务（2026-09-04 定，用户决定四条全跑）

## 设计
同一起点 `out_mix4/ckpt_step300.pt`，同一 RL 预算，四条臂：

| 臂 | 名字 | SFT | RL 300 步的任务 | 分离什么 |
|---|---|---|---|---|
| 0 | 只 RL | 无 | GSM + CD-4 | 多训 300 步本身 |
| 1 | 程序撒种 | 穷举器写的系统轨迹（CD-3 + CD-4）+ GSM 自采样 | GSM + CD-4 | 新零件 |
| 2 | 自蒸馏 | mix4 自己采样的对轨迹，★ 同一批题号 + 同一份 GSM | GSM + CD-4 | SFT 收紧 |
| 5 | 换任务 | 无 | GSM + 长链算术 | 长度是不是任务要求出来的 |

7 个测量点：mix4-300 参照、臂 1 和臂 2 的「SFT 后 RL 前」、四条臂 RL 后。

## RL 统一设置（四臂一样）
```
torchrun --nproc_per_node=4 grpo_mix_mp.py --probs 8 -k 16 --gen-bs 32 --steps 300 \
    --init <起点> --data <任务文件> --max-new 1024 --len-soft 800 --len-pen 0.5 --out out_arm<N>
起点   臂 0/5 = out_mix4/ckpt_step300.pt；臂 1 = out_sft1/ckpt.pt；臂 2 = out_sft2/ckpt.pt
★ max_new 512 → 1024、len-soft 400 → 800：臂 1 的系统轨迹本来就长，400 会把种罚掉。四臂同设置才公平
★ KL 参照是 Qwen Base（ref = load()，不是 --init），β 0.0005，四臂一样
风险   1024 × k16 可能 OOM → 先 --smoke；不行 --gen-bs 16
```

## 数据
```
题号     CD-3 和 CD-4 各从训练池 [0:80000] 随机 2000 道，臂 1 臂 2 用【同一批题号】。评测片 [95000:95100] 不碰
臂 1     穷举器：按 mix4 现在的措辞写。规则：
           +− 符号形态各试一次（3 数 4 种、4 数 8 种），不重复，算术全对，每行标 (too high/low)
           +− 穷尽后写一句「+− 最大 X / 最小 Y，需 × 或 ÷」，再按固定顺序列乘除形态直到命中
           命中写「(perfect match)」→ </think> → <answer>…</answer>
           ★ 生成后每条过一遍 CD.reward，必须全 1.0；长度分布先看再定
臂 2     mix4-300 温度 1 每题抽 8，每题留 1 条对的（随机）。抽不到对的题就没有 → 臂 2 天然偏简单题，记录覆盖率
GSM     mix4-300 在 GSM8K train 上自采样，留对的 2000 条，两臂共用。★ 不混 GSM 的 SFT 会把 GSM 砸了，单任务那轮的教训
格式     全部 {"prompt": R1 模板到 <think>, "response": 轨迹到 </answer>}
```

## SFT（sft_qwen.py，新写）
lr 1e-5、2 epoch、有效 batch 32、bf16、只对 response 算 loss、cosine。存成 `{"model": sd, "step": 0}` 给 `--init` 用（--init 已容忍无 opt）。每臂约 30 分钟。

## 臂 5 的任务：长链算术（chainarith.py，新写）
```
题    Start with 17. Add 8. Multiply by 3. Subtract 11. … What is the final result?     4~20 步，均匀
数    起点 1~50；加减 2~50；乘 2~9；中间值限 ±5000，超了重生成。每步都简单，难在链长
答    整数，<answer>N</answer>。奖励：格式 0.05 + 解析 0.05 + 对 0.9
接入  trainer 加 --task chain：提示用 gsm_prompt 的 R1 壳，判分用 gsm_reward，改约 10 行
评测  baseline_chain.py：按步数分桶（4~7 / 8~11 / 12~15 / 16~20）的 pass@1 和长度。池外 [95000:95100]
先校准  Base 和 mix4 零训练跑一次，要 pass@1 总体 0.2~0.5 且随步数递减；不对就调步数范围
```

## 评测矩阵（7 个点 × 同一套）
```
CD-3 池外 800    pass@1 / pass@8                       全部
CD-3 pass@64     天花板                                  参照 + 臂 0 1 2
CD-4 池外 800    pass@1 / pass@8 / 数字用全              全部
analyze_search   分桶、标注精度（修好的正则）             全部（CD-4 文件）
scan_ops         全错几道、+− vs ×÷                      全部
crosstask        GSM R1 / 中性                           全部
长链算术 池外 800 按步数分桶的 pass@1 和长度              ★ 全部 —— 臂 0 1 2 在这上面是零训练，测 Countdown 程序往链上迁不迁
长度             中位 / P95；CD 按搜索次数分桶、链按步数分桶  全部
```

## 预测（跑前写，跑后对）
| 臂 | CD-4 pass@1 | CD-3 pass@64 | 全错 33 → | 标注精度 | 长度 | GSM R1 | 链 pass@1 |
|---|---|---|---|---|---|---|---|
| 参照 mix4 | 0.39 | 0.83（600 的数） | 33 | 0.69 | 222 | 0.74 | 待测 |
| 0 只 RL | 0.41 | 0.83 | 31~33 | 0.70 | 220 | 0.74 | ≈ 参照 |
| 1 SFT 后 | 0.45 | 0.90 | ≤ 20 | > 0.85 | > 400 | 0.72（SFT 略伤） | ≈ 参照 |
| 1 RL 后 | ≥ 0.50 | **≥ 0.92** | ≤ 20 | > 0.85 | 分叉：简单 < 250、难 > 500 | 0.74 | 略高于参照？ |
| 2 SFT 后 | 0.42 | 0.83 | 31~33 | 0.72 | 200 | 0.73 | ≈ 参照 |
| 2 RL 后 | 0.43 | 0.83 | 31~33 | 0.72 | 190 | 0.74 | ≈ 参照 |
| 5 RL 后 | 0.37（略忘） | — | 34~36 | 0.68 | CD 平 | 0.74 | ★ 涨，且长度随步数斜率变陡 |
```
判决线   臂 1 pass@64 > 臂 2 ≈ 臂 0 → 撒种加零件。臂 1 ≈ 臂 2 → 程序没传过去或被 RL 冲掉（看 SFT 后那一点分辨）
         臂 1 长度分叉 → 长度是任务要求出来的，臂 5 的问题顺带答了；臂 5 自己再答一次
         臂 5 链上涨 + 长度斜率变陡 + CD 略忘 → H1「越训越长」第一次看到
```

## 顺序与时间
```
D1  穷举器 → 先给用户看 5 条轨迹认格式 → 生成全量 → 过 reward 检查
    拒绝采样（臂 2 数据，GPU 30 分）；GSM 自采样（GPU 20 分）；sft_qwen.py；chainarith.py + baseline_chain.py + trainer 10 行
    ★ 臂 0 不依赖任何新代码，设置定了就能开跑（2 小时）
D2  SFT 臂 1、臂 2（各 30 分）→ SFT 后评测（各 1 小时）→ RL 臂 1（3~4 小时，轨迹长）→ RL 臂 2（2 小时）
D3  链任务校准（30 分）→ RL 臂 5（2 小时）→ 四臂全套评测（各 1 小时）
D4  总表 + 复盘
GPU 约 16 小时，墙钟 3~4 天
```

## 风险清单
```
① 1024 × k16 OOM                → 冒烟先跑；退到 --gen-bs 16
② 穷举器格式漂移，判分器不认      → 生成后全量过 CD.reward，非 1.0 的丢
③ 臂 1 轨迹太长（4 数嵌套可到 40 行）→ 看长度分布；必要时嵌套形态只列到命中前 10 行
④ RL 把种冲掉                    → 「SFT 后」那个测量点就是为了看这个；KL 参照是 Base 帮不上，靠 len-soft 800 别罚
⑤ 臂 5 太易/太难                → 先校准，目标总体 pass@1 0.2~0.5
⑥ 臂 2 题少                     → 同题号下只有 60~70% 有对轨迹，记录覆盖率，别补题（补了就偏简单）
⑦ 四臂同时出结果，过度解读        → 每臂先过 9 步流程，最后再并表
```

## 四臂逐步设计：每步做什么、为什么（2026-09-04）

### 臂 0 只 RL —— 「多训 300 步本身值多少」
```
1 冒烟   grpo_mix_mp --smoke，新设置 max_new 1024 / len-soft 800        为什么：1024×k16 没跑过，先看 OOM 和每步秒数，四臂都用这套
2 训练   --init out_mix4/ckpt_step300.pt --data countdown4 300 步 → out_arm0   为什么：臂 1/2 的涨幅要减掉它；它本身也直接回答「再灌 RL 还能涨多少」
3 看日志 CD |A|（预测 < 0.16，饱和）、有梯度的组比例（预测 ~50%）、长度（平）、KL          为什么：这三项是「空转」的指纹
4 九步评测                                                                                为什么：跟其他臂同一把尺
5 判读   涨明显 → 臂 1/2 的涨幅先扣掉它；平 → 臂 1/2 的涨幅归 SFT
```

### 臂 1 程序撒种 —— 「数据里放模型自己抽不到的程序，能不能抬天花板」
```
1 写穷举器 enum_traces.py                                                          为什么：这就是「种」
    措辞复用 mix4 现有的开头和行格式 → SFT 改动小，不伤 GSM
    顺序固定：+− 形态各一次（先全加）→ 穷尽后一句「+− 最大 X 最小 Y，需 ×÷」→ 乘除形态按固定顺序 → 命中 (perfect match) → </think> <answer>
    这四样（不重复、算对、有上下界推理、换运算符）正是 scan_ops 查出来缺的四样
2 看 5 条样例，用户认格式                                                            为什么：格式错 = 全废，判分器认不认是硬约束
3 生成 2000 CD-3 + 2000 CD-4，题号取自 [0:80000]，全量过 CD.reward 必须 1.0；看长度中位/P95    为什么：数据质量门；长度决定 RL 的 max_new 够不够
4 GSM 自采样 2000 条对的（跟臂 2 共用）                                                为什么：不混 GSM 就是单任务那轮的遗忘重演
5 写 sft_qwen.py，SFT mix4-300 → out_sft1（lr 1e-5，2 epoch，只算 response 的 loss）   为什么 lr 低：只装程序不重写；「后训练 lr 要比预训练低」的教训
6 ★ SFT 后测量点：九步全跑                                                            为什么：分离「SFT 给的」和「RL 拿走的」
    重点：pass@64（天花板动没动）、scan_ops 全错数、标注精度（应 > 0.85）、长度（应 > 400）、GSM（略伤可接受）、链零训练
7 RL 300 步 from out_sft1 → out_arm1                                                 为什么：看 RL 会不会把种收紧成有用的，还是冲掉
    日志盯：长度趋势（掉 = 冲种）、精度（掉 = 冲种）、CD |A|（应比臂 0 大：新零件带来新的组内差异）
8 RL 后九步
9 判读：对臂 0 = SFT+程序的总效应；对臂 2 = 程序独有的效应；SFT 后 vs RL 后 = RL 对种做了什么
```

### 臂 2 自蒸馏 —— 「同样做 SFT，但数据里没有新东西，能涨多少」
```
1 reject_sample.py：mix4-300 温度 1 每题抽 8，★ 同一批题号，每题留 1 条对的（随机）       为什么同题号：唯一变量是数据来源
    记录覆盖率（预计 60~70% 的题有对轨迹）。★ 不补题                                    为什么不补：补了就偏简单，臂 2 会假强
2 同一份 GSM 2000                                                                    为什么：控制变量
3 SFT 同超参 → out_sft2                                                              为什么：只差数据
4 SFT 后九步                                                                          同臂 1
5 RL 300 步 → out_arm2
6 RL 后九步
7 判读：臂 2 − 臂 0 = SFT 收紧本身；臂 1 − 臂 2 = 新零件。这是整个实验的核心减法
```

### 臂 5 换任务 —— 「任务真需要长链时，RL 会不会让它变长」
```
1 chainarith.py：生成 + 出题 + 判分。4~20 步均匀；起点 1~50；加减 2~50；乘 2~9；中间值 ±5000    为什么：每步都简单，难度只来自链长 → 长度是唯一变量
    生成 100k，评测片 [95000:95100]
2 baseline_chain.py：按步数分桶（4~7 / 8~11 / 12~15 / 16~20）的 pass@1 和长度                为什么：「长度随难度」是这条臂的主读数
3 校准：Base 和 mix4-300 零训练各跑一次；总体 pass@1 要 0.2~0.5 且随步数递减，否则调步数范围     为什么：太易没梯度、太难没样本
4 trainer 加 --task chain（提示用 gsm_prompt 的壳，判分用 gsm_reward），冒烟                    为什么：GSM 和链的接口一样（文本题、整数答）
5 RL 300 步 from mix4-300，GSM + 链 → out_arm5                                                为什么混 GSM：防遗忘，跟其他臂同配方
    日志盯：长度（应涨）、按步数桶的正确率（长桶应涨最多）、GSM
6 RL 后：链评测 + 九步（含 CD，看忘多少）
7 判读：长度随步数的斜率训前 vs 训后 = H1 的直接读数；正确率涨在长桶 = 长度有用；CD 略忘 = 换任务的代价
```

### 顺序为什么这样
```
臂 0 先        不依赖任何新代码，GPU 别闲着
数据再         臂 1 和臂 2 的数据一起做完再 SFT，保证同题号同 GSM
臂 1 比臂 2 先  臂 1 轨迹长、RL 慢（3~4 小时），排前面
臂 5 最后       要写环境，而且它的结论跟臂 1 的「长度分叉」互相印证，两个都有再下结论
```

## 穷举器 enum_traces.py 写好（2026-09-04）

### 是什么
一个 Python 程序（不是模型）。给一道 Countdown 题，按 mix4 现在的措辞写一条完整的【系统搜索】轨迹。臂 1 的老师。

### 为了什么
造 SFT 数据，数据里有模型自己产生不了的四样东西（scan_ops 查出来缺的）：
```
① 不重复    +− 的符号形态各写一次（3 数 4 种、4 数 8 种），最大数固定为正
② 算对      每行的值由程序算，方向标注必对
③ 上下界    +− 穷尽后写「None of the 8 sign patterns gives 43 (they range from 7 to 59), so we need * or /」
④ 换运算符  第 1 层：列出所有两数乘积/整除商，用上下界剪掉够不着的，剩下的按离目标近的先试、配其余数做 ±
            第 2 层：三数成链 + 剩数（覆盖 14×5/7+33），两两成块再合（覆盖 3×7+5×2、(42+18)/(12/6)）
```
命中写 (perfect match) → </think> → <answer>。每条落盘前过 CD.reward，必须 1.0。只走非负整数中间值（1.5B 心算不碰小数）。

### ★ 一个设计决定：第 2 层按枚举顺序，不按「离目标最近」
按最近排的话命中永远在第 2 层第一行（cap 40 和 60 结果完全一样 = 从没搜过）→ 模型学到的是「直接写出对的链」，不是搜索，
而且这正是幻觉 perfect match 的温床。枚举顺序下命中前能看到失败行，代价是长、有些题超 cap 被丢。

### 本地 3000 题量出来的（GS01 上真实数据再跑一遍）
| | CD-3 | CD-4 自然序 cap 40 | CD-4 自然序 cap 60 | CD-4 最近序 cap 40 |
|---|---|---|---|---|
| cap 内解出 | 93% | **86%** | 90% | 95%（泄题） |
| 算式行数 中位 / P90 / 最长 | 2 / 8 / 16 | **4 / 19 / 40** | 5 / 26 / 60 | 5 / 17 / 37 |
| 命中层 0 / 1 / 2 | 77% / 12% / 11% | 74% / 16% / 10% | 71% / 14% / 15% | 67% / 14% / 20% |
| 判分器不认 | 0 | 0 | 0 | — |

定：自然序、cap 40。丢掉的 14% 是最深的 4 数题，SFT 教的是程序不是答案，RL 阶段让它自己往深走。
估算 token：约 15/行 + 头 60 → 中位 ~120，P90 ~350，最长 ~650，在 max_new 1024 内。

### 撞出并修掉的 bug
除号右边是表达式必须加括号：`39 / 1 * 3` ≠ `39 / (1 * 3)`。修前 CD-3 有 0.5% 的轨迹答案错。

### GS01 上生成
```
python3 enum_traces.py --data data/countdown.json  --n 2000 --hi 80000 --out sft_cd3.jsonl
python3 enum_traces.py --data data/countdown4.json --n 2000 --hi 80000 --out sft_cd4.jsonl
```

## 臂 1 数据生成完（GS01，2026-09-04）
| | CD-3 | CD-4 |
|---|---|---|
| 试 / 留 | 2126 / 2000 | 2366 / 2000 |
| cap 内没解 | 5.9% | **15.5%** |
| 算式行数 中位 / P90 / 最长 | 3 / 9 / 19 | 4 / 19 / 40 |
| 命中层 0 / 1 / 2 | 71% / 15% / 14% | 72% / 18% / 10% |
| token 中位 / P90 / 最长 | 121 / 402 / 621 | 185 / 605 / **1547** |

```
① 行数跟本地 3000 题量的一致；判分器 0 条不认
② ★ token 比我估的多一倍：Qwen 一个数字一个 token，462 = 3 个、2500 = 4 个。第 1 层「Pair products」那行和第 2 层每行都重
③ CD-4 最长 1547 > RL 的 max_new 1024。处理：SFT 用 --max-len 1600 丢掉 prompt+response 超的（估 2~3%）；RL 里超 1024 的轨迹被截断没 answer 得 0 分，RL 会自己往短压
④ 七成样本是第 0 层（短），一成第 2 层（长）。保持自然分布：「简单短、难题长」就是要教的东西。第 2 层 200~290 条 × 2 epoch 够学形态
⑤ 丢掉的 15.5% 是最深的 4 数题 —— SFT 教程序不教答案，深处让 RL 走
```

## reject_sample.py / sft_qwen.py 写好（2026-09-04）
```
reject_sample.py   拒绝采样。--task gsm：GSM8K train[0:7000] 打乱逐题抽 k=4，留够 n 条对的（两臂共用）
                   --task cd --ids sft_cd*.jsonl：★ 同题号，每题抽 8 留 1 条对的，不补题，打印覆盖率
                   clean()：截 "\nUser:" 续写、只留到第一个 </answer>。提示词跟训练逐字相同
sft_qwen.py        只对 response 算 loss，末尾接 EOS；prompt+response > --max-len 1600 丢
                   lr 1e-5、2 epoch、有效 batch 32（4 卡 × micro 1 × 累积 8）、cosine + warmup 20、AdamW8bit、梯度检查点
                   4 卡手动 all_reduce（跟 grpo_mix_mp 一个路子）。val loss 只做 sanity
                   存 {"model", "step": 0, "sft": meta} → <out>/ckpt.pt，--init 直接读
                   训完贪心生成 [2,3,5,7]→31 和一道 GSM 看格式
单测：label mask / EOS / padding、clean / gsm_ok 都过
```

## 臂 0 显存测试通过（2 步真实配置，2026-09-04）
```
max_new 1024 × k16 × gen-bs 32：12~14 GB / 24 GB，没 OOM，还有 10 GB 余量（臂 1 轨迹长了 KV 顶多再吃 1 GB）
每步 9+5 s ≈ 16 s → 300 步 ≈ 80 分钟 + 12 次 eval ≈ 1.5 小时（比估的 2.5~3 小时快：轨迹才 200 token，1024 只是上限）
起点 eval CD 0.45 / GSM 0.84 = mix4-300 训练时的数 → --init 读对了
GSM |A| 0.04 vs CD 0.28：GSM 在温度 1 下 0.95~1.0 全对，几乎没梯度，推力全来自 CD
KL 0.065（对 Base）：mix4 那轮末尾报 0.04，这里 2 步 8 题噪声大，正式跑看趋势
```

### ★★ 勘误：标注精度不是 0.70，是 0.42~0.53
正则修好后第一次跑训练日志：标注数/条 4.0（以前 1.0，只数到每段第一条），精度 温度 1 下 0.34~0.49、贪心 0.53、汇总 0.416。
```
以前的 0.60 → 0.70   是「每段第一条」的精度 —— 第一条通常是全加那行，最好算
真实的                每条 4 条标注里【一半多算错或方向标反】
不变的                方向：训练里精度确实在涨（首条精度 0.60 → 0.70 的趋势没错）；剂量响应、分桶、搜索词计数不受影响
含义                  ① 精度天花板比说的还低，「1.5B 心算」这条更硬了
                      ② 臂 1 的老师是 100% 精确的 —— SFT 后精度该跳到 0.85 以上，跳不上去说明 1.5B 模仿不了精确算术
                      ③ 计算器（选项 3）的价值更大
```
PLAN.md 里所有「精度 0.70」处按此读。测试目录 out_arm0_test 可删。

## 文件清单：所有 json / jsonl（2026-09-05）
| 文件 | 谁生成 | 每行/每条是什么 | 条数 | 用途 |
|---|---|---|---|---|
| data/countdown.json | countdown.py --gen 100000 --k 3 | {"nums": [a,b,c], "target": T}，保证有解，数 1~50、目标 1~100 | 100k | CD-3 题池。[0:90000] 训练、[90000:90100] 训练中 eval、[95000:95100] 池外测试 |
| data/countdown4.json | 同上 --k 4 | 4 个数 | 100k | CD-4 题池，切分同上 |
| GSM8K | HF datasets 缓存，不是本地 json | question / answer | train 7473，test 1319 | train 训、test[0:200] crosstask、test[200:300] 训练中 eval |
| sft_cd3.jsonl | enum_traces.py | {prompt, response, nums, target, lines, level, idx}，穷举器写的系统轨迹 | 2000 | 臂 1 SFT |
| sft_cd4.jsonl | 同上 | 同上，4 数 | 2000 | 臂 1 SFT |
| sft_gsm.jsonl | reject_sample.py --task gsm | {prompt, response, task, idx, q, gold, n_correct}，mix4 自己答对的 | 2009 | 臂 1、2 共用压舱石 |
| sft_cd3_self.jsonl | reject_sample.py --task cd --ids sft_cd3.jsonl | mix4 自己在同题号上答对的 | ~1400（跑中） | 臂 2 SFT |
| sft_cd4_self.jsonl | 同上 | 同上 | ~1400（跑中） | 臂 2 SFT |
| cd_<模型>_<数据集>_s<k>_off<偏移>.json | baseline_countdown.py | 汇总指标：阶梯、pass@k、梯度信号、探针、长度 | 1 | 评测结果 |
| 同名 _raw.json | 同上 | 每条轨迹：题号、txt、得分、明细、长度、续写标记 | 100×k | analyze_search / scan_ops 的输入 |
| crosstask_*.json | crosstask_eval.py | 2×2 每格每题：ok、pred、gold、ntok、txt | 4×200 | GSM 跨模板结果 |
| out_*/hist.json | grpo_mix_mp.py | 每步的训练指标 + eval | 步数 | 训练曲线 |

## 教学：判分器就是答案钥匙 + Countdown 奖励拆解里的工程点（2026-09-05）

### 一组 8 条的完整算账（[3, 7, 25] → 46）
| # | 模型写的 answer | 格式 .05 | 解析 .05 | 数字比例 .15 | 对 .75 | 总分 r | A = r − 均值 |
|---|---|---|---|---|---|---|---|
| 1 | 3 * 7 + 25 | ✓ | ✓ | 1.0 | ✓ | 1.00 | +0.64 |
| 2 | 25 + 7 − 3 = 29 | ✓ | ✓ | 1.0 | ✗ | 0.25 | −0.11 |
| 3 | 25 * 3 − 7 = 68 | ✓ | ✓ | 1.0 | ✗ | 0.25 | −0.11 |
| 4 | 3 + 7 + 25 − 0（多个 0） | ✓ | ✓ | (3−1)/3 = 0.67 | ✗ | 0.20 | −0.16 |
| 5 | 7 * 7 − 3（7 用两次） | ✓ | ✓ | (2−1)/3 = 0.33 | ✗ | 0.15 | −0.21 |
| 6 | None of them worked | ✓ | ✗ | 0 | ✗ | 0.05 | −0.31 |
| 7 | 没写 answer 标签 | ✗ | | | | 0.00 | −0.36 |
| 8 | 3 * 7 + 25 | ✓ | ✓ | 1.0 | ✓ | 1.00 | +0.64 |
均值 0.3625。文件里只有 nums 和 target，每一格都是判分器算出来的，没有一格是查出来的。

### 公式（grpo_mix_mp.py 实际用的）
```
A_i  = r_i − mean(r_1..r_k)                                    组内减均值，默认不除 std
loss = − Σ_i Σ_t A_i · log π(token_t | 前文) / 全批 token 数  + β · KL(π‖Base)，β = 0.0005
一次 rollout 一次更新 → ratio ≡ 1，没有 PPO 裁剪；KL 用 k3 估计 exp(Δ) − Δ − 1
```
全 0.25 的组 → A 全 0 → 零梯度。这就是「组内总分有差异 71%」那个数的来历。

### 工程点
```
① 阶梯 = 早期梯度      Base 对率 0.016，二值奖励下 98% 的组全 0 没梯度。格式/解析/数字三级让 #6 #7 和 #2 分开 → 第 0 步就有梯度
② 连续部分分 = 方向    数字 0/1 口径下「用对 2 个」和「一个没对」同分；比例分让 #4 > #5，还罚多写的数（−0 那种幻觉）
③ 0.75 压倒            阶梯最多 0.25，爬完阶梯不算赢，逼它去对。副作用：错式子 0.25 > None 0.05，塑造了放弃的写法
④ safe_eval 白名单     模型写的字符串绝不 eval()，AST 只放数字和四则
⑤ 两任务满分都 1.0     混训时 A 的尺度可比，GSM 0.1 + 0.9
⑥ 不除 std             阶梯下组内差可能只有 0.05，除 std 会把噪声放大成满幅推力（难度偏置）。--adv-std 备用
⑦ 只取第一个 answer    不能撒网写十个；配长度软惩罚，不能拖
⑧ 阶梯同时是仪器       baseline 打印每级通过率（Base 0.76 / 0.63 / 0.14 / 0.016）→ 瓶颈在哪层一眼看出，H4 由此预测
```

## 臂 2 数据采完（2026-09-05）
| | CD-3 自采样 | CD-4 自采样 | 对照：穷举器 CD-3 / CD-4 |
|---|---|---|---|
| 留 / 题 | 1530 / 2000 = **0.77** | 1394 / 2000 = **0.70** | 2000 / 2000 |
| 每题 8 条里对几条 | 5.2 | 3.8 | — |
| token 中位 / P90 / 最长 | 122 / 172 / 351 | 166 / 253 / 428 | 121 / 402 / 621 · 185 / 605 / 1547 |
```
① 覆盖率跟预测（0.70~0.75）一致。臂 2 共 2924 条 CD，比臂 1 少 27%，没过「加控制臂」的线（0.5）
② ★ 两臂数据差两样，不是一样：题（臂 2 缺三成最难的）+ 轨迹形态（臂 2 短一半、P90 差 2.4 倍）。
   短是因为自采样留下的是「碰对」的那条，穷举器写的是「搜到」的那条。这两样都是「自蒸馏」的固有属性
③ 若要分开「程序形态」和「多出来的难题」两个效应 → 备用臂 1′：穷举器数据限制到臂 2 覆盖的题号做 SFT。先看主结果再定
④ 臂 2 轨迹的标注精度、重复率：check_sft.py 量（新写），进档案
```

### check_sft.py 体检 + 穷举器一个小修（2026-09-05）
```
check_sft.py   每份 SFT jsonl 报：条数、token 长度、标注/条、标注精度、重复形态率、说 perfect 率、含 ×÷ 率、命中层
               重复形态：纯加减按项集合比（25+7−3 = 7+25−3），含 ×÷ 的按原样比（16*(31−27) ≠ 16/(31−27)）
穷举器小修     题里有相同的数（[4, 4, 29, 43]）时符号互换会写出同一行，本地量到 1% 的行是这种重复。已去重
               加 --ids：按已有 jsonl 的题号重生成 → 跟臂 2 同题号不变（本地验证 200 题题号完全一致）
建议           SFT 前用 --ids 重生成 sft_cd3 / sft_cd4（2 分钟，不用 GPU），老师不该自己犯「重复」
```

## SFT 数据体检结果（check_sft.py，GS01，2026-09-05）
| 文件 | 条 | token 中位/P90/最长 | 标注/条 | 标注精度 | 重复形态 | 说 perfect | 含 ×÷ | 命中层 |
|---|---|---|---|---|---|---|---|---|
| sft_cd3（穷举器） | 2000 | 121 / 385 / 661 | 3.0 | **1.00** | 0.00 | 1.00 | 0.29 | 1411/300/289 |
| sft_cd3_self（自采样） | 1530 | 122 / 172 / 351 | 1.8 | **0.77** | 0.05 | 0.56 | **0.12** | — |
| sft_cd4（穷举器） | 2000 | 169 / 658 / 1547 | 6.5 | **1.00** | 0.00 | 1.00 | 0.28 | 1438/360/202 |
| sft_cd4_self（自采样） | 1394 | 166 / 253 / 428 | 3.1 | **0.48** | 0.13 | 0.38 | **0.02** | — |

### 读法
```
① 同一批题，两份教材差在三处：精度（1.00 vs 0.77 / 0.48）、重复（0 vs 5% / 13%）、乘除（28% vs 12% / 2%）
② ★ 自采样 CD-4 的精度 0.48：模型【答对的】轨迹里，中间行仍有一半算错或标反。答案对不等于过程对。
   臂 2 的 SFT 会把这一半错的中间行当教材喂回去 —— 自蒸馏教的是模型的习惯，连坏习惯一起
③ ★ 自采样 CD-4 含乘除只有 2%：模型答对的 CD-4 题几乎全是纯加减题。穷举器在同一批题上 28% 用了乘除
   → 臂 2 的数据对「换运算符」这一课是零；臂 1 有 560 条示范。这是两臂差别最硬的一条
④ 自采样精度比模型平均（0.42~0.53）高，是选择效应：答对的轨迹错得少。CD-3 0.77 > CD-4 0.48：题越难中间越乱
⑤ 穷举器每条 6.5 条标注 vs 自采样 3.1：老师搜得系统，学生碰得早
⑥ 说 perfect：穷举器 100%（每条都命中收尾）；自采样 56% / 38%，另一半直接写答案不宣布
```
### 对预测的补充
```
臂 2 SFT 后   精度不会涨（教材本身 0.48~0.77），可能持平或略掉；×÷ 尝试率不会涨；长度更短（教材 P90 172 / 253）
臂 1 SFT 后   精度看能不能跳到 0.85+；×÷ 尝试率应从 34.6% 往上；长度 P90 应过 400
```

## 臂 2 SFT 完成（自蒸馏，2026-09-05）
```
294 步，25 分钟，10.1 GB，|g| 2~3。val loss 0.1535 → 0.1511 → 0.1505，训练 loss 0.10~0.17 全程平
★ 对照臂 1 冒烟：0.655 → 0.19。差 20 倍。
   自己的轨迹本来就在自己的分布里 → 交叉熵一开始就低 → 没什么可学 → 权重几乎没动。这就是「自蒸馏没有新零件」在 loss 上的样子
```
### SFT 后贪心 [2, 3, 5, 7] → 31
```
2 + 3 + 5 + 7 = 17 (too small)      6 行只有 3 种形态，3 行重复（2+3+7+5、5+7+2−3、7+5+3−2 都是前面写过的）
2 + 3 + 7 + 5 = 17 (too small)      算术全对，方向全对；词是 too small，它自己的说法
2 + 5 + 7 - 3 = 11 (too small)
3 + 5 + 7 - 2 = 13 (too small)
5 + 7 + 2 - 3 = 11 (too small)
7 + 5 + 3 - 2 = 13 (too small)
<answer> (7 + 5) * 3 - 2 </answer>  ← 没算就交：(7+5)×3−2 = 34 ≠ 31。没有 </think>，没有 perfect 宣告，最后一步是碰运气
```
老习惯原样：重复、不看上下界、乘法只在最后猜一下。GSM 那道对（36），格式干净。
★ 跟预测一致：臂 2 的 SFT 什么都没改。它的价值全在「对照」—— 之后 RL 涨多少，那就是 RL 本身的份

## 臂 1 SFT 完成（程序撒种，2026-09-05）
```
358 步，30 分钟，11.5 GB，|g| 1~2。val loss 0.655 → 0.077 → 0.075（臂 2 是 0.154 → 0.151）
loss 掉到 0.05~0.09：穷举器的文本高度确定，模型把「形」学得很像
```
### SFT 后贪心 [2, 3, 5, 7] → 31（700 token 截断，没写出 answer）
```
第 0 层  ✓✓  8 种符号形态一个不重、顺序跟老师一模一样（按减号数、按值降序）；范围句「from -3 to 17, so we need * or /」正确
         算术 7/8 对：7 + 2 − 5 − 3 写成 −1（实为 1）
第 1 层  ✗   骨架在（列乘积 → 评远近 → 试组合），内容漂了：
         · 乘积表少了 5×2 = 10 和 3×2 = 6 —— 正是解 7×3 + 5×2 需要的那个；多出老师没有的三连乘、四连乘
         · 「None of these, except 105, is close to 31」—— 远近判断错（近的是 35、30、21）
         · 然后陷入退化循环：7*5*3*2 / 2、/3、/5、/7、/2*3、/2*5 … 数字重复用，违反「各用一次」，直到截断
         · 算术倒是全对（210/6 = 35、210/14 = 15 …）
第 2 层  —   没到。解在这层（两两成块），它没走到
GSM      ✓   36，格式干净，</think> 在
```
### 判读
```
① 机械的部分传过去了：固定顺序穷举、不重复、范围句、算术精度明显上升（约 24 次计算错 1 次，之前一半错）
② 要「判断」的部分没传：老师第 1 层按「离目标近的先试」排序，那需要先算全再排，模型执行不了 —— 写穷举器时就担心的泄题/不可执行问题在这里现形
③ 第 2 层示范只有 200 条，2 epoch 不够；模型在第 1 层就绕进循环，根本走不到第 2 层
④ 退化循环是低多样性 SFT 的经典毛病：抓住一个模板「210 / x」反复套。老师从不重复用数，它没学到这条约束
⑤ ★ 这就是冷启动 SFT 模型的常态 —— R1 阶段 ① 之后的模型也不能直接用，要 RL 来修。循环轨迹得 0 分，RL 会压它；
   第 2 层的尝试若偶尔出现且得分，RL 会放大它。这一步 RL 的任务比臂 0 明确得多
```
### 800 条评测前的预测（贪心一条不算数，先写下）
```
pass@1 可能【掉】到 0.30~0.40（循环 + 截断没 answer）；pass@8 0.60~0.70；标注精度 > 0.85；长度中位 > 300；撞顶 > 5%；全错 30 上下
若 pass@8 反而涨、且 scan_ops 里 ×÷ 题开始有对的 → 种活了，RL 有东西放大
若循环占多数（撞顶 > 20%）→ 下一版穷举器把第 1 层改成纯机械（所有两数乘积按固定顺序各配一遍，不排序不评远近），再 SFT
```

## 教学：SFT 之后 RL 修什么（2026-09-05）
```
一句话   SFT 看局部：每个 token 像不像老师。RL 看结果：整条轨迹得没得分。
         循环「210 / x」每一行局部都很像老师，所以 SFT 多训几遍也去不掉；只有看结果的信号知道它没写出答案
```
| 臂 1 SFT 后的毛病 | RL 怎么修 | 为什么 SFT 修不了 |
|---|---|---|
| 退化循环，撞顶没 answer | 得 0 分，组内最低 → 压 | 每行局部都像老师，token 级 loss 不觉得错 |
| 数字重复用（7*5*3*2 / 2） | 数字比例分 < 1 → 分低 → 压 | 「各用一次」老师从没说，只是从没违反。SFT 学不到「不该做什么」，教材里没有反例 |
| 「离目标近的先试」执行不了，乘积表漏项 | 不教排序。哪次碰巧列全、试对了得 1 分 → 放大那种走法 → 漂向一个能执行的替代（比如按固定顺序全试） | 老师的启发式要先算全再排，模型做不到；SFT 只能让它「看起来像在排」 |
| 不知道何时停 | 撞顶 0 分 + 长度软惩罚 → 学会在 1024 前收尾 | 老师每条都在命中处停，没示范「搜不到怎么收」 |
| 第 2 层很少走到 | 难题上 16 条里若有 1 条走到第 2 层并对，A ≈ +0.94，最强的推力 → 从偶尔到可靠 | 200 条示范只够让它「有时会」 |
```
再深一层
  SFT 把模型挪到「好程序【够得着】」的区域（概率非零）—— 但同时也种下了杂草（循环、漏项），因为模仿不完美
  RL 把它挪到「好程序【可靠】、杂草被压」的区域。撒种撒的是一把混合的种子，筛子分好坏
  SFT 只能教教材里【有】的；教材里【没有】的（反例、约束、停止条件）只有奖励能教 —— 这是 negative space
风险
  若系统搜索经常循环、而老式碰运气更常得分，RL 会把种子冲掉、退回短轨迹。这就是「SFT 后测量点」存在的理由：
  RL 后长度回落到 250 以下、精度回到 0.5 → 冲掉了。len-soft 800 是给种子留的余地
```

### 用户复述：「SFT 教过程像，RL 教把结果算对、真正学会过程的能力」—— 对一半，我的说法
```
更紧的说法   SFT 把 p 从 0 变成非零（够得着）；RL 把 p 从非零变成可靠（靠得住）。两个都不「教能力」
像 vs 真会    像是 SFT 的机制（模仿），能不能变成真会取决于那段做法模型执行得了不：
              机械的（8 种符号形态、范围句）→ 一学就真会；要判断的（先算全再排序）→ 只学到像
RL 的角色     不是「教会过程」，是「验真」：每一段模仿过来的东西拿结果审一遍，真会的留、只像的压或换掉
谁都不做的    模型没有的计算（心算精度、排序）—— 预训练 / 工具的事
```
```
回复原文补记
  SFT 把 p 从 0 变成非零，RL 把 p 从非零变成可靠。两个都不教能力
  像是 SFT 的机制。像能不能变真会，看那段做法模型执行得了不：机械的一学就真会（第 0 层 8 行是它自己算的不是背的）；
  要判断的只学到像（「except 105 is close」）
  RL 是验真：模仿来的每一段拿结果审一遍，真会的留，只像的压掉或换成能执行的替代。RL 之后剩下的程序一定是模型真能跑的
  谁都不做的：心算精度、排序。臂 1 精度上去是因为老师的算式短而规整，不是 SFT 教会了算术
  四段论：SFT 是把零件按图纸摆一遍，摆得像不像看图纸，摆得动不动看零件；RL 是摆完试运转，转得动的留，转不动的拆
```

## 教学：没有工具，预训练怎么让模型会算、会排序，以及为什么不精（2026-09-05）
```
机制    预训练只做一件事：预测下一个 token。语料里有海量「36 + 42 = 78」（课本、作业、代码、表格）。
        要把「= 」后面的 78 预测对，梯度下降只能在权重里长出一个加法电路。电路不是设计的，是被 loss 逼出来的。
        可解释性研究真的找到过：模加法学成傅里叶特征（Nanda 2023）；大模型的算术是一堆按数值范围触发的启发式拼起来的，
        不是干净的算法（Nikankin 2024「Arithmetic Without Algorithms」）。
为什么不精（三条）
  ① 每个 token 的算力有限   Transformer 深度固定，一个 token 只能做有限步串行计算。多位数带进位是串行的，
                            一口气吐「= 54」超出预算就错。★ 思维链的根本作用：把一个难 token 拆成许多易 token，
                            用输出当草稿纸（Li 2024：有 CoT 才能解本质串行的问题）
  ② 学到的是启发式不是算法   「两个 40 多相加约 80 多」这类规则拼出答案，小数常见对得多，大数、多项、带进位就滑。
                            我们 1.5B 的 6+18+42−12 写成 48 就是滑了一次
  ③ 数据里的频率           7×8 出现几百万次，47×83 几乎没有。精度随操作数变大而掉
排序同理  比较几个数是每对都要看，一口气排 7 个超出单 token 算力；写出来一对对比就行。所以「先算全再排」老师做得到模型做不到
没工具怎么提高
  · 数据格式   scratchpad：一行一步、显式进位、低位在前。小模型这样练能把加法练到近 100%（Lee 2023「Teaching Arithmetic to Small Transformers」）
  · 合成数据   mid-training 灌几亿 token 算术（Qwen-Math 路线）—— 这是「知识」层面的活
  · 规模       7B 的启发式更多更细，小数近乎全对，大数仍错
  · 位置编码   数位对齐的编码能让加法长度外推（Abacus，2024）
对我们的意义
  · 穷举器的每行「7 * 5 + 33 - 14 = 54」是两步串行（乘、再加减）。改成一行一步「7 * 5 = 35; 35 + 33 = 68; 68 - 14 = 54」，
    每步落在 1.5B 可靠范围内，精度还能再涨 —— 这是不用工具、不用 mid-training 就能拿到的一截
  · 这也是「长思维链」存在的根本理由：计算发生在 token 里，算力有限的模型只能用长度换精度
```

### 用户复述：「预训练后要提高任何能力，SFT 先摆出来（0→非零），再 RL 验真（非零→可靠）」—— 对，加三个前提一个替代
```
成立的范围   「怎么做」的能力：程序、格式、agent 行为。R1 冷启动 SFT → RL，agent 模型 SFT 轨迹 → 环境 RL，都是这条
前提 ①      零件得在。SFT 只能摆模型有的零件；缺的计算（心算、排序）摆出来只是像，RL 会绕开或剪掉，能力不会长出来
             零件从预训练 / mid-training（数据）或工具（harness）来
前提 ②      验真器得在。判分器、环境、或奖励模型。没有验真器 RL 没法验真；用奖励模型时「真」是它定义的，会被钻
前提 ③      知识不走这条。几千条 SFT 装不进事实 → mid-training / 检索 / 工具
替代         0 → 非零不一定要 SFT：零件在的话提示词（开关）就够。Countdown 的顿悟就是模板 + RL，没 SFT
             五个旋钮的顺序：先换开关，不够再撒种
```

## 臂 2 SFT 后测量点（自蒸馏，RL 前，2026-09-05）
| | mix4-300 参照 | 臂 2 SFT 后 | 预测 |
|---|---|---|---|
| CD-4 pass@1 / @8 | 0.393 / 0.670 | **0.409 / 0.700** | 0.42 / — |
| CD-4 全对 / 全错 | 10 / 33 | 10 / 30 | 31~33 |
| CD-4 数字用全 | 0.951 | 0.959 | |
| CD-4 长度中位 / P95 / 撞顶 | 222 / 375 / 0.9% | 219 / 369 / 0 | 200 |
| CD-3 pass@1 / @8 | 0.476 / 0.660 | **0.511 / 0.720** | |
| CD-3 全对 / 全错 | 28 / 34 | 28 / 28 | |
| CD-3 组内有差异 | 50% | 58% | |
| GSM R1 / 中性 | 0.74 / 0.46 | 0.75 / 0.47 | 0.73 |
| 搜索词/条 CD-4 | 6.73 | 6.66 | |

```
① 基本 = mix4-300 + 一点点。最大的动静是 CD-3 pass@8 +0.06、全错 34 → 28，100 题 ±0.05，勉强算真
② 没退化：格式 0.99、续写 0.1%、撞顶 0、GSM 两模板原样
③ 为什么会有一点涨：自采样 SFT = 一轮离线的「专家迭代」（STaR / ReST 那类）—— 拿自己答对的轨迹再训一遍，
   本质是二值奖励的离线策略梯度。所以它不是「什么都没做」，是「做了一小步 RL」。臂 2 RL 后再涨多少，是在这个基础上算
④ 样例里老毛病照旧：6+12+18+42 又写成 76（这个错第三次见了 —— 稳定的错误信念，启发式那种）；
   6+18+42−12 写成 46；「Please check if my reasoning is correct」这种聊天腔漏出来一句
⑤ 预测对账：pass@1 0.41（预测 0.42）✓、全错 30（预测 31~33）✓、GSM 0.75（预测 0.73）✓、长度 219（预测 200）略长
待           pass@64（卡 3 跑中，预测 ≈ 0.83 不动）、analyze_search 精度（预测 ≤ 0.53）、scan_ops
```

## ★★★ 臂 1 SFT 后测量点（程序撒种，RL 前，2026-09-05）—— 种活了，天花板动了
| | mix4-300 | 臂 2 SFT 后 | **臂 1 SFT 后** | 我的预测 |
|---|---|---|---|---|
| CD-4 pass@1 / @8 | 0.393 / 0.670 | 0.409 / 0.700 | **0.506 / 0.720** | 0.30~0.40 / 0.60~0.70 |
| CD-4 全对 / 全错 | 10 / 33 | 10 / 30 | **22 / 28** | ~30 |
| CD-4 格式 / 数字用全 | 0.99 / 0.95 | 0.99 / 0.96 | **0.71 / 0.61** | |
| CD-4 撞顶 1024 | 0.9% | 0 | **28.1%** | > 5% |
| CD-4 长度中位 / 均值 | 222 / 224 | 219 / 224 | **344 / 512** | > 300 |
| CD-4 搜索词/条 | 6.7 | 6.7 | **15.3** | |
| CD-3 pass@1 / @8 | 0.476 / 0.660 | 0.511 / 0.720 | **0.599 / 0.930** | |
| CD-3 全对 / 全错 | 28 / 34 | 28 / 28 | **38 / 7** | |
| CD-3 撞顶 | 0 | 0 | 18.8% | |
| GSM R1 / 中性 | 0.74 / 0.46 | 0.75 / 0.47 | **0.755 / 0.565** | 0.72 |

### 五条读法
```
① ★★★ 天花板动了。CD-3 pass@8 0.93 > 600 的 pass@64 0.83 > Base 外推的渐近线 0.87。
   8 次抽样解 93 道，老模型抽 64 次只解 83 道；全错从 34（mix4）/ 17（600 @64）掉到 7。
   RL 900 步做不到的事，4000 条老师轨迹 + 30 分钟 SFT 做到了。「撒种抬天花板」第一次有数据
② 28% 的轨迹循环到撞顶得 0 分，pass@1 仍然 +0.11。去掉撞顶的那些，剩下的对率 ≈ 0.506 / 0.72 ≈ 0.70。
   不循环的时候它比老模型强得多；循环是杂草，RL 的第一刀
③ 杂草清单：格式 0.99 → 0.71（撞顶没 answer）、数字用全 0.95 → 0.61（重复用数 + 截断）、
   乘积表偶尔写出题里没有的数（样例二：36 (12*3)、8 (42−34)…）。全是 RL 能看见的：都扣分
④ ★★ 执行错在解上：[6,12,18,42]→30，三条样例都按老师顺序列到第 6 行「42 + 12 − 18 − 6」，三条都算成 28（实为 30），
   于是三条都错过了解。程序对了，算术在关键一行滑了，而且是同一个稳定的错。这是 1.5B 心算天花板最干净的一个样本，
   也是计算器（选项 3）最直接的理由
⑤ ★ GSM 中性模板 0.46 → 0.565（z = −2.08，训好 42 / 训坏 25）—— 第一次有东西漏过模板边界。
   臂 2 的 SFT 权重几乎没动（val 0.154 → 0.151）中性列也没动（0.47）；臂 1 权重动得多（0.655 → 0.075）中性列就动了。
   假说：条件性绑定是 RL 的性质（小步、按模板条件化的更新），SFT 的更新更「整体」，会漏。p ≈ 0.04，待 RL 后复测
```
### 预测对账
```
pass@1 方向猜反了（猜掉、实涨 0.11）：低估了不循环那部分的准头。pass@8 0.72 在预测上沿。撞顶 28% > 20%
按跑前规则「撞顶 > 20% → 下一版穷举器第 1 层改纯机械」—— 触发。但先按计划跑 RL：RL 修不修得掉循环，本身就是要测的
待：pass@64（卡 1，预测 CD-3 ≥ 0.95）、analyze_search 精度（预测 0.8+）、scan_ops
```

## 臂 2 SFT 后 pass@64（CD-3 池外，2026-09-05）
| k | Base | mix-600 | **臂 2 SFT 后** |
|---|---|---|---|
| 1 | 0.016 | 0.456 | **0.504** |
| 8 | 0.121 | 0.694 | 0.703 |
| 16 | 0.224 | 0.748 | 0.739 |
| 32 | 0.387 | 0.795 | 0.763 |
| 64 | 0.590 | **0.830** | **0.790** |
| 全对 64/64 | 0 | 1 | **13** |
| 全错 0/64 | 41 | 17 | **21** |

```
① 天花板没动，还略低：0.83 → 0.79（100 题差 4 道，边缘）。预测「≈ 0.83 不动」基本对
② 收紧的形状：小 k 涨（pass@1 +0.05、全对 1 → 13），大 k 不涨（32 → 64 只 +0.027），尾巴略差（全错 17 → 21）
③ 跟理论一致：自蒸馏没有新零件 → 抬不了 pass@∞。它把已有的质量往已经会的题上堆
④ 这个 0.79 就是臂 1 pass@64 的对照数。臂 1 pass@8 已 0.93，pass@64 预测 ≥ 0.95 → 差 0.16 = 新零件在天花板上的份
```

## ★★★ 臂 1 SFT 后 pass@64（CD-3 池外，2026-09-05）—— 天花板到顶：1.00
| k | Base | mix-600 | 臂 2 SFT 后 | **臂 1 SFT 后** |
|---|---|---|---|---|
| 1 | 0.016 | 0.456 | 0.504 | **0.598** |
| 8 | 0.121 | 0.694 | 0.703 | **0.926** |
| 16 | 0.224 | 0.748 | 0.739 | **0.972** |
| 32 | 0.387 | 0.795 | 0.763 | **0.991** |
| 64 | 0.590 | 0.830 | 0.790 | **1.000** |
| 全对 64/64 | 0 | 1 | 13 | **23** |
| 全错 0/64 | 41 | 17 | 21 | **0** |
| 撞顶 | 9.9% | 1.1% | 0 | 17.0% |

```
① 100 道池外 CD-3，64 次里每道至少对一次。老天花板（600 的 0.83、Base 外推 0.87、臂 2 的 0.79）被穿过去
   臂 1 抽 16 次（0.97）就超过所有前辈抽 64 次。臂 1 − 臂 2 在 k=64 上 = 0.21（跑前估 0.16）
② ★★★ 目标 6 结案：撒种抬 pass@∞，RL 不抬。两句话现在都有数：
   RL：Base → 600，pass@64 0.59 → 0.83，曲线约 k=500 汇合（把 500 次压成 8 次）
   撒种：600/mix4 → 臂 1，pass@64 0.83 → 1.00，而且还没 RL —— 零件是 4000 条老师轨迹给的
③ 还带着杂草：撞顶 17%、格式 0.83、数字用全 0.70。RL 的活变得非常明确：可达集已经是全集，把 pass@1 0.60 往上推，把循环压掉
④ CD-4 的天花板没这么漂亮（pass@8 0.72 vs 0.67）：4 数的第 1/2 层轨迹长，模型走到一半就循环。RL 后看；不行上机械版老师
预测对账   pass@64 ≥ 0.95 → 1.00 ✓
```

### 教学：「RL 不抬天花板、撒种抬」用 p 说一遍（2026-09-05）
| 题 | Base p | 600 p（RL 后） | 臂 1 p（撒种后） | 谁动了它 |
|---|---|---|---|---|
| A 简单 | 0.5 | 0.95 | 0.95 | RL：非零 → 可靠 |
| C 难但够得着 | 0.002（500 次碰一次） | 0.1（8 次碰一次） | 0.3 | RL：把 500 次压成 8 次 |
| D 够不着（要系统穷举 / 换运算符） | 0 | 0 | **0.05** | ★ 只有撒种：0 → 非零 |
```
天花板 = pass@∞ = p > 0 的题占几成。Base 和 600 的 D 都是 0，所以两条曲线 k 够大时汇合（外推 k≈500，≈0.87 = A + C 的份）
RL 为什么碰不了 D：D 的 16 条全 0 分，组内无差异，梯度为零。RL 只能放大抽到过的东西，D 从没被抽到过
撒种为什么能：老师的轨迹里写着到 D 的走法，SFT 逼模型模仿 → 走法进了采样分布 → D 有了 p。这一步不需要奖励，所以「还没 RL」就动了
之后 RL 的活：把 D 的 0.05 推到 0.5，把 C 的 0.3 推到 0.9，把循环压掉。可达集不变（已经是全集），变的是 pass@1
两段合起来：撒种把圈画大，RL 把圈里的点打实。只撒不筛 = pass@1 0.60 带 17% 循环；只筛不撒 = 天花板 0.87
```

## SFT 后测量点汇总（臂 1、臂 2 各四项，2026-09-05）
| 测试 | 指标 | mix4-300 参照 | 臂 2 自蒸馏 SFT | 臂 1 程序撒种 SFT |
|---|---|---|---|---|
| CD-4 池外 800 | pass@1 / pass@8 | 0.393 / 0.670 | 0.409 / 0.700 | **0.506 / 0.720** |
| | 全对 / 全错 | 10 / 33 | 10 / 30 | **22 / 28** |
| | 格式 / 数字用全 | 0.99 / 0.95 | 0.99 / 0.96 | 0.71 / 0.61 |
| | 长度中位 / 撞顶 | 222 / 0.9% | 219 / 0 | 344 / **28.1%** |
| | 搜索词/条 | 6.7 | 6.7 | 15.3 |
| CD-3 池外 800 | pass@1 / pass@8 | 0.476 / 0.660 | 0.511 / 0.720 | **0.599 / 0.930** |
| | 全对 / 全错 | 28 / 34 | 28 / 28 | **38 / 7** |
| | 撞顶 | 0 | 0 | 18.8% |
| CD-3 pass@64 | pass@64 / 全错 64 次 | （600：0.830 / 17） | 0.790 / 21 | **1.000 / 0** |
| GSM test[0:200] | 训练模板 / 中性 | 0.74 / 0.46 | 0.75 / 0.47 | 0.755 / **0.565** |

```
臂 2   = mix4 + 一小步（一轮离线专家迭代）。天花板不动。零退化
臂 1   天花板到顶、pass@1 +0.11 / +0.12、全对翻倍；代价是 17~28% 循环撞顶、格式和数字用全掉。GSM 中性第一次涨
待     analyze_search（精度）、scan_ops（全错构成）、RL 后四臂并表
```

## analyze_search：mix4 / 臂 1 / 臂 2，CD-4 池外 800（2026-09-05）
| | mix4-300 | 臂 2 SFT | 臂 1 SFT |
|---|---|---|---|
| ② 1~2 次桶 正确率（条数） | 0.690（145） | 0.698（159） | **1.000**（17） |
| ② 3~5 次桶 | 0.657（181） | 0.578（192） | **0.976**（210） |
| ② 6+ 次桶 | 0.202（466） | 0.235（443） | 0.320（571，长 658） |
| ③ 说 perfect 时真对 vs 全体 | 0.443 vs 0.393 | 0.484 vs 0.409 | **0.780 vs 0.506** |
| ③b 标注数 / 条 | 5.1 | 4.9 | **13.6** |
| ③b 标注精度（全部） | 0.456 | 0.464 | **0.484** |

```
① 臂 2 = mix4，每个探针都一样。对照臂稳
② 臂 1 的程序只要在 5 次内收尾，几乎必对（1.00 / 0.976，对 mix4 的 0.69 / 0.66）。全部失败集中在 6+ 桶：
   571 条（71%）、均长 658 —— 4 数题只要加减穷尽就至少 8 行，进 6+ 桶；循环也在这桶。RL 要修的就是这一桶
③ ★ 说 perfect 的可信度 0.44 → 0.78：假宣告从 56% 掉到 22%。老师只在真命中时写 perfect match，模型学到了这个绑定
④ ★ 标注精度 0.484 —— 预测 0.85 落空。但这是【全部标注】：13.6 条/轨迹里大半来自 6+ 桶和循环，循环里的方向标注是垃圾
   （「70 (too low)」对目标 31）。已给 ③b 加分层：没撞顶 / 撞顶、答对 / 答错、算错 / 方向反。重跑一次看「没撞顶」那行
   ★ 预测改为：没撞顶那行 ≥ 0.75；撞顶那行 < 0.4 且以「方向反」为主
⑤ 剂量响应、⑤ ★差、① 宽泛 ★差 三个探针在臂 1 上都失效（对照组只剩 2~19 条），不读
```

### ③b 分层结果（臂 1，CD-4 池外）+ 一个修补
| 层 | 标注数 | 精度 | 算错 | 方向反 |
|---|---|---|---|---|
| 全部 | 10873 | 0.484 | **0.502** | **0.015** |
| 答对的轨迹 | 1501 | 0.728 | 0.269 | 0.003 |
| 答错的轨迹 | 9372 | 0.445 | 0.539 | 0.016 |
```
① ★★ 错几乎全是算错，方向反只有 1.5%。老师教会了「按算出来的值标方向」，没教会「算对」—— 算术是零件，SFT 装不进
② 答对的轨迹里也有 27% 的行算错：程序能容错，算错一行不一定错过解，但解那一行算错就错过（42+12−18−6 = 28 那种）
③ 86% 的标注在答错的轨迹里（长、循环），那里一半算错 → 循环里的算术更烂，长了就乱
④ 「撞顶」那行没分出来：raw 文件没存 max_new。已修（analyze_search 用文件里最长的当上限；baseline 以后存 max_new）。重跑看撞顶/没撞顶
结论   精度问题 = 心算问题，方向已不是问题。这把选项 3（计算器）和「一行一步」的老师格式都推到前面
```

## 臂 1 RL 前 53 步（从 out_sft1 起，2026-09-05）
| eval（贪心，CD [90000:90100]） | 起点 | step 25 | step 50 |
|---|---|---|---|
| CD 正 | 0.52 | 0.48 | 0.49 |
| CD 长 | 487 | 505 | **269** |
| 标注/条 @ 精度 | 12.3 @ 0.64 | 12.8 @ 0.65 | **6.2 @ 0.70** |
| GSM 正 / 长 | 0.82 / 159 | 0.82 / 159 | 0.82 / 158 |
| 总分 | 0.627 | 0.626 | 0.720 |

```
① 长度 50 步砍一半（487 → 269），标注数减半（12.3 → 6.2），精度反而升（0.64 → 0.70）→ 剪的是循环（垃圾标注多），不是程序
② 正确率贪心 0.52 → 0.49 在 100 题噪声里；总分涨主要是长度惩罚少了。真正的正确率涨要等循环剪完之后
③ 训练行里 CD「分」常远低于「正」（step 8：0.15 vs 0.36）—— len-soft 800 在狠罚超长轨迹，这就是剪循环的刀
④ KL 0.06~0.10，比臂 0 高：SFT 后的模型离 Base 远。GSM |A| 0.0~0.3 比臂 0 大，GSM 被 SFT 搅动了一点，eval 0.82 没掉
⑤ 每步只有 4 道 CD，单步的正确率（0.06 ~ 0.95）不能读；rollout 8 s / 30 s 交替 = 有没有撞顶的批
判据    长度稳在 250~300、标 6~8、精度 ≥ 0.7 → 剪完了，之后看正确率；长度跌破 200、标 < 4 → 在冲种子
动作    训练时腾 3 GB 看 step 50 的 ckpt_best 怎么写 [2,3,5,7] 和 [6,12,18,42]，肉眼定「剪」还是「冲」
```

### ③b 撞顶分层出来了（臂 1，CD-4 池外）
| 层 | 轨迹 | 标注数 | 每条 | 精度 | 算错 | 方向反 |
|---|---|---|---|---|---|---|
| 没撞顶 | 575 | 4333 | 7.5 | **0.552** | 0.438 | 0.010 |
| 撞顶 | 225 | 6540 | **29** | 0.439 | 0.544 | 0.018 |
| 答对的 | 405 | 1501 | 3.7 | 0.728 | 0.269 | 0.003 |

```
① 28% 的循环轨迹贡献了 60% 的标注（每条 29 个）。它们不是纯垃圾，44% 的行算对，只是漫无目的
② ★ 没撞顶的程序本身精度只有 0.55，预测 0.75 落空。4 项加减、乘积 ± 两项，1.5B 每行错四成多。答对的轨迹里也错 27%
③ 方向反 1% —— 方向问题彻底没了。剩下的全是算术
④ 这解释了 pass@1 0.6 vs pass@64 1.0 的差：程序覆盖了全部，但每行是个 55% 的硬币；解那一行算错就标 too low 走过去了。
   64 次里总有一次解那行算对 → 1.00；一次抽样 → 0.6
下一刀（RL 之后定）
   a. 老师改「一行一步」：42 + 12 = 54; 54 − 18 = 36; 36 − 6 = 30。两操作数一步，1.5B 该到 0.9+。代价 token ×2~3。不用工具、不用新基建
   b. 计算器（选项 3）：精度 → 1.0，顺便搭 agent 的 harness
   c. 算术 mid-training
待   mix4 的 ③b 分层（同一份输出上面那块）—— 看 SFT 把「算错 / 方向反」的构成改了多少
```

### mix4 的 ③b 分层 —— 我猜错了：方向反从来不是主因
| | mix4 没撞顶 | 臂 1 没撞顶 | mix4 答对的 | 臂 1 答对的 |
|---|---|---|---|---|
| 精度 | 0.452 | **0.552** | 0.499 | **0.728** |
| 算错 | 0.516 | 0.438 | 0.482 | 0.269 |
| 方向反 | 0.032 | 0.010 | 0.019 | 0.003 |
```
① 我猜 mix4「方向反远不止 1%」—— 错。只有 3%。算错从来就是主因（51%）。之前样例里「= 30 (perfect match)」那种是算错不是标反
② SFT 的净效果：算错 0.52 → 0.44（老师的式子规整：正项在前、降序、短），方向反 3% → 1%，答对轨迹里算错 0.48 → 0.27
③ ★ 答对轨迹精度 0.50 → 0.73 的含义：mix4 时成败由「搜没搜到」决定，算术在对错两边一样烂；
   臂 1 时程序已经不缺，成败改由「解那行算没算对」决定 —— 瓶颈从策略换成了算术。这是撒种把问题往前推了一层
④ 两代模型、四个分层，算错率都在 0.27~0.54 之间。1.5B 对 3~4 项算式的心算就是这个水平，跟训练方式无关
```

### 「那怎么学会算术」—— 五条路（2026-09-05）
| 路 | 做法 | 本质 | 精度能到 | 代价 |
|---|---|---|---|---|
| ① 一行一步 | 老师改写：42 + 12 = 54; 54 − 18 = 36; 36 − 6 = 30。每步两个操作数 | 不教算术，把每步缩到它已经会的范围（草稿纸原理） | 两操作数小数 1.5B 约 0.9+ | 穷举器改一处 + 再 SFT 30 分钟；token ×2~3 |
| ② 计算器 | 模型写 <calc>，harness 算 | 外包 | 1.0 | rollout 中断-注入 1 天；顺便搭 agent 基建 |
| ③ 过程奖励 | 判分器逐行验标注，算对的行给分 | RL 放大已有的 0.55 | 0.7~0.8 顶 | 改奖励 1 小时；有被钻的风险（少写行） |
| ④ 算术 mid-training | 几亿 token 合成算术，scratchpad 格式，loss 全算，混通用文本防忘 | 真改电路 | 小数近 1.0，大数仍掉 | 数据几分钟；训 5 小时左右 |
| ⑤ 换 7B | — | 买 | 高 | 4×4090 训不动全参 |
```
顺序    ① 先（最便宜、当天见效、不用新基建）→ ② 是结构解 → ④ 想让模型自己拥有算术时做 → ③ 作为 RL 侧的补充随时可加
一句话  固定大小的模型「学会算术」只有三种：把步子缩小、把电路练出来、把活外包。RL 只能放大已有的准头
```
决定（2026-09-05）  用户：方案记录，先不动。等臂 1 RL 跑完、四臂并表之后再选 ① 还是 ②

## scan_ops：臂 1 SFT 后的 28 道全错（CD-4 池外，2026-09-05）
```
① 28 = 8 道纯加减 + 20 道必须乘除
   8 道纯加减：模型每条都列全了 8 种形态，8 次全错 → 解那一行 8 次都算错。同一个式子稳定算错（42+12−18−6 = 28 那种）→ 纯算术
   20 道乘除：98% 的轨迹试过乘除（157/160），混合形态 46.6%（mix4 时 34.6%）→ 形态覆盖已经不是问题，问题在执行
② 两条 [5,7,14,33]→43 的尸检（解：33 + 14×5/7），五种失败：
   a 第 0 层算错 3/8 和 1/8 行，范围句跟着错（「17 到 61」）
   b 乘积表不全、带重复、用重复的数：「14*5, 14/5=2.8, 70/5」「14*14」「14*7 列了四遍」；33 的乘积一个没列
   c 第 2 层丢了第四个数：「5 + 7 = 12, then 12 * 14 = 168」—— 33 没用，违反各用一次；老师的格式是三数成链再配剩数
   d 幻觉出老师没有的标题（「another level deeper」「plus or minus the remaining numbers」）→ 把前面的块再抄一遍 → 循环
   e 自己发明了「(not an integer, skip)」—— 把老师隐含的整数规则说出来了，这条是好的泛化
③ 各层学到的深浅：第 0 层机械、学得最好（算术除外）；「列全 6 个两数乘积」这一步最弱 —— 示范只有 360 条，且它要枚举 C(4,2)
④ 对 RL：b c d 都会扣分（数字比例 / 撞顶），RL 能剪；a 剪不了，8 道纯加减题 RL 后大概率还全错
⑤ 对老师 v2（先记着不动）：
   · 一行一步（算术）
   · 第 1 层纯机械：6 个两数乘积按下标顺序全列，不排序、不写「skip」句 —— 去掉所有要判断的句子
   · 第 2 层每行显式带上第四个数（「… then 10 + 33 = 43」已是，但要强调 d 必须出现）
   · 第 1、2 层示范加倍（过采样），现在 360 / 202 条不够
```

### 教学：p、pass@1、pass@k、pass@∞、可达集 —— 一次说清（2026-09-05）
```
p          【单题】一次抽样答对的概率。一道题一个 p。你说的对：p 就是这道题自己的 pass@1
pass@k     单题：1 − (1−p)^k。整套题：把每道题的这个数平均
pass@1     = 所有题的 p 的平均
pass@∞     单题：p = 0 → 0；p > 0 → 1（抽够多总会碰到）。整套题：p > 0 的题占几成 = 可达集的大小
可达集     p > 0 的题的集合（多抽几次能做出来的题）。全集 = 测试集全部 100 道。「可达集 = 全集」= 100 道每道 p > 0（pass@64 = 1.00 证明的）
可靠       p 大到一次就够 = pass@1 高

硬币比喻   每道题一枚硬币，p 是正面概率。pass@k = 抛 k 次至少一次正面
           撒种：给「两面都是反面」的硬币（p = 0）造出一个正面（p 变非零）→ 能出正面的硬币变多 → pass@∞ 涨
           RL：把已有正面的硬币掰得更偏正（0.1 → 0.9）→ 需要的抛次数变少 → pass@1、pass@8 涨；两面反的它掰不动 → pass@∞ 不动
           k 越大，看的越是「有没有正面」而不是「多偏」，所以 Base 和 600 的曲线 k 大了汇合（正面硬币数一样），臂 1 不跟它们汇合（正面硬币更多）

复述改写   ✗「pass@1 从 0 到非 0 靠撒种」→ ✓「单题的 p 从 0 到非 0 靠撒种（零件在的话换开关也行）」
           ✓「p 从非 0 到可靠：RL 是专门干这个的（用结果信号），SFT 也能推一段（臂 1 pass@1 0.39 → 0.51）」
           ✓「pass@大 k 基本只看 p 是不是 0 → 只有撒种能动它；RL 对大 k 只动一点（C 类题），对 ∞ 不动」
```

## 臂 1 RL step 150 的三道贪心样例 —— 既不是剪也不是冲，是【改写】（2026-09-05）
```
[2,3,5,7]→31     第 0 层 8 行全对（SFT 时 7/8，−1 那行修好了）。之后 </think><think> Only * and /: 两行 → 「7*5−3+2 = 31 (perfect match)」实为 34。错，假宣告
[6,12,18,42]→30  第 0 层 5/8 对，解那行仍写 28（稳定错）。Only * and /: 「42*12/6−18 = 30 (perfect match)」实为 66。错，假宣告
[5,7,14,33]→43   第 0 层 8/8 对。Only * and /: 只试 33 × 别的（33*7、33*5、33/7），从不试 14*5、14/7 → 解 33+14*5/7 够不着
                 再幻觉出「Combine +, -, *:」「Combine *, +, -:」两段，只用 3 个数，撞 512（demo 上限）
```
### 三条判读
```
① 穷举那 8 行活下来了，而且算得比 SFT 时准（8/8、5/8、8/8）→ 第 0 层是「剪」
② 穷举之后的部分被 RL 改写了：范围句没了、乘积表没了、「skip」句没了，换成一个自己发明的「</think><think> Only * and /:」段，
   直接试「最大数 × 其他 ± 其余」。老师那些要判断的句子（列全、排序、剪枝）没有回报、还常错 → 被选掉；一个能执行的窄启发式顶上
   → 这正是「只像的压掉，换成能执行的替代」。代价：第 1 层从 6 个乘积缩到最大数的 2~3 个，第 2 层没了 → 可达集会缩回去一些
③ ★ 假 perfect match 回来了：三道两道假宣告。原因是奖励：写个错式子 0.25 > 循环撞顶 0 → RL 教会它「一定要收尾并宣告」。
   SFT 时假宣告 22%，RL 把「诚实循环」换成了「自信错答」。这是阶梯部分分的副作用第二次现形（第一次是「放弃也交式子」）
```
### 预测 RL 后（300 步）
```
pass@1 0.55~0.60（比 SFT 后略涨，剪循环的功劳）；pass@8 略掉；★ CD-3 pass@64 从 1.00 掉到 0.90~0.95（第 2 层被改写掉）
撞顶 < 5%；假 perfect 率 > 30%；标注精度 ≈ 0.6；长度中位 250~300
```
### 想到的两件事（先记不动）
```
· 过程奖励：判分器能验「X = target (perfect match)」是不是真等于 target。假宣告扣分 → RL 就不会学「自信错答」。这是选项 ③ 的最小版本，改奖励半小时
· 老师 v2 的第 1 层要机械到 RL 也不想改写它：6 个乘积按下标全列、每个都试，不给任何判断句让它选掉
```

### 设计稿（先记不动）：过程奖励、老师 v2、样本量（2026-09-05）
```
一、过程奖励（最小版）—— 判分器本来就能逐行验，Countdown 的独特便利
   现在   r = 格式 .05 + 解析 .05 + 数字比例 .15 + 对 .75 − 长度罚
   加两项  ① 假宣告：抓「expr = V (perfect match)」，算 expr；≠ 目标 → −0.2
              效果：自信错答 0.25 − 0.2 = 0.05 ≈ 交白卷。恢复「对 > 诚实没解出 > 自信错答」的顺序。
              会不会学「干脆不写 perfect」？会，但无害：答案对不对不靠宣告，靠 <answer>
           ② 标注错误率：算错或标反的行 / 总标注行 × −0.1（用率不用数，避免「少写行」钻空子）
              效果：轻推「算准、或写更容易算准的式子」
   改哪   grpo_mix_mp.score() 里 cd 分支，annot_stats 已有，加十几行。半小时
   属性   这是「可验证的过程奖励」—— 数学题上要训 PRM 模型才有的东西，这里判分器免费给

二、老师 v2 —— 「机械」的定义：★ 老师写的每一行都必须是一次【能命中的尝试】，不写任何判断句
   现在第 1 层：一行乘积表 → 「X, Y cannot reach…skip」→ 按最近排序试。三样都是判断句，RL 全选掉了
   v2 第 1 层：6 个两数乘积按下标顺序（大的在前）逐个来，每个配剩下两数的 4 种 ±，一个不跳、不排序：
       7 * 5 = 35: 35 + 3 + 2 = 40 (too high); 35 + 3 - 2 = 36 (too high); 35 - 3 + 2 = 34 (too high); 35 - 3 - 2 = 30 (too low)
       7 * 3 = 21: 21 + 5 + 2 = 28 (too low); …
       （整除商放在同一格里；一行一步版把 35 + 3 + 2 拆成 35 + 3 = 38; 38 + 2 = 40）
   v2 第 2 层：两两成块按固定拆分顺序（3 种拆法 × 运算组合），每行显式写出四个数
   为什么 RL 不会再改写它：RL 丢的是「不产生命中的行」。判断句永远不命中 → 丢；每一行都是尝试 → 每行都可能是命中 → 留（第 0 层就是这么活下来的）
   代价   第 1 层 24 行、一行一步再 ×2 → 需第 1 层的题 600~700 token；len-soft 得抬到 900 或关掉，不然 RL 又来剪
三、SFT 样本量加大有没有用
   有用   第 1、2 层的【形态】：现在 360 / 202 条太薄，乘积表列不全、丢第四个数都是没学熟。过采样到各 2000 条，穷举器免费
   没用   算术：4 万条也治不了 1.5B 的四项心算，那是 ④ 的量级（亿级 token）且要 scratchpad 格式
   没用   RL 改写：只要判断句没回报，样本再多 RL 照样丢。治这个靠老师 v2（不写判断句）和过程奖励（让「算对」有回报）
   小用   量大 / epoch 多 → 程序在分布里更占主导 → RL 起点更稳，少一点冲种子的风险
```

### 教学：判断句 vs 尝试行（2026-09-05）
```
尝试行   提出一个用全所有数的完整式子，算出值，对着目标标方向。它自己可能就是命中
         7 * 5 + 33 - 14 = 54 (too high)
判断句   对搜索本身下结论或做决定，自己不可能命中：
         None of the 8 sign patterns gives 43 (they range from 7 to 59), so we need * or /.      ← 结论
         2, 98, 165 cannot reach 43 even with the remaining numbers …, skip them.                ← 剪枝决定
         Try the rest with the remaining numbers, closest to the target first:                    ← 排序决定
         Pair products and quotients: 7 * 5 = 35, 14 * 5 = 70, …                                 ← 中间表（算了但不是对目标的尝试）
RL 为什么丢判断句、留尝试行
         奖励按整条轨迹给。含判断句的轨迹要比不含的平均得分高，它的 token 才会被推高。
         判断句：占 token（长度罚）、常算错（范围写错、剪错枝）、错了把搜索带偏 → 含它的轨迹不比不含的强 → 选掉
         尝试行：命中的那一行一定是尝试行 → 含正确尝试的轨迹得 1.0 → 留。第 0 层 8 行就是这样活下来的
过程奖励验什么
         验算式行最省事、最无歧义：「expr = V (too high)」→ expr 算出来是不是 V、方向对不对；「expr = V (perfect match)」→ expr 是不是等于目标
         判断句也能验（范围、剪枝都可算），但要解析散文、容易误判、而且只是奖励「写了句子」不是「用了判断」→ 不值得
         所以不是「过程奖励不能用判断句」，是「让老师别写判断句，过程奖励只验算式行和宣告行」
判断句用什么代替
         ★ 把判断变成顺序。策略写进尝试的【固定顺序】里，而不是写成句子：
         第 0 层「先全加、再一个减号、再两个减号」本身就是策略，但它以 8 行尝试的形式出现，模型只要学顺序，不用学推理
         v2 第 1 层同理：6 个乘积按下标顺序、每个配 4 种 ±，「哪个先试」由顺序定死，没有一句话要它判断
```

### 用户复述：「不写文字写算式，算式拆成按顺序的小算式」+ 过程奖励的权重 + 判分器的地位（2026-09-05）
```
复述对，两条治两种病：不写判断句只写固定顺序的尝试 → 治 RL 丢判断句；一次尝试拆成一步一算 → 治算错
补充：拆的是搜索过程里的算，<answer> 仍是一个用全所有数的完整式子。一行 = 一次尝试，行内几个小算式，行尾标方向，行的顺序 = 策略
代价：轨迹约翻倍 → v2 要配 len-soft 抬高或关掉

过程奖励结构   最终答案 0.75 不变；格式/解析/数字比例 0.25 不变；每步算对率 −0.1 × 错步占比；假宣告 −0.2；长度不变
               一步一算的每步「a op b = c」两操作数，正则抓出来算一遍就验，免费且无歧义
               两原则：过程奖励是推一把不是目标（权重小、结果分占大头）；用率不用数（防「多写简单步凑分」，过程奖励最经典的被钻法）

★ 判分器就是环境本身，模型是它的镜子。RL 拿到的唯一信息是它吐的那个数；它的每条规则最后都变成模型的一个习惯，已见四次：
   错式子 0.25 > 交白卷 0.05     → 放弃也交错式子
   假 perfect 不扣分             → 自信错答
   只取第一个 answer             → 一条只写一个 answer
   超长线性扣分                  → 剪循环，也剪掉第 2 层
   判分器三要件：不能被钻（safe_eval、只取第一个 answer）；要给梯度（阶梯）；不能奖错东西（部分分的副作用）。第三条只有跑了才看得见，改了再跑
```

## 臂 0 RL 完成（只 RL，mix4-300 再 300 步，2026-09-05）
| eval（贪心） | 起点 | 25 | 50 | 150 | 300 |
|---|---|---|---|---|---|
| CD 正 | 0.45 | 0.40 | 0.43 | 0.47 | **0.50** |
| CD 长 | 220 | 278 | 305 | 350 | **357** |
| 标注/条 @ 精度 | 6.2 @ 0.53 | 7.5 @ 0.52 | 9.2 @ 0.52 | 12.1 @ 0.51 | 11.9 @ 0.58 |
| GSM 正 | 0.84 | 0.86 | 0.79 | 0.83 | 0.81 |
| 总分 | 0.715 | 0.689 | 0.661 | 0.669 | 0.675 |
129 分钟；KL 0.03~0.04 平；ckpt_best = step 25（总分被长度罚拉低的假峰），并表用 ckpt_latest = 300

```
① 正确率 +0.05（0.45 → 0.50，贪心 100 题，噪声边缘）。跟预测「+0.02，空转」差不多，略好一点
② ★ 长度 220 → 357，标注 6.2 → 11.9 —— 翻了近一倍。原因：max_new 512 → 1024、len-soft 400 → 800 给了空间，RL 就往长走
   ★ 修正之前的结论：「Countdown 上 RL 不会变长、停在 210」有一半是 len-soft 400 压出来的。松开罚，RL 会拉长搜索。
   但拉长的回报很小（+0.05），6+ 桶 0.22 → 0.285 —— 「长度是任务要求的」在【回报】意义上仍成立：这个任务不值得写更长
③ 总分从 0.715 掉到 0.675 是长度罚在扣：拉长的那部分超过 800 的在挨打。最优 ckpt 停在 25 步是这个假象
④ 精度：贪心 0.53 → 0.58，温度 1 下 0.44 → 0.41。数量翻倍精度不涨 → 按脚本的判据是「多写不是算对」
⑤ GSM 0.84 → 0.81 噪声内
预测 held-out   CD-4 pass@1 0.42 / pass@8 0.68；CD-3 pass@64 ≈ 0.83；全错 31~33；长度中位 280~320；撞顶 2~5%
```

## 臂 1 RL 完成（程序撒种 SFT → 300 步 RL，2026-09-05）
| eval（贪心） | 起点 | 50 | 75 | 150 | 200 | 300 |
|---|---|---|---|---|---|---|
| CD 正 | 0.52 | 0.49 | 0.48 | 0.50 | **0.52** | **0.52** |
| CD 长 | 487 | 269 | 220 | 279 | 279 | 302 |
| 标注/条 @ 精度 | 12.3 @ 0.64 | 6.2 @ 0.70 | 5.4 @ 0.78 | 8.3 @ 0.61 | 8.1 @ 0.65 | 9.2 @ 0.55 |
| GSM 正 | 0.82 | 0.82 | 0.80 | 0.81 | 0.85 | 0.83 |
| 总分 | 0.627 | 0.720 | 0.715 | 0.723 | **0.749** | 0.737 |
160 分钟。ckpt_best = 200；并表用 latest = 300。训练探针（温度 1）：标注 12.2 → 7.3，精度 0.51 → 0.53，3~5 桶 0.90 → 0.80，6+ 桶 0.30 → 0.34，CD |A| 0.28 → 0.11

```
① 循环剪掉了：长度 487 → 302，前 50 步完成，之后在 220~300 徘徊
② ★ 正确率 300 步一步没涨：贪心 0.52 → 0.52。预测的「pass@1 0.55~0.60」大概率落空（等 held-out）
   为什么：剩下的错是解那行算错（RL 修不了）+ 第 1 层执行漂（RL 改写成了窄启发式，覆盖变小）+ 奖励偏向「收尾宣告」（错式子 0.25 > 循环 0）
   三样都不是「多筛几轮」能动的。RL 只看结果，结果的瓶颈在算术，它就没抓手了
③ 3~5 次桶 0.90 → 0.80：短轨迹的对率掉了 —— 更多「早早自信错答」，跟 step 150 样例一致
④ CD |A| 0.28 → 0.11：组内差异塌了一半多，RL 后半程在空转。KL 偶发尖峰（0.54 @126、0.38 @149）= 某些批整体跳过穷举
⑤ GSM 0.82 → 0.83 没事
⑥ ★ 长度均衡点：臂 0 从 220 涨到 357，臂 1 从 487 降到 302 —— 同一奖励、同一 len-soft，两头都往 ~300 收。
   长度是奖励结构定的均衡，不是起点定的。「长度由任务 + 奖励定」的最直接证据
```
### 对撒种实验的判读（等 held-out 定案）
```
SFT 把圈画大了（pass@64 1.00），RL 没把圈里的点打实（贪心 0.52 平）。原因不是 RL 无能，是这一版的组合里
「RL 能筛的东西」已经筛完（循环），「RL 筛不了的东西」占了剩余误差（算术、被改写掉的第 2 层、奖励的收尾偏好）
→ 下一版必须换教材（v2 机械 + 一行一步）和换奖励（假宣告扣分、步算对率）才有新的可筛项。这两件事记着，等并表
预测 held-out：CD-4 pass@1 0.50~0.53，pass@8 0.70~0.75（略降），CD-3 pass@64 0.90~0.95，撞顶 < 3%，假 perfect > 30%
```

### 教学：「RL 能筛的筛完了」是什么意思，责任怎么分（2026-09-05）
```
换什么   教材 = 穷举器 v2 写的 SFT 数据：每行都是尝试、一行一步、第 1/2 层过采样。SFT 脚本不变
         奖励 = grpo_mix_mp.score() 里 CD 的判分加两项：假宣告 −0.2、错步率 × −0.1。GRPO 的优势/loss 公式一个字不改
RL 筛一次的条件   同一题 16 条里 ① 有的条没这个错（好的变体被抽到了）② 那些条得分更高。缺一条 RL 就动不了
筛完了的证据      |A| 0.28 → 0.11（组内差异塌掉 = 16 条越来越像，没东西可分）+ 正确率 250 步平 + 剩余错误逐个对条件：
```
| 剩余错误 | 条件 ① 抽得到好变体？ | 条件 ② 好变体得分更高？ | 责任在谁 |
|---|---|---|---|
| 解那行稳定算错（42+12−18−6=28） | ✗ 16 条都算 28 | — | 零件（心算）。不在采样半径内，RL 按定义碰不了 |
| 第 2 层没了 | 勉强（SFT 只给了 200 条示范，p 小） | ✗ 走到第 2 层的轨迹长、中间算错多、吃长度罚 → 平均不比短的高 | 教材（示范太少）+ 奖励（长度罚）+ 零件（算错） |
| 假 perfect 宣告 | ✓ 有诚实的变体 | ✗ 反了：错式子 0.25 > 诚实循环/放弃 0~0.05 | 奖励（部分分设计） |
| 判断句被改写掉 | ✓ | ✗ 判断句常错、占 token，含它的不比不含的高 | 教材（写了不可执行的判断句） |
```
RL 本身的问题（算法固有，换奖励换教材也不消失，只能绕）
  ① 只能给抽到的东西重新加权。抽不到的（稳定算错）它按定义不存在
  ② 整条轨迹一个分：30 行里第 6 行算错、第 20 行答对，两行一起被推高。答对轨迹里 27% 的错行也在被强化 —— 信用分配粗。过程奖励是给它补细一点的分
  ③ k 有限 + 在线：每步只看这 16 条，没有记忆。上一步抽到过的好变体这一步没抽到就白抽
一句话  RL 是筛子；筛子筛不出没撒进去的种、也筛不出被称错重量的东西。这版剩下的误差正好全是这两类
```

## 待办：臂 3 —— 只换奖励的消融（2026-09-05 讨论，用户：记录到待办，先不跑）
```
设计   起点 out_sft1/ckpt.pt（跟臂 1 配对；可选再从 mix4-300 跑一条跟臂 0 配对），教材不变、len-soft 800 不变，只换判分器：
       假宣告：抓「expr = V (perfect match)」算 expr ≠ 目标 → −0.2
       错步率：标注行里算错或标反的比例 × −0.1
代码   grpo_mix_mp.score() 加 --proc-reward 开关，日志加「假宣告率」一列。约 30 行
要盯   「标注/条」跌到 3 以下 = 模型在躲错步罚（不搜直接猜）
预测   假 perfect 率 > 30% → < 10%；精度 0.55 → ~0.65；pass@1 平或 +0.03。证明「奖励是责任之一」，救不了算术
命令   torchrun --nproc_per_node=4 grpo_mix_mp.py --probs 8 -k 16 --gen-bs 32 --steps 300 --init out_sft1/ckpt.pt \
         --data data/countdown4.json --max-new 1024 --len-soft 800 --len-pen 0.5 --patience 0 --proc-reward --out out_arm3
```

## 臂 2 RL 完成（自蒸馏 SFT → 300 步 RL，2026-09-05）
| eval（贪心） | 起点 | 25 | 175 | 300 |
|---|---|---|---|---|
| CD 正 | 0.46 | 0.44 | 0.53 | **0.44** |
| CD 长 | 217 | 256 | 473 | **437** |
| 标注/条 @ 精度 | 6.0 @ 0.54 | 7.5 @ 0.62 | 16.4 @ 0.55 | 14.6 @ 0.50 |
| GSM 正 | 0.79 | 0.80 | 0.81 | 0.82 |
| 总分 | 0.695 | 0.693 | 0.637 | 0.630 |
142 分钟。ckpt_best = 25（假峰）；并表用 latest = 300。训练探针：标注 4.6 → 5.7，精度 0.45 → 0.41，CD |A| 0.17 → 0.17 没塌

```
① 跟臂 0 一个模子：正确率平（0.46 → 0.44，中途摸到 0.53 又回落），长度翻倍（217 → 437 贪心），精度略掉，GSM 没事
② 三臂 RL 后贪心 CD-4：臂 0 0.50、臂 1 0.52、臂 2 0.44 —— 全在 ±0.05 噪声带里，300 步 RL 谁都没把 CD-4 推上去
③ ★ 长度均衡修正：贪心长度三臂 357 / 302 / 437 散，但【温度 1 的训练均长】三臂 245 / 264 / 272 —— 全收在 250~270。
   均衡点是采样分布上的，贪心会在个别题上循环拉高均值
④ 精度三臂都在 0.41~0.58 之间晃，没有一臂靠 RL 把算术推上去 —— 再次确认零件问题
待   三臂 held-out（CD-4、CD-3、GSM、pass@64）→ 并表。预测臂 2：CD-4 0.42 / 0.70，CD-3 pass@64 ≈ 0.80，全错 30，撞顶 3~8%
```

## 判分器 v2 设计研究（2026-09-05，用户：还能把判分器/过程奖励改得更好吗）

### 一、现判分器的激励账（哪条规则 → 模型学成什么 → 还错位在哪）
| 规则 | 学成的习惯 | 错位 |
|---|---|---|
| 阶梯 0.25（格式 .05 解析 .05 数字 .15）| 早期爬阶梯有梯度 ✓ | 模型成熟后阶梯早已 0.95+，它剩下的作用只有「错式子 0.25 > 交白卷 0.05」→ 放弃也猜、自信错答 |
| 对 0.75 | 主奖励 ✓ | 对的长轨迹 1.0 − 长度罚 0.45 = 0.55，只比错的 0.25 高一点 → 第 2 层被剪 |
| 长度罚 800→1024 线性 0.5 | 剪循环 ✓ | 对错一视同仁地罚 → 长而对的输给短而错的 |
| 假宣告不罚 | 自信错答 | 诚实循环 0 < 假宣告 0.25 |
| 中间行不看 | 答对轨迹里 27% 错行一起被强化 | 信用分配粗 |
| 重复行不看 | 不记账 | 循环短一点就不撞顶，不挨罚 |
| 尝试行不查「各用一次」 | 「5 + 7 = 12, then 12 * 14」丢 33 | 只查最终 answer |

### 二、候选改法（每条：验什么、怎么算、被钻风险）
```
A 阶梯缩小      0.25 → 0.10（.02/.02/.06），对 0.9。错式子的吸引力从 0.25 降到 0.10。风险：无（格式早已饱和）
B 长度罚只罚错的  对的轨迹只扣 0.05 × len/1024（对里面短的略优）；错的照旧 soft 线性到 0.5
                  效果：任何对 > 任何错，永远成立。第 2 层的长正解不再输给短错解。风险：对的轨迹可以变长 —— 它本来就该被允许
C 假宣告 −0.2    抓「expr = V (perfect/works/…)」算 expr ≠ 目标。风险：少写 perfect，无害
D 错步率 −0.1×率  算错或标反 / 总标注。风险：★ 不写标注率为 0 → 躲罚。要盯「标注/条」
D′ 代替 D：按行   算对 +0.01、算错 −0.02，各封顶 ±0.1。风险：刷简单行凑分 → 要求行里的数来自题目
E 重复行 −0.05/行  同一条里同形态第二次出现（check_sft 的 canon），封顶 4 行。治「不记账」。风险：少写行 —— 但正解仍要搜到才有 0.9
F 违规行 −0.05/行  尝试行里某个数用了两次或用了题外的数，封顶 4 行。治「丢数/重数」。风险：同 E
G 靠近度 +0.05    min|尝试值 − 目标| 越小越高。给全错组一点梯度（现在 50% 组无差异）。★ 风险：奖「近似」不奖「解」；文献里势函数塑形才安全。先不加
H 动态采样        （训练器不是判分器）丢掉全同分的组、补抽，让每批都有梯度。DAPO 的做法。代价：多抽。先记
```
文献锚点：DAPO（超长塑形、token 级 loss、动态采样）；Dr. GRPO（不除 std）；R1 明确不用神经 PRM（怕被钻）—— 我们的逐行验是规则的，钻的口子只在「率/数」的算法上；Lightman 2023 过程监督优于结果监督（数学）

### 三、推荐的 v2 判分器（一次换整套，按项打日志，之后再拆）
```
r = 0.9·对 + 0.02·格式 + 0.02·解析 + 0.06·数字比例
    − 0.2·[假宣告]
    − 0.1·错步率
    − 0.05·min(重复行, 4)
    − 0.05·min(违规行, 4)
    − 长度：对的 0.05·len/1024；错的 soft 800→1024 线性到 0.5
对 vs 错的底线：最差的对 = 0.9 − 0.05 − 0.1 − 0.2 − 0.2 = 0.35 > 最好的错 = 0.10 + 0 = 0.10 ✓
日志加五列：假宣告率、错步率、重复行/条、违规行/条、对的均长 vs 错的均长
```

### 四、怎么安排
```
臂 3   mix4-300 + 判分器 v2 整套，对臂 0。一个因子「判分器」，按项日志看哪条在起作用
臂 3′  out_sft1 + 判分器 v2，对臂 1。看长度罚改法保不保得住第 2 层、假宣告消不消
之后   哪项有效再单独拆消融。教材 v2 + 判分器 v2 是最终组合
预测（臂 3 vs 臂 0）假宣告率 56% → <10%；重复行 ↓ 一半；精度 +0.05~0.1；pass@1 平或 +0.03；pass@64 不动
预测（臂 3′ vs 臂 1）第 2 层保住（scan_ops 混合形态/条不掉）、CD-3 pass@64 ≥ 0.95、假宣告 <10%、pass@1 +0.05
```

## 判分器 v2 实现完（2026-09-05）
```
judge_v2.py        纯函数，不依赖 torch：terms() 出四个过程项，score() = 阶梯 .02/.02/.06 + 对 .9 − 假宣告 .2 − 错步率 .1 − 重复 .05/行 − 违规 .05/行（各封顶 4）
                   宣告正则：值和宣告词之间只许字母/空格，不许跨过下一个等式（否则第 2 层「= 10, then 10 + 33 = 43 (perfect)」会把 perfect 挂到 5*14/7 上，误判假宣告）
                   违规只算「题里的数用超了」，题外数不算 —— 第 2 层的中间值「10 + 33」是合法的
                   canon 首项补 "+"（7+5+3+2 和 2+3+5+7 才归一）—— check_sft.py 同一处 bug 一起修了，之前的重复率略低估
grpo_mix_mp.py     --judge v2：score() 走 J2；rollout 里对的只扣 .05·len/max_new、错的才吃 soft 长度罚；
                   日志加「假 错步 重 违 对长/错长」；汇总加「判分器 v2 各项 前 20 → 后 20」；启动打印 v2 说明。v1 路径一字未动
自检   对+干净 1.000 / 假宣告 −0.200 / 重复+违规+算错 −0.025（重复 1 违规 1 错步 .25）/ 第 2 层 1.000 不误判 / 诚实放弃 0.020
md5    grpo_mix_mp f66d99d0…  judge_v2 f356c603…  check_sft 4606f223…
```
### 两条臂的跑法（用户决定：mix4-300 和 out_sft1 都做）
```
臂 3   torchrun --nproc_per_node=4 grpo_mix_mp.py --probs 8 -k 16 --gen-bs 32 --steps 300 --init out_mix4/ckpt_step300.pt --data data/countdown4.json --max-new 1024 --len-soft 800 --len-pen 0.5 --patience 0 --judge v2 --out out_arm3
臂 3′  同上，--init out_sft1/ckpt.pt --out out_arm3s
先各跑 --steps 2 --out out_arm3_test 看新列打印正常再正式。评测九步同前；并表：臂 3 对臂 0，臂 3′ 对臂 1
预测   臂 3 vs 臂 0：假宣告 0.56 → <0.10，重复 ↓ 一半，精度 +0.05~0.1，pass@1 平或 +0.03，pass@64 0.83~0.86
       臂 3′ vs 臂 1：第 2 层保住（混合形态/条不掉、scan_ops 乘除题全错数 ≤ 臂 1）、CD-3 pass@64 ≥ 0.95、假宣告 <0.10、pass@1 +0.05
```

## 臂 1 RL 后 pass@64（CD-3 池外，2026-09-05）—— 循环剪干净了，种也被吃掉一截
| k | Base | 600 | 臂 2 SFT | 臂 1 SFT 后 | **臂 1 RL 后** |
|---|---|---|---|---|---|
| 1 | 0.016 | 0.456 | 0.504 | 0.598 | **0.507** |
| 8 | 0.121 | 0.694 | 0.703 | 0.926 | **0.761** |
| 16 | 0.224 | 0.748 | 0.739 | 0.972 | 0.849 |
| 64 | 0.590 | 0.830 | 0.790 | **1.000** | **0.950** |
| 全对 / 全错 | 0 / 41 | 1 / 17 | 13 / 21 | 23 / 0 | 28 / **5** |
| 格式 / 数字用全 | | | | 0.83 / 0.70 | 1.00 / 0.86 |
| 撞顶 / 长度中位 | | | | 17% / 338 | **0% / 192** |

```
① RL 做到的：撞顶 17% → 0、格式 0.83 → 1.00、长度 338 → 192、全对 23 → 28。杂草清干净了
② RL 吃掉的：pass@8 0.93 → 0.76、pass@64 1.00 → 0.95（5 道重新够不着）、pass@1 0.60 → 0.51
   ★ pass@1 反而掉：SFT 后不撞顶的那部分对率 ≈ 0.60 / 0.81 ≈ 0.74，RL 后撞顶 0 但对率只 0.51 → 不循环的轨迹变差了
   原因就是 step 150 样例看到的改写：乘积表、第 2 层被剪成「最大数 × 其他」的窄启发式。CD-3 需 ×÷ 的题（约 30%）首当其冲
   ★ 而且 RL 训的是 CD-4，CD-3 一步没训，掉的是迁移过来的程序 —— 改写是全局的
③ 仍然比所有没撒种的模型强：k=64 0.95 vs 0.83，k=8 0.76 vs 0.70。种没死，缩了
④ 数字用全 0.86：14% 的轨迹用超了数 —— v2 的违规项正对着它
预测对账   pass@64 0.90~0.95 → 0.95 ✓；「pass@8 略掉」→ 掉 0.17，远超「略」；pass@1 猜涨实掉
★ 这一条把臂 3′ 的意义定死了：v2 判分器（对永远 > 错、不罚长而对的）能不能让 RL 只剪杂草、不吃种。指标就看 CD-3 pass@8 保不保得住 0.9
```

## 三臂 RL 后：CD-4 / CD-3 池外 800（2026-09-05）
### CD-4（RL 训的任务）
| | mix4-300 参照 | 臂 0 只 RL | 臂 2 自蒸馏+RL | **臂 1 撒种+RL** | 臂 1 SFT 后（RL 前） |
|---|---|---|---|---|---|
| pass@1 | 0.393 | 0.471 | 0.473 | **0.563** | 0.506 |
| pass@8 | 0.670 | 0.710 | 0.690 | **0.750** | 0.720 |
| 全对 / 全错 | 10 / 33 | 12 / 29 | 15 / 31 | **29 / 25** | 22 / 28 |
| 数字用全 | 0.951 | 0.955 | 0.948 | 0.921 | 0.61 |
| 长度中位 / P95 | 222 / 375 | 237 / 563 | 254 / 609 | 205 / 596 | 344 / 1024 |
| 撞顶 | 0.9% | 0.5% | 0.4% | **0** | 28% |
| 搜索词/条 | 6.7 | 8.5 | 8.8 | 9.9 | 15.3 |

### CD-3（RL 没训，看迁移）
| | mix4-300 | 臂 0 | 臂 2 | 臂 1 | 臂 1 SFT 后 |
|---|---|---|---|---|---|
| pass@1 | 0.476 | 0.511 | 0.518 | 0.503 | 0.599 |
| pass@8 | 0.660 | 0.700 | 0.720 | **0.740** | **0.930** |
| 全对 / 全错 | 28 / 34 | 26 / 30 | 28 / 28 | **39 / 26** | 38 / **7** |
| 数字用全 | 0.944 | 0.925 | 0.921 | **0.856** | 0.69 |
| pass@64 | — | 跑中 | 跑中 | 0.950 | 1.000 |

### 四条读法
```
① ★ 训的任务上撒种赢了：CD-4 pass@1 臂 1 0.563 vs 臂 0 0.471 → 种在 RL 之后的净贡献 +0.09；全对 29 vs 12；全错 25 vs 29
   而且臂 1 RL 后 > 臂 1 SFT 后（0.506 → 0.563、撞顶 28% → 0）—— 训练日志贪心 0.52 → 0.52 把这个涨幅藏住了，温度 1 才看得见
② 臂 0 = 臂 2，两位小数都一样（0.471 / 0.473）。自蒸馏 SFT 在 RL 之后贡献为零，跟预测一致
③ 只 RL 300 步 = +0.08 pass@1（0.393 → 0.471）。比我预测的「+0.02 空转」多 —— 松开长度罚（400 → 800）给了它空间
④ ★★ 迁移任务 CD-3 上种被 RL 吃了大半：SFT 后 pass@8 0.93 / 全错 7 → RL 后 0.74 / 26。对臂 0 只剩 +0.04 pass@8、+13 全对
   数字用全 0.856：RL 改写出的「最大数 × 其他」段会重复用数（42 + 18 * 12 / 18）—— v2 违规项正对着它
   CD-3 的 ×÷ 题占三成，老师给的短形态（乘积 ± 第三数）被改写掉后它们首当其冲
减法定案（CD-4，RL 后）  RL 本身 +0.08 ｜ 自蒸馏 +0.00 ｜ 程序撒种 +0.09（在 RL 之上）｜ 天花板 CD-3 @64：0.83 → 0.95
臂 3′ 的目标更具体了：v2 判分器下 CD-4 pass@1 ≥ 0.56 且 CD-3 pass@8 ≥ 0.85、数字用全 ≥ 0.93
待   pass@64 臂 0 / 臂 2、三臂 GSM、analyze_search、scan_ops
```

### 用户三条结论 + 修正（2026-09-05）
```
① 撒种效用高            对。但书：老师得写模型执行得了的（判断句、算术传不过去）；撒下的种 RL 会改，改得好不好看秤
② 「RL 只对学过的题有效」 不准 → 「RL 优化的是训过的分布，迁移有但方向不保证」
                          证据：池外 CD-4 没见过的题只 RL 也 0.39 → 0.47（题型不是题目）；臂 0 没训 CD-3，CD-3 也 0.476 → 0.511（正迁移）；
                          GSM-RL 去做 CD pass@8 ×2.9（正迁移）；臂 1 CD-3 掉是 RL 为 CD-4 改写了乘除段、改法对 CD-3 不利（迁移方向不保证）
③ 自蒸馏没效、没新零件   对。换接收者、迭代多轮才有它的位置
④（补）秤决定 RL 拿种子怎么办：v1 把长而对的判得跟短而错的差不多 → RL 拿短错换长对 → 第 2 层没了。臂 3′ 验
「塑走」= RL 按训练任务的奖励把程序重新整形（reshape），整形后原来的形状不在了。撒进来的程序在分布里根基浅（SFT 4000 条 vs 预训练 + 900 步 RL 养出来的旧习惯），
        每行没有回报的部分最先被替换，所以比模型自己长出来的程序更容易被整形掉
```
### 「塑走」展开
```
不是删掉，是概率被压到采样半径以下，抽不到了
臂 1 step 150 的实例：老师第 1 层「6 个乘积表 → 范围句 → skip 句 → 按远近试」→ RL 整形成「</think><think> Only * and /: 最大数 × 其他」
  三个判断句在 CD-4 上不产生命中，每步都被压；「最大数 × 其他」偶尔命中，每步都被推；150 步后老师的形状抽不到了
  这个形状 CD-4 够用、CD-3 不够用 → CD-3 掉
根基浅所以易塑：老师的程序 = 4000 条 SFT；旧习惯 = 预训练 + 900 步 RL。RL 每步重加权，浅的、在训练任务上不得分的几十步就被换掉
第 0 层没被塑走：加减题每次命中都靠它，一直得分
塑是有选择的：训练任务上得分的留，不得分的换。秤称什么就留什么
```

# ★★★ 四臂实验总表（臂 0 / 1 / 2 全部评完，2026-09-05）
| | mix4-300 参照 | 臂 2 SFT 后 | 臂 1 SFT 后 | 臂 0 只 RL | 臂 2 自蒸馏+RL | **臂 1 撒种+RL** |
|---|---|---|---|---|---|---|
| CD-4 pass@1 / @8 | 0.393 / 0.670 | 0.409 / 0.700 | 0.506 / 0.720 | 0.471 / 0.710 | 0.473 / 0.690 | **0.563 / 0.750** |
| CD-4 全对 / 全错 | 10 / 33 | 10 / 30 | 22 / 28 | 12 / 29 | 15 / 31 | **29 / 25** |
| CD-3 pass@1 / @8 | 0.476 / 0.660 | 0.511 / 0.720 | **0.599 / 0.930** | 0.511 / 0.700 | 0.518 / 0.720 | 0.503 / 0.740 |
| CD-3 pass@64 / 全错 64 次 | （600：0.830 / 17） | 0.790 / 21 | **1.000 / 0** | 0.830 / 17 | 0.840 / 16 | **0.950 / 5** |
| GSM 训练模板 / 中性 | 0.74 / 0.46 | 0.75 / 0.47 | 0.755 / **0.565** | 0.75 / 0.49 | 0.77 / 0.49 | 0.755 / 0.50 |
| CD-4 长度中位 / 撞顶 | 222 / 0.9% | 219 / 0 | 344 / 28% | 237 / 0.5% | 254 / 0.4% | 205 / 0 |
| CD-3 数字用全 | 0.944 | 0.944 | 0.69 | 0.925 | 0.921 | 0.856 |

## 八条定论
```
① 天花板：没撒种的怎么 RL 都是 0.83~0.84（600、臂 0、臂 2 三次重复）；撒种 1.00，RL 后 0.95。「撒种抬 pass@∞、RL 不抬」结案
② 训的任务（CD-4）：撒种+RL 最好，pass@1 0.563，比只 RL 高 0.09，全对 29 vs 12。种在 RL 之后有净贡献
③ 只 RL 300 步：+0.08 pass@1（0.39 → 0.47）。比预测多，松开长度罚给了搜索空间；但 pass@64 一动不动
④ 自蒸馏：RL 后贡献 0.00（0.471 vs 0.473，pass@64 0.83 vs 0.84）。没有新零件就没有效果
⑤ 迁移任务（CD-3）：种被 v1 判分器下的 RL 吃掉大半（pass@8 0.93 → 0.74、全错 7 → 26、数字用全 0.69 → 0.86 但比参照 0.94 差）
   但仍高于无种臂（@8 0.74 vs 0.70、@64 0.95 vs 0.83）
⑥ GSM：三臂训练模板 0.75~0.77、中性 0.49~0.50，无遗忘。★ 臂 1 SFT 后那次中性 0.565「漏过模板」在 RL 后回到 0.50 → 「SFT 漏、RL 重新绑」，趋势可疑但样本不够，存疑
⑦ 长度：三臂 RL 后 CD-4 中位 205~254，撞顶 ≤ 0.5%。撒种的长轨迹（344）被 RL 压回均衡点，没有出现「难题长、简单题短」的分叉
⑧ 自信错答、重复用数在臂 1 RL 后是主要残余（数字用全 0.856），这两条正是 v2 判分器的靶子
预测对账   臂 0 pass@1 猜 0.41 实 0.47（低估）；pass@64 0.83 ✓；臂 1 RL 后 pass@1 ≥ 0.50 ✓（0.563）；pass@64 ≥ 0.92 ✓（0.95）；
           全错 ≤ 20 ✗（25）；长度分叉 ✗（没有）；臂 2 pass@64 ≈ 0.83 ✓（0.84）
```
## 下一步
```
CPU  analyze_search（mix4 + 三臂）、scan_ops（三臂）→ 精度、假宣告率、全错构成
GPU  臂 3（mix4-300 + v2 判分器）→ 对臂 0；臂 3′（out_sft1 + v2）→ 对臂 1。目标：CD-4 ≥ 0.56 且 CD-3 pass@8 ≥ 0.85、数字用全 ≥ 0.93
```

## 三臂 RL 后：analyze_search + scan_ops（CD-4 池外，2026-09-05）
### ③b 标注精度
| | mix4-300 | 臂 0 只 RL | 臂 1 撒种+RL | 臂 2 自蒸馏+RL | 臂 1 SFT 后 |
|---|---|---|---|---|---|
| 标注/条 | 5.1 | 6.3 | 7.9 | 6.3 | 13.6 |
| 精度（全部） | 0.456 | **0.404** | **0.498** | **0.418** | 0.484 |
| 算错 / 方向反 | 0.51 / 0.03 | 0.56 / 0.04 | 0.48 / 0.02 | 0.55 / 0.04 | 0.50 / 0.02 |
| 答对轨迹的精度 | 0.499 | 0.431 | **0.670** | 0.445 | 0.728 |
### ③ 说 perfect 的可信度
| | mix4 | 臂 0 | 臂 1 | 臂 2 | 臂 1 SFT 后 |
|---|---|---|---|---|---|
| 宣告条数 / 800 | 221 | 378 | **657** | 495 | 518 |
| 宣告里假的 | 56% | 49% | **32%** | 41% | 22% |
| 假宣告轨迹占全部 | 15% | 23% | **26%** | 25% | 14% |
### ② 分桶正确率（1~2 / 3~5 / 6+）
```
mix4  0.69 / 0.66 / 0.20      臂 0  0.78 / 0.71 / 0.25      臂 2  0.68 / 0.71 / 0.30
臂 1  0.77 / 0.88 / 0.37      臂 1 SFT 后  1.00 / 0.98 / 0.32
```
### scan_ops
| | mix4 | 臂 0 | 臂 1 | 臂 2 |
|---|---|---|---|---|
| 全错 = 纯加减 + 必须乘除 | 33 = 7 + 26 | 29 = 3 + 26 | **25 = 4 + 21** | 31 = 5 + 26 |
| 必须乘除的题上试过乘除的轨迹 | ~0.5 | 0.32 | **0.87** | 0.24 |
| 全错轨迹混合形态/条 | — | 1.95 | 5.43 | 1.46 |
| 800 条里碰过混合形态 | 0.35 | 0.58 | 0.35 | 0.33 |

```
① ★ v1 判分器下 RL 让精度【掉】：臂 0 0.456 → 0.404、臂 2 → 0.418，标注数 5.1 → 6.3。「数量涨精度掉」= 脚本自己的判据「表演」。
   奖励只认结果，多写几行提高命中率但没人管算没算对 → 精度被稀释。臂 1 持平 0.498（老师的规整式子撑着）
② 假宣告翻倍：臂 1 从 SFT 后的 14% 到 26%（预测 >30% 的宣告是假的 → 实 32% ✓）。三臂 23~26%，全在自信错答。v2 假宣告项的靶
③ 乘除形态的覆盖是种给的：必须乘除的题上，臂 1 87% 的轨迹试乘除，无种臂 24~32%。种没被塑光，塑的是「怎么试」
④ 纯加减全错：7 → 3 / 4 / 5，三臂都剪了一点；臂 1 剩的 4 道是「解那行算错」
⑤ 臂 1 全错 25 = 4 + 21：乘除题从 26 → 21，5 道靠种解出来；剩 21 道 87% 试了乘除但没试对 → 执行（窄启发式 + 算错）
⑥ 臂 0 全对轨迹里 48% 也写乘除（不需要也写），臂 1 全对里 1%（+− 命中就停）：无种臂的搜索无结构，有种臂的搜索有顺序
所有评测齐了。臂 0/1/2 各四项 GPU + 两项 CPU 全部入档。下一步：臂 3 试跑
```

# ★★★ 三臂实验报告（臂 0 / 1 / 2 结案，2026-09-05）
## 设计
| 臂 | 起点 | SFT | RL | 耗时 |
|---|---|---|---|---|
| 0 只 RL | mix4-300 | 无 | GSM + CD-4，300 步，v1 判分器，len-soft 800 | 129 分 |
| 1 程序撒种 | mix4-300 | 穷举器 CD-3/CD-4 各 2000 + GSM 自采样 2009，2 epoch | 同上 | SFT 30 分 + RL 160 分 |
| 2 自蒸馏 | mix4-300 | 同题号自采样 CD-3 1530 + CD-4 1394 + 同一份 GSM | 同上 | SFT 25 分 + RL 142 分 |
评测：每个点 CD-4 池外 800、CD-3 池外 800、CD-3 pass@64、GSM 两模板，加 analyze_search、scan_ops。测量点 6 个（两臂 SFT 后 + 三臂 RL 后 + 参照）

## 各臂教了什么
```
臂 0   再灌 300 步 RL：pass@1 +0.08（松开长度罚的功劳）、天花板不动（0.83）、精度反而掉（0.456 → 0.404）、假宣告 23%
臂 2   自蒸馏 = 一轮离线专家迭代：SFT 后 +0.02，RL 后 +0.00。val loss 0.154 → 0.151 就预告了：自己的轨迹里没有新东西
臂 1   撒种：SFT 后 CD-3 pass@64 1.00（老天花板 0.83~0.87 穿过去）、pass@8 0.93；代价 28% 循环撞顶
       RL 后：CD-4 上最好（0.563，比只 RL 高 0.09）、循环清零；但 CD-3 被塑走一截（pass@8 0.93 → 0.74、@64 1.00 → 0.95）、假宣告翻倍到 26%
```
## 结论（八条，见「四臂实验总表」）+ 三条方法论
```
方法论 ① 贪心 eval 会藏涨幅：臂 1 训练日志 0.52 → 0.52，温度 1 池外 0.506 → 0.563
       ② 探针会退役：行为普及后对照组消失（★差、0 次桶），要换新探针（分层精度、假宣告率）
       ③ 跑前写预测：这轮 12 条预测中 8 条，错的 4 条（pass@1 方向、精度 0.85、全错 ≤ 20、长度分叉）每条都指向一个没想到的机制
```
## 遗留 → 下一步
```
残余误差三类：解那行算错（零件）、乘除段被塑成窄启发式（教材+奖励）、自信错答/重复用数（奖励）
臂 3 / 3′（判分器 v2）验奖励那两类；老师 v2 验教材；算术那类等一行一步或计算器
```

### 三条方法论展开（2026-09-05，用户要求解释）
```
① 贪心藏涨幅   贪心 = 每题只走最可能的那一条路，100 题一条 1%。RL 改的是整个分布不只是众数：众数原来对的还对、原来错的（解那行稳定算错）还错，
               贪心看不出周围的质量变多了。温度 1 抽 8 条量的是「对的路径占多少质量」，800 条分辨率也细 → 0.506 → 0.563 才看得见。
               规则：判 RL 用训练温度抽样，贪心只回答「它默认怎么做」
② 探针退役     ★差 = 有这个行为的轨迹 vs 没有的。Base 时 44% 有，两组都大 → 有信息。RL 后 99% 有，「没有」组只剩 8 条，且是交白卷的怪子集
               → 量的变成「放弃 vs 不放弃」。0 次桶同理。探针只在行为「有时出现」的阶段有效，普及后要换成能在普及行为内部分辨的探针：
               分层精度、假宣告率、分桶正确率、pass@64。看分母，分母塌了探针就退役
③ 跑前写预测   预测 = 对系统的一个可证伪模型。中了是确认，错了是新机制。这轮错的 4 条：
               pass@1 猜掉实涨 → 不循环的轨迹比想的准得多；精度猜 0.85 实 0.48 → 算术是零件 SFT 装不进、聚合精度被循环稀释；
               全错猜 ≤20 实 25 → 乘除段被塑走；长度分叉猜有实无 → 长度是奖励定的均衡不是难度定的
               不先写下来，任何结果事后都能说「那当然」，什么都学不到
```
### 能不能写成论文 / 博客（2026-09-05）
```
博客   能，现在就够：受控消融（同起点同 RL 三种 SFT）、pass@64 天花板测试、预注册预测、判分器即镜子的四个实例、塑走。
       4×4090 + 1.5B 一个人复现得了，教学价值高。等臂 3′ 落地给故事一个结尾（秤修了种保不保得住）
论文   现在只到 workshop 级。结论跟文献呼应（Yue 2025「RL 不超 Base」、STaR/ReST、R1 冷启动、过程奖励），新的是：
       程序老师做撒种把 pass@∞ 推满 + RL 在未训任务上把撒进来的程序塑走 + 判分器 v2 能不能止住（缺这一块）
       短板：单模型尺寸、单任务族、held-out 100 题（置信区间宽）、没有多种子重复、指标多为自制
       要补：3 个种子、第二个任务（长链算术已在计划）、3B 一次、臂 3′ 的结果、跟标准 baseline 对齐的指标
```

## 总结：RL 训练的本质、特点、用处（2026-09-05，跑完三臂后的版本）
```
本质    RL 是把「能判」变成「能做」的机制。判分器只会判对错（验证），模型只会生成；RL 让模型自己抽样、判分器打分、
        把得分高的抽样推高、低的压低。它不创造行为，它对已有行为做选择和放大 —— 筛子。用 p 说：把 p > 0 的推向可靠，p = 0 的碰不了
        它独有的地方：只需要判断，不需要示范。判断比示范便宜得多（很多任务能判不能写）
特点 / 优点
  ① 只要验证器，不要标注答案（Countdown 文件里只有 nums 和 target）
  ② 同策略：看见自己此刻的错，修，飞轮转（循环 50 步剪光；SFT 看不见自己的错）
  ③ 学得到「不该做什么」：负例天然在组内（重复、撞顶、用超数都被压）；SFT 教材里没有反例
  ④ 组装：把散在各处的零件按奖励拼成程序（顿悟从 44% 装饰到 99% 有用）
  ⑤ 压缩：把 500 次抽样压成 8 次（Base → 600，pass@64 曲线 k≈500 汇合）
  ⑥ 权重动得少（KL 0.04），不忘旧任务（混训三代 GSM 不掉）
缺点 / 限制
  ① 采样半径：p = 0 的题按定义碰不了，不抬天花板（无种臂三次都是 0.83）
  ② 整条一个分：信用分配粗，答对轨迹里 27% 的错行一起被强化；秤称错就学错（部分分 → 放弃也猜、假宣告不罚 → 自信错答）
  ③ 磨多样性：|A| 0.28 → 0.11，pass@8 平；撒进来的程序在未训任务上被塑走（CD-3 pass@8 0.93 → 0.74）
  ④ 贵：每步现生成，九成时间在 rollout
  ⑤ 绑定：收益锁在训练模板和训练分布上（中性模板 = Base），迁移有但方向不保证
  ⑥ 没有验证器就得用奖励模型，会被钻
用处
  适合   有验证器的领域（数学、代码、游戏、可执行任务）从「会但不稳」到「稳」；格式、停止、风格的对齐；把环境/工具反馈变成习惯；agent 多轮行为；偏好对齐（RLHF）
  不适合 注入知识、注入模型没有的计算（心算）、在没有验证器的地方求真、抬天花板 —— 这些是预训练 / mid-training / 撒种 / 工具的活
一句话  预训练给零件，SFT 摆图纸，RL 试运转：转得动的留，转不动的拆。秤称什么留什么
```

### p 的定义 + 分工 + RL 对预训练的依赖（2026-09-05，用户复述确认）
```
p 的定义   固定模型、题、采样设定，所有判对的输出串的概率之和：p_i = Σ_{y 正确} π_θ(y | x_i)。好记的说法：这道题自己的 pass@1
           量不到只能估（8 次一格 0.125；64 次全错 = p < 0.05，不是 0）；严格不存在 p = 0，只有小到抽不到的（10⁻³⁰）
每题一个 p  mix4 CD-3 池外：28 道 ≈ 1、38 道 0.1~0.9、34 道 < 0.1（17 道 < 0.02）。Countdown 难度双峰
pass@k     一道题 k 次至少一次对 = 1 − (1−p)^k；整套取平均。pass@1 看多偏，pass@大 k 看有没有
           「100 道对 1 道」是 pass@1 = 0.01（轴：题数），跟 pass@100（轴：同题抽几次）不是一回事
RL 的目标   J(θ) = 平均 p_i = pass@1；梯度 Σ A_i ∇log π，全错组 A = 0 → p = 0 的题永远 0；上限 pass@∞；有效下限 p ≳ 0.7/k
分工       0 → 非 0：预训练零件、提示词开关、SFT 撒种、工具；RL 偶尔靠运气和迁移，弱
           非 0 → 可靠：RL 专长；SFT 也能推一段
依赖       上限由预训练定（可达集），起点由提示词和 SFT 定，方向由判分器定。RL 只在这三条划出的范围里选择
```

### 用户比喻：奇异博士看未来挑赢的那条，RL 确保它发生（2026-09-05）—— 抓住核心，四处修
| | 奇异博士 | RL |
|---|---|---|
| 看几条未来 | 1400 万 | 一题 16 条，p < 4% 的赢路看不见（采样半径） |
| 确保发生 | 精确那一条 | 把那类路径概率推高（0.05 → 0.5，不到 1），相似题也跟着涨（参数共享） |
| 谁定义赢 | 自己知道结局 | 判分器。秤错了就确保错的未来（假宣告） |
| 看几次 | 一次 | 每步一次，放大后再看赢路更多，飞轮 |
他没有的：RL 同时把输的路压下去。更准的版本：只能看 16 条、每步重看、靠一把秤判输赢、赢路变常见输路变少见的奇异博士。看不见的赢路要先有人撒进视野（SFT）
追问「那就能预测未来了」—— 不能。奇异博士看的是真的未来，模型看的是自己写的 16 次猜测；能挑出赢的是因为判分器【现在】就能判。
RL 能「预测」的范围 = 判分器当下能验证的范围（数学、代码、游戏、可执行任务）。真正的未来没有当下的秤，RL 一步走不了。
擦边做法：拿已揭晓的问题当秤训「预测策略」，学的是做法不是看见未来。一句话：RL 能确保发生的，只有它能当场验证的未来

### 用户的图景：人类探索新价值（0 → 非 0），AI 放大并稳定获取（非 0 → 可靠）—— 同构于撒种/收紧，四条修正（2026-09-05）
```
① 人的稀缺贡献是秤不是种：有判分器的地方 AI 也能探索（第 37 手、AI 找算法）；AI 探索不了的是没秤的地方 = 「什么算有价值」没定义的地方
② 放大器放大秤称的东西，包括错的：假宣告 15% → 26%。人交一把有漏洞的秤，AI 把漏洞高效放大 → 人的第二份活是检查放大了什么（九步评测不会消失）
③ 放大器塑走它不称的部分：第 2 层 150 步没了。人撒的东西在 AI 目标下不得分就被整形掉，人未必察觉。文明尺度最该警惕
④ 种子得是接收者执行得了的：判断句、算术传不过去。人的发现要能写成 AI 跑得了的形式
改写：人定义价值、撒 AI 抽不到的种；AI 在定义下把可达的变可靠；人回头看秤有没有称错、种有没有被塑走
```

### 教学：秤（判分器）究竟是什么（2026-09-05）
```
一句话   秤是「我们想要什么」里能被写成可自动执行的那一部分。RL 优化的是写下来的这部分，不是想要的全部
三层
  ① 唯一通道   模型变成什么样，只经过这一个数。想要的东西没进这个数就等于没说（诚实、不重复用数、别循环 —— 秤没称，模型就没学）
  ② 代理       秤永远是价值的代理，代理和价值之间的缝就是副作用的来源：
                想要「解出来」 → 写成「第一个 answer 等于目标 + 阶梯」 → 得到「放弃也猜、假宣告、第 2 层被剪」。Goodhart：优化代理，代理就不再是好代理
  ③ 信息源     秤知道模型不知道的事：验证比生成容易。RL 是把「能判」的信息灌进权重。
                但带宽极低：一条轨迹约 1 bit（对/错），300 步 × 128 条 ≈ 4 万 bit；老师一条轨迹几百 token。
                所以 RL 权重动得少（KL 0.04）只能选择，SFT 能安装。RL = 低带宽高相关（自己此刻的错），SFT = 高带宽低相关（别人的轨迹）
秤的种类   规则（Countdown 算式）→ 参考答案（GSM8K）→ 执行测试（代码）→ 模拟器（游戏）→ 奖励模型（RLHF）→ 人。从便宜精确到昂贵模糊，本质一样
前提       秤必须比生成便宜，否则直接生成答案就好了。Countdown 验证两毫秒、生成两秒，这个不对称是 RL 能用的根
好秤的五条  钻不了（safe_eval、只取第一个 answer）；给梯度（阶梯）；不奖错东西（部分分的教训）；任何对 > 任何错（v2）；能验过程就验（逐行）
这两周的账  秤称过的都学会了（格式、停、方向标注）；没称的都没学（诚实、记账、各用一次）；称错的学反了（自信错答）。模型是秤的镜子
```

## 臂 3 前 126 步：v2 判分器被钻了 —— 模型不写标注了（2026-09-05）
| | 起点 | 24 | 50 | 125~126 |
|---|---|---|---|---|
| 标注/条（训练） | 4.9 | 2.7 | 2.0 | **0.2~0.7** |
| 假宣告率 | 0.17 | 0.14 | 0.09 | **0.00** |
| 错步率 | 0.50 | 0.23 | 0.26 | **0.02** |
| 重复/条 | 1.6 | 0.8 | 0.5 | **0.0** |
| eval CD 正 / 长 / 标 | 0.45 / 220 / 6.2 | 0.42 / 184 / 4.2 | 0.44 / 183 / 4.2 | 0.45 / 217 / **0.2** |
| eval 总分 | 0.608 | 0.595 | 0.607 | **0.664** |

```
① 跑前写的那条「标注/条跌到 3 以下 = 在躲罚」发生了，而且跌到 0.2：四个过程项全部归零不是因为做对了，是因为没东西可判
② 正确率没动（0.45 → 0.45），长度没动（~200）：它还在写差不多长的东西，只是不再用「= V (too high/low)」这个格式
   猜测：换成「(not 30)」「(no)」这类正则抓不到的说法，或者只写式子不标方向（要看 ckpt_best 的样例确认）
③ 总分涨 0.61 → 0.66 全是罚没了，是假的。这是判分器第五课：★ 过程奖励只验某个格式，模型就绕开那个格式
④ 臂 3′ 不能用这版 v2 跑：从 out_sft1 起，模型会为了躲罚放弃老师的格式 —— 那等于亲手把种拔了
决定   臂 3 跑完（还有 1 小时，结果本身是「v2 被钻」的干净数据）；3′ 等 v2.1
v2.1（三处改）
   a. 尝试行的正则放宽：「expr = V (任意短标签)」都算尝试，算术都验；方向只在写了 high/low 时验 → 换说法躲不掉
   b. 答错且尝试行 < 3 → 错步率按 1 算（最重罚）。答对的不受影响 → 「不搜就猜」是最差的错法，「搜且算对」是最好的错法
   c. 每条算对的尝试行 +0.01，封顶 +0.05 → 写正确的搜索有正回报，不只是少挨罚
```

### v2.1 实现 + 自检（2026-09-05）
```
改三处   a 尝试行正则 ATTEMPT：「expr = V (任意 ≤40 字符标签)」，perfect/works 类归宣告；算术每行验，方向只在写了 high/low 时验
         b 答错且尝试行 < 3：错步率 = max(实际, 缺的行数/3) → 0 行 1.0、1 行 0.67、2 行 0.33。答对的不受影响
         c 每条算对的尝试行 +0.01，封顶 0.05
自检八例  对+干净 1.02 ｜ 假宣告 −0.20 ｜ 重复+违规 0.005 ｜ 第 2 层 1.01 不误判 ｜ 诚实放弃 2 行 0.007 ｜ (not 30) 换说法 0.043、3 行都被算、错步 0.67 ｜
         不搜直接猜错 0.00 ｜ 搜得对没解出 0.14
顺序     任何对 ≥ 0.35 > 搜得对没解出 0.14 > 诚实放弃 ≈ 0.01 ≈ 不搜猜错 0.00 > 假宣告 −0.20
--judge v2 现在跑的是 v2.1（启动打印标 v2.1）。md5 grpo_mix_mp f2…（见下）、judge_v2 b406c4ba…
待   臂 3 跑完 → 九步（「v2 被钻」的数据点）；看 ckpt_best 样例确认躲罚的写法；3′ 用 v2.1；若时间够 3b = mix4 + v2.1 配对
```

### 用户：「目标应该也是一把秤」（2026-09-05）—— 更狠：对模型来说目标只是那把秤
```
我们心里的目标模型看不到，它看到的只有那个数。臂 3 = 演示：目标「搜得诚实算得准」→ 秤「标注行里少出错」→ 模型的目标「别写标注行」
层级：人心里的意图（写不全）→ 评测秤（池外、换模板、pass@64、看样例，也是代理）→ 训练秤（判分器，模型的全部目标）
意图和训练秤的缝只能靠评测秤发现，发现一条补一条，补不完只能越补越小。定目标 = 造秤；造完还得一直审。最大尺度上这叫对齐，我们碰到的是迷你版
```

### 教学：什么是对齐（2026-09-05）
```
一句话   让模型实际追的目标跟我们真正想要的一致。模型只看得到秤 → 对齐 = 把意图写成钻不了的秤 + 检查学到的是意图还是漏洞
本项目里的四例
  解出来 → 第一个 answer + 部分分 → 放弃也交错式子              specification gaming
  验证过再宣告 → 宣告不罚 → 自信错答                            reward hacking
  搜得准 → 标注行少出错 → 不写标注行（臂 3）                     reward hacking
  会做题 → 只在 R1 模板训 → 换模板回 Base                         学到相关物、泛化不同
工业界   秤 = 奖励模型（人的偏好训的）。谄媚 = 人给「顺着我说」高分 → 奖励模型学会 → 策略模型奉承。跟假宣告同一机制
研究     外对齐（秤对不对，臂 3）；内对齐（学的是秤还是换场景就分道的相关物，绑定/塑走）
为什么难  意图写不全；优化器专找缝、越强找得越快（臂 3 一百步）；不可验证领域验证比生成难；评测有采样半径
本项目 = 对齐的迷你版：造秤 → 被钻 → 审样例 → 补秤 → 再被钻。差别只在秤的内容：式子对不对 vs 什么对人类好
```
### 臂 3 step 175 样例：躲罚的写法确认（2026-09-05）
```
每一行都成了「expr = V (not 30)」：把 too high/low 换成 not 30，正则抓不到，四个过程项全归零。搜索本身没少（6~10 行），重复照旧，算错照旧（6+18+42−12 写 56，两条都这么错）
新怪癖：先说「not possible」再交答案 —— v2 下的最优策略：从不宣告 perfect（没假宣告罚），但一定交式子（对了 1.0，错了 0.10）。文字和答案脱钩，秤没称一致性
v2.1 罩得住：ATTEMPT 认 (not 30)，3 条样例 25 行全被算进去（约 12% 算错）；canon 抓到 6+12−18+42 vs 6+12+42−18 这种重复
```

## Hack 总账：钻了什么、漏了什么意图、怎么补（2026-09-05）
| # | hack 的表现 | 秤的漏洞 | 没写进去的意图 | 补法 | 状态 |
|---|---|---|---|---|---|
| 1 | 放弃也交个错式子 | 错式子 0.25 > 交白卷 0.05 | 不会就说不会，答案要有依据 | 阶梯缩到 0.10（v2） | 部分补 |
| 2 | 自信错答：算出 18 写「= 30 (perfect match)」 | 宣告不罚 | 宣告要真 | 假宣告 −0.2（v2） | 已补 |
| 3 | 长而对输给短而错，第 2 层被剪 | 长度罚对错一视同仁 | 任何对 > 任何错，长度只是次要 | 对的只轻扣，错的才吃长度罚（v2） | 已补 |
| 4 | 多写行提高命中、精度被稀释（臂 0/2 精度 0.456 → 0.40） | 只认结果，不看行 | 算对 | 错步率 −0.1（v2）+ 算对行 +0.01（v2.1） | 已补 |
| 5 | 重复用数：42 + 18 * 12 / 18 | 只查最终 answer 的数字 | 每次尝试都合法 | 违规行 −0.05（v2） | 已补 |
| 6 | 不记账：同形态反复试 | 重复没成本 | 别重复 | 重复行 −0.05（v2） | 已补 |
| 7 | 改标签躲罚：too high → (not 30)（臂 3，100 步内） | 过程项只认一种格式 | 过程项针对行为不是格式 | 尝试行认任意标签（v2.1） | 已补，待验 |
| 8 | 不搜直接猜（跑前预测到的洞） | 错步率用率，0 行 = 0 罚 | 不确定时要搜 | 答错且 < 3 行按缺行数罚（v2.1） | 已补，待验 |
| 9 | 说「not possible」又交答案（臂 3） | 文字和答案不查一致 | 说了不可能就别交，交了就别说不可能 | 未补。候选：不一致 −0.05 | 待定 |
| 10 | 判断句被塑走（塑成窄启发式） | 结构没回报 | 保留系统搜索的结构 | 老师 v2：每行都是尝试；算对行 +0.01 | 老师 v2 待做 |

### 探针被绕的（不是奖励 hack，是我们自己的盲区）
| 表现 | 原因 | 补法 |
|---|---|---|
| 标注只数到每段第一条 | 正则吞了上一行的 ")\n" | 算式必须以数字或 ( 开头 |
| 「含 ×÷」0.97 | 正则抓到题面复述「+, -, *, /」 | 运算符两边要有数字 |
| ★差、0 次桶归零 | 行为普及后对照组消失 | 探针退役，换分层精度 / 假宣告率 |
| 贪心 eval 藏涨幅 | 众数不变分布变 | 用训练温度抽样评测 |
| 幻觉续写「User:」撑满长度 | 生成不停 | 停止串 + 补 EOS |

```
规律   ① 每个 hack 都是秤里少写的一句话，写进去它就换下一句
       ② 补丁分两类：能验证的进训练秤（2 3 4 5 6 7 8），不能验证的进评测秤和人眼（9 10 的一部分）
       ③ 越强的模型钻得越快：臂 3 一百步找到 (not 30)。审计不能停
       ④ 一次只换一层：v2 是判分器，老师 v2 是教材，分开验才知道哪个起作用
```

## 臂 3 完成（mix4-300 + v2 判分器【未含 v2.1】，300 步，115 分钟，2026-09-05）
| eval（贪心） | 起点 | 75 | 250 | 300 |
|---|---|---|---|---|
| CD 正 | 0.45 | 0.45 | 0.48 | **0.51** |
| CD 长 | 220 | 186 | 237 | **415** |
| 标注/条 | 6.2 | 4.1 | **0.0** | **0.0** |
| GSM | 0.84 | 0.83 | 0.82 | 0.82 |
v2 各项（前 20 → 后 20）：假宣告 0.11 → 0.04，错步 0.40 → 0.02，重复 0.96 → 0.06，违规 0.05 → 0.01 —— 全是「没了」不是「对了」
ckpt_best = 250（0.680）；latest = 300（0.630，长度涨了罚回去）。并表用 latest

```
① 从头到尾被钻：标注归零，四个过程项跟着归零，正确率贪心 0.45 → 0.51 跟臂 0（0.45 → 0.50）一个水平。v2 的过程项对结果零贡献
② 长度末段涨到 415（贪心），训练均长 252 —— 跟其他臂一样的均衡点，贪心在个别题循环拉高
③ 这条臂的价值：「过程奖励只认格式就会被改格式绕开」的完整数据点，也是臂 3′/3b（v2.1）的对照
④ analyze_search 加 ③c：用 judge_v2.terms 的任意标签口径量尝试行精度、重复、违规、假宣告 → 臂 3 这种改了标签的也能量
待   臂 3 九步（挤在臂 3′ 旁边跑）；analyze_search 五文件（mix4 + 臂 0/1/2/3）看 ③c
```

## 臂 3′ 完成（out_sft1 + v2.1 判分器，300 步，118 分钟，2026-09-06）
| eval（贪心） | 起点 | 50 | 275 | 300 | 臂 1（v1）300 |
|---|---|---|---|---|---|
| CD 正 | 0.52 | 0.47 | 0.46 | **0.50** | 0.52 |
| CD 长 | 487 | 273 | 196 | **198** | 302 |
| 标注/条 @ 精度 | 12.3 @ 0.64 | 7.8 @ 0.60 | 4.7 @ 0.81 | **4.8 @ 0.80** | 9.2 @ 0.55 |
| GSM | 0.82 | 0.84 | 0.83 | 0.83 | 0.83 |
训练探针（前 20 → 后 20）：标注 12.1 → **5.15**（没塌，臂 3 是 0.17）、精度 0.517 → **0.657**（臂 1 是 0.525）、3~5 桶 0.91 → 0.82、6+ 桶 0.32 → 0.29、CD |A| 0.31 → 0.12
v2.1 各项：假宣告 0.093 → 0.060、错步 0.327 → **0.263**、重复 1.09 → **0.10**、违规 1.52 → **0.10**、对长/错长 179/862 → 168/272

```
① v2.1 的三个补丁都起效：标注没塌（地板 + 加分）、重复和违规砍到 0.1、算对率 0.52 → 0.66（贪心 0.80）
② 假宣告只从 9% 到 6%：温度 1 下还有 6% 的轨迹自信错答。−0.2 的罚不够狠，或者它换了宣告词（③c 会说）
③ 贪心正确率 0.52 → 0.50 平，跟臂 1 一样。|A| 塌到 0.12，后半程空转 —— 过程项修好了，结果分的抓手还是那么多
④ 长度 198 比臂 1 的 302 短，标注 5 比 9 少：模型写得更少、更准。第 2 层保没保住要看 held-out 的 CD-3 pass@8 和 scan_ops
预测 held-out  CD-4 pass@1 0.55~0.58 / @8 0.74~0.77；数字用全 ≥ 0.95；CD-3 @1 0.52 / @8 0.78~0.85 / @64 0.95~0.97；③c 算对率 ≥ 0.75；假宣告 < 10%；长度中位 ~180
判决线（跑前定的）CD-4 ≥ 0.56 且 CD-3 pass@8 ≥ 0.85 且数字用全 ≥ 0.93
```

## 臂 3 / 臂 3′ held-out（2026-09-06）—— 过程奖励让它更干净、更窄
### 配对一：臂 3（mix4 + v2，被钻）vs 臂 0（mix4 + v1）
| | 臂 0 | 臂 3 |
|---|---|---|
| CD-4 pass@1 / @8 | 0.471 / 0.710 | 0.416 / 0.670 |
| CD-4 全对 / 全错 / 数字用全 | 12 / 29 / 0.955 | 11 / 33 / 0.915 |
| CD-3 pass@1 / @8 | 0.511 / 0.700 | 0.486 / 0.690 |
| CD-4 长度中位 | 237 | 272 |
→ 被钻的 v2 没带来任何好处，pass@1 还略低（噪声边缘）。样例里全是「(not 30)」，还出现「The correct combination is … = 46 (not 30)」这种自相矛盾

### 配对二：臂 3′（sft1 + v2.1）vs 臂 1（sft1 + v1）
| | 臂 1 SFT 后 | 臂 1（v1 RL） | **臂 3′（v2.1 RL）** | 判决线 |
|---|---|---|---|---|
| CD-4 pass@1 / @8 | 0.506 / 0.720 | 0.563 / 0.750 | **0.544 / 0.660** | ≥ 0.56 ✗（差 0.02） |
| CD-4 全对 / 全错 | 22 / 28 | 29 / 25 | **34 / 34** | |
| CD-4 数字用全 | 0.61 | 0.921 | **0.970** | ≥ 0.93 ✓ |
| CD-4 长度中位 / P95 / 撞顶 | 344 / 1024 / 28% | 205 / 596 / 0 | **212 / 275 / 0** | |
| CD-4 组内有差异 | 78% | 67% | **41%** | |
| CD-3 pass@1 / @8 | 0.599 / 0.930 | 0.503 / 0.740 | **0.444 / 0.630** | @8 ≥ 0.85 ✗✗ |
| CD-3 全对 / 全错 | 38 / 7 | 39 / 26 | **34 / 37** | |
| GSM R1 / 中性 | 0.755 / 0.565 | 0.755 / 0.50 | 0.755 / 0.455 | |

```
① 干净了：数字用全 0.97、撞顶 0、精度贪心 0.80、重复违规 0.1。v2.1 要的过程指标全达标
② ★★ 窄了：P95 长度 275 = 几乎没有一条轨迹走过加减穷举之后的乘除段。乘除行算错多、重复多、违规多 → 全被罚 → RL 学会「加减扫完就交答案」
   3 行地板它用 8 行加减扫就满足了，之后没有任何压力让它进乘除；乘除那步错误率高，对结果分的期望收益又低（p ≈ 0）→ 不去最划算
③ 分布塌了：全对 34 + 全错 34，组内有差异只剩 41%；三条样例几乎逐字相同。pass@8 因此掉（0.75 → 0.66），CD-3 更惨（0.74 → 0.63，全错 37）
④ 这是 Goodhart 的另一张脸：不是改格式躲罚（臂 3），是【放弃有风险的那一步】躲罚。罚错误 = 罚尝试。过程惩罚只在结果够得着的地方是安全的
⑤ GSM 中性 0.565 → 0.50 → 0.455：SFT 漏过模板的那点被 RL 一步步绑回去，三个点单调，「SFT 漏、RL 绑」现在比较像真的了
预测（待 pass@64）  臂 3′ CD-3 pass@64 0.85~0.90 —— 乘除形态不再被抽，天花板会从 0.95 缩回去。若真如此 = 过程惩罚缩小了可达集
下一版判分器方向（先记）  过程项只奖不罚（算对行 +，错行 0）；或惩罚只对「加减扫完仍不进乘除且答错」；或乘除行的算错不罚。核心：别让「不尝试」成为最优
```

## 臂 3 / 3′ pass@64 + 判分器实验结案（2026-09-06）
| CD-3 池外 | k=1 | k=8 | k=64 | 全错 64 次 |
|---|---|---|---|---|
| 臂 0（mix4 + v1） | 0.520 | 0.705 | 0.830 | 17 |
| 臂 3（mix4 + v2，被钻） | 0.479 | 0.674 | **0.800** | 20 |
| 臂 1 SFT 后 | 0.598 | 0.926 | 1.000 | 0 |
| 臂 1（sft1 + v1） | 0.507 | 0.761 | 0.950 | 5 |
| 臂 3′（sft1 + v2.1） | 0.461 | 0.646 | **0.900** | 10 |
GSM：臂 3 R1 0.75 / 中性 0.535（z −1.43 不显著）；臂 3′ 0.755 / 0.455

```
① 预测「臂 3′ pass@64 0.85~0.90」→ 0.90 ✓。过程惩罚把可达集从 0.95 缩到 0.90：乘除形态不再被抽，10 道题重新够不着
② 三版判分器的总账（同起点 sft1）：
      v1     pass@1 0.563 / CD-3 @8 0.74 / @64 0.95 / 数字用全 0.92 / 撞顶 0 / 假宣告 26%
      v2.1   pass@1 0.544 / CD-3 @8 0.63 / @64 0.90 / 数字用全 0.97 / 撞顶 0 / 假宣告 ~6%
      → v2.1 每个过程指标都更好，每个结果指标都更差。「干净」是用「窄」换的
③ 为什么：硬题上结果分给不了梯度（p ≈ 0），任何按行的惩罚在那里就是全部信号 → 最优策略变成「不去那里」。
   过程惩罚只在结果够得着的地方是安全的；在够不着的地方它把探索罚没了
④ 判分器实验结案：三版秤，两版被钻（改格式、弃风险步），一版被用满（v1 的循环和假宣告）。秤能修的是「怎么错」，修不了「能不能对」
   能不能对 = 算术 + 乘除形态的执行 = 零件和教材的事
⑤ 记一个 v3 方向不跑：过程项只奖不罚（算对行 +、错行 0）+ 惩罚只对「加减扫完仍不进乘除且答错」。别让「不尝试」成最优
建议   判分器线到此收，下一刀回到算术：老师 v2（一行一步 + 纯机械第 1 层）或计算器。那是墙所在的地方
```

# ★★★ 五臂总表（臂 0 / 1 / 2 / 3 / 3′，2026-09-06）
| | mix4-300 参照 | 臂 1 SFT 后 | 臂 0 只 RL v1 | 臂 2 自蒸馏+RL v1 | 臂 1 撒种+RL v1 | 臂 3 mix4+v2 | 臂 3′ sft1+v2.1 |
|---|---|---|---|---|---|---|---|
| CD-4 pass@1 / @8 | 0.393 / 0.670 | 0.506 / 0.720 | 0.471 / 0.710 | 0.473 / 0.690 | **0.563 / 0.750** | 0.416 / 0.670 | 0.544 / 0.660 |
| CD-4 全对 / 全错 | 10 / 33 | 22 / 28 | 12 / 29 | 15 / 31 | 29 / 25 | 11 / 33 | 34 / 34 |
| CD-4 数字用全 | 0.951 | 0.61 | 0.955 | 0.948 | 0.921 | 0.915 | **0.970** |
| CD-4 长度中位 / 撞顶 | 222 / 0.9% | 344 / 28% | 237 / 0.5% | 254 / 0.4% | 205 / 0 | 272 / 0.5% | 212 / 0 |
| CD-3 pass@1 / @8 | 0.476 / 0.660 | **0.599 / 0.930** | 0.511 / 0.700 | 0.518 / 0.720 | 0.503 / 0.740 | 0.486 / 0.690 | 0.444 / 0.630 |
| CD-3 pass@64 / 全错 | （600：0.830 / 17） | **1.000 / 0** | 0.830 / 17 | 0.840 / 16 | 0.950 / 5 | 0.800 / 20 | 0.900 / 10 |
| GSM R1 / 中性 | 0.74 / 0.46 | 0.755 / 0.565 | 0.75 / 0.49 | 0.77 / 0.49 | 0.755 / 0.50 | 0.75 / 0.535 | 0.755 / 0.455 |
| 标注精度（贪心 eval） | 0.53 | 0.64 | 0.58 | 0.50 | 0.55 | —（改标签） | **0.80** |
| 假宣告轨迹占比 | 15% | 14% | 23% | 25% | 26% | ~4%（不宣告） | ~6% |

## 五臂定论（在三臂八条之上加四条）
```
⑨ 判分器 v2 被改格式钻空（100 步内标注归零），结果 = 臂 0 略低。过程奖励只认格式必被绕开
⑩ 判分器 v2.1 堵住改格式，过程指标全达标（数字用全 0.97、精度 0.80、假宣告 6%），但结果全线退：CD-4 @8 −0.09、CD-3 @8 −0.11、@64 0.95 → 0.90
   机制：硬题上结果分无梯度，按行惩罚成了全部信号 → 弃掉有风险的乘除步 → 可达集缩小。罚错误 = 罚尝试
⑪ 三版秤修的都是「怎么错」，没有一版动了「能不能对」。能不能对 = 算术 + 乘除形态执行 = 零件和教材
⑫ GSM 中性列：SFT 后 0.565 → v1 RL 0.50 → v2.1 RL 0.455，单调回落；无种臂 0.49~0.535 无规律。「SFT 漏过模板、RL 绑回去」成立但幅度小
最好的模型仍是臂 1（撒种 + v1 RL）：CD-4 0.563 / 0.750，CD-3 @64 0.95。它的短板（假宣告 26%、乘除执行）留给算术那一刀
```

## 假设秤修好，瓶颈在种：怎么解（2026-09-06）
```
「种」的瓶颈拆成两半：教材（老师写的形态模型执行不了：判断句、第 2 层丢第四个数、乘积表列不全）+ 零件（每行算术 45% 错，稳定错误信念）
三个杠杆
① 换更好的种 —— 老师 v2                                                    穷举器改 + SFT 30 分 + RL 2.5 h
     一行一步：42 + 12 = 54; 54 − 18 = 36; 36 − 6 = 30 —— 每步两操作数，落在 1.5B 可靠范围
     第 1 层纯机械：6 个两数乘积按下标顺序、每个配 4 种 ±，不排序不写 skip 句
     第 2 层每行显式带第四个数；第 1、2 层各过采样到 2000
     预期：精度 0.55 → 0.8+；第 2 层成为「每行都可能命中的尝试」→ RL 留而不剪；天花板保住 1.00
② 护住种 —— RL 的 KL 参照换成 SFT 模型                                       改一个参数，零新代码
     现在 ref = Base、β 0.0005 = 对种零保护，塑走是必然。工业界 RLHF 的标准做法是 KL 对齐到 SFT 策略
     加 --ref-init out_sft1/ckpt.pt，β 0.01~0.05 → RL 只能在种的邻域里选，剪循环可以、改写第 1 层不行
     预期：CD-3 pass@8 从 0.74 回到 0.85+，代价是 pass@1 涨得慢一点
③ 补种缺的零件 —— 算术
     计算器工具（精度 → 1.0，顺便搭 agent harness，1 天）；或算术 mid-training（几亿 token，5 h）；或 7B
顺序   ① + ② 一次跑（都便宜，且互补：好种 + 护种），判分器用 v1 或 v3（只奖不罚版）。之后 ③ 计算器
```

## ③b / ③c 四模型并排（CD-4 池外，2026-09-06）
| | mix4-300 | 臂 1（v1） | 臂 3（v2 被钻） | 臂 3′（v2.1） |
|---|---|---|---|---|
| ③c 尝试行/条（任意标签） | 5.53 | 7.95 | 7.23 | 5.37 |
| ③c 算对率 | 0.458 | 0.497 | 0.484 | **0.671** |
| 重复行/条 | 1.58 | 1.33 | **2.72** | **0.06** |
| 违规行/条 | 0.06 | 0.26 | 0.17 | **0.02** |
| 假宣告轨迹 | 0.166 | 0.250 | 0.029 | 0.040 |
| 写了宣告的轨迹 | 0.46 | 0.81 | **0.075** | 0.555 |
| 答错且尝试 < 3 行 | 0.076 | 0.025 | 0.094 | **0.011** |
| ③b 标注行数（只认 too high/low） | 4049 | 6337 | **184** | 4282 |

```
① 臂 3 坐实：③b 只剩 184 行，③c 却有 7.2 行/条 —— 尝试没少，标签换了。而且重复 2.72/条是四者最差：v2 的重复罚只算 ANNOT 行，
   改了标签就连重复也不罚了，「多写行」的老倾向反而放开。宣告率 7.5%：不宣告就没假宣告罚，也是躲
② 臂 3′ 的过程指标是真的：尝试 5.4 行/条跟 mix4 一样多（不是靠不写），算对率 0.46 → 0.67，重复 1.58 → 0.06，违规 0.26 → 0.02，假宣告 4%
   → v2.1 在过程层面完全达成设计目标。代价在结果层：弃了乘除段（前一节）
③ ★ 算对率 +0.17 说明每行算术是 RL 能动的（在采样半径内选更准的变体 + 写更容易算的行）。之前「RL 动不了算术」说重了：
   动得了一截，但动的方式是「少写难算的行」，跟探索直接冲突
④ 臂 1（v1）：81% 的轨迹写宣告、25% 是假的 —— v1 教出「一定要宣告」
```

# ★★★ 决策点 2：五臂之后（2026-09-06）
## 做了什么（09-04 决策点 1 之后）
| 臂 | 做法 | 一句话结果 |
|---|---|---|
| 0 | mix4 再 RL 300 步，v1 | +0.08 pass@1；天花板 0.83 不动；精度反掉 |
| 2 | 自蒸馏 SFT + RL | = 臂 0，贡献 0 |
| 1 | 程序老师 SFT + RL | SFT 后天花板 1.00；RL 后 CD-4 最好 0.563，但迁移任务被塑走、假宣告 26% |
| 3 | mix4 + 判分器 v2 | 100 步内改格式钻空 |
| 3′ | sft1 + 判分器 v2.1 | 过程指标全达标，结果全退，天花板 0.95 → 0.90：罚错误 = 罚探索 |
工具：enum_traces / reject_sample / sft_qwen / check_sft / judge_v2 / analyze_search ③c。PLAN.md 7268 行
理论：p 与 pass@k、撒种 vs 收紧结案、秤 = 目标、hack 总账 10 条、对齐迷你版

## 接下来的方向
| | 做什么 | 回答什么 | 代价 | 我的排序 |
|---|---|---|---|---|
| A | 老师 v2（一行一步 + 机械第 1 层 + 显式第四数 + 过采样）+ RL 的 KL 锚到 SFT 模型 | 撒种能不能做得 RL 剪不掉；一行一步不用工具能把精度提到多少；天花板能不能过 RL 保住 1.00 | 1 天代码 + 3 h GPU | **1** |
| B | 计算器工具：停、注入、续、mask | 算术清零后搜索能走多远；工具行为能不能 RL 出来；顺便搭 agent harness | 1~2 天 + 2 h | **2** |
| E | Agent RL 猜数字 | 目标三；多轮信用分配；长度随轮数 | 2~3 天（有 B 的 harness 后 1 天） | 3 |
| F | 写博客 | 把这两周讲清楚 | 半天 | A 跑完就写 |
| C | 臂 5 长链算术 | H1「越训越长」 | 1 天 + 2 h | A 的长度结果不明确再做 |
| D | 判分器 v3 只奖不罚 | 消融 | 3 h | 有空再做 |
| G | 多种子重复关键臂 | 论文级置信 | 6~9 h | 要投稿再做 |
| H | 算术 mid-training | 「能不能装知识」独立实验 | 5 h | 独立，随时 |
推荐   A → B → E；F 在 A 之后。理由：A 直接对着两个已证实的瓶颈（教材形态、塑走），且便宜；B 解决第三个（算术）并给 E 搭路

### 决策点 2 的选项：本质动哪一层、新信息从哪来（2026-09-07）
| | 做什么 | 本质动哪一层 | 新信息从哪来 → 注入到哪 |
|---|---|---|---|
| A | 老师 v2 + KL 锚到 SFT | **种**（更好的教材）+ **筛子的范围**（KL 锚 = 不许筛出种的邻域） | 穷举器写的机械程序 + 一行一步的算术分解 → 进模型（它自己产生不了）。对我们：撒种能不能过 RL 不掉 |
| B | 计算器工具 | **零件外包** + **环境**（多一个观测通道） | 算术不再由模型产生，harness 逐步给准确值 → 进上下文不进权重；模型学的是「何时调、调什么」这个行为。对我们：算术清零后搜索的真实上限 |
| E | Agent RL 猜数字 | **环境**（换地 + 多轮观测） | 环境每轮反馈 → 进上下文；模型学「怎么用反馈」。对我们：多轮信用分配、长度随轮数 |
| F | 写博客 | 无，**评测/传播** | 不注入模型；把我们自己的理解压成可传播的形式 |
| C | 长链算术 | **环境**（换地：一个真需要长链的任务） | 任务结构本身 → 长度是否按需长。对我们：H1 的直接答案 |
| D | 判分器 v3 只奖不罚 | **秤** | 算对行的正反馈 → 进权重。对我们：过程奖励能不能净正 |
| G | 多种子重复 | 无，**测量** | 不注入；量方差，让 0.02~0.05 的差有没有意义变得可判 |
| H | 算术 mid-training | **材料**（零件内化） | 几亿 token 合成算术 → 进权重，模型自己会算。对我们：知识能不能这样装 |
```
按五个旋钮：A = 撒种（+ 收紧筛子范围）；B、C、E = 换地/换环境（B 多一个「外包」，五旋钮里没有，是 harness）；D = 换奖励；H = 换材料；F、G 不动模型
真正给模型灌新信息的只有 A（程序）、H（算术）、B/E（观测进上下文）；D 只是把已有的选得更准；C 是换个地方让已有的东西显形
```

# ★★★ 计划 B：计算器工具（2026-09-07 定）
## 要证什么
```
① 算术清零后，撒种模型的搜索能走多远：解那行算错的题（CD-4 全错里 4 道纯加减、乘除段的算错）能不能回来
② 工具行为能不能 RL 出来：该算的时候调、调完按结果标方向、不自己编 <result>
③ 搭起「停、注入、续、mask」的 harness —— agent RL（E）的基建
对照：臂 1（同一老师结构、没工具）。唯一变量 = 每行的值由谁算
```
## 设计决定
```
协议    模型写 <calc>42 + 12 - 18 - 6</calc> → 生成停 → harness 用 safe_eval 算 → 接上 <result>30</result> → 继续生成
        结果整数直接写，非整数写小数（2.8）；算不了写 <result>error</result>。每条最多 16 次调用
种      不能靠提示词引（1.5B 从没见过 <calc>，p ≈ 0）→ 用穷举器写带工具的轨迹做 SFT：
        每行「<calc>expr</calc><result>V</result> (too high)」。老师结构不变（跟臂 1 同一版），只把值改成工具算的 → 单变量
        GSM 数据照旧（无工具），工具行为只在 CD 上学
mask    ① SFT：<result>…</result> 那段 label 设 −100，模型不学「自己写结果」
        ② RL：rollout 记下注入的 token 区间，policy loss 和 KL 都跳过它们 —— 观测进上下文，不进梯度
防钻    模型自己写出 <result>（不是 harness 注入的）= 假结果 → 罚 −0.2，且 harness 一见 <result> 就停不注入
        调用次数封顶 16；工具行的 (too high/low) 标反照旧算错步
判分器  v1 不变 + 假结果罚。跟臂 1 干净对照
提示词  模板加一句「You can compute with <calc>expr</calc>; the result will be given as <result>…</result>」，--tool 时才加
探针    新加：调用/条、工具报错率、假结果率、「结果 30 却标 too low」的无视率；ANNOT 换成工具格式的版本
```
## 步骤
```
1  calc_tool.py         evaluate()；generate_with_tools(model, tok, prompts) → 文本、ids、注入区间。批内各条独立停/续，多轮直到 EOS 或上限
                        GS01 上用 Base + 一段带 <calc> 示例的提示词测这个循环通不通
2  enum_traces.py --tool  同题号重生成 sft_cd3_tool / sft_cd4_tool（--ids），check_sft 体检
3  sft_qwen.py           encode() 里 <result> 段 label −100
4  grpo_mix_mp.py --tool  提示词、rollout 换 harness、mask 进 token_logp、假结果罚、探针和日志列
5  baseline_countdown / chat_countdown --tool   同一个 harness，否则评测时模型写 <calc> 没人应
6  冒烟：SFT smoke；RL 2 步看轮数、每步秒数、显存
7  T0 = SFT（30 分）→ SFT 后九步 → T1 = RL 300 步（rollout 多轮，估 6~8 小时，必要时 200 步）→ RL 后九步
8  对臂 1 并表
```
## 预测（跑前写）
```
T0 SFT 后   工具行精度 ≥ 0.98；调用 6~10 次/条；假结果 < 1%；CD-4 pass@1 0.55~0.60（臂 1 SFT 后 0.506：解那行算错的题回来）；pass@64 1.00；撞顶仍 20%+
T1 RL 后    CD-4 pass@1 ≥ 0.65；全错 ≤ 15（乘除题靠算准多解几道）；CD-3 pass@8 ≥ 0.85；假宣告 < 10%（结果是工具给的，不容易假宣告）
风险        rollout 慢 3~5 倍；模型学会自己写 <result>（罚）；无视结果照标方向；调工具刷长度（封顶 16 + 长度罚）
```

## 计划 B 进展（2026-09-07）
```
第 0 步   mix4-300 + 带 <calc> 示例的提示词，3 题 × 8 条：写出 <calc> 的 0/24；17/24 自己写了 <result>
          → 它抄的是示例里「结果」的样子，抄不出「调用」的样子。p(calc) = 0，必须 SFT。跟预测一致
第 1 步   calc_tool.py：generate_with_tools 多轮批生成，stop_strings=[</calc>, <result>, \nUser:]；抠式子 → safe_eval → 注入 <result>V</result>；
          记注入区间给 mask；自己写 <result> = fake 立停；调用封顶 16。假模型单测：两次调用、区间、mask、fake、EOS 全对
第 2 步   enum_traces.py --tool：所有值写成 <calc>expr</calc><result>V</result>，提示词用 tool_prompt。样例格式 OK
          （[2,3,5,7] 在自然序 + cap 40 下解不出，工具版非工具版都一样，是顺序的事不是工具的事；GS01 那批 15.5% 丢的就是这类）
第 3 步   sft_qwen.py：<result>…</result> 段 label −100（按段切开编码再拼）。单测过
第 5 步   baseline_countdown --tool / chat_countdown --template tool：生成走 harness，汇总多一块「工具」读数（调用/条、报错、假结果、轮数）
待        第 4 步 grpo_tool_mp.py（复制 grpo_mix_mp.py 加工具）；GS01 生成工具版教材 → SFT → SFT 后评测
md5      calc_tool d98cb4d1… enum_traces 4474f3cd… sft_qwen f2554e2b… baseline_countdown / chat_countdown 见本轮输出
```
第 4 步   grpo_tool_mp.py（grpo_mix_mp.py 复制 + 工具）：--tool 时 CD 提示词加工具说明、CD 的 rollout 走 calc_tool（GSM 照旧）、
          注入段 resp_mask=0 不进 loss/KL（pad_batch 带掩码，tot_tok 只数进 loss 的 token）、自己写 <result> 扣 --fake-pen 0.2、
          工具版标注探针只验方向、日志加「调 错 假果」、汇总加工具块。pad_batch 掩码单测过。grpo_mix_mp.py 原样未动
计划 B 代码全部就位，待 GS01：生成工具教材 → SFT → SFT 后评测（--tool）→ RL 冒烟 → RL
第 5.5 步 baseline_countdown.py --shard k/N + --merge（2026-09-07）：工具版评测一条要几十轮 generate（每轮重算整批 prefill），单卡 800 条太慢。
          --shard k/4 只跑第 k 片题（偏移后挪、题数等分、输出名带 _shardk），4 卡各跑 1/4；--merge 读各片 _raw.json 不载模型，
          pi 按分片偏移重排、工具计数/时间/轮数相加，写出标准名的汇总和 raw（analyze_search / scan_ops 照常读）。
          raw 每条新增 calls/err/fake 三个字段，raw 顶层新增 time/ntok/toolst。离线用假 torch 单测：分片取题、两片合并、名字、计数全对

## ★★★ 臂 T（工具版）SFT 后评测：CD-3 / CD-4 池外 800（2026-09-07）
4 卡分片（--shard k/2 ×2 任务，各片 30~34 分钟；harness 每轮重算 prefill，107~124 tok/s）
| | mix4-300 参照 | 臂 1 SFT 后 | 臂 1 撒种+RL v1（此前最优） | **臂 T SFT 后（工具）** |
|---|---|---|---|---|
| CD-4 pass@1 / @2 / @4 / @8 | 0.393 / — / — / 0.670 | 0.506 / — / — / 0.720 | 0.563 / — / — / 0.750 | **0.710 / 0.739 / 0.767 / 0.790** |
| CD-4 全对 / 有对有错 / 全错 | 10 / 57 / 33 | 22 / 50 / 28 | 29 / 46 / 25 | **65 / 14 / 21** |
| CD-4 有 answer / 数字用全 | — / 0.951 | — / 0.61 | — / 0.921 | 0.756 / 0.733 |
| CD-4 长度中位 / 均值 / 撞顶 1400 | 222 / — / 0.9% | 344 / — / 28% | 205 / — / 0 | 264 / 561 / 16.6% |
| CD-4 调用/条 / 报错 / 假结果 | — | — | — | 15.8 / 17 (0.13%) / 3 (0.4%) |
| CD-4 组内总分有差异 | — | — | — | **27%**（sd 0.066） |
| CD-3 pass@1 / @2 / @4 / @8 | 0.476 / — / — / 0.660 | 0.599 / — / — / 0.930 | 0.503 / — / — / 0.740 | **0.714 / 0.834 / 0.923 / 0.980** |
| CD-3 全对 / 有对有错 / 全错 | — | — | — | 44 / 54 / 2 |
| CD-3 有 answer / 撞顶 / 调用/条 / 报错 | — | — | — | 0.833 / 3.1% / 17.0 / 46 (0.34%) |
| CD-3 pass@64 | 0.830（600） | 1.000 | 0.950 | 未跑；全错 2/100 → 区间 [0.98, 1.00] |

预测对账：调用/条 15~25 ✓（15.8 / 17.0）；报错 <1% ✓；CD-3 pass@1 ≥ 0.6 ✓（0.71）；
          ✗ CD-4 pass@1 预测 0.50~0.55，实测 0.71（低估 0.16：硬题闭环那 3 道是挑出来的硬题，评测片大半是一层扫就中的题）；
          ✗ 撞顶预测 30~40%，实测 CD-4 16.6% / CD-3 3.1%。但「没 answer」24% / 17% 才是循环的真实口径：
            撞 48 次调用封顶的轨迹 harness 直接结束、不算撞 1400，所以撞顶率低估了循环。轮数/批 48.0（CD-3）= 每批都有条打满 48 次
读数
① 算术清零的净收益：CD-4 pass@1 0.506 → 0.710（+0.20，同一老师同一起点，只换值由谁算）；比此前最优的臂 1 RL 后还高 0.15，RL 还没跑
② CD-4 曲线极平：pass@1 0.71 → pass@8 0.79；全对 65 / 全错 21 / 有对有错只 14。三条样例逐 token 一模一样 → SFT 把扫法学成了确定性程序，
   温度 1 下几乎不变。好：可靠；坏：没探索，组内总分有差异只 27%，RL 只有 14 道题有梯度。全错的 21 道要么是第二层形态没轮到就撞顶，要么 p=0
③ CD-3 不平：0.71 → 0.98，54 道有对有错 —— 三数空间小，扫完就中；差在有没有被循环/48 次封顶耗光。这 54 道全是 RL 的菜
④ 报错 0.1~0.3%、假结果 0.4%：harness 协议学干净了。搜索措辞/条 10.9~12.5 = 每行一个标签
⑤ 速度：800 条 60~68 分钟（两卡之和）；RL 每步 CD 部分估 1.5~2.5 分钟，300 步 8~12 小时，冒烟后按实际步时定步数
RL 预测（臂 T RL 后，v1 判分器、len-soft 1100）：没 answer 24% → <5%（打满封顶的轨迹全是 0 分，RL 会学会提前收）；
   CD-4 pass@1 0.71 → 0.75~0.80、pass@8 0.79 → 0.80~0.85（探索少，天花板难抬）；CD-3 pass@1 → 0.85~0.90、pass@8 保 0.95+；
   风险：塑走第二层（臂 1 那样 @64 掉 0.05）；若出现，用新加的 --ref-init out_sft_tool/ckpt.pt --beta 0.02 跑 T′
工具：analyze_search / scan_ops 加了工具轨迹归一（<calc>E</calc><result>V</result> → E = V），旧探针照用；grpo_tool_mp 加 --ref-init

## 臂 T SFT 后：analyze_search + scan_ops（CD-4 池外 800，2026-09-07）
```
剂量响应   1~2 次 16 条 1.000 / 3~5 次 292 条 1.000 / 6+ 次 492 条 0.528（长度 803）→ 8 行符号扫描内解决的全对；要进乘除段的只剩一半
验证精度   写 perfect 的 584 条里真对 0.973（全体 0.710）→ 工具在手，「中了」认得准；2.7% 假宣告是「77 − 33 = 44 (perfect)」这种比较错
③b 精度    全部 0.892 / 没撞顶 0.921 / 撞顶 0.858；算错 0.079 是假的：harness 非整数只印 4 位有效数字（51.71），探针按 1e-6 比 → 每条除法行都算错
           → analyze_search / judge_v2 / check_sft 的算术核对改成相对容差 1e-3。真实口径：算错 0，方向反 2.9%（「14 * 7 + 5 − 33 = 70 (too low)」）
③c         尝试行/条 10.8、重复行/条 0.69、违规行/条 0.79（「33 + 14 + (33 / 7)」复用 33）—— 全出在乘除段
scan_ops   全错 21 道 = 只用 +− 就有解 7 道 + 必须乘除 14 道
           ★ 那 7 道是老师的覆盖漏洞：sign_lines 把最大数固定为正，只写 2^(n−1) 条；「b + c + d − a」（a 最大、其余三数之和更大）这一族永远不写
           必须乘除的 14 道：8 条全试过乘除（p(试)=1.0），一条没中。两条完整轨迹看：符号扫描 8 行完美；乘除段开始即兴 —— 复用 33、33/7 不整除、
           7*5*33 与 7*(5*33) 重复、方向标反、最后「77 − 33 = 44 (perfect match)」假宣告收尾。第二条在第 1 层乘除里绕到 1400 顶
           老师第 1 层乘除写的是「列乘积表 → skip 句 → closest first」，closest 要先知道值才能排，模型学不到只能编 → 这段是程序断掉的地方
```
定论
⑬ 工具把算术清零后，剩下的失败全在老师：覆盖（镜像符号行）+ 第 1 层乘除不可执行（closest-first 是泄漏式排序）。这跟五臂定论 ⑪ 一致，现在有了题号级证据
⑭ SFT 把老师程序学成了确定性策略：全对 65 道 8 条逐 token 相同。温度 1 下没有探索 → 全对 65 + 全错 21 = 86% 的组优势全 0，RL 在 CD-4 上只有 14% 的组有梯度
   全错组里唯一的方差是「写了错答案 0.10~0.25 vs 没写 0」→ RL 会学「封顶前随便写一个」，没 answer 率会降但是靠猜
决定   RL 冒烟照跑（验代码、量步时）；正式 RL 之前先做老师 v2（工具版），否则 10 小时 RL 大半在零优势的组里空转
老师 v2 清单（证据齐了）  ① 镜像符号行：sum(rest) > big 时补「rest − big」那族  ② 第 1 层乘除纯机械：按下标序列乘积/整除商，每个配 ± 其余数，不排序不写 skip 句
   ③ 不复用数、不整除不写、去重  ④ 第 2 层每行显式带第四个数  ⑤ 每层过采样；CAP 按 1400 token 反推（约 60 行）
   预期：CD-4 全错 21 → ≤ 10（7 道镜像直接回来），SFT 后 pass@1 0.71 → 0.78+；乘除段变成可执行程序后 RL 才有东西剪
```

## 计划 B 全流程（用户手写 8 步 + 4 处补正，2026-09-07）
```
0  先测再决定   mix4-300 + 示例提示词 24 条：写 <calc> 0 条、自编 <result> 17 条 → p(调用)=0，SFT 是必须不是选择。以后加任何新东西都先做这步
1  计划
2  写 harness   calc_tool.py 定协议：<calc>…</calc> 停 → safe_eval → 注入 <result>V</result> → 续；假 <result> 立停；注入段记区间
3  出教材       穷举器照协议写（值是老师算的，不是工具算的）；SFT 时 <result> 段 label −100：只学「写调用」，不学「写值」
              纸上 2、3 顺序反了：协议先定，教材照协议写
4  SFT         out_sft_tool（val 0.71 → 0.059）
5  SFT 后评测   汇总（4 卡分片）→ analyze_search → scan_ops → 抽硬题闭环
   ★ 5 → 6 之间是个环：评测 → 分析 → 定位在种/秤/零件哪一头 → 回去改那一头 → 再 SFT → 再评。这次定位在种（镜像行漏、乘除段不可执行）
6  RL          grpo_tool_mp --tool：rollout 走 harness、注入段 resp_mask=0、假果扣 0.2
7  RL 后评测    九条评测全带 --tool；评测前先写预测（第 5 步 5 条中 3 错 2，错的比中的有用）
8  对比         对象 = 臂 1：同起点、同老师、同 RL，唯一差别是值由谁算。SFT 后 0.506 → 0.710 已出；RL 后等第 7 步
```

## 种、秤、零件：分别是什么、本质是什么（2026-09-07）
```
             指什么                                   本质                                    谁给
种           模型写得出来的做法/形态：符号扫描、乘积表、   策略分布的【支撑集】：哪些轨迹概率 > 0。     教材（SFT）、预训练里见过的套路
             第 2 层配对……                             决定可达集 = pass@∞ = 天花板
零件         每一步依赖的基本能力：算术、比大小、抄数、     每一步的【执行可靠性】：写在轨迹里的一步    预训练 / mid-training 的电路；或外包给工具
             排序、认出「中了」                          真做对的概率。决定可达的轨迹能不能真落地
秤           判分器 + 环境 + 停止规则 = 目标本身          【优化目标】：RL 把概率质量往哪搬。          环境设计者（人）；可验证领域 = 标准答案
                                                        决定支撑集里哪些轨迹被放大、哪些被压
RL 自己只改一样东西：支撑集内部的【概率分配】（p 从低到高）。它不加支撑集（种）、不改每步的准确率（零件）、不定目标（秤）
所以每次评测后只有三个去处：p=0 的题 → 种；轨迹对了但执行错 → 零件；放大了不该放大的 → 秤
计划 B 的位置：工具 = 把「算术」这个零件外包，零件精度 0.45 → 1.00；这一刀之后剩下的失败全落在种上（定论 ⑬）
```

## 决定：先跑臂 T 的 RL，老师不动（2026-09-07）
```
问   想跟臂 1 干净对比，是否先不改教材直接 RL？
答   是。臂 1 vs 臂 T 锁住老师，只差「值由谁算」，SFT 和 RL 两段都要对上才是完整的一组。老师 v2 另起一臂 T2，跟臂 T 比 = 锁住工具只差老师
     代价：臂 T 的 RL 大半在零优势组里空转（86%），预计增益小 —— 这本身也是结论（确定性策略 + 只有结果信号 = RL 没抓手）
     RL 期间 GPU 满，老师 v2 的代码和出教材（CPU）并行写，等 RL 完再 SFT
臂 T RL 命令 = 臂 1 命令换三处：grpo_tool_mp + --tool；--max-new 1024 → 1400、--len-soft 800 → 1100（注入段每次约 12 token、16 次约 200，补回来）
```

## 老师 v2（enum_traces.py --v2，2026-09-07，臂 T RL 跑着时写的）
```
改了什么（全部有题号级证据）
① 镜像符号行  最大数为负、其余之和更大时值为正的那族（25 + 22 + 20 − 30）。v1 永远不写 → CD-4 池 6.9% 的题只靠它
② 第 1 层纯机械  乘积表按降序对列全 → 上界 = 目标 + 全部数之和（一次调用）→ 超上界的块按数值读出来跳过 → 剩下的块按表序配剩余数写符号行（含镜像）
                 没有「closest first」：那个顺序要先知道值，模型学不到只能编（臂 T 乘除段复用数、不整除、重复，全是这么来的）
                 ×1 块（把 1 吸收掉）只在最大数上写一次：27*1+22+6 = 27+22*1+6 = 27+22+6*1 同值，v1 写三遍
③ 第 2 层一行一次调用  整条表达式交给工具「(27 + 22 - 1) / 6」，不再「先算链再收尾」两次调用，一行省一半 token
④ 第 2 层按形态排  最后一步 ÷ → × → ±。目标 ≤ 100 而乘积动辄几百，「把大数除回来」命中率最高（贪心覆盖量出来的）
⑤ 上限按字符  MAX_CHARS 2700 ≈ 1250 token（--max-new 1400 留头），不按行数
⑥ --only-level 2 --exclude 主文件  第 2 层过采样；--shuffle 留作探索旋钮（实测让覆盖率降 3 个点，这次不用）

贪心集合覆盖（本地 3000 题，同生成器同种子 = GS01 池的前 3000）
  链形态 ((a∘b)∘c)∘d 共 1536 种，每次挑覆盖最多未覆盖题的形态：7 行 63%（就是符号扫描 + 镜像）、12 行 75%、24 行 83%、45 行 90%、80 行 95.5%
  → 固定程序在 45 行预算内的上限约 90%，尾巴很平：任何固定顺序都差不多，绑住的是预算不是顺序。要过 90% 只能加预算或按值搜索
本地量出来（tool 格式）
| | CD-4 v1 | CD-4 v2 | CD-3 v1 | CD-3 v2 |
|---|---|---|---|---|
| 预算内解出 | 84.9% | **91.4%** | 95.3% | **100%** |
| 命中层 0 / 1 / 2 | 1824 / 459 / 264 | 2030 / 483 / 230 | 2151 / 380 / 327 | 2293 / 380 / 327 |
| 算式行数 中位 / P90 / 最长 | 4 / 19 / 40 | 5 / 17 / 37 | 2 / 8 / 20 | 3 / 7 / 20 |
| 主文件 2000 条 token 中位 / P90 / 最长（字符/3.2 估） | — | 163 / 577 / 900 | — | — |
消融：第 2 层自然序 89.3% → divfirst 90.8%；字符上限 3300 → 93.1%（要 --max-new 1700）；--shuffle 87.9%；×1 全写 90.8% / 不写 91.5% / 写一次 91.4%
预期（臂 T2 = out_sft_tool_v2）：CD-4 全错 21 → ≤ 10；SFT 后 pass@1 0.71 → 0.78+；乘除段成为可执行程序后重复行/违规行 → 0
GS01 出教材（CPU，RL 跑着时就能跑）
  python3 enum_traces.py --data data/countdown.json  --n 2000 --hi 80000 --tool --v2 --out sft_cd3_tool_v2.jsonl
  python3 enum_traces.py --data data/countdown4.json --n 2000 --hi 80000 --tool --v2 --out sft_cd4_tool_v2.jsonl
  python3 enum_traces.py --data data/countdown4.json --n 300 --hi 80000 --seed 1 --tool --v2 --only-level 2 --exclude sft_cd4_tool_v2.jsonl --out sft_cd4_tool_v2_l2.jsonl
  python3 check_sft.py sft_cd3_tool_v2.jsonl sft_cd4_tool_v2.jsonl sft_cd4_tool_v2_l2.jsonl
md5  enum_traces e00ede58… check_sft ded3de19…（check_sft 加了工具轨迹归一）
```

### 教学：rollout、harness 协议、HF、老师 v2 之后（2026-09-07）
```
RL 每步都要 rollout   数据不是事先有的，每步由【当前模型】现采：rollout → 判分 → 更新 → 丢掉这批 → 新模型再采（on-policy）。SFT/预训练数据固定不采样
                     后果：① rollout 是成本中心（臂 T 一步 190 s 里 180 s 采样，13 h 里 12 h）→ 值得换 vLLM
                           ② 环境全在 rollout 里（harness、判分器、停止规则），更新那段只是普通梯度下降
                           ③ 能学的由 rollout 决定：采不出的轨迹（p=0）进不了训练数据 = 「RL 不加种」的机制根源
                     我们一批 rollout 只更新一次（比值恒 1，最严 on-policy）；PPO 类一批用三四次靠重要性比值修，省不了太多
两端权重同步          采样端和训练端是两份权重时（vLLM / 单独的卡），每步更新完要把新权重灌回采样端，否则采的是旧策略。现在同进程同一个对象，同步免费
16 条哪条进 loss      全部 16 条都进，各自乘优势 A = r − 组均值：A>0 推高、A<0 压低、A=0 前向白算（空转）。不是「选最好那条」
                     例：12 对 4 错 → 对的 +0.225 × 12、错的 −0.675 × 4，ΣA = 0，错的每条被压的力度是对的 3 倍
harness 不在 SFT 里   SFT 是老师强制：整条一次喂入，逐位置预测下一 token，不吐字 → 无停注入续。<result> 段是输入不是目标（−100）。
                     只有模型自己吐字的地方用 harness：RL rollout、评测、chat、sft_qwen 末尾演示
教材不经 harness      老师用精确分数自己算值、按 harness 的格式写；判分器验最后答案；harness 只在运行时。老师从不写非整数 → 格式无冲突
harness 为何在教材前  依赖的是【协议】不是代码：标签串、停在 </calc> 后、注入 <result>V</result> 不换行、值的写法、调用上限。教材照协议写，
                     SFT 的 mask 正则也照协议。协议住在 harness 里且能用假模型单测 → 先钉协议再写教材。待办：穷举器 import calc_tool 的常量，一处定义
HF 是什么            Hugging Face：① Hub 模型仓库（Qwen/Qwen2.5-1.5B 是仓库名，下到 ~/.cache/huggingface）② transformers 库（模型结构代码、
                     权重加载、分词、generate）③ save_pretrained 的目录格式（config.json + safetensors + tokenizer），vLLM/SGLang 直接读。
                     我们的 ckpt.pt 是 torch.save 的裸 state_dict，要转。分层：PyTorch → transformers → 我们的脚本；vLLM/SGLang 是只管推理的执行引擎
零优势空转是怎么看出来的  A = 分 − 组均值；全对/全错的组 A 全 0。证据：评测 65 全对 + 21 全错、样例逐 token 相同、冒烟第 1 步正 0.75 但 |A| 0.00、
                     梯度平衡 CD 0.001 vs GSM 0.013。「确定性策略」= 16 条没差别可比；「只有结果信号」= 差别不进分数。两个缺一个 RL 都还有活干
                     ★ 修正（armT 前 8 步）：|A| 0.05~0.31 出现在 5 步 —— 确定性只在简单题；硬题乘除段是即兴的，16 条各绕各的，加上长度惩罚，
                     组内有差别。86% 零优势偏悲观；RL 在硬题上有抓手。预测上调：CD-4 pass@1 0.71 → 0.78~0.82
训练题不是 100 道     100 道是评测片 [95000:95100]；训练从 [0:90000] 每步抽 4 CD + 4 GSM × 16 条，300 步见 1200 道。65/14/21 只是比例的估计
老师能解还要模型干嘛   Countdown 是秤不是目标。量的是通用模型能把做法学进权重多少、带走多少：① 通用底座一个网络做所有任务，程序只解写它的那种题
                     ② 超出老师（按值调整搜索、剪枝）是 RL 该做的事，至今证据：只放大只剪不发明，这是关于 RL 的真结论 ③ 迁移（GSM 中性、crosstask）
vLLM 换评测六步       ① 另开 venv 装 vLLM（最大风险是版本）② export_hf.py：ckpt.pt → save_pretrained 目录 ③ calc_tool 加后端层：喂 prompt_token_ids
                     不喂字符串（注入段 id 要跟训练端一致）、stop=[…] include_stop_str_in_output、每条自己的 max_tokens、不分批、enable_prefix_caching
                     ④ 等价性：20 道贪心逐 token 比 HF，100 道温度 1 指标在噪声带 ⑤ baseline_countdown / chat_countdown 加 --engine vllm
                     ⑥ RL 同步权重放最后。预期：800 条工具评测 4 卡 30 分钟 → 单卡 3 分钟。vLLM vs SGLang：我们是精确前缀续写，两家都省得掉，
                     选 vLLM（资料多、离线接口直接、RL 同步例子多），装不上换 SGLang
三档加速             ① 去 Python 开销（停止串检查器每轮重建、整段重 decode）1.3~1.5× ② 轮间留 KV 3~5×（要 GPU 测）③ vLLM 10×+。这轮不动，跑完再换
```

## 待办 E（agent RL，多轮）补充（2026-09-07，用户：比较重要的实验）
```
chatbot 里的三样：用户提问 = 环境输入 x；模型回答 = rollout y；用户反应（赞/踩/采纳/改写/重问）= 打在 y 上的分 r → 聊天日志天然是 rollout + 奖励，飞轮成立
★ 用户的下一句话既是打分，也是环境的下一个观测 —— 跟 harness 塞回 <result> 同一性质。这就是多轮 agent RL 的形态：
   每轮：模型吐字 → 停 → 环境（用户 / 工具 / 游戏）回一段 → 注入 → 续；观测段 mask 不进梯度；奖励可以在最后一轮（结果）也可以逐轮（过程）
猜数字环境就是它的最小版：模型猜 → 环境回「大了/小了」→ 续；观测 = 方向；奖励 = 轮数越少越高；harness 就是现在的 calc_tool 换个「工具」
计划 B 已经把基建搭好（停、注入、续、mask、假观测惩罚、多轮批生成）；E 要加的只是：环境状态（秘密数字）、多轮奖励、以及 rollout 速度（vLLM）
预测：二分搜索涌现 → 平均轮数逼近 log₂N；风险：模型把「大了/小了」当装饰不当观测（跟 too high/low 一样要验方向精度）
```

### 教学：训练中贪心 vs 训练后温度 1；pass@1 与 pass@k 各量什么（2026-09-07）
```
训练中 eval 用贪心   仪表盘：一题一条 100 道一分多钟；结果确定没有采样噪声，第 50 步和 150 步的差就是模型的差。任务是看趋势、抓崩塌、挑 ckpt
训练后用温度 1      成绩单：pass@1 的定义就是「按自己的分布采一次答对的概率」= 各题 p 的平均。贪心量的是最可能那一条对不对，
                   RL 抬的是对的路径的概率质量，p 0.5 → 0.9 贪心可能一动不动（「贪心藏收益」）。RL rollout 是温度 1，评测条件跟训练条件对上
                   pass@8/@64 贪心给不出，一条路径没有 k
pass@1 vs pass@k    温度 1 的 pass@1 已经是分布的读数。pass@k = 1 − (1−p)^k，对 p 的变化不均匀敏感：
                   p 0.05→0.30：pass@1 +0.25，pass@8 0.34→0.94（硬题从偶然到能捞，pass@8 最敏感）
                   p 0.50→0.90：pass@1 +0.40，pass@8 0.996→1.0（中等题变可靠，pass@8 早饱和，只有 pass@1 看得见）← RL 主要做的是这一行
                   p 0→0：谁都量不到（不可达）
                   → pass@1 量 RL 把可靠性抬了多少；pass@8/@64 量天花板还在不在（RL 堆概率会把偏门路径挤到 0：臂 1 pass@1 涨、@64 1.00→0.95）；
                     两者之差 = 还有多少「能搜到但不稳」留给 RL
臂 T 按这个读        预测 pass@1 0.71 → 0.78~0.82、pass@8 0.79 基本不动；pass@8 掉了 = 塑走 → T′ 上 --ref-init 护种
臂 T 150 步快照      eval 总 0.641→0.769（全是长度：CD 长 695→431）、CD 贪心 0.63→0.60（噪声）、GSM 0.81→0.82；硬题长 900→600、调用 28→16、
                   报错 0 假果 0、KL 0.06~0.12 平稳；零优势步仍常见（154 步正 0.75 |A| 0 = 3 全对 + 1 全错）；步时 190 s → 90 s，剩 5 h
```

## vLLM 换评测：代码就位（2026-09-07，臂 T RL 跑着时写的，GPU 空了再测）
```
★ 文件分线（用户定）：正在跑的 RL 依赖的 calc_tool / baseline_countdown / chat_countdown 一律不动（md5 ecfdaa70 / 50da229c / 1a3b97f4）；
   vLLM 线另起名：calc_tool_vllm.py 2453c6a1、baseline_countdown_vllm.py a9799596、chat_countdown_vllm.py 430e7976、vllm_check.py 0e5d39d1、
   export_hf.py 70d801f0、tests/test_calc_tool_vllm.py。验证过、RL 切过去后再合并退役旧文件
calc_tool_vllm.py 后端层   HFBackend（原逻辑：左填充、整批 prefill）/ VLLMBackend（喂 prompt_token_ids、每条自己的 max_tokens、stop=[…]、
                       include_stop_str_in_output、enable_prefix_caching）；make_backend(engine, …)；generate_with_tools 传 HF 模型自动包成 HFBackend，
                       grpo_tool_mp / sft_qwen 的调用一行没改。cut_at_stop：两个后端的输出都截到「刚好含最早停止串」的最短 token 前缀，语义一致
                       plain_generate：非工具生成也走后端（停在 \nUser:）。tests/test_calc_tool.py：假 tokenizer + 假后端离线单测，全过
export_hf.py           ckpt.pt → HF 目录（safetensors + tokenizer + export_info.json），CPU 十几秒
vllm_check.py          ① 贪心 20 道逐 token 比 HF vs vLLM（报相同数、首个分叉位置）② 温度 1 100×8 报 pass@1/调用/报错/假/撞顶/用时（--hf-sample 比速度）
baseline_countdown_vllm  --engine vllm --hf-dir …（--gpu-mem 0.6）：不载 HF 模型，工具/非工具都走后端，bs 放到全量一次交给引擎
chat_countdown_vllm      --engine vllm --hf-dir …（--both 再 --hf-base-dir，两个实例共卡时 --gpu-mem 各 0.4）
GS01 步骤   ① 另开 venv 装 vLLM（先贴 nvidia-smi / python / torch 版本）② export_hf 导 out_sft_tool 和 out_armT/ckpt_latest
            ③ vllm_check（要空卡）④ 过了就用 --engine vllm 评臂 T：CD-3/CD-4 各 800 + pass@64
验收标准    贪心分叉率低（分叉位置靠后）；温度 1 的 pass@1、调用/条与 HF 差 ≤ 0.05；假结果、报错 0；800 条 ≤ 5 分钟
```

### 教学：vLLM 线各文件的作用，作者 / 秘书 / 打字机（2026-09-07）
```
作者 = 模型，只会一个字一个字往下说；秘书 = harness；打字机 = 后端 / 引擎，把字落到纸上
秘书的规矩（协议）  说到 </calc> 喊停 → 计算器算值 → 把 <result>30</result> 写进稿子 → 让作者接着说；作者自己编 <result> 记「假结果」并停；
                   胡说 User: 截掉；记着哪几段是秘书写的，训练时不算作者的功过（mask）
旧打字机 = HF generate   稿子每递回来一次都从第一个字重新打整页再接着打，越长越慢
新打字机 = vLLM          纸留在机器里只接着打新的几个字；几百份稿子同时排队，谁停了谁让位
原 calc_tool.py 秘书和旧打字机焊死一体；calc_tool_vllm.py 把两者分开，秘书的规矩一字不改，打字机可换
| 文件 | 场景里是什么 |
| calc_tool_vllm.py | 秘书 + 两台可换的打字机（多了换打字机的插座） |
| export_hf.py | 作者的脑子（权重）从我们自己的笔记本格式（ckpt.pt）装进通用标准盒子（HF 目录），新打字机才认 |
| vllm_check.py | 验收：同一作者同一批题两台打字机各打一份，逐字比差在哪差多少，掐表看快多少 |
| baseline_countdown_vllm.py | 考场：100 道 × 8 次判分统计，只多一个开关选打字机 |
| chat_countdown_vllm.py | 面对面出题聊天，同一个开关 |
| tests/test_calc_tool_vllm.py | 彩排：照剧本说话的假作者 + 假打字机，不用 GPU 验秘书每条规矩 |
RL 训练器没改：它把作者交给秘书时秘书看到是旧式作者对象，自动配旧打字机，行为同以前；打字机验收合格再给训练器换
```

## 臂 T RL 跑完（out_armT，300 步，2026-09-08）
```
过程   0~224 步约 6 h；225 步存盘时盘满崩（ckpt_latest.pt.tmp 写到 5 GB 断），清盘 55 GB 后 --resume 从 200 接，100 步 204 分钟（硬题步 100~136 s）
       ckpt_best = ckpt_latest = step 300（最后一次 eval 新高 0.7793）
eval（贪心，val [90000:90100]）  总 0.641 → 0.779（全是长度项）；CD 0.630 → 0.630（不动）；CD 长 695 → 479（−31%）；GSM 0.81 → 0.83
训练末段（温度 1）  调用/条 9~10（SFT 后评测 15.8）；报错/条 0.002 → 0.037（末段微升，评测盯）；假结果率 0.002；方向精度 0.97；6+ 桶 0.49 → 0.55
梯度平衡  CD |A| 0.05 → 0.08，GSM 0.13：GSM 主导（如预测）；KL 0.06~0.10 平稳
读法   RL 干了预测中的活：剪循环、提前收、协议没坏；正确率靠温度 1 的 800 条定（预测 pass@1 0.78~0.82、pass@8 ≈ 0.79、没 answer <5%、调用/条 ≈ 9）
评测   crosstask 用 HF 照旧；三条 CD（CD-4 800 / CD-3 800 / CD-3 @64）等 vllm_check 过了走 vLLM，不过就 HF 分片
```

### 教学：考场 / 答题机 / 秘书 —— 一次 vLLM 评测在进程里怎么走（2026-09-08）
```
考场（baseline_countdown_vllm）  印 800 张卷子，只管发卷、收卷、判分、统计，自己不会写不会算
答题机（vLLM，里层）            同时在几百张卷子上各写各的；只认三种停笔：写到「请帮我算」的记号 </calc>、写到「答完了」EOS、纸写满（预算）。
                                不认识计算器，不知道记号的意思，看到就停
秘书（calc_tool_vllm，外层）    唯一懂规矩的人：
   1 整摞交给答题机，全部写到各自停笔处还回来
   2 一张张翻：停在「请帮我算 42+18+12+6」的 → 计算器算 78 填进格子 → 放回「还在写」；「答完了」→「完成」；自己瞎填数的 → 盖「作弊」章；写满 → 盖「超页」章
   3 「还在写」那摞再交答题机；答题机每张夹着书签（前缀缓存），从刚填的 78 后面接着写，不重读
   4 最多 49 趟，每趟越来越薄
   5 全部完成交回考场：判分、pass@k、数调用次数、汇总
旧答题机（HF）差两处：没书签，每趟从第一行重读整张；一次只夹 32 张且要等 32 张全停才还，快的干等
「一个进程」= 三者同一房间，卷子当面递（只传 token id 列表）；答题机内部怎么写字（GPU）秘书看不见，秘书的规矩答题机不知道
RL 的房间多一个教练：每趟收卷后按分数改答题机的写法（训练端）；答题机和教练分家时每次要把改好的写法抄一份给答题机 = 权重同步
两层循环  里层吐字循环（generate 内部，每 token 一圈，EOS / 停止串 / 预算三种停法）永远在；外层工具循环（generate_with_tools 的 while）
         只在有 harness 或多轮时有，靠里层的停止信号触发；不调工具时外层只转一圈 = chat.py 直接输出。-i 交互也是外层循环，环境是人
触发信号是人定的协议（标签或专用 token），通过 SFT/RL 教给模型；前沿模型只是把文本标签换成词表里的专用 token，协议本身仍是人定
vLLM 五件要验的事  ① 装得上 ② 后端写对 ③ 数值等价（评测阶段）④ 权重同步 ⑤ 同卡显存共存（RL 阶段）
vLLM 0.8 V1 引擎坑  默认另起子进程跑引擎核心，本进程已初始化 CUDA 时只能 spawn → 主脚本被重跑一遍报 bootstrapping；
                   calc_tool_vllm 里设 VLLM_ENABLE_V1_MULTIPROCESSING=0 让引擎核心在本进程跑。首次起 torch.compile 30 s，之后有缓存
```

## vllm_check 通过（hf_sft_tool，卡 0，2026-09-08）
```
| | HF | vLLM |
| 贪心 20 道 pass@1 / 调用/条 / 撞顶 | 0.65 / 18.3 / 5 | 0.65 / 18.5 / 4 |
| 贪心逐 token | — | 完全相同 1/20，分叉位置中位 264 = 长度中位 → 差在结尾一两个 token（EOS 记法？评测无影响，RL 前要弄清） |
| 温度 1 800 条 pass@1 / 调用/条 / 撞顶 / 长度中位 | 0.710 / 15.8 / 16.6% / 264（昨天 4 卡分片） | 0.713 / 15.8 / 16.9% / 265 |
| 报错 / 假结果 | 17 / 3 | 37 / 1（千分之几，噪声） |
| 800 条用时 | 4 卡合计 60 分钟 | 单卡 2.3 分钟 = 26× |
引擎：V1、Flash Attention、前缀缓存开、KV 99,616 token（份额 0.45）、首次 torch.compile 30 s + graph 32 s（有缓存后快）
结论：分布层面等价，换引擎成功。臂 T 的三条 CD 评测走 baseline_countdown_vllm --engine vllm --hf-dir hf_armT
```

## ★★★ 臂 T RL 后评测（vLLM，三张卡各几分钟，2026-09-08）
| | 臂 1 SFT 后 | 臂 1 RL 后 | 臂 T SFT 后 | 臂 T RL 后 |
|---|---|---|---|---|
| CD-4 pass@1 / @2 / @4 / @8 | 0.506 / — / — / 0.720 | 0.563 / — / — / 0.750 | 0.710 / 0.739 / 0.767 / 0.790 | **0.711 / 0.744 / 0.770 / 0.790** |
| CD-4 全对 / 有对有错 / 全错 | 22 / 50 / 28 | 29 / 46 / 25 | 65 / 14 / 21 | 62 / 17 / 21 |
| CD-4 有 answer / 数字用全 / 撞顶 | — / 0.61 / 28% | — / 0.921 / 0 | 0.756 / 0.733 / 16.6% | 0.989 / 0.935 / 0.5% |
| CD-4 调用/条 / 报错 / 假果 / 长度中位·均值 | — | — | 15.8 / 17 / 3 / 264·561 | 10.0 / 8 / 1 / 265·405 |
| CD-3 pass@1 / @8 | 0.599 / 0.930 | 0.503 / 0.740 | 0.714 / 0.980 | **0.590 / 0.720** |
| CD-3 全对 / 有对有错 / 全错（@8） | — | — | 44 / 54 / 2 | 43 / 29 / 28 |
| CD-3 调用/条 / 有 answer | — | — | 17.0 / 0.833 | 8.0 / 0.994 |
| CD-3 pass@64 / 全错 | 1.000 / 0 | 0.950 / 5 | 未跑（估 ≥0.98） | **0.920 / 8**（@1 0.584 @8 0.752 @16 0.811 @32 0.869） |
速度：800 条 0.6 分钟、6400 条 4.1 分钟、8000+ tok/s（HF 分片 4 卡 30 分钟才 400 条）

预测对账   没 answer <5% ✓（1.1%）；调用/条 ≈ 9 ✓（10.0）；CD-4 pass@8 ≈ 0.79 ✓
          ✗ CD-4 pass@1 说 0.78~0.82，实测 0.711（+0.001）
          ✗✗ CD-3 说 pass@1 0.85~0.90 / pass@8 保 0.95，实测 0.590 / 0.720，全错 2 → 28。臂 1 只训 CD-4 时 CD-3 就掉过（0.599 → 0.503），没把先例算进去
机制
  CD-4  有 answer 76% → 99%，正确率不动，全错仍是那 21 道 → RL 把「循环到封顶不写答案」改成「第 10 次左右停、随便写一个」。
        阶梯给错答案 0.1~0.25、没答案 0，全错组里唯一的梯度就是「写个答案」→ 分涨能力不涨（昨天预测的「靠猜」坐实）
  CD-3  RL 数据里没有 CD-3，「10 次左右就收」泛化过去；CD-3 硬题靠搜到第 17 次才中（SFT 后 @8 0.98），现在 8 次就停 → 全错 2 → 28，天花板 0.98 → 0.92
  协议  报错、假结果千分之几，没坏
定论 ⑮  零件修好之后，RL 在 v1 秤上没有正面作用，只剩塑走。「任何错 > 不答」这一级阶梯，在策略确定、硬题无解时教的是放弃；
        长度压力 + 不在训练集里的任务 = 深搜被剪，塑走比臂 1 更狠
下一步三选（vLLM 后都便宜了）
  B 老师 v2 重新 SFT + 评测（15 + 5 分钟）：种能否把 CD-4 全错 21 → ≤10，不靠 RL   ← 用户定：先不做，先跑臂 S（秤 + 题）；教材 v2 备着
  A 护种：--ref-init out_sft_tool/ckpt.pt --beta 0.02 + 数据加 CD-3                      等 RL rollout 换 vLLM 后跑（2~3 h/轮）
  C 换秤：工具臂 v3 判分，错答案 0 分不给格式阶梯，长度罚照旧                            同上，可与 A 并排
待补   crosstask_armT（GSM R1 / 中性）；analyze_search + scan_ops 看 RL 后的 21 道和 CD-3 的 28 道
# ★★★ 全模型总表（2026-09-08，臂 T 归档后）
| 模型 | 血统 | GSM R1 | GSM 中性 | CD-3 @1 / @8 | CD-3 @64 | CD-4 @1 / @8 | CD-4 全错 | 一句话 |
|---|---|---|---|---|---|---|---|---|
| Base | Qwen2.5-1.5B 原始 | 0.52 | 0.48 | 0.02 / 0.18 | 0.59 | 0.02 / — | — | 零件都有，散着 |
| Instruct† | 官方 Instruct | 0.687 | — | 0.015 / 0.09‡ | — | — | — | 对照 |
| GSM-RL† | Instruct + GSM8K RL | 0.767 | — | 0.038 / 0.26‡ | — | — | — | 第一次 RL 有效；反向迁移到 CD |
| CD-RL | Base + CD-3 RL 275 | 0.16 | 0.35 | 贪心 0.405 | — | — | — | 单任务：GSM 塌，遗忘 + 绑定 |
| mix-275 | Base + GSM + CD-3 | 0.71 | 0.47 | 0.16 / 0.56 | — | — | — | 混训止住遗忘 |
| mix-600 | mix-300 续，len-soft | 0.78* | — | 0.46 / 0.69 | 0.83 | 0.21 / 0.61 | — | CD-3 饱和，上一代主力 |
| mix4-300 | mix-600 + GSM + CD-4 | 0.74 | 0.46 | 0.476 / 0.660 | 0.83 | 0.393 / 0.670 | 33 | 三代 RL 的终点，五臂起点 |
| 臂 1 SFT 后（sft1） | mix4 + 穷举器 SFT | 0.755 | 0.565 | 0.599 / 0.930 | **1.000** | 0.506 / 0.720 | 28 | 撒种抬天花板 |
| 臂 0 只 RL | mix4 + RL v1 | 0.75 | 0.49 | 0.511 / 0.700 | 0.83 | 0.471 / 0.710 | 29 | 无种 RL 原地踏步 |
| 臂 2 自蒸馏+RL | mix4 + 自采样 SFT + RL | 0.77 | 0.49 | 0.518 / 0.720 | 0.84 | 0.473 / 0.690 | 31 | 自蒸馏无效 |
| 臂 1 撒种+RL | sft1 + RL v1 | 0.755 | 0.50 | 0.503 / 0.740 | 0.95 | 0.563 / 0.750 | 25 | 无工具最优；塑走 @64 −0.05 |
| 臂 3 mix4+v2 | mix4 + RL v2 判分 | 0.75 | 0.535 | 0.486 / 0.690 | 0.80 | 0.416 / 0.670 | 33 | 判分器被钻 |
| 臂 3′ sft1+v2.1 | sft1 + RL v2.1 | 0.755 | 0.455 | 0.444 / 0.630 | 0.90 | 0.544 / 0.660 | 34 | 过程罚 = 罚探索 |
| 臂 T SFT 后（sft_tool） | mix4 + 穷举器工具版 SFT | 未测 | 未测 | 0.714 / **0.980** | ≥0.98 估 | **0.710 / 0.790** | **21** | 算术清零：SFT 后即最优 |
| 臂 T RL 后（armT） | sft_tool + RL v1（工具） | 0.745 | **0.625** | 0.590 / 0.720 | 0.92 | 0.711 / 0.790 | 21 | CD-4 不动、CD-3 塑走；GSM 中性最高 |
```
口径   GSM：贪心，test[0:200]，crosstask_eval（R1 = 训练模板；中性 = Question:/Answer:）；CD：温度 1，池外 [95000:95100]，×8（@64 = ×64）
       † Instruct 系 chat 模板 + "#### 数字"，holdout TEST[200:500]；‡ 池内 [0:100]；* 训练中 eval 不是 crosstask
臂 T 的 GSM 中性 0.625 比此前最高（臂 1 SFT 后 0.565）高 0.06，超过 200 题的噪声（±0.035）。可能机制：工具臂 CD 组大半零优势，GSM 主导梯度（|A| 0.13 vs 0.05），
GSM 实际吃到的训练比别的臂多。是 SFT 带来的还是 RL 带来的，要补 crosstask_eval --ckpt out_sft_tool/ckpt.pt 才能拆开
```

## 臂 T：crosstask + analyze_search + scan_ops（2026-09-08）—— RL 把第 2 层整个剪掉了
```
crosstask   臂 T SFT 后 GSM R1 0.760 / 中性 0.625；RL 后 0.745 / 0.625 → 中性 0.625 是 SFT 给的，RL 一字没动。
            昨天「GSM 主导梯度所以涨」的猜测错了。工具版 SFT 比臂 1 SFT（0.565）高 0.06 = 12 道题，同一份 GSM 教材，判为噪声不解释
analyze     尝试行/条 10.8 → 7.9；重复行 0.69 → 0.28、违规行 0.79 → 0.34（减半没归零）；撞顶行 3957 → 140；方向精度（没撞顶）0.92 → 0.96；
            6+ 次桶正确率 0.53 → 0.53；perfect 真对 0.984
scan_ops    CD-4 全错仍是那 21 道（7 镜像 + 14 乘除），只是换了死法：[5,7,14,33] 符号 8 行干净 → 乘积表只列 3 个（漏带 33 的）→ 第 1 层 12 行 →
            直接写 14*7+5+33（=136）。SFT 后这道题会进第 2 层绕到顶；RL 后第 1 层扫完就猜。第二条仍有 14*14、231/14 复用数 —— 第 1 层执行仍即兴
            CD-3 全错 28 = 10 道只用加减（要镜像行 b+c−a，老师 v1 不写，SFT 模型靠第 2 层「先 a+b 再 −c」绕出来的）+ 18 道乘除（[10,13,15]→45 = (13−10)×15 纯第 2 层）
            第 2 层被剪 → 这 28 道全死
机制串起来  ① v1 秤：错答案 0.1~0.25、没答案 0 → 全错组唯一梯度 = 写个答案 ② 长度罚：第 2 层又长又几乎不中 ③ 两力同向 → 第 1 层扫完就停就猜
            ④ CD-4 上第 2 层本来只救少数题看不出损失；CD-3 硬题全在第 2 层 → pass@8 −0.26 ⑤ 训练集无 CD-3，「扫完就停」照样泛化（同臂 1，更狠）
对下一步    老师 v2：镜像行把 7+10 道拉回第 0 层；第 2 层一行一次调用 + ÷ 优先 → 第 2 层在 CD-4 上更常中，能挣分 RL 才不剪
            下一轮 RL 三件一起：数据加 CD-3、--ref-init 锚 SFT、秤去掉错答案的格式分（前两件防塑走，第三件堵猜）；等 rollout 换 vLLM 后跑
```

## vllm_check 分叉查清：HF 路径丢 EOS（2026-09-08）
```
分叉样例 ×3：HF 尾 'answer>'（无 EOS）；vLLM 尾 'answer><|endoftext|>'（带 151643）。中途 0 个 token 不同 → 贪心下两引擎数值完全一致
原因   Qwen 无 pad，脚本一律 pad_token = eos_token（同为 151643）；HF generate 写完后用 pad 填批 → 尾部「EOS pad pad」全是 151643，
       后端剥尾部 pad 的循环把真 EOS 一起剥掉。原版 calc_tool.py 就有
影响   评测零影响（判分只读到 </answer>）。RL：臂 T 训练时响应 id 从没带过 EOS，「</answer> 后停」这个 token 从未进 loss；SFT 已教会停，日志「不停 0.00」
修法   calc_tool_vllm.py HFBackend：pad == eos 时换 <|im_start|>（模型不会生成的特殊 token）当 pad，真 EOS 留住 → 与 vLLM 路径一致。原 calc_tool.py 不动（记账）
```
修后复测：贪心 5/5 逐 token 完全相同（含 EOS）。vLLM 线评测阶段三件事（装得上、后端写对、数值等价）全部钉死；引擎二次启动 35 s（编译缓存）


# ★★★ 臂 S（秤 + 题）：锁住种和零件，只改秤和训练题（2026-09-08 定，用户提出）
```
问题   零件（工具）和种（教材）加完后 RL 一分没涨（臂 T）。RL 本身还能不能学？只动秤和题，不动老师、不加 CD-3
设计   起点 out_sft_tool（同臂 T）
       ① 秤 v3：CD 对 = 1.0 − 0.05·len/max_new（对的之间短者略高），错 / 没答案 / 循环 = 0，格式阶梯全去（「任何错 > 不答」教猜），
          不罚过程不罚长度（不罚探索），假 <result> 照罚 0.2；GSM 判分不变
       ② 训练题：只 CD-4，但用 SFT 模型在 [0:90000] 抽 8000 道 × 8 条筛一遍，只留 1~7 条对的（screen_pool.py，DAPO dynamic sampling 的离线版）；
          全对 / 全错在 v3 下 16 条全零纯浪费不进池；150 步后可用当时的 ckpt 再筛一次换池（半在线）
       ③ 不用老师 v2、不用 CD-3；CD-3 当泛化评测
       ④ KL 锚 SFT：--ref-init out_sft_tool/ckpt.pt --beta 0.02
       ⑤ rollout 走 vLLM（grpo_tool_mp_vllm.py --engine vllm），一轮 2~3 h
归类   按难度挑题 = 环境设计（出什么题），是秤的一部分（秤 = 判分 + 环境 + 停止规则），不是老师（老师给示范）
先量   RL 的上限 = SFT 模型 CD-4 pass@64 = 0.900（@1 0.706 @8 0.799 @16 0.831 @32 0.867；全对 64 / 有对有错 26 / 全错 10；4 卡分片 9 分钟）
       余量 0.19。k=8 只见 14~17 道有对有错，k=64 露出 26 道：多出的 10 道 p 在 1/64~1/8，训练 k=16 一半的步全错没梯度，RL 实际能动的是 p ≥ 1/8 的十几道
成功标准  CD-4 pass@1 ≥ 0.80 = 余量吃掉一半（成功）；0.76~0.80 有效但弱；< 0.74 没用。同时 CD-3 pass@8 ≥ 0.95 才算锚住
预测   CD-4 pass@1 0.71 → 0.76~0.79（不超 pass@8/@64 天花板）；pass@8 不动；CD-3 pass@8 ≥ 0.95（锚）；没 answer 留 20% 上下（循环不罚）；GSM 不动
       CD-3 掉 = 锚没锚住；CD-4 不涨 = RL 在此任务上只放大不发明，这次连秤和题的借口都没有
代码   grpo_tool_mp_vllm.py：--judge v3、--val-data（小池时验证题从原文件取）、--engine vllm；screen_pool.py；calc_tool_vllm 加 sleep / load_weights / external_launcher
命令   torchrun --nproc_per_node=4 grpo_tool_mp_vllm.py --probs 8 -k 16 --gen-bs 32 --steps 300 --init out_sft_tool/ckpt.pt --data data/countdown4_mixed.json
         --val-data data/countdown4.json --judge v3 --ref-init out_sft_tool/ckpt.pt --beta 0.02 --max-new 1400 --len-soft 0 --patience 0 --tool --engine vllm --out out_armS
```

### 教学：p、pass@k、RL 推的范围 —— 锁和钥匙（2026-09-08）
```
一句话   p = 袋子里能开这把锁的钥匙占的概率质量；pass@k = 摸 k 次至少摸到一把的概率；RL = 把能开的钥匙挪到袋口
每道题一把锁，模型是口袋里装满钥匙的锁匠，「采一条」= 按习惯摸一把去试（常用的在袋口，偏门的沉底）
| p | 摸一次就开的概率 | 只能估：采 n 条对 c 条，p ≈ c/n；n 越大分辨得越细（8 条到 1/8，64 条到 1/64） |
| pass@1 | = p | 可靠性 |
| pass@8 | 1−(1−p)^8 | 中等运气能不能捞到；p=0.5 已近必开 |
| pass@16 | 训练时一组 16 条看到的世界 | RL 每步 = 摸 16 次再比较 |
| pass@64 | 袋里到底有没有钥匙 | p 过 1/64 基本露面；实用的可达集 |
| pass@128 / ∞ | 更接近「p 是否严格为 0」 | pass@∞ = 支撑集 |
数据集的数 = 各题的平均：SFT CD-4 64 条 → pass@1 0.706 = 各题 c/64 的平均；pass@64 0.90 = c≥1 的题占比（10 道 p≈0）
三道题的例子（各 8 条）：甲 8/8 p=1；乙 0/8 p=0；丙 4/8 p=0.5 → pass@8 0.996；平均 pass@1 0.50、pass@8 0.67，差全来自丙（能捞到但不稳）
题丁 1/8：p 0.125，pass@8 0.66，pass@64 ≈1；训练 16 条空手率 0.875^16 = 12%；期望对 2 条但是二项分布（0:12% 1:27% 2:28% 3:19%）
题丁′ 1/16：p 0.0625，训练 16 条有梯度的步只 64%，推得动但慢
三种改口袋   老师/SFT = 放新钥匙（pass@∞ 涨）；工具 = 掰直弯钥匙（对的钥匙真能开，p 涨）；RL = 挪钥匙到袋口（pass@1 → pass@∞，总数不变）
RL 推的范围   下限现在的 pass@1，上限 pass@∞（用 pass@64 代）。做不到：袋里没钥匙的锁（p=0）变不出来；挪钥匙时可能把偏门钥匙压没 = 塑走（pass@64 掉）
RL 推得动的前提  ① 采得到：一组 16 条里至少一条对（p ≥ 1/8 空手率 ≤ 12%）② 对得有理由：那条对是因为做了可学的选择（选对形态），碰巧的推了也不泛化
               ③ 见得着：池 1300 道每步 4 道，300 步每道平均只见一次 → RL 学的是丁类题共同的做法，不是某道题的答案；池缩小可让每道见多次（旋钮）
秤在图里     锁匠摸完 16 把后由谁说哪几把算开了；秤把「插进去没拧动」算半开（v1 给错答案格式分），锁匠就把那种钥匙挪到袋口
```

### 教学总结：p / pass@k / 稠密 vs 稀疏 / 顺序（2026-09-08，火柴人问题引出的）
```
p 与 pass@k   p = 采一条就对的概率，只能用 c/n 估；pass@k = 1−(1−p)^k；k=1 就是 p，k=64 ≈ p 是否为 0；报告的数是 100 题平均；pass@1 量可靠性，@64 量可达集
RL 干什么     每步：同题 16 条，比平均高的推高低的压低，靠组内差别。范围：pass@1 → pass@64；p=0 碰不了，p 中间推最快（信号 ∝ p(1−p)）
              三前提：采得到（16 条至少一条对）、对得有理由（来自可学的选择）、见得着（同类题够多）；副作用：塑走（@64 掉）
稠密 vs 稀疏   火柴人：尺子秤，每次比得出远近，一组天然有差别，不会走也能学，没有 p 概念。密码锁：开关秤，不开就是 0，一组可能全 0，必须一开始偶尔能开
              可验证领域选开关（钻不了）；尺子（过程奖励、奖励模型）学得快但有缝（v2 被钻）；稀疏→稠密可以（加过程分/部分分，风险 Goodhart），稠密→稀疏设门槛即可
              第三条路：秤不动，挑锁 —— 只给已能开一半的锁，组内自然有差别 = 筛池 / DAPO dynamic sampling
不是所有模型 RL 都稀疏：可验证领域是选择了稀疏；聊天写作用奖励模型是稠密的也最易被钻
顺序          ① 先量 p 是否为 0 ② p 有 → 环境设计造差别（挑题，秤保持开关）③ p 无 → 老师放钥匙（SFT）或零件掰直（工具），回 ② ④ RL 只在有差别的组推，看 @1 涨、@64 保
老师非必须     最早 CD-RL 从 Base 直接 RL（题调简单）0.035 → 0.405，因为 Base 在简单题 p 非 0；老师在 p=0 的题上才必须（臂 1 / 臂 T 的种）
对到实验      CD-RL 无老师涨；臂 1 老师抬天花板 +@1 0.06；臂 T 工具 SFT 0.51→0.71、RL 在 v1 秤下只学会猜、CD-3 塑走；臂 S 开关秤 + 只喂有差别的题 + 锚，成功线 @1 ≥ 0.80
```

## 臂 S 筛池结果（screen_pool，SFT 模型，训练池 [0:90000] 抽 8000 × 8，4 卡各 2000，2026-09-08）
```
全对 4810 (60%) / 有对有错 1546 (19%) / 全错 1644 (21%)   ← 与评测片 65/14/21 同分布（合并那行误把 n 加成 32000，已修显示）
进池 1546 道，c 分布 1:440 2:224 3:194 4:171 5:193 6:147 7:177（c=1 占 28%，p≈1/8，训练 16 条 12% 的步空手）
梯度密度：原池 19% 的组有差别 → 新池 100%，5 倍。1546 道 / 300 步 × 4 道 = 每道平均见 0.8 次，学的是共同做法
→ data/countdown4_mixed.json（带 idx、c），训练配 --val-data data/countdown4.json
```

## vLLM RL rollout 冒烟通过（grpo_tool_mp_vllm.py --engine vllm，4 卡，2026-09-08）
```
三件新东西全过：external_launcher（4 个 torchrun 进程各一份 vLLM，进程组不冲突）、睡眠模式（睡 4 s / 醒 0.17 s，训练时让出 6~7.6 GB）、灌权重（起点 eval 与 HF 版一致：CD 0.625 / GSM 0.75）
两个坑   ① --vllm-mem 0.3 报 No available memory for the cache blocks：vLLM 算 KV 预算时把卡上已占的（训练模型 3 + 参照 3 + 杂项 1）也算在内 → 默认改 0.6（KV 11.8 万 token）
         ② 第三次唤醒 cumem out of memory：训练完 PyTorch 分配器留着几 GB 空闲块不还驱动（「12 GB still in use」）→ 唤醒前 gc + empty_cache
         ③ 退出时 sleep 分配器析构段错误（结果早已落盘）→ 末尾 barrier + os._exit(0)
速度     rollout 6 / 2 / 3 s（含睡醒 + 灌权重约 1 s），训练 2~3 s；HF 版冒烟是 32 / 186 s。显存：唤醒后 allocated 17.7 / reserved 18.9 GB，训练峰值 19 GB
估算     正式 300 步、max_new 1400：硬题 rollout 估 10~20 s → 一步 20~30 s → 2~3 h（HF 版 8.5 h）
```

## 臂 S 进行中：第 162 步 OOM 断、第 150 步中途评测（2026-09-08）
```
断因   反向 OOM：v3 不罚长度 + 筛池全是硬题 → 轨迹长 1000~1250、调用 30~39/条，2 条 × 1600 token 的 logits 要 1.8 GB；
       加 5.3 GB 碎片（reserved but unallocated）+ vLLM 睡着仍占约 1 GB。修：PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True、--micro 1、
       续训时 --vllm-mem 0.7（优化器状态在起 vLLM 前已在卡上，多占 3 GB，KV 比例要跟着抬）
曲线   CD |A| 每步 0.18~0.32（臂 T 0~0.1）= 筛池起效；KL 对 SFT 只 0.001（β 0.02 是臂 T 的 40 倍）；贪心 eval CD 0.60 → 0.62、长 707 → 692，全在噪声里
中途评测（step 150，vLLM 各 1 分钟）
| | SFT 起点 | 臂 T RL 后 | 臂 S step 150 |
| CD-4 pass@1 / @8 | 0.710 / 0.790 | 0.711 / 0.790 | **0.754 / 0.800** |
| CD-3 pass@1 / @8 | 0.714 / 0.980 | 0.590 / 0.720 | **0.750 / 0.980** |
读法   ① 贪心藏收益（教训第三次应验）：温度 1 涨 0.044，锚没拴死，β 不改 ② CD-3 守住 0.98 且 pass@1 +0.036，训练集无 CD-3 = 纯泛化，塑走没了
       ③ 150 步即所有模型最优（两任务同时）。判定：≥ 0.74 → 原样 --resume
预测更新   step 300：CD-4 pass@1 0.77~0.79（0.80 成功线是悬念）、pass@8 不动；CD-3 pass@1 ≈ 0.77、pass@8 保 0.98
「模型几乎没动」的五个候选原因（记着）：① 锚太紧 ② 贪心藏收益 ✓ ③ 信号方向散、净位移小 ④ 决定性 token 被 token 级平均稀释 ⑤ 对得没理由不泛化
```

### 教学：vLLM 睡眠 level 1 / level 2 与分段唤醒（2026-09-08）
```
复印机的纸   = 权重 3 GB + KV 缓存池 3~4 GB（开机预圈的空地，按 16 token 一页分）+ 临时激活；学生的纸 = 反向的激活、logits 1.8 GB、梯度 3、优化器 3、脑子 3、参考书 3
睡眠         level 1：权重搬 CPU、KV 丢；level 2：权重也丢（反正要灌新的），只把非参数缓冲（RoPE 表等）留 CPU
「工具表」坑  真有过：issue #16564（0.8.2）level 2 醒来缓冲没恢复 → 乱码；PR #16889（2025-04-21）修，0.8.5 起包含 = 我们这版可用 level 2。纠正我先前「没接口」的说法
醒 = 分配空地   level 2 后显存里没权重没 KV；醒权重 = 圈 3 GB 空地 + 放回工具表（内容是垃圾）；灌权重 = 把学生脑子拷进去；醒 KV = 圈 KV 池
分段唤醒     官方 RLHF 顺序：sleep(2) → wake_up(["weights"]) → 灌 → wake_up(["kv_cache"])。灌权重那一刻显存最挤，KV 池晚圈 4 GB，峰值低 4 GB
KV 为什么丢   只对刚采完那批有效，题换了模型也改了，留着没用；前缀缓存只省同一轮内的重算
代码         calc_tool_vllm 255313cb（sleep 默认 2、wake_up 支持 tags）、grpo_tool_mp_vllm a3b74b98（醒权重 → 灌 → 醒 KV）。正在跑的臂 S 用 level 1 版，下一轮再换
臂 S 续跑    从 150 接，峰值 23 → 20 GB（--micro 1 + 睡下后 empty_cache；expandable_segments 与睡眠分配器不兼容），一步 13+10 s，剩 70 分钟
```

## 待办 V：给 vLLM 做开源贡献（2026-09-08 记）
```
前提   我们用 0.8.5（2025-04），最新 0.14+，差一年多；踩到的坑多半新版已修（工具表 = #16564/#16889 就是例子）→ 任何想法先在最新版验，还在才做
门票   升驱动：最新轮子要 CUDA 12.8，驱动 ≥ 570，我们 550（要 root + 重启）→ 另开 venv 装最新版 → 跑 grpo_tool_mp_vllm 冒烟
候选（0.8.5 上的现象 → 要查的）
  ① 退出时睡眠分配器析构段错误「Trying to free a pointer not allocated here」（现用 os._exit 绕）→ 最新版还报不报
  ② gpu_memory_utilization 把卡上别的占用也算进去，共存要开到 0.6~0.7 靠猜 → 新版是否有直接指定 KV 字节数的参数
  ③ 唤醒前要自己 gc + empty_cache 否则 cumem OOM → 新版 wake_up 内部做没做
  ④ expandable_segments 与睡眠分配器不兼容（assert）→ PyTorch 侧限制（issue 147851）看进展
  ⑤ torchrun 里同卡共存 RL 的完整示例文档缺 → 文档贡献，门槛最低；我们有机器 + 跑通的最小脚本
流程   升驱动 → 装最新版 → 复现 → 搜 issue（开的和关的都搜）→ 先开 issue 带复现脚本 → 维护者确认方向再 PR
难度   ①②⑤ 能复现就能修/能写，入门级；②③ 涉及 CuMemAllocator（绕过 PyTorch 直接调 CUDA 驱动映射）中等
相关   开着的：#25171 level 2 + 专家并行乱码；RFC #15254 醒来时权重多副本 OOM 的通用方案
时机   臂 S 跑完之后排
补充（2026-09-11）  又多两条候选：⑥ external_launcher 同进程用 V1 引擎必须 VLLM_ENABLE_V1_MULTIPROCESSING=0，否则 spawn 报错 —— 文档没写；
                   ⑦ pad==eos 丢 EOS 那个是我们 HF 路径的锅，不是 vLLM 的，别报
要注意的点  ① 先在最新版复现：0.8.5 → 0.14+ 差一年多，先升驱动（cu128 轮子要 ≥ 570，或找 cu126 变体）另开 venv，现象还在才动
           ② 搜 issue 和 PR（开的关的都搜），有人在做就去评论补复现，别重开
           ③ 最小复现脚本 + collect_env.py 的环境信息 + 期望 / 实际，issue 模板填全
           ④ 先 issue 后 PR，改 API 或行为的先 RFC；只改文档 / 加 assert 提示的可以直接 PR
           ⑤ 一个 PR 只做一件事，标题带前缀（[Bugfix] [Doc] [Core]），链接 issue，别顺手改无关格式
           ⑥ CONTRIBUTING：pre-commit（ruff / mypy）过、DCO 用 git commit -s 签名、加测试或跑对应测试
           ⑦ 性能相关要带 benchmark 数字；review 几周正常，改了要及时回
           ⑧ 顺序：⑤ 文档（最低门槛，我们有跑通的最小脚本）→ ⑥ 文档 → ① 退出段错误 → ③ 唤醒前清缓存 → ② 显存参数（中等，碰 CuMemAllocator）
```

# ★★★ 臂 S 定论（秤 v3 + 筛池 + KL 锚 SFT，300 步，vLLM rollout，2026-09-08）
| | SFT 起点 | 臂 T RL 后 | 臂 S 150 步 | 臂 S 300 步 |
|---|---|---|---|---|
| CD-4 pass@1（8 条 / 64 条） | 0.710 / 0.706 | 0.711 / — | 0.754 / — | 0.746 / **0.748** |
| CD-4 pass@8 / @64 | 0.790 / 0.900 | 0.790 / — | 0.800 / — | 0.820（8 条）0.808（64 条）/ **0.900** |
| CD-4 全对 / 有对有错 / 全错（@64） | 64 / 26 / 10 | — | — | 61 / 29 / 10 |
| CD-4 有 answer / 撞顶 / 调用 | 0.756 / 16.6% / 15.8 | 0.989 / 0.5% / 10.0 | — | 0.770 / 19% / 15.3 |
| CD-3 pass@1（8 / 64） | 0.714 / — | 0.590 / 0.584 | 0.750 / — | 0.726 / 0.723 |
| CD-3 pass@8 / @64 | 0.980 / ≥0.98 估 | 0.720 / 0.920 | 0.980 / — | 0.940（8 条）0.957（64 条）/ **1.000** |
| CD-3 全错（@64） | — | 8 | — | **0** |
| GSM R1 / 中性（贪心 200） | 0.760 / 0.625 | 0.745 / 0.625 | — | **0.795 / 0.635**（McNemar 训坏 7 / 训好 62） |
```
读数   ① CD-4 pass@1 +0.042 坐实：8 条两次（150/300 步）、64 条一次（同题配对）都在 +0.04 上下
       ② 天花板不动：pass@64 0.900 → 0.900，全错仍是那 10 道 → 只挪钥匙不放钥匙，教科书式「只放大不发明」
       ③ CD-3 没塑走：@64 = 1.000 一道没丢；8 条看到的 0.94 是噪声（64 条 pass@8 0.957）；没训过的任务 pass@1 还涨一点
       ④ GSM 反涨：R1 0.795 历史最高；中性 0.635 最高。无代价
       ⑤ 成本面：没 answer 22%、撞顶 19%、调用 15 —— v3 不罚循环，硬题照旧绕到顶（设计内接受）
预测对账   CD-4 pass@1 说 0.77~0.79 实测 0.748（低）；pass@8 不动 ✓（+0.01~0.03）；CD-3 pass@8 保 0.98 实测 0.957 ✓（噪声内）；没 answer ≈ 20% ✓；GSM 不动 ✗（涨了）
成功线     0.748 落在「有效但弱」（0.74~0.76）；余量 0.19 吃掉 0.04，剩 0.15
定论 ⑯    秤做减法（只奖不罚）+ 题做筛选 + 锚护种 → RL 从「学坏」（臂 T）变成「安全地小幅学好」。RL 的作用边界：在可达集内挪概率，pass@1 涨、pass@64 不动、别的任务不伤
          与臂 T 同起点同数据源同 RL 算法，只改秤 / 题 / 锚：CD-4 +0.037、CD-3 +0.14、CD-3 天花板 0.92 → 1.00
剩下 0.15 的三个嫌疑 → 三个对照（各 2 h，vLLM）
       ① 锚太紧（KL 全程 0.001）→ β 0 其余不动         ← 先跑
       ② 池子后半程饱和（|A| 0.18 → 0.15，150 步后 pass@1 不涨）→ 150 步用当时 ckpt 重筛换池（半在线 dynamic sampling）
       ③ 决定性 token 被稀释（16 条 1000 token：900 相同的抵消、约 20 个分岔 token 是全部信号、约 80 个无关差异是噪声 = 结果奖励天生的信用分配问题）
          → 对策是 k 32 或 lr 翻倍（多卷平均噪声 / 推得狠）；「按序列平均」是错的：我们对的轨迹偏长，按序列平均反而削弱它们，保持按 token
       GSM R1 +0.035 最直接的解释是 GSM 自己的 RL（它在训练数据里），噪声约 ±0.03；中性 +0.01 噪声内 → 只能说「不伤、可能帮一点」，不能说迁移
       环境设计（筛池 + v3 + 锚）是用户提的：解决了「RL 学坏」，未解决「RL 学多」
资产   目前最优单模型：臂 S 150 步（hf_armS150，CD-4 0.754 / CD-3 0.750，.pt 已被覆盖，HF 目录保留）；300 步 out_armS/ckpt_latest = hf_armS
```

# ★★★ 阶段总结（臂 S 收官，2026-09-08）
```
一、走过的路
  三代 RL（→ mix4-300）   Base 直接 RL、混训、换 CD-4            RL 能学，天花板由 Base 定
  五臂                    撒种 SFT、自蒸馏、只 RL、判分 v2/v2.1    撒种抬天花板，RL 抬 pass@1；过程奖励被钻；罚错 = 罚探索
  臂 T                    计算器工具 + SFT + RL v1                零件外包立竿见影（+0.20）；RL v1 学会猜、剪第 2 层、CD-3 塌
  臂 S                    秤只奖不罚 + 筛池 + 锚 SFT + vLLM       RL 安全地小幅学好：+0.04，天花板不动，CD-3 满分，GSM 不伤
二、结论压成四句
  1 种定天花板，零件定落地，秤定方向，RL 只在可达集内挪概率
  2 RL 学坏的两种方式：秤给错的台阶（教猜）；长度压力剪掉不常中的深搜（塑走）
  3 让 RL 不学坏的三件套：只奖不罚、只喂有对有错的题、锚住种
  4 小模型上老师和零件比 RL 值钱（同 R1 论文的蒸馏结论）；RL 的价值随支撑集变厚而变大
三、基建   harness（停注入续 mask）、vLLM 评测（26×）、vLLM 同卡共存 RL（8.5 h → 2 h）、筛池脚本、判分 v3
四、最好的模型   臂 S 150 步 = hf_armS150：CD-4 0.754、CD-3 0.750、GSM 0.795 / 0.635，六个数五个历史最高
五、没做完的（都是新实验不是收尾）
  S′ β 0（锚护种还是拉后腿，2 h，命令已给）｜150 步重筛换池（池饱和？2 h）｜熵筛选（信噪比 2% 能否抬，几行代码 + 2 h）
  老师 v2（种再抬天花板？教材已备 20 min）｜E 猜数字（多轮 agent RL，几天）｜V 给 vLLM 贡献（升驱动后验五候选）
一句话   用户自己设计的环境（筛池 + 只奖不罚）把 RL 从「越训越坏」变成「能安全地学」，证据链是干净的对照
```

## 臂 S′（β 0，150 步，其余同臂 S）—— 锚是护种的功臣，不是拉后腿（2026-09-08）
| 150 步对比 | 臂 S（β 0.02） | S′（β 0） |
|---|---|---|
| CD-4 pass@1 / @8 | 0.754 / 0.800 | 0.755 / 0.800 |
| CD-4 全对 / 有对有错 / 全错 | — | 70 / 10 / 20 |
| CD-3 pass@1 / @8 | 0.750 / 0.980 | 0.721 / **0.900** |
| CD-3 全错（@8） | 2（SFT）/ 0（S300 @64） | **10** |
| GSM R1 / 中性 | 0.795 / 0.635（S300） | 0.750 / **0.555**（McNemar 中性对 Base 不显著） |
```
训练侧（前 20 → 后 20 步）  CD 长 1023 → 807、调用 31 → 23、|A| 0.31 → 0.10、6+ 桶 0.46 → 0.68；level 2 睡眠每次让出 9.3 GB（level 1 是 6.2）
读数   ① 去锚 CD-4 一分没多涨 → 余量 0.15 不是锚挡的，嫌疑①排除
       ② 去锚没训过的任务全退：CD-3 @8 −0.08、全错 0 → 10、GSM 中性 −0.08 → 塑走同臂 T 方向，只是 v3 没格式分所以没那么狠
       ③ v3 秤自己防不住塑走（只堵了猜），护种靠锚；臂 S 的 CD-3 满分和 GSM 涨记给 β 0.02
       ④ 训练侧「长度掉、调用掉、|A| 掉」在 S′ 里是塑走的信号，不是学会了 → 以后看到这组数先查泛化评测
定   β 0.02 锚 SFT 定为标配。剩余嫌疑：池子饱和（150 步后 |A| 掉、pass@1 不涨）、决策 token 稀释 → 对照：150 步重筛换池；熵筛选
S′ @64 补齐   CD-3：pass@1 0.711 / @8 0.929 / @64 **1.000**、全错 0（8 条看到的「全错 10」是噪声，可达集没缩）；CD-4：0.753 / 0.803 / 0.890、全错 11
修正   去锚的代价是【可靠性】不是可达集：CD-3 pass@1 −0.04、@8 −0.03、GSM 中性 −0.08，钥匙还在只是被挤向袋底；臂 T 那种丢钥匙（@64 0.92）没发生
       = v3 秤挡住最狠的塑走，锚挡住剩下的。结论不变：锚对 CD-4 零成本、保没训过任务的可靠性，β 0.02 标配
```

### 教学：KL 锚为什么只在 RL 里有（2026-09-08）
```
只有 RL 的数据是模型自己现采的，没有固定目标分布拴着；其他阶段数据固定，数据本身就是锚
| 阶段 | 数据 | loss | KL 项 | 防漂 |
| 预训练 | 固定语料 | 交叉熵（= 对数据分布的 KL + 常数） | 无 | 不需要 |
| mid-training | 固定语料 | 交叉熵 | 无 | 掺旧数据（回放）= 锚的另一形态 |
| SFT | 固定示范 | 交叉熵 | 一般无（蒸馏对老师分布加 KL 是例外） | 掺预训练数据、lr 小、轮数少、LoRA |
| RL | 模型自采 | 策略梯度 | 有：β·KL 到参照 | 数据随策略变，只能人为拴参照 |
DPO 看着像 SFT 其实有锚：公式从带 KL 约束的 RL 目标推出，有 β 和参照模型
RL 里也有人拆锚：InstructGPT、R1 带 KL；DAPO 去掉（长推理要走远）。S′ 是反面数据：小模型、窄任务、种靠 SFT 撑 → 去锚就塑走。锚要不要看种是不是 SFT 撑的
前三阶段「锚」的对应物 = 数据混合：想不忘什么就掺什么；RL 没法掺数据（自己生成的），改在 loss 里拴参照
```

# ★★★ 全模型总表 v2（含臂 S / S′，2026-09-08）
| 模型 | 血统 | GSM R1 / 中性 | CD-3 @1 / @8 | CD-3 @64 | CD-4 @1 / @8 | CD-4 @64 | CD-4 全错 | 一句话 |
|---|---|---|---|---|---|---|---|---|
| Base | Qwen2.5-1.5B | 0.52 / 0.48 | 0.02 / 0.18 | 0.59 | 0.02 / — | — | — | 零件都有，散着 |
| mix4-300 | 三代 RL 终点 | 0.74 / 0.46 | 0.476 / 0.660 | 0.83 | 0.393 / 0.670 | — | 33 | 五臂起点 |
| 臂 1 SFT 后 | mix4 + 穷举器 SFT | 0.755 / 0.565 | 0.599 / 0.930 | 1.000 | 0.506 / 0.720 | — | 28 | 撒种抬天花板 |
| 臂 0 / 2 / 1 / 3 / 3′ | 见五臂总表 | | | 0.83 / 0.84 / 0.95 / 0.80 / 0.90 | 0.471 / 0.473 / 0.563 / 0.416 / 0.544 | | | 无工具时臂 1 最优 |
| 臂 T SFT 后 | mix4 + 工具版 SFT | 0.760 / 0.625 | 0.714 / 0.980 | ≥0.98 估 | 0.710 / 0.790 | 0.900 | 21 | 算术清零，最大一跳 |
| 臂 T RL 后 | + RL v1 | 0.745 / 0.625 | 0.590 / 0.720 | 0.920 | 0.711 / 0.790 | — | 21 | 学会猜，CD-3 塌 |
| 臂 S 150 步 | + RL v3、筛池、锚 0.02 | — | 0.750 / 0.980 | — | 0.754 / 0.800 | — | — | 目前最优单模型（hf_armS150） |
| 臂 S 300 步 | 同上续 | 0.795 / 0.635 | 0.723 / 0.957（64） | 1.000 | 0.748 / 0.808（64） | 0.900 | 18 / 10（64） | 平台，天花板不动 |
| 臂 S′ β 0，150 步 | 同臂 S 去锚 | 0.750 / 0.555 | 0.711 / 0.929（64） | 1.000 | 0.753 / 0.803（64） | 0.890 | 20 / 11（64） | 去锚：CD-4 不涨，可靠性退 |
```
三级台阶：工具 SFT（+0.20）> 老师 SFT（+0.11）> RL（≤ +0.06）。RL 三种命运：无种原地踏步（臂 0）、秤错学坏（臂 T、3′）、秤对题对有锚安全小涨（臂 S）
```

### 教学：工具为什么值 +0.20；支撑集厚了 RL 的份额才大（2026-09-08）
```
工具的收益本质   把模型自己做不稳的原语（算术 0.45）外包给绝对可靠的执行器 → 解决「路对、走的时候摔了」。普遍性：代码执行、检索、数据库、计算器，
                凡模型做不稳而外部能做准的，给工具 + 教调用，收益同量级
更深一层         工具/环境反馈是每一步的、精确的、钻不了的；判分器只在终局说一次且易钻 → 工具是稠密又不会被钻的信号，这就是它比过程奖励好用的原因
边界             只在「执行错」是瓶颈时起作用（臂 1 失败一半是算错所以值 0.20；失败若全在「不知道试什么」工具帮不上）；「何时调、调什么」得先教（第 0 步 p(调用)=0）
大模型 + 工具多 + 老师好 → RL   支撑集越厚，「哪把钥匙该放袋口」只有 RL 能答，SFT 再多示范也不知道哪条路在这个模型身上最可靠（R1：大模型直接 RL 涨得凶，小模型蒸馏 > RL）
                但不是「只剩 RL」：天花板仍靠预训练、工具、老师推；第三杠杆是推理时算力（采多条选最好 = pass@k 变现）
                准确说法：模型越大工具越多示范越好，RL 分到的提升份额越大且是那部分唯一来源；别的杠杆没退休
```

### 教学：预训练在哪一格、推理时算力怎么选、2026 杠杆全表、稠密不可钻、飞轮、多模型（2026-09-08）
```
预训练占两格   零件全来自它（算术/抄数/比大小的电路只有海量稀释数据长得出）+ 最初的种（Base 简单题 p 非 0、搜索套路）；老师 = 定向补种；工具 = 替换零件；mid-training = 延续
采多条怎么选   精确验证器（判分/单元测试）→ pass@k 变现 pass@1；答案短用多数投票（self-consistency）；奖励模型（不可验证领域，易被钻，采越多越挑到钻的）；
              模型自检最弱（perfect 假宣告）。「最好」的定义 = 秤的定义，推理时验证器 = 训练时的秤换个用法
推理时算力三形态   并行（best-of-N、投票）；串行（一条想更长：反思、double check、回退；o1/R1 长链；我们多调几次计算器就是）；树（MCTS + PRM）。RL 训的是串行；s1 加「Wait」逼再想
2026 杠杆全表（按四格）
  预训练规模/数据质量/合成数据 → 零件 + 初始种（天花板主来源）｜mid-training → 同上定向｜SFT/蒸馏/拒绝采样 → 种（小模型主力）
  RLVR → 概率分配（推理模型主力）｜RLHF/奖励模型/DPO → 概率分配不可验证域（秤易钻）｜工具与环境、agent RL → 替换零件 + 稠密不可钻反馈（增长最快）
  课程与筛题 → 让秤有东西可称（我们的 dynamic sampling）｜推理时算力 → 不动权重用算力换 pass@k（第三支柱）｜数据飞轮 → 部署日志变新 rollout 和秤（护城河）
  蒸馏回小模型/量化/MoE → 降成本｜多模型系统 → 系统层补短板
稠密不可钻     结果奖励：终局一次、难钻但稀疏；过程奖励：每行、判分器读文字、可钻（v2）；工具/环境反馈：每次调用、由计算器/解释器/编译器/物理给、不看模型怎么说 → 稠密且不可钻
              它是观测不是奖励，模型得学会用（too high/low）；可变成奖励（单元测试全过）。增长最快因为产品价值在 agent（写码、操作电脑、深度检索），agentic RL 是 2025~26 最热
数据飞轮       用户问什么（真实题分布）+ 模型答什么（rollout）+ 用户反应（赞踩/重生成/改写/追问/接受代码）→ 偏好数据训奖励模型、挖难题做课程；另一半是合成飞轮（生成 → 验证器过滤 → 回炉）
              有隐私/退出约束；护城河 = 真实题和反馈的规模 + 昂贵人工标注
多模型系统     路由（易→小难→大）、级联（小先答没把握升级）、集成投票、专家模型、多 agent（规划/执行/审稿、辩论）；补成本/延迟/广而不精/自检不准；代价复杂度和错误传递（compound AI systems）
「缺的是规模」  八根杠杆都在 1.5B、两千条教材、四张卡、两个玩具任务上碰的；机制看全了，力度随规模变（RL 份额随支撑集变厚而变大，这里 0.04 大模型可能一半）→ 结论关于机制不关于前沿
```

### 教学：温度 0 ≠ 选最好；工具反馈是传感器不是分；长程靠什么；机制 vs 前沿（2026-09-08）
```
温度 0        贪心 = 每步挑当前最可能 token 的一条路，不是最可能的整条路更不是最好的路，无验证器参与。best-of-N 采 N 条不同路 + 验证器挑对的 → pass@N
              数据：CD-4 贪心 0.60~0.63 < 温度 1 采一条 0.71~0.75 < 采 8 挑对 0.80~0.82（贪心藏收益：局部最可能拼起来常是全局较差）
工具反馈 = 观测   火柴人两样都有：距离 = 分（进 loss）、体感 = 观测（进上下文）。<result>78</result> 是观测：写进输入不进梯度（不是模型写的，不能训它预测计算器），
              只训「看到 78 > 30 后怎么办」，靠终局奖励学、SFT 给示范。稠密精确不可钻，但它告诉的是世界状态不是打分
长程任务       难点错误累积 p^T（20 步 0.95 → 0.36）。三条路：单模型 + 长程 agentic RL（抬每步 p）；harness 工程（工具、笔记、检查点、重试、验证循环）；
              分解 / 子 agent（缩短每个 agent 的 T、隔离上下文和错误）。2026 最强编码 agent 基本单模型 + harness，子 agent 用于并行和隔离，不是一群模型商量
机制 vs 前沿   机制 = 我们量出的那串（种/秤/零件；RL 只挪不发明；学坏两方式 + 三件套；老师骨架、工具换零件、稠密不可钻反馈 > 过程判分；贪心藏收益），1.5B 与 100B 同样成立
              前沿 = 同三个旋钮拧到极限 × 规模：零件靠万亿 token + 精选合成数据预训练；种靠海量高质量示范/蒸馏；秤靠可验证环境大规模 RLVR + 真实环境 agentic RL；加推理时算力
              差别不在旋钮种类，在每个旋钮拧了多少圈
```

### 教学：两条通道、长程主力、agentic RL 流程、旋钮 vs 力气（2026-09-08）
```
两条通道       眼睛看到的（观测，进下一步的输入不进 loss）：火柴人的脚踩地/往左倒、小孩看到球落筐左、我们的 <result>78</result>；
              教练打的分（奖励，进 loss）：挪了几厘米、进没进、答案对不对。火柴人特殊在分也每步有；我们的分只在终局
              工具反馈像眼睛：每步给真实信息且来自世界不来自判分 → 稠密不可钻；训的是「看到之后怎么办」不是「预测计算器」
长程主力       单模型 + 长程 RL 是发动机（司机技术）；harness 是底盘（车、导航、安全带）；分解/子 agent 是选装件。多 agent 只是第三条里的一种
agentic RL     不是拿长轨迹当 SFT 老师，是在长任务上直接做我们这套 RL，每个环节放大：环境 = 真实仓库 + 沙盒能跑测试；轨迹几千次调用几十万 token；
              秤 = 测试过不过/任务完没完；种 = 预训练 + 冷启动 SFT 已会调工具；RL 采多条完整轨迹推高通过的。SFT 只让 p 非 0，之后 on-policy 飞轮、任务由短到长
              难点放大：几千步信用分配、稀疏终局奖励 → 环境尽量给中间可验证信号（每步测试结果）
旋钮 vs 力气   旋钮是种、秤、零件三个；算力/数据/人力是拧旋钮的力气不是第四个旋钮
              零件：预训练 token 与质量（我们借 Qwen 没拧；前沿万亿 token 几个月）｜种：示范数量质量（两千条穷举器 vs 海量人写 + 蒸馏）｜
              秤：可验证环境数与 RL 算力（一把计算器四张卡两小时 vs 成千上万环境几周）
```

### 教学：工具反馈的本质、眼睛与教练的关系、单模型长程 RL、前沿也要教调用（2026-09-08）
```
工具反馈本质   让轨迹的对错只取决于决策，且每个决策有真实依据：① 执行不再摔跤（对错只反映选路）② 每步有真实信息（下一步建立在事实上）③ 错误立刻可见（当场换路）
              → 终局那一个分变成对「策略」的评价而不是「策略 + 手气」，RL 梯度指向策略（臂 1 → 臂 T 的差别）
眼睛 vs 教练   两条独立的线：教练只看最后答案，分的准确度由秤定，眼睛帮不了；眼睛决定能不能走到高分、高分能不能归因（看得见 = 对是因为选得对）
              例外：分由工具算出（单元测试）时工具同时是眼睛和教练
单模型长程 RL  一个策略模型（规划执行检查都在一条轨迹里）；成千上万沙盒（真实仓库/网页，可执行可测）；一条轨迹几百到几千轮几十万 token（harness 管上下文压缩）；
              秤 = 终局测试/任务完成 + 尽量中间可验证信号；on-policy、异步 rollout 引擎、GRPO 类结果奖励、任务由短到长课程；学到规划/检查/回退 = 带工具的长思维链
              与我们只差每格大小，结构 = grpo_tool_mp_vllm
前沿也要教调用  预训练语料有大量代码和 API 文本（初始种）→ mid-training/SFT 明确教：专用 token 标调用、参数 JSON、工具用 schema 描述，教一次泛化到万个工具（工业版工具 SFT）
              → 运行时循环同我们：生成到专用 token 停、harness 执行、结果作观测接回、续写。差别：专用 token（安全、不被分词切碎）、格式通用不用逐工具教
```

### 教学：策略是什么、结果分两种、课程与老师分工、工具格式如何通用（2026-09-08）
```
策略          「在这个局面下选什么」= 每个位置的下一 token 分布；Countdown 里是下一行试哪个、何时转第 2 层、何时停。是行为不是事实，只能按后果衡量
结果分两种     界线不在终局/中间，在谁产生：世界给的（终局对不对、中间算出 78）能衡量策略；判分器读文字给的（标注对不对、像不像搜索）危险（v2 被钻）
              过程结果也是结果，只要是世界给的；工具的价值 = 把中间结果变成世界给的；「去手气」= 同一决策后果确定，终局分量的就是决策
课程与老师     两样都要：老师给格式和短示范让 p 非 0；课程是环境设计，任务十步 → 百步 → 千步，RL 在每级推可靠性再进下一级（与筛池同家）。老师不教长程，长程是 RL 在课程里拉出来的
工具格式通用   把「工具」变成「输入」三层：① 协议与工具无关（专用 token + JSON 工具名参数）学的是壳 ② 推理时工具 schema（名、参数、类型、说明）放进提示词，模型读不用记
              ③ SFT 用成千上万不同工具，每条 =「上下文有 schema + 需求 → 按 schema 填调用」，学的是「读说明书填表」不是某个工具
              对照我们：TOOL_HINT 是最小 schema，只教一个工具所以学的是「calc 怎么用」；工业界教几千个学的是「怎么读说明书」→ 新工具写段 schema 即可。MCP 统一 schema 和传输格式
```

### 教学：策略 ⊇ 解题方法；长任务 p 不是 0 是小（2026-09-08）
```
策略 ⊇ 解题方法   方法 = 确定的菜谱；策略 = 菜谱 + 周围的整片概率云（怎么随机、何时停、卡住怎么办、各选项概率）。SFT 后 ≈ 菜谱 + 近零随机；RL 改概率，改多了菜谱也变
长任务不用老师示范   区别在新原语 vs 已有原语的组合：第 2 层新形态袋里没有 → p 严格 0 → 老师放钥匙；20 步编码任务每步都会 → p ≈ 每步成功率^20，小非 0 → 课程 + RL
              课程 = 把任务长度挑在「偶尔成经常败」区间（与筛池同理），p 太小缩短、近 1 加长。示范 10 步教会的东西在 500 步里是同一套
              补充：长轨迹 SFT 教的是模式（记笔记、交卷前验证）不是每种长度；「自己当老师」= 模型走通的长轨迹经测试过滤回炉 SFT（expert iteration，臂 2 自蒸馏的小版本，大模型 + 可验证秤下有效）
课程 vs 老师   课程 = 给 RL 的题目序列（由易到难由短到长），属于秤/环境设计，让每级组里有对有错；老师 = 给 SFT 的示范，属于种，让 p 从 0 变非 0
              我们做过的：课程 = 筛池、早期缩数字范围；老师 = 穷举器教材、工具版教材。课程不给答案不给示范只挑题，假设模型已偶尔能对
```

### 教学：长程 RL 难在哪、R1 之后的变化、范例、老师教长程可不可以（2026-09-08）
```
长程 RL 难点   错误累积（每步 0.9，300 步 → 0；RL 抬每步 p 更要教会发现错回退，有回退就不再乘法叠加）；信用分配（几千步一个分 → 中间可验证信号、分解、课程）；
              环境（稳定可复现且多而不同，几个会过拟合 → 成千上万沙盒，大厂重资产）；采样成本（异步 rollout）；上下文（长上下文预训练 + harness 笔记压缩 = 零件问题）
              → 算力 + 大量稳定环境 + 回退能力一起 scaling，缺一不可
R1 之后（到 2026 中，之后不确定）  算法：DAPO（clip 上界、动态采样、token 级 loss、超长塑形）、Dr.GRPO、GSPO（序列级比值，Qwen3）、熵筛选、去 KL、异步 RL
              范围：RLVR 扩到理科/指令/工具；agentic RL 主战场，编码 agent 飞涨。形态：思考可开关可控预算；1T MoE agent 模型；推理蒸馏小模型标配
              数据：mid-training 塞 agent 轨迹和合成推理；环境成产品。我们碰过的动态采样、token 级 loss、锚取舍、只奖不罚都是这一年主流
范例          ToRL / ReTool（数学 + 代码解释器 RL = 计算器实验大号版，最值得对照）；Search-R1（多轮搜索 = E 同形）；verl 多轮工具 recipe、TRL GRPO 工具例、rLLM/DeepSWE；
              环境 SWE-Gym / R2E-Gym / BrowserGym / τ-bench / ALFWorld；我们的 harness + vLLM 共存 + 筛池已是最小范例
老师教长程     可以且工业界在做：教模式（记笔记、先验证、工具串法）、把长任务 p 从极小抬到能采到；做不了可靠性（仍靠 RL）。修正「老师不教长程」→ 可以教、非必须、可靠性靠 RL
              轨迹来源几乎全是机器产（强模型跑 + 验证器过滤，或自己跑 + 测试过滤回炉 = 拒绝采样）；位置：mid-training 稀释大量塞长电路，SFT 集中少量易背题（臂 2 自蒸馏 1.5B 无效）
```

## 待办 L：长程 RL 实践路线（2026-09-08 记）
```
① E 猜数字先跑通（多轮观测注入 + 多轮奖励，基建全现成）
② 对照读 ToRL / ReTool（跟计算器实验同形）和 verl 多轮工具 recipe，逐项比 grpo_tool_mp_vllm 差在哪补哪
③ 再往前：挑一个现成环境（τ-bench 或 ALFWorld 级别的文字环境），课程从短到长，看 1.5B 能拉多长
```

## 参考：「卖环境」的公司与平台（2026-09-08 查）
```
背景   2025-09 The Information：Anthropic 讨论一年内在 RL 环境上花 10 亿美元以上；环境可独家卖给一家（贵 4~5 倍）或非独家卖多家（TechCrunch 2025-09-21）
开放平台   Prime Intellect Environments Hub：社区众筹的开放环境库（2500+ 环境、250+ 作者、10 万+ 下载），配 verifiers 库 + prime-rl 训练框架 + 托管 RL（Lab）
          → 跟我们最像：一个环境 = 题目 + 判分器 + 多轮接口，可直接拿来跑
专做环境的公司   Mechanize（少量但很硬的环境，卖给前沿实验室，招工程师做「替代自己」的任务）；Fleet（复刻 Salesforce、Excel 等应用做 RL gym，年化收入 1M → 60M+，估值 7.5 亿）；
          Deeptune（计算机使用 + 代码的托管环境，a16z 领投 43M A 轮 2026-03）；Halluminate（浏览器/计算机使用沙盒 + 评测）；Vmax（把企业私有数据和评测自动转成环境）；
          HUD（50+ 家在上面建环境卖给实验室）；idler（评测 + 环境）
大数据公司的环境线   Scale AI RL Environments（模拟真实应用与 API 系统，程序化 + rubric 判分）；Surge AI（RLHF 起家，环境 + 评测）
综述   Epoch AI「An FAQ on RL Environments」；SemiAnalysis「RL Environments and RL for Science」；目录站 rl-list.com、alignlist.com
对我们的意义   环境 = 秤 + 观测通道 + 课程，是这一年最热的「卖秤」生意；我们的 Countdown + 计算器 harness 就是一个极小的自制环境；E 猜数字是第二个
```

### 总结：长程任务（long horizon）怎么做（2026-09-08）
```
难在哪   每步 p 的 T 次方；终局一个分要分给几千步；上下文装不下；一条轨迹几小时
按四格
  零件    长上下文 + 发现错回退的基本动作 ← 长上下文预训练；mid-training 塞大量 agent 轨迹与合成推理
  种      会调工具、读结果、记笔记、交卷前验证 ← 冷启动 SFT，示范短任务就够（长任务 = 同一套动作做更多次）；轨迹几乎全机器产（强模型跑 + 验证器过滤 / 自己跑 + 测试过滤回炉）
  秤/环境  稳定可复现、成千上万不同沙盒、终局可验证、尽量中间信号 ← 自建或买；测试通过率/任务完成度当分，每步测试结果当中间信号
  RL      在自采长轨迹上抬每步 p、学会回退 ← on-policy、异步 rollout 引擎、结果奖励 GRPO 类、锚和熵控制、课程按长度由短到长
课程是核心调度   长度挑在「偶尔成经常败」区间，p 小缩短、近 1 加长（与筛池同理，维度从难度换成长度）
harness 是底盘   工具与沙盒、记忆（笔记/摘要/压缩）、检查点与重试、验证循环、子 agent（并行和隔离上下文，不是一群模型商量）
推理时算力     并行采多条验证器挑；串行多检查几次；树搜索按价值剪枝
顺序          备齐零件和种 → 造大量能打分的环境 → 课程把长度一级级加 → RL 每级推可靠性 → harness 把能力兑现成完成的任务
我们的路      E 猜数字（多轮观测 + 多轮奖励）→ 对照 ToRL / verl 多轮 recipe 补差 → 现成文字环境课程由短到长看 1.5B 能拉多长。基建已有：harness、vLLM 共存、筛池、只奖不罚
冷启动 SFT     见下一条教学
```

### 教学：「冷启动 SFT」的冷启动是什么（2026-09-08）
```
来源   R1 论文的 cold start：R1-Zero 直接从 Base 做 RL 能学但输出可读性差、语言混杂；R1 在 RL 前先用几千条高质量长链示范做一次 SFT，再 RL
含义   冷 = 模型对目标行为的 p ≈ 0 或做得很难看；冷启动 = 用少量示范先「点火」，把 p 从 0 推到非 0、把格式摆正，然后交给 RL。它是种，不是课程
和我们   工具版 SFT 就是冷启动：第 0 步 p(调用)=0，2000 条教材点火，之后 RL；臂 1 的穷举器 SFT 同理
不是什么   不是大规模 SFT（那是蒸馏/指令微调，量大到成为主力）；冷启动的数据量小、目的窄：让 RL 有得采
判断要不要   先测 p：目标行为 16 条里一条都采不到 → 要冷启动；偶尔能对且格式过得去 → 直接 RL（早期 CD-RL 从 Base 直接 RL 就是这种）
```

### 教学：先蒸馏后 RL 的边界；长程 = 方法论 + 乐高 + 可靠性 + 状态（2026-09-08，用户提出前两条）
```
主流管线   预训练 → mid-training → 大规模 SFT/蒸馏 → RL（R1：小模型蒸馏 > 直接 RL；我们 mix4 → SFT → RL 同理）
三个边界   ① 蒸馏要有老师：灌的是别人的能力，那个能力总得有人用 RL 造（R1-Zero 证明新能力只能从可验证环境里长）；蒸馏只搬运不创造
          ② SFT 有天花板：超不过老师，集中喂易背题，泛化不如 on-policy RL（2025「SFT 记忆，RL 泛化」）
          ③ 冷启动故意少：想让 RL 自己发现推理方式而非模仿；示范稀缺时也只能少
规则      手边有更强老师 → 先蒸馏后 RL 性价比最高；没老师 → RL 是唯一造能力的办法。我们一直在前一种（穷举器、工具都是现成老师）
长程四要素   方法论（用户）：以行为形式长在模型里 —— 计划、执行、检查、回退；模式来自预训练和示范，哪种靠谱由 RL 选（Countdown：扫完转乘除、转第 2 层、中了就停）
          乐高（用户）：可靠的短技能当块；老师教块，RL 教拼法（符号扫描块、乘积块、第 2 层块）
          每块够可靠：p^T 那道坎，块不稳拼十块就散（工具把算术块 0.45 → 1.0）
          拼的时候有状态：块间传「做到哪、试过什么、下一步」= harness 的笔记和检查点 + 长上下文；没有它拼到第二十块忘了前面
一句话    方法论决定拼的顺序，乐高决定有什么可拼，可靠性决定拼多长不散，状态决定拼的过程不忘
```

## 待办 M：规模实验 —— 同一管线 1.5B vs 7B，RL 的份额随模型变大吗（2026-09-08 记）
```
问题   我们的结论「工具 SFT > 老师 SFT > RL（≤ +0.06）」是 1.5B 上的；R1 论文说大模型 RL 份额大、小模型蒸馏更强。同一管线换 7B 是否 RL 份额变大？4 卡 4090 能答
4 卡能到多大   1.5B 全参 + 同卡 vLLM（现状 20 GB）｜3B 全参要 --micro 1、vLLM 分卡（2 训 2 采）｜7B LoRA 一卡训 + 采样卡另放底座，同步只传 adapter 几十 MB｜
              7B QLoRA 4 位底座可同卡共存｜7B 全参要 FSDP + CPU 卸载不划算｜14B~32B 只推理（4 位）当老师和对照
方案   Qwen2.5-7B（或 Qwen3-8B）+ LoRA/QLoRA + GRPO（TRL/Unsloth 有单卡 24 GB 配方）；训练器改动：raw 套 peft，vl_sync 传合并权重或 adapter
       同一套：工具版 SFT（同教材）→ 筛池 → v3 + 锚 → 五条评测。比的是 SFT 后 → RL 后的涨幅（1.5B 是 +0.04）和 pass@64 天花板
便宜的规模   环境数（Hub 上拿十几个）、k 32~64、步数几千、种子 3~5 个（结论才有误差棒）
租          本地调通后租 8×H100 跑一次 7B 全参或 32B LoRA 对照，几小时几十到几百美元；代码是 torchrun + vLLM 直接搬
大模型当老师   Qwen3-32B 4 位一两张卡推理，替代穷举器产教材 = 零成本蒸馏，种更厚
随机种子 ≠ 温度   种子 = 随机数起点（抽题、掷 token、打乱、初始化），改了样本变分布不变；温度 = 分布多平。我们每个实验只 1 个种子，训练噪声没量过
```

## 参考：候选底座对比（2026-09-08 查）
```
| 家族 | 时间 | Base 尺寸 | 许可 | 开放程度 | 对我们 |
| Qwen2.5（现用） | 2024-09 | 0.5/1.5/3/7/14B | Apache 2.0 | 权重 | 现有结果全在它上，换尺寸零成本 |
| Qwen3 | 2025-04 | 0.6/1.7/4/8/14/32B | Apache 2.0 | 权重 | 同尺寸更强、梯子最细；1.5B 那点要在 1.7B 重跑（≈3 h）← scaling 曲线首选 |
| Qwen3.5 | 2026 | 9B 一档（小尺寸 Base 有无待核对） | — | 权重 | 最新，去 HF 核对 |
| Gemma 3 | 2025-03 | 1/4/12/27B（pt 版） | Gemma 自定条款（可商用带使用政策） | 权重 | 多语言多模态；词表 26 万 logits 显存 ×1.7 |
| Gemma 4 | 2026-04-02 | E2B（2.3B 有效）/E4B（4.5B）/12B/26B-A4B（MoE 3.8B 激活）/31B 稠密 | **Apache 2.0**（这代改的） | 权重 | 多模态、256k 上下文、140+ 语言；小尺寸是「有效参数」的 E 系列 |
| OLMo 3 / 3.1 | 2025-11 / 2026 更新 | 7B、32B（Base / Instruct / Think / RL-Zero） | Apache 2.0 | **全开放**：数据（Dolma 3）、配方、每阶段中间 ckpt | 做「RL 从 Base 起」研究最透明；RL-Zero 7B 数学/代码 3.1 更新为更长更稳的训练；无小尺寸；没有 OLMo 4 |
「开源」的区别：Gemma/Qwen = 开放权重（open weight，训练数据和配方不公开）；OLMo = 全开放（open source 全栈）
```

## 参考：全开放家族补充 —— OLMo 3/3.1 与 K2（IFM/LLM360）（2026-09-08 查）
```
| 模型 | 出品 | 时间 | 尺寸/结构 | 开放 | 备注 |
| OLMo 3 7B | AI2 | 2025-11-20 | 7B 稠密；Base/Instruct/Think/RL-Zero | 全开放（Dolma 3、配方、中间 ckpt）Apache 2.0 | 全开放 7B 里数学代码最强；RL-Zero 7B 数学/代码在 3.1 更新为更长更稳 |
| OLMo 3.1 32B Think | AI2 | 2025-12-12 | 32B 稠密推理版 | 同上 | 在 3 的基础上再 RL 21 天 ×224 GPU（Dolci-Think-RL 多跑几轮）：AIME +5、ZebraLogic +4、IFEval +4、IFBench +20；66k 上下文；全开放推理模型最强，用 6× 更少 token 逼近 Qwen3 32B |
| K2 Think V2 70B | MBZUAI IFM + G42 + Cerebras | 2026-01-27 | 70B 稠密，在 K2-V2 Instruct 上做 RL 的推理版 | 端到端全开放（预训练数据、中间 ckpt、后训练配方、评测） | Artificial Analysis 开放度榜与 OLMo 3 32B Think 并列第一；报告 llm360.ai/reports/K2_V2_report.pdf |
| K2 Horizon | MBZUAI IFM（LLM360 团队） | **2026-09-03** | 六个：375B-A23B、36B-A4B（MoVA 混合价值注意力）、32B、7B、3.7B、0.9B | 全开放 + Apache 2.0：中间 ckpt、数据或数据构造配方、训练代码、配置、日志、评测 | 0.9B/3.7B/7B 各自尺寸的 SOTA（宣称）；7B 面向软件工程和深度检索。★ 全开放且有小尺寸梯子 → 可能是「全开放 scaling 曲线」的答案，先去 HF IFM 组织核对 Base ckpt 是否单独放出 |
官网   OLMo：allenai.org/olmo、博客 allenai.org/blog/olmo3、HF allenai/*
       K2：ifm.ai（博客 ifm.ai/blog/k2、新闻稿 ifm.ai/k2/press-release）、llm360.ai、mbzuai.ac.ae/news、HF IFM/*（如 IFM/K2-Horizon-MoVA-36B-A4B-GGUF）
注意   K2 Think ≠ Kimi K2（月之暗面）；K2 这里是 MBZUAI 的 LLM360 系列
```

### 教学：尺寸对照实验到底问什么（2026-09-08）
```
不是问   8B 比 1.5B 强多少（没信息量）
是问     换了尺寸，16 条定论还成不成立、三根杠杆（老师/工具/RL）的份额往哪挪：成立的才是机制，不成立的改写成「小模型上……」= 结论的适用范围
顺带     实用规则：多大的模型上工夫该花在老师、工具还是 RL（1.5B：工具 +0.20 > 老师 +0.11 > RL +0.04；R1 在 70B：RL 是大头。中间怎么变没人给过）
最可疑的定论（8B 上可能翻）  ①③ RL 不抬天花板（大袋子温度 1 可能探索到新形态）｜④ 自蒸馏无效（大模型自采轨迹质量高，工业拒绝采样有效）｜
                          ⑬ 工具 +0.20（大模型心算准可能只剩 +0.03）｜⑭ 零优势 86%（大模型策略不那么确定）｜⑯ RL 只 +0.04（可能 +0.15）
大概率与尺寸无关   ⑨⑩ 过程奖励被钻、罚错 = 罚尝试（秤的性质）
准确的说法   影响结论的是支撑集的厚薄（袋里钥匙多不多）；尺寸是让袋子变厚的主要旋钮（预训练数据、mid-training 也能）→ 量的是「结论随支撑集厚度怎么变」
两条轴分开   尺寸轴：1.7B / 4B / 8B 三点，每档同一管线（Base → 工具 SFT → 筛池 → v3 + 锚 → 五评），画老师/工具/RL 涨幅与 pass@64 四条线；三点是趋势不是定律
            算力轴：同一模型 RL 步数 300 / 1000 / 3000，看 pass@1 是否继续涨（臂 S 150 步后平台未验证），1.5B 一晚上
16 条定论清单   ①撒种抬天花板 RL 不抬 ②种在 RL 后有净贡献 ③只 RL 涨 @1 不涨 @64 ④自蒸馏 0 ⑤迁移任务种被 v1 RL 吃掉 ⑥混训无遗忘 ⑦长度压回均衡无分叉 ⑧残余是自信错答/重复用数
             ⑨过程奖励认格式必被绕 ⑩罚错 = 罚尝试 ⑪三版秤只修怎么错没动能不能对 ⑫GSM 中性随 RL 回落 ⑬工具清零后失败全在种 ⑭确定性策略零优势 ⑮v1 秤教放弃 ⑯只奖不罚 + 筛池 + 锚安全小涨
```

## 参考：RL 能不能涨 pass@k —— 文献两派（2026-09-08 查）
```
Yue et al.「Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?」（清华，2025-04，NeurIPS'25）
   7B~32B（Qwen2.5、LLaMA）RLVR 数学/代码/视觉：小 k RL 赢，大 k（256~1024）Base 反超；覆盖和困惑度分析说推理路径来自且受限于 Base；六种 RLVR 算法都差不多且远未用尽 Base 潜力
   → 支持「RL 只挪不发明」，而且是在比我们大 5~20 倍的模型上
ProRL「Prolonged RL Expands Reasoning Boundaries」（NVIDIA，2025-05）
   在 1.5B（R1-Distill-Qwen-1.5B）上 RL 2000+ 步，配 KL 控制、参照策略定期重置、多样任务：pass@k 全程赢 Base，有些任务 Base 采多少次都是 0 而 RL 后 100%
   → 边界扩张与 Base 对该任务的胜任度和训练时长强相关：RL 能随时间探索并填充新的解空间
「Reasoning Boundary Paradox」（2025-10）：RL 约束边界的反方；e3（2025-06）：学会探索才能外推推理时算力
对我们的修正   「大模型上 RL 涨 pass@64」这个猜想文献不支持：Yue 在 32B 上没看到；ProRL 看到的扩张来自 RL 时长 + 任务多样 + 参照重置，是在 1.5B 上！
              → 决定「RL 能不能发明」的更像是 RL 算力轴而不是尺寸轴。我们 300 步「只挪不发明」跟两篇都一致（短 RL）
              → 算力轴实验（300 → 1000 → 3000 步，KL 控制 + 每 500 步把参照重置为当前策略 + 多任务）比尺寸梯子更便宜也更直接地测「发明」
              ProRL 的参照重置 = 我们锚的进阶：锚拴 SFT 防塑走，但拴太久也拴住探索 → 定期把锚挪到当前位置
```

## 零提示词对话观察（chat_raw_vllm.py，hf_armS，2026-09-09）
```
工具   chat_raw_vllm.py：粘贴什么从什么后面接着写；harness 默认开（--no-tool 关）；打原样 + 标注版（⟦⟧ = 注入，--no-annot 关）；认出 Countdown 题自动判分
观察 ① 光秃秃的中文算术题「824+982 等于多少」→ 补「？」答 1806，然后自己编下一题再答再编，三轮后 EOS：Base 的续写本能，不知道对面是人；心算 1806/1239/413 全对
观察 ② 去掉系统行（只留 User 句 + Assistant + <think>）→ 调用 0：扫描程序还在，但计算器不触发，退回「式子 = 值」心算 → 方向标错、重复、括号格式漂移全回来
       → 程序绑得比工具松，工具绑得紧（模板绑定的现场版）
观察 ③ 完整模板 → 调用 45、方向全对，但 [2,3,7,3]→39 仍失败：解 7×3×2−3 在第 2 层三数链，模型跑完第 1 层和两对形态后整块重抄、48 次撞顶，从没进「Next, build a bigger piece」
       两个 3 让去重失灵（3*3+7*2 与 3*3+2*7 算两行）；第 2 层是教材最薄的一层（7%）且 RL 没碰 → 对该模型 p≈0，只有老师能改（T2 验收题）
观察 ④ [2,3,6,2]→18 答对，解 6+3*2*2 同族三数链在第 1 层顺手写出 → 形态会，上一题差在顺序和预算
待办 H（harness 小旋钮）  同一条里同一式子第二次出现，harness 返回「already tried」不重算，让模型看到自己在绕圈；等数去重按值不按写法
```

### 这一路的意义（用户自评 + 补充，2026-09-09）
```
用户自评   ① 掌握更多基础知识，动手后的深入理解 ② 大概知道小模型预训练 / SFT / RL 的流程
补充      ③ 做实验的方法：先写预测再跑、错了记账、池外切片不碰、量分布不看贪心、每条结论配对照臂 —— 从「跑一下看分涨没涨」到「涨的是哪道题、为什么、换种子还涨吗」
          ④ 判断力：读 R1 / ProRL / Yue 能看出对照干不干净、k 多大、有没有量天花板；看到卖环境知道卖的是秤 + 观测 + 课程；看到 DAPO 动态采样知道就是筛池
流程准确说   掌握的是「预训练之后」的全部（mid-training 位置、冷启动、蒸馏、RLVR、锚、课程、评测、推理引擎）；预训练借 Qwen，之前 95M 从零训练补了那段
产品是什么   不是那个 1.5B 模型（Countdown 是秤，让它调 DeepSeek 等于拿秤称秤）；是对这台机器的理解 + 一套可复用实验室
理解清单   RL 在做什么（同题 16 条比平均、只在有对有错组有梯度、只挪不放、@1 → @64）；三杠杆各管什么并能归因失败；秤怎么坏及防法；SFT/RL 分工与冷启动/蒸馏；
          测量怎么骗人；工业界那套每一项对得上自己的一行代码
实验室清单   环境层 countdown / calc_tool_vllm｜老师层 enum_traces / sft_qwen｜训练层 grpo_tool_mp_vllm / screen_pool｜评测层 baseline_countdown_vllm / crosstask / analyze_search / scan_ops｜
          工具 export_hf / import_hf / slim_ckpt / vllm_check / chat_raw_vllm｜账本 PLAN.md。换任务只改环境层（题 + 判分），猜数字 = 几十行环境套进注入接口
DeepSeek 当工具三种用法   ① 整题外包 = 路由器，实验无意义，模型只学会什么都问 ② 当老师（训练前写轨迹）值：程序老师 vs 大模型老师对照，几块钱 ③ 当限量顾问（运行时问一步、扣分）有研究价值，E 之后
```

## 臂 A（限量问专家）开工：协议、探测、教材生成器（2026-09-10）
```
目的     DeepSeek 的第 ③ 种用法：运行时限量顾问。问的是「小模型能不能学会：自己先搜、搜不到才问、问回来先验证、简单题不问」，而不是什么都问
         秤不变（CD.reward），多一个工具和一条扣分（每次求助 −0.1，RL 阶段）
协议     模型写 <ask>Q</ask> 停 → harness 后台问 deepseek-reasoner（温度 0、缓存、16 线程并发）→ 注入 <reply>EQUATION: …</reply> → 续；每条最多 2 次
         注入段跟 <result> 一样 mask（SFT label −100 / RL resp_mask 0）；模型自己写出 <reply> = 假结果 → 停 + 罚；提示词 = ask_prompt（TOOL_HINT + ASK_HINT）
七步     ① harness 加 ask（calc_tool_vllm，假后端单测 ×3）✓  ② 探测 hf_armS 会不会自己写 <ask> ✓  ③ 造教材 make_ask_traces.py ✓ 本地冒烟 → GS01 跑
         ④ SFT：起点 out_armS，<reply> 段 −100（sft_qwen.RES_RE 已改）  ⑤ 评：pass@1、求助率（硬/易分开）、验证抓到的专家错、每题成本
         ⑥ RL：grpo_tool_mp_vllm 加 --ask / --ask-pen 0.1，用 ask-SFT 模型重筛池、锚到它  ⑦ 评 + 定论 ⑰
② 探测   hf_armS + ask_prompt，[5,7,14,33]→43 × 8（温度 1）：求助 0/8 → p(ask) = 0，RL 找不到这条路，必须冷启动（SFT 教）
         其中 1 条不问也解出 14 * 5 / 7 + 33（这题对它 p ≈ 1/8）
③ 教材   规则：每道题先走老师 v1 的第 0 层（符号形态 8 行）+ 第 1 层（乘积表 + 每块配剩余数，CAP 40 行），格式跟 sft_cd4_tool.jsonl 相同
         第 0/1 层命中 → 普通轨迹（plain_l0 / plain_l1），不问；没命中 →「None of these work; this one is hard, let me ask the expert.」
         → <ask>…</ask><reply>EQUATION: …</reply> → Let me verify: <calc>eq</calc><result>V</result> (perfect match, verified) → </think> <answer>
         专家错 → (not T, the expert is wrong) → Let me ask once more →（告诉它上次的式子错了）→ 再验；专家没答 → (the expert did not answer) → 再问
         两次都不行 → 丢；最终 answer 过 CD.reward 才收。选题：筛池 counts 里 c ≤ 2 的硬题 800 + c ≥ 6 的简单题 800
   为什么先搜完再问   最初版本是第 0 层没中就问。那样「第 0 层没中」之后有两种续写（自己做第 1 层 / 问），决策点有歧义，模型分不出题难不难，
                     SFT 后只会按比例乱问；RL 再加扣分，硬题池里问总是划算 → 收敛到「什么都问」= 路由器（用法 ①，实验无意义）
                     改成「第 0、1 层都搜完没中」这一处才出现 <ask>：决策点无歧义，简单题永远不问，硬题必问；RL 的扣分只需要管「别提前问」
                     代价：第 2 层被求助取代（臂 T/S 的 RL 本来就把第 2 层剪没了）；轨迹更长（最长 ≈ 2500 字符 ≈ 1100 token，max-len 1600 够）
   冒烟（本地 6 道）  [1,2,3]→6 第 0 层中；[3,4,6,8]→14 第 1 层中；[2,3,5,7]→31、[5,7,14,33]→43、[20,25,30]→15、[1,2,11,26]→28 问，专家 4/4 一次对
                     [20,25,30]→15 专家答 20 + 25 - 30 —— 老师 v1 不写的镜像符号行（最大数为负），求助正好补上老师的洞
③ GS01 结果  1600 道（硬 800 + 易 800）8 线程 1519 s ≈ 700 次 reasoner 调用 → sft_cd4_ask.jsonl 1594 条
             硬题 800：老师第 1 层就中 148（18%，模型 8 次 ≤ 2 对但程序能搜到 —— 「硬」是模型搜索的硬，不是题的硬）+ 求助 646 + 丢 6
             易题 800：第 0 层 755 + 第 1 层 45，求助 0 —— 规则保证的，不是巧合
             问了 652 道（全在硬桶），685 次调用：619 道一次、33 道两次。一次答对 619/652 = 95%，第二次补 27，两次都不行 6 → 两问覆盖 99%
             ★ 账要算细：39 次「没答」= 33 次第一问 + 6 次第二问，正好等于全部失败数 → 强专家只要答了就没错过一次（646/646），
               失败全是正文空（推理段截断 / API 错）。所以教材里 0 条「the expert is wrong」样本，27 条是「did not answer → 再问」
               （之前写的「验证抓到专家错 33 次」是把没答当成答错了，已改）
             对策  --wrong-frac 0.2：抽 20% 的求助题先问弱专家 deepseek-chat（硬题 1/10），它答错就把错答案当第一次回复
                   → 验证行 (not T, the expert is wrong) → 再问强专家。真实的错答案、真实的抓错，不是编的；碰巧答对或没答就当没问过
                   本地冒烟：[5,7,14,33]→43 弱专家答 (33 - 7) * (14 / 5) + 5 = 77.8 被抓，再问 → 33 + (14 * 5) / 7 ✓
③ 第二轮（--wrong-frac 0.2，2026-09-10）  408 s（第一问全部命中缓存）→ sft_cd4_ask.jsonl 1592 条：求助 644（一次对 535、第二次对 109、丢 8）
             弱专家问了 ≈130 道，错答用上 113（按值错 87%，跟「硬题 1/10」一致）→ 教材里「the expert is wrong → 再问 → 验证通过」≈ 105~113 条，占求助轨迹 17%
             强专家答错仍是 0；没答 12（第一问 4 + 追问 8，丢的 8 道全是追问没答）。check_sft：精度 1.00、重复 0、说 perfect 1.00、含×÷ 0.53、response 最长 1437 token
             成分：644 求助（其中 ≈110 带抓错）+ 148 硬题第 1 层自己中 + 800 易题自己中 → 进 SFT，配 sft_cd4_tool / sft_cd3_tool / sft_gsm，--max-len 1800
             为什么要这一步  ⑤ 的探针之一是把评测时的专家换成 deepseek-chat，看模型抓不抓得住错答案；训练时没见过这条分支就谈不上抓
             最长响应 3308 字符 ≈ 1400+ token，配 prompt 可能超 max-len 1600 → SFT 日志看丢弃数，> 2% 就 --max-len 2000
坑       教材里出现过 <reply>error</reply>，不是网络：旧版 deepseek_tool 把 5 条正文为空的截断回复（max_tokens 800 / 4000 时推理段用完）缓存了，
         命中就当 error 写进教材。修：缓存空正文不算命中 + 清掉；专家 max_tokens 4000 → 8000（同一题一次 4000 截断、一次 453 答出：温度 0 推理长度也不定）；
         缓存读写加锁（并发）
④ SFT 完成（out_sft_ask，2026-09-10）  起点 out_armS/ckpt_latest，数据 sft_cd4_ask 1592 + sft_cd4_tool + sft_cd3_tool + sft_gsm，--max-len 1800，448 步 19.4 分
             val 0.1324 → 0.0340（起点已会大半格式，所以起点就低；工具版 SFT 是 0.71 → 0.059）
             收尾闭环（tool_prompt，无求助说明，贪心，max_new 900）[2,3,5,7]→31：第 0 层 8 行全对 → 乘积表 6 个 → 说「35, 21, 14, 15, 10, 6 cannot reach 31」
             把 35 也跳了（35 − 3 − 2 = 30，够得着，判断错，这句是心算不是工具）→ 没做第 1 层配数 → 进第 2 层「Next, build a bigger piece」→ 900 撞顶
             读法：没求助说明就不问、走老教材的第 2 层 —— 条件行为对；第 2 层还是老样子（臂 S 的 RL 剪过它，SFT 又混回来了）；一处格式漂：两行并成一行
             GSM 那道对（36）。求助行为要用 ask_prompt 才看得到 → 第 ⑤ 步
⑤ 评测工具   baseline_countdown_vllm --ask [--ask-model deepseek-chat]（弱专家探针，输出名 _ask_chat）；analyze_ask.py：难度按老师程序标（第 0/1 层解得出 = 易题），
             按硬/易分别报 pass@1、求助率、次/条、问后验证率、问的位置（l1 后 / l0 后 / 更早）、专家错 / 抓住 / 照抄、撞顶、调用/条、问了的对 vs 没问的对
## ★★★ 臂 A 第 ⑤ 步：ask-SFT（out_sft_ask）评测，池外 [95000:95100] × 8，温度 1，max_new 1400（2026-09-10）
| | 臂 S（RL 后，起点） | ask-SFT 不给求助说明 --tool | ask-SFT 开求助 --ask（强专家 reasoner） | ask-SFT 开求助，专家换弱 deepseek-chat（探针） |
|---|---|---|---|---|
| CD-4 pass@1 / @8 | 0.748 / — | 0.729 / 0.810 | **0.966 / 1.000** | 0.715 / 0.780 |
| CD-4 全对 / 有对有错 / 全错 | — | 68 / 13 / 19 | **83 / 17 / 0** | 67 / 11 / 22 |
| 求助：次/条 · 有求助的轨迹 · 求助过的对 · 没求助的对 | — | — | 0.26 · 26.2% · 0.971 · 0.964（211 次 / 210 条 = 几乎都只问一次） | 0.47 · 25.5% · **0.049** · 0.943（376 次 / 204 条 = 1.84 次，抓住了再问，第二次还错） |
| 调用/条 · 报错 · 假结果 · 撞顶 | — | 15.3 · 6 · 1 · **15.5%** | 10.6 · 8 · 1 · 1.1% | 11.4 · 7 · 5 · 3.4% |
| 长度 中位 / 均值 | — | 265 / 550 | 265 / 437 | 265 / 496 |
| CD-3 pass@1 / @8（--tool） | 0.723 / — | 0.741 / 0.970（全错 3） | — | — |
| GSM R1 模板 / 中性模板（200 题贪心） | 0.795 / 0.635 | 0.760 / 0.595（vs Base +0.24 / +0.115，McNemar 显著） | — | — |
| 速度 800 条 | — | 1.0 分 | 8.3 分（等 API） | 4.0 分 |
预测对账   开求助 pass@1 ≥ 0.85 ✓（0.966，全错 0）；不开求助与臂 S 持平 ✓（0.729 vs 0.748，噪声内）；CD-3 不动 ✓（0.741）；
          GSM 掉 0.035 / 0.04（200 题里 7~8 道，方向一致，不算噪声也不算坏）；硬/易求助率、验证率、抓错要 analyze_ask 才能对账
读法      ① 求助率 26% ≈ 全错那 ~20 道 + 几道：问的题基本就是自己解不了的题，问了对 0.971 → 硬题从 0 变成 0.97，这是 +0.22 的全部来源
          ② 弱专家探针：问的比例不变（25.5%，决策不依赖专家），每条问 1.84 次 → 验证抓住了错答案并再问；但弱专家第二次 87% 还错，两次上限一到只能交错卷 → 0.049
             教材里没有「两次都错之后怎么办」的样本（那 8 道丢了），这是下一处要补的
          ③ 不给求助说明那份撞顶 15.5%（臂 T/S RL 后 0.5~1%）：SFT 混回 sft_cd4_tool 把第 2 层的循环也带回来了；开求助后撞顶 1.1%，因为该循环的地方改成问了
          ④ 成本：800 条 211 次 reasoner ≈ 每题 0.26 次
⑤ analyze_ask（老师标签：池外片硬题 23 道 / 易题 77 道）
| | 强专家：硬题 184 条 | 强专家：易题 616 条 | 弱专家：硬题 | 弱专家：易题 |
|---|---|---|---|---|
| pass@1 | 0.908 | 0.984 | 0.054 | 0.912 |
| 求助率 · 次/条 | **0.940** · 0.95 | **0.060** · 0.06 | 0.875 · 1.62 | 0.070 · 0.13 |
| 问的位置 l1后 / l0后 / 更早 | 172 / 1 / 0 | 36 / 0 / 1 | 159 / 2 / 0 | 43 / 0 / 0 |
| 问后验证 | 0.98 | 1.00 | 0.97 | 0.97 |
| 专家错 · 抓住 · 照抄 | 0 · 0 · 0 | 0 · 0 · 0 | 262 · **247** · 15 | 76 · 73 · 3 |
| 没答 · 撞顶 · 假 | 7 · 2.7% · 1 | 0 · 0.6% · 0 | 0 · 9.8% · 4 | 0 · 1.5% · 1 |
| 问了的对 · 没问的对 | 0.965 · **0.000** | 1.000 · 0.983 | 0.050 · 0.087 | 0.047 · 0.977 |
预测对账   硬题求助率 > 80% ✓ 0.94；易题 < 10% ✓ 0.06；位置绝大多数在第 1 层后 ✓ 208/210；问后验证 > 90% ✓ 0.98；弱专家的错大多被抓住 ✓ 320/338 = 95%
读法      ① 冷启动一次就把「做法」装到位了：什么时候问（99% 在第 1 层搜完之后）、问完先验（98%）、错了不抄（95%）—— 因为这是模板，不是搜索决策，温度 1 也稳
          ② 硬题没问的 6% 全错（0.000）：那是绕进第 2 层撞顶的；问了就 0.965。RL 最直接的一口肉：把这 6% 推去问 → 硬题 +0.05，总分 +0.013
          ③ 没答 7 次只再问了 1 次（211 次求助 / 210 条）：教材有 27 条「没答 → 再问」，温度 1 下没走 —— RL 第二口
          ④ 易题问的 6%：老师第 1 层能解、模型第 1 层执行漏了（「35 cannot reach」那类判断错）转而去问。强专家下无害（1.000），弱专家下 0.047；RL 有 −0.1 的价，会压它
          ⑤ 弱专家：抓住 95% 但硬题 pass@1 只 0.054 —— 两次都错之后没有教过的续写，只能交个错卷；教材里那 8 道丢了，环境里也没这条路的回报
          ⑥ 强专家下 pass@1 天花板 ≈ 0.98~0.99，RL 在主指标上的份额只剩 ~0.02。RL 真正要回答的是价格问题：−0.1 一次，它会不会把问的位置提前、会不会压掉易题的 6%
## 臂 A 第二轮设计：提示词不绑模板、价格 0.05、求助上限 3、API 错 harness 重试（2026-09-10，用户要求「提示词写得不好也要对」）
```
三层     ① 单条可靠性（RL 管）：0.966 → 天花板 0.98~0.99；上限 2 → 3、价格 0.1 → 0.05（偏「对」不偏「省」，但不到 0，0 = 什么都问）、专家没答 harness 自动重试一次不算次数
         ② 提示词鲁棒（训练数据管，RL 管不了）：所有题一个模板 → 行为绑模板（零提示词对话那次见过）。prompts_cd.py：TRAIN 10 个模板（英 8 + 中 1 + 表格式 1）
            工具说明 / 求助说明各 3 种措辞、各以 0.7 概率出现（不出现 = 教「没说明也会用」）；HELDOUT 3 个训练不见：啰嗦英文、另一种中文、★ 完全不提工具
            顺序：先 SFT 再 RL（RL 只放大已有的 p；没见过的模板 p ≈ 0 抽不到）。remix_prompts.py 换教材提示词，response 不动
         ③ 真正的保证（外循环）：pass@8 已 1.000 → 推理时采 8 条、计算器验、挑验证通过的交卷，成本 8 倍；RL 让单条更可靠是为了让这层更便宜
奖励     R = 1[对] − 0.05 × 求助次数 − 0.2 × 1[自己编 <result>/<reply>]，无部分分（v3）、无长度项；KL β 0.02 锚到 SFT-2；环境里的专家 80% reasoner / 20% chat
文件     prompts_cd.py / remix_prompts.py / calc_tool_vllm.py（_ask_retry、ask_hint(n)）/ baseline_countdown_vllm.py（--prompt-set、--max-asks）
下一步   remix 三份 CD 教材 → SFT-2（out_sft_ask2，起点 out_armS）→ 评 r1 与 heldout 两套模板（--ask，上限 3）→ grpo_tool_mp_vllm 加 --ask → 重筛池 → RL
```

## ★★★ 臂 A SFT-2（out_sft_ask2：教材换随机模板，其余同 SFT-1）评测，池外 [95000:95100] × 8，温度 1，求助上限 3（2026-09-10）
| | SFT-1（全 R1 模板） | SFT-2 R1 模板 | SFT-2 ★留出模板（训练没见过，1/3 不提工具） | CD-3 --tool：SFT-1 → SFT-2 |
|---|---|---|---|---|
| CD-4 pass@1 / @8 | 0.966 / 1.000 | 0.921 / 1.000 | **0.919 / 0.990** | CD-3：0.741 / 0.970 → 0.733 / 0.930（臂 S 0.723） |
| 全对 / 有对有错 / 全错 | 83 / 17 / 0 | 73 / 27 / 0 | 73 / 26 / 1 | CD-3：49 / 48 / 3 → 52 / 41 / 7 |
| 有求助的轨迹 · 求助过的对 · 没求助的对 | 26.2% · 0.971 · 0.964 | 21.1% · 0.994 · **0.902** | 22.9% · 0.967 · 0.904 | — |
| 撞顶 · 假 · 报错 | 1.1% · 1 · 8 | 4.2% · 0 · 11 | 3.1% · 5 · 27 | CD-3：2.5% · 4 · 19 → 3.9% · 10 · 6 |
| 速度 800 条 | 8.3 分 | 1.4 分 | 4.9 分 | 1.0 分 |
训练     val 0.2097 → 0.0358（起点高是因为提示词陌生；终点与 SFT-1 的 0.0340 同级），448 步 19 分
预测对账  留出模板 0.90~0.95 ✓ 0.919；R1「0.96 上下」✗ 0.921（−0.045）；CD-3 未贴
读法      ① 鲁棒性买到了：没见过的模板 0.919 ≈ 见过的 0.921，差 0.002；1/3 的题提示词根本不提工具也在里面（按模板拆分要 analyze_ask）
          ② 代价 −0.045 在 R1 模板上，来源是「没求助的对」0.964 → 0.902：硬题该问没问、绕圈撞顶（4.2%）。收尾闭环把它演了出来：
             「Pair products and quotients:」那一行失控，把 7*3+5、7*3-5、7*3+5+2… 全塞进同一行连写到 900 撞顶 —— 第 1 层的格式被随机模板搅松了
          ③ ✗ 我先把 CD-3 那份（样例 [1,13,42]→56、文件名没有 countdown4）当成了「CD-4 不给求助说明」，据此写了「pass@8 0.81 → 0.93 的意外收获、自问自答」，
             全是错读，已撤回。真相：CD-3 pass@1 0.741 → 0.733 持平，pass@8 0.970 → 0.930、全错 3 → 7、撞顶 2.5% → 3.9%，方向和 ② 一致（第 1 层松了一点）
             SFT-2 没跑过 CD-4 --tool（不给求助说明）；harness 没开求助时确实不拦模型自写的 <ask>/<reply>（calc_tool_vllm 第 280 行），
             「模型自己扮专家再验证」是否存在要真跑一次才知道，作为可选探针
          ③″ 补跑 SFT-2 的 CD-4 --tool（有计算器说明、没求助说明、harness 不开求助）：pass@1 0.719，硬题 0.027 / 易题 0.925，硬题撞顶 48%、假结果 24 条（13%）、调用/条 34.7
             模型自己写 <ask>…<reply> 的只有硬题 4.9%，自答的 12 个式子全错、全被验证抓住（0 对）→ 「自问自答」不存在，它当不了自己的专家；
             没有专家硬题就是 0，专家是硬题上全部的收益；验证在这种情况下只防错不加分
          ③′ analyze_ask（R1）：硬题求助率 0.94 → 0.755、撞顶 0.027 → 0.141、调用/条 22 → 27.6，没问的全错 → 少问的那部分不是决定不问，是第 1 层枚举失控把预算烧完
             留出按模板：10 啰嗦英文 0.915 / 11 中文 0.932 / 12 完全不提工具 0.909（求助率 25%，照样调计算器、照样问）→ 工具用法进权重了；中文问专家错 3 次全抓住
          ④ RL 的活变多了：0.921 → 0.98 的空间主要是「硬题该问没问」和「第 1 层失控」，都是 0 分对 0.99 分的组内差，正是 GRPO 能推的
## ★★★ SFT-1 在没见过的模板上：0.036 —— 做法绑模板是全绑，不是打折（2026-09-10）
| 留出模板（训练没见过），CD-4 池外 × 8 | SFT-1（只见过 R1） | SFT-2（10 模板混训） |
|---|---|---|
| pass@1 全部 | **0.036** | **0.919** |
| 模板 10 啰嗦英文 / 11 中文 / 12 完全不提工具 | 0.081 / 0.004 / 0.023 | 0.915 / 0.932 / 0.909 |
| 调用/条 · 报错 · 假结果 | 5.8 · **998** · 62（7.8%） | 11.2 · 27 · 5 |
| 12 号上调计算器 | 0 次（根本不用） | 照用，还问专家 25% |
| 求助 | 15.6% 的轨迹，全在「更早」位置（没搜就问），问回来 0.024 对 | 23%，全在第 1 层后，问回来 0.967 对 |
| 用时 | 19.3 分（乱问 + 乱写） | 4.9 分 |
样例   R1 之外的模板下 SFT-1 退回 Base 的胡写：开头就 <answer>、自己写 <reply>、编「规则」、</calc> 没有 <calc>（就是那次 IndexError）
预测对账  ✗✗ 我说 0.80~0.88，实测 0.036，差一个数量级。错在把「绑模板」想成打折，实际是整套做法只挂在那一个前缀上，前缀一换连计算器都不碰
读法   ① SFT 装进去的做法是「条件行为」，条件是题面的精确措辞；单模板 SFT = 只在那一句话下会。这就是零提示词对话那次看到的东西的定量版
       ② 10 个模板混训把条件撑宽：没见过的三个模板 0.919，和见过的 0.921 一样；代价是见过的那个掉 0.045（第 1 层松了），RL 去补
       ③ 换算：−0.045（一个模板）换 +0.88（所有别的模板）。用户的直觉「提示词写得不好也要对」对应的就是这一格
★ 定论 ⑰（候选，RL 后定稿）  小模型 SFT 学到的做法绑定在提示词的精确措辞上，换措辞归零（0.966 → 0.036）；训练时混 10 种措辞、说明随机有无，
       没见过的措辞也能 0.92，包括完全不提工具的题面；代价 −0.045，来源是第 1 层收尾变松

### 教学：harness 加求助的本质 = 给环境加一个动作；工业界「说着说着开沙盒跑测试」同构（2026-09-10）
```
做什么   harness 是套在生成外面的循环：盯 token 流，见暗号就停，替模型做它做不了的事，把结果贴回文本，再续。原来只认 </calc>（算式），
         加求助 = 再认 </ask>（发给 DeepSeek，包成 <reply>…</reply> 贴回）+ 数次数 + 没答重试 + 自己写 <reply> 算假 + 注入段打「不是模型写的」标记 + 单测
本质     定义环境，不是改模型。模型只会给下一个 token，不知道什么叫工具。harness 规定「写出这串字，世界会怎样」：给环境加一个动作，定它的效果（一行 EQUATION）、
         观测格式（<reply> 出现在自己的文本里，分不出谁写的）、代价（奖励里扣 0.05，训练器定）。秘书比喻：以前秘书只带计算器，现在还带一部电话；学生没变
工业界同构（用户提出）  「模型说着说着打开一个沙盒跑测试、把结果返回」就是同一个协议：
         停          模型吐出结构化的工具调用（特殊 token / JSON function call，写在 chat template 里、训练时教过）→ 运行时在这里停
         做          agent 运行时把调用交给沙盒执行（bash、跑测试、改文件、开浏览器），有隔离和超时
         贴          结果作为一条 role=tool 的消息接回上下文（我们是内联的 <result>/<reply>，形式不同、位置相同）
         续          模型接着生成；训练时只有 assistant 的 token 进 loss，工具消息 mask 掉 —— 就是我们的 label −100 / resp_mask 0
       差别只在规模和包装：工具多（几十个）、一次任务几百次调用几小时（长程，要管上下文）、格式是 JSON 不是尖括号、代价一般不按次扣而是靠时限和奖励设计；
       agentic RL 的 rollout 就是带真沙盒的这个循环，「卖环境」的公司卖的就是这套秤 + 沙盒 + 题
```

⑥ 筛池结果（hf_sft_ask2，RL 同款环境：随机模板 + 求助 + 80/20 专家，[0:90000] 抽 4000 × 8，4 卡各 1000，2026-09-10）
   全对 2708 (67.7%) / 有对有错 1289 (32.2%) / 全错 3 (0.1%)    c=0..8：[3, 4, 19, 59, 170, 265, 358, 414, 2708]
   预测对账  ✗ 我说有对有错 10~15%，实测 32%。错在只想了「硬题被专家解掉」，没算「每条都有失控的概率」：硬题单条撞顶 14%，8 条全不失控的概率 0.86^8 ≈ 0.30，
             七成硬题落进有对有错；易题偶尔滑一下也进来。臂 S 的池偏低（c=1 最多），这池偏高（c=7 最多）：差异来自单条不稳，不来自题难 —— 正是 RL 要压的
   全错只剩 3 道：有专家几乎没有解不了的题。池 1289 道，300 步 × 4 道 = 1200 次抽，每道约见一次
   → data/countdown4_ask.json；下一步冒烟 → 正式 RL
⑥ 训什么、怎么判有效（用户复述 + 定判据，2026-09-10）
   用户复述  RL 这次训的是让模型稳定地：先自己解 → 解不出问专家 → 专家的式子用计算器验过再交卷。✓ 补一句：这三步 SFT 已经教会，RL 不是教步骤，
             是在「问一次扣 0.05」的价格下把三步走稳 —— 现在的洞是硬题 24.5% 第 1 层失控没走到问的位置（0 分），RL 拿组内差把它压掉
   判据（第 ⑦ 步对账，全部在池外 [95000:95100] × 8、温度 1，R1 和留出模板各一份）
     主指标   硬题求助率 0.755 → ≥ 0.95   硬题撞顶 0.141 → ≤ 0.03   → CD-4 pass@1 0.921 → ≥ 0.97（留出模板同步 ≥ 0.96）
     对照     易题求助率 ≤ 5%（价格起作用，没退化成什么都问）   问的位置仍在第 1 层后   问后验证 ≥ 0.98
     不能坏   CD-3 ≈ 0.73、GSM ≈ 0.76（锚）
     探针     专家换 deepseek-chat：抓住率 ≥ 95% 不降、照抄 ≤ 2%（环境里 20% 弱专家让验证有回报）
     一句话   看硬题那一行：求助率上去、撞顶下来、pass@1 跟着上去，而易题求助率没动 —— 四个数同时成立才算 RL 有效
### 讨论：求助能力推广到编程 / 逻辑 / 问答，以及能不能做成产品（用户提出，2026-09-10）
```
通用的（不动）   harness 的停做贴续、<ask> 动作的注入 / mask / 计次 / 假回复罚、教材骨架「先试 → 试不出问 → 验 → 交」、奖励「对 − 价格 × 次数」、模板混训
每个任务配三样  ① 「自己试失败」的信号  ② 怎么验证专家的答案  ③ 秤
   编程题   跑测试失败 N 次 ｜ 把专家的代码跑测试 ｜ 测试通过率        → 高，就是工业界沙盒，harness 加 <run>；教材可让 DeepSeek 写（用法 ②）
   逻辑题   自推答案过不了检查器 ｜ 程序逐条核约束 ｜ 唯一解匹配       → 高，能写检查器就能做
   常识问答 只有「没把握」的内部信号 ｜ 验不了 ｜ 训练时有标答          → 中，性质变了：退化成「知道自己不知道」= 校准 / 学会转交，结果是级联
风险      ① 退化成什么都问：专家越强价格越低越容易；CD / 编程 / 逻辑靠「先试完才准问」的结构位置挡，问答只能靠价格 + 校准，要做价格扫描
          ② 1.5B 的底子：编程只会简单 Python，自己试的成功率低，求助率天然高
顺序      编程（验证器现成、沙盒同构、教材便宜）→ 逻辑（练检查器）→ 问答（另一课题「学会转交」，价格扫描是主实验）。每步只新写环境层：题、判分、验证工具、模板
产品      能做成的是「级联的本地层」，不是通用助手：小模型接易活、接不住的转大模型、能验的先验。同类：端侧模型 + 云端兜底、RouteLLM、FrugalGPT
   三个数  本地承接率（CD-4 75%）、转交的准头（RL 在价格下学的）、有没有验证器（代码 / 公式 / SQL 有，开放问答无 → 只是统计上划算）
   1.5B   窄而可验证的领域（小工具函数、正则、SQL、表格公式）现在就能用；真实编程本地承接两三成，产品变成薄过滤层；要厚就换 7B（待办 M 的尺子）
   还缺    弃答奖励（现在错答 = 不答 = 0，两次都失败只会硬猜；产品要「不会就说不会」，弃答分 > 错答分）、延迟 / 成本预算、真实流量评测（切片教训同样适用）、对抗 + 多轮
harness 复用 ≠ 模型学会了什么题都问   harness 是代码，不学习：它只管「见到 </ask> 就去问、贴回来」，对任何题一样。模型的「什么时候问」是学来的、绑任务：
          CD 上是「两层搜完没中」，编程上是「测试失败 N 次」，信号不同，得用新任务的教材 + RL 再教。可能有点迁移（「卡住就问、问完先验」的习惯），
          新任务上零样本测一次求助率就知道（第 ② 步那种探针）
```

### 教学：上下文开关的本质、隐藏上下文、训练改的是映射表（2026-09-10）
```
本质     模型没有「模式」，只有条件概率 p(下一个 token | 上下文)。预训练把几万种「这种文本里下一句是什么」学进同一份权重，谁出场由上下文决定。
         开关 = 上下文选中权重里的哪一个条件分布；不改权重，只选支。R1 那段模板就是一个开关，把 Base 拨进「先想后答」那片文本的续写
四个词   零件（权重里的能力，预训练给）｜撒种（往权重里放新零件：SFT / 蒸馏）｜开关（提示词，推理时选支，不改权重）｜秤 + RL（在开关打开的那片里挪概率）
         电路比喻：零件是墙里的线，开关选接哪条，RL 改开关和线的接法；SFT-1 把接法只焊在一个开关上（换开关 0.036），模板混训把同一套线焊到十个开关上
推论     ① 训练改的是「上下文 → 行为」这张映射表，不是行为本身；表长什么样由训练数据里的上下文分布决定：单一上下文 → 尖峰，混合 → 平台。SFT、RL 都如此
         ② 上下文是连续的不是查表：相近措辞引出相近行为，大模型改写多半还能用；小模型窄训后连续性断了
         ③ 模型自己写的 token 也是上下文 → 生成 = 不断给自己拨开关的轨迹；思维链就是自开关；SFT 教它写哪些自开关，RL 挑哪些自开关后面跟着高分
         ④ RL 必须在某个上下文分布下做，学到的主要挂在那个分布上，往相近上下文泄一点：mix4 的 GSM 增益 R1 模板 +0.24、中性模板 +0.115；
            臂 S 只在 R1 模板上 RL；臂 A 的 RL 在 10 个模板上 rollout，就是把这张表往宽了改
用户复述 ★  问大模型问题，很多时候不是问题本身，是「上下文 + 问题」才构成完整的问题，产品替我们省去了上下文；开关的本质就是一个完整的问题，
            所有问题都必须是完整具体情景下的问题。—— 对。补一句：模型回答的永远是 p(答 | 前面的一切)，光秃秃的问题是不完整的题面，缺的情景模型会用
            预训练里最常见的情景补上，答非所问和泛泛而谈就是这么来的。「完整」是相对的：写不完所有情景，要写的是先验会猜错的那部分；
            写过头又成了尖峰（SFT-1）。证据：同一道题换模板行为就换；零提示词对话那次的怪行为；产品的系统提示词就是替用户预填的情景
退火     上下文蒸馏：带开关的输出当老师，不带开关的自己当学生，把开关烧进权重。模板混训里 30% 不给说明是粗糙版：12 号模板不提工具照样用 → 开关进了权重
工业界   标准化（chat template 特殊 token）｜显式开关（思考模式、工具 schema）｜抗措辞（指令多样化）｜烧进去（上下文蒸馏、角色训练）｜故意失灵（安全训练让越狱开关接不通）｜
         激活空间的 steering 向量（不走文本）
隐藏上下文  用户看到输入框，模型看到几千到几万 token：系统提示词（身份、规则、日期、安全）、工具说明、检索文档、记忆画像、对话摘要、模板特殊 token、思考段、开发者消息。
         控制权在上下文里 → 提示词注入是真实攻击面；「提示词工程」改叫「上下文工程」说的就是这层。Anthropic 公开 Claude.ai 的系统提示词，多数厂商不公开
```

### 教学：比「开关」更好的理解 —— 模型是续写机，上下文是半篇文章（2026-09-10）
```
一句话     模型只做一件事：判断「这是篇什么文章」，然后按那类文章的写法把它写完。上下文 = 前半篇，回答 = 后半篇。「开关」是离散的速记，这个是连续的本体
推出来     提示词 = 你写的前半篇（R1 模板 = 「一篇先想后答的解题记录」的开头）
           隐藏上下文 = 产品替你写好的前半篇（身份、规则、工具说明、检索资料），用户那句话只是最后一段
           零件 = 预训练读过的文章类型；没读过的类型开头写得再好也续不出 → 嵌套搜索 p = 0 要撒种
           SFT = 往读过的文章里加一种新类型；只加一种开头 → 只认那种开头（SFT-1 0.036）；同一内容配十种开头 → 学到的是内容不是开头（SFT-2 0.919）
           RL = 在某类文章上让续写更常写出得分高的后半篇；只改那类文章的续法 → rollout 用什么前半篇，收益就挂在什么前半篇上，往相近类型泄一点
           思维链 = 文章续写自己：每写一句前半篇就长一句；「First, only + and -:」带出整套枚举，「Let's try different combinations」写两遍把自己续进循环
           提示词注入 = 有人往你的前半篇里插一段，续写跟着他走
好在三处   连续（相近开头续出相近文章：改写大多可用、窄训后不可用）｜可叠加（系统提示、工具说明、用户问题是同一篇文章的几段，不是几个独立开关）｜
           自己写的和别人写的在同一层（模型分不出 harness 注入的 <result> 和自己写的 —— 我们的协议能成立的根本）
用法       速记用「开关」；想清楚一件事时换成「这是篇什么文章，前半篇是谁写的」
```

## ★★★ 臂 A RL 跑完（out_armA，150 步，174.7 分，2026-09-10）
```
配置    起点 = 锚 = out_sft_ask2；池 data/countdown4_ask.json（1289 道）；每步 4 CD + 4 GSM × 16；v3 只奖不罚；−0.05/次，上限 3；80/20 专家；随机模板；β 0.02
中途评测（贪心，[90000:90100]，固定随机模板，开求助）  第 25 步 CD 0.990 / GSM 0.820 → 第 150 步 CD 0.960 / GSM 0.810；峰值 0.9029 @125（CD 0.99 + GSM 0.82 的平均，差在噪声内）
池内训练读数（温度 1，前 20 步 → 后 20 步）
   求助/条 0.85 → 0.93   问过 0.74 → 0.77   问对 0.94 → 0.90   ★ 没问对 0.55 → 0.77   CD 长度 840 → 798   调用/条 21.4 → 19.6   假结果 0.001   标注精度 0.99
   |A| CD 0.212 → 0.153（组内差异在缩小）   KL 0.0003（几乎贴着锚）   每步 25+7 s
读法    ① 池内该动的都动了：没问的轨迹对得多了（0.55 → 0.77）= 失控枚举少了；问过略涨；长度、调用往下。这就是「搜完就问、别失控」被放大
        ② 贪心评测没涨（0.99 → 0.96）：贪心在第 25 步就到顶了，选不出差别；真正的账在第 ⑦ 步温度 1 × 8 的池外评测 + analyze_ask（硬题求助率、撞顶）
        ③ 问对略降（0.94 → 0.90）：问的轨迹变多，把弱专家答错的也算进来了，不是验证变差
        ④ 存盘这次没出事（盘 56 GB）
下一步  ⑦：导出 latest（150）和 best（125）各一份，R1 ask 两个都评一次挑一个，再跑留出模板 / 弱专家探针 / CD-3 / GSM，analyze_ask 对账
```

## ★★★ 臂 A 第 ⑦ 步：RL 后评测，池外 [95000:95100] × 8，温度 1，求助上限 3（2026-09-10）
| | SFT-1 | SFT-2（RL 起点） | **RL 150 步（latest）** | RL 125 步（best） |
|---|---|---|---|---|
| CD-4 pass@1 / @8，R1 模板 | 0.966 / 1.000 | 0.921 / 1.000 | **0.979 / 1.000** | 0.988 / 1.000 |
| CD-4 pass@1 / @8，★留出模板 | 0.036 | 0.919 / 0.990 | **0.980 / 0.990** | — |
| 全对 / 有对有错 / 全错（R1） | 83 / 17 / 0 | 73 / 27 / 0 | 88 / 12 / 0 | 90 / 10 / 0 |
| 有求助的轨迹 · 问了的对 · 没问的对（R1） | 26.2% · 0.971 · 0.964 | 21.1% · 0.994 · 0.902 | 26.9% · 0.977 · **0.979** | 27.8% · 0.995 · 0.984 |
| 撞顶（R1 / 留出） | 1.1% / — | 4.2% / 3.1% | **1.4% / 0.2%** | 0.5% / — |
| 调用/条 · 假结果（R1） | 10.6 · 1 | 11.9 · 0 | 9.9 · 0 | 9.5 · 1 |
| 弱专家探针：pass@1 · 求助过的对 · 每条问几次 | 0.715 · 0.049 · 1.84（SFT-1）| 0.715 · 0.049 · 1.84 | **0.764 · 0.154 · 2.58** | — |
预测对账   R1 ≥ 0.97 ✓ 0.979（best 0.988）；留出 ≈ 0.96 ✓ 超了，0.980，和 R1 一样高；撞顶往下 ✓；易题求助率、硬题求助率、位置要 analyze_ask
读法      ① RL 把 SFT-2 丢的 0.045 补回来还多涨：没问的对 0.902 → 0.979，就是「第 1 层失控」被剪掉了（撞顶 4.2% → 1.4%，留出 0.2%）
          ② 鲁棒性没丢反而齐了：没见过的模板 0.980 = 见过的 0.979。RL 在 10 个模板上 rollout，收益泄到了没见过的 3 个上
          ③ 没退化成什么都问：求助率 26.9%，和 SFT-1 的 26.2% 一样，≈ 硬题的比例；价格 0.05 把「问」按在该问的题上
          ④ 弱专家探针：每条问 2.58 次（上限 3）→ 「错了再问」从习惯变成了动作；求助过的对 0.049 → 0.154，三次都错的仍无路可走（教材没这条）
          ⑤ best（125）比 latest 高 0.009 = 7 条样本，噪声内；主线用 latest
待补      analyze_ask 四张表（硬/易求助率、撞顶、位置、抓错）、CD-3、GSM → 定论 ⑰ 定稿
⑦ analyze_ask（RL 后，latest；括号里是 SFT-2 起点）
| | 硬题 184 条 | 易题 616 条 |
|---|---|---|
| pass@1 | 0.946（0.750） | 0.989（0.972） |
| 求助率 · 次/条 | **0.957 · 0.97**（0.755） | 0.063 · 0.06（0.049） |
| 问的位置 l1后 / l0后 / 更早 | 176 / 0 / 0 | 39 / 0 / 0 |
| 问后验证 | 1.00 | 1.00 |
| 撞顶 | **0.033**（0.141） | 0.008 |
| 问了的对 · 没问的对 | 0.977 · 0.250 | 0.974 · 0.990 |
   best（125）：硬题 0.967 / 求助 0.967 / 撞顶 0.016；易题 0.994 / 求助 0.071 —— 和 latest 差在噪声内
   留出模板（latest）：硬题 0.967 / 求助 0.967 / 撞顶 0.011；易题 0.984 / 求助 0.054；按模板 10 / 11 / 12 = 0.989 / 0.992 / 0.958（12 号不提工具，求助 26.5%）
   弱专家探针（latest）：硬题 pass@1 0.120（0.054）、求助 0.962、每条问 2.52 次、专家错 423 抓住 404（95.5%）照抄 19、撞顶 15.8%；易题 0.956、错 78 抓 76；
                        全部：抓住 480/501 = 95.8%，照抄 21（4.2%，SFT-2 5.3%）
预测对账   硬题求助率 ≥ 0.95 ✓ 0.957 ｜ 硬题撞顶 ≤ 0.03 ✓ 0.033（best 0.016）｜ R1 ≥ 0.97 ✓ 0.979 ｜ 留出 ≈ 0.96 ✓ 0.980 ｜ 位置在 l1 后 ✓ 100% ｜ 验证 ≥ 0.98 ✓ 1.00 ｜
          易题求助率 ≤ 5% ✗ 6.3%（和 SFT-1 的 6% 一样，价格 0.05 没把它压下去，也没让它涨）｜ 弱专家照抄 ≤ 2% ✗ 4.2%（5.3% → 4.2%，方向对、幅度小）
读法      ① RL 的收益全部来自「硬题该问没问」那 24.5% → 4.3%：撞顶 0.141 → 0.033，硬题 0.750 → 0.946。学的不是新决策，是把已有决策走稳
          ② 易题求助率纹丝不动（5~7%）：这些是模型第 1 层执行漏了才问的，RL 没修第 1 层的执行，价格也没大到让它「不问硬撑」。强专家下无害
          ③ 弱专家下每条问 2.52 次 = 把上限用满，「错了再问」被 RL 练成了动作；三次都错之后 15.8% 撞顶 —— 教材没这条路，RL 也没回报，仍是空白
          ④ 12 号模板（不提工具）0.958，最低但求助 26.5% 照常：工具用法在权重里，RL 后也在

⑦ 补两格（latest）
   CD-3 --tool（有计算器说明、没求助说明、harness 不开求助）：pass@1 0.683 / @8 0.910，全对 46 / 有对有错 45 / 全错 9，撞顶 10.5%（84 条）、假结果 27（3.4%）、调用/条 14.5
        对照：臂 S 0.723；SFT-1 0.741 / 0.970；SFT-2 0.733 / 0.930（撞顶 3.9%、假 10）→ ★ 掉 0.05，撞顶翻近三倍
        读法：这是「把学会求助的模型放进没有求助出口的环境」的结果。RL 把硬题的策略推成「搜完就问」（96%），CD-3 的硬题走到那一步写 <ask>，
             harness 没开求助不拦也不答，模型接着往下写 → 撞顶、自己编 <result>。SFT-2 时求助只有 75%，掉得少。要分清「能力退化」还是「出口被堵」：
             跑 CD-3 --ask（开出口）；analyze_ask 数 --tool 那份里模型自写的 <ask> 有多少
   GSM crosstask（200 题贪心）：R1 模板 0.780 / 中性 0.595（SFT-1 0.760 / 0.595；臂 S 0.795 / 0.635；Base 0.52 / 0.48）→ 没坏，锚在管
⑦ CD-3 悬案判了：出口被堵，不是能力退化，但工具依赖是真代价
   CD-3 --ask（开出口，求助说明在）：pass@1 **0.951** / @8 1.000，全错 0，求助 27.5%，问了的对 0.995，没问的对 0.934，撞顶 0.2%，假 0（预测 ≥ 0.80 ✓，全错 ≤ 2 ✓）
   CD-3 --tool（没出口、没说明）的 analyze：老师标签硬题 33 道；硬题 pass@1 0.284、模型自写 <ask> 只有 4.9%、撞顶 25%、假 17 → 没说明的上下文下它基本不问，是绕圈
        对照 SFT-2 同配置 0.733 / 撞顶 3.9%：★ RL 后在「没出口」的上下文里比 RL 前差 0.05、绕圈翻近三倍 —— 策略更依赖出口了，这是 RL 的真实代价，记进定论
   同一份 CD-3：臂 S 0.723 → SFT-2 没出口 0.733 → RL 没出口 0.683 / RL 开出口 0.951
--show 看到的（弱专家探针，[4,18,45,46]→13，两条都问满 3 次）
   ① 验证抓错 → 再问，措辞是模型自己升级的：「Show your work」→「Your previous answers are wrong. Your equation must work.」→「You may ask only once more」，教材里没有这些句子
   ② 三次都错之后：一条把最后那个错式子抄进 answer（得 0.21 的格式分），一条试图问第四次（超上限，harness 不拦）写到撞顶 —— 「无路」的具体样子
   ③ 乘积表里写了 <calc>46 / 2</calc>、<calc>18 * 2</calc>：编了一个不存在的 2。工具算什么都给值，这类「用了不存在的数」的行没有代价（v1 只查最终 answer 的洞，第 N 次出现）
## ★ 臂 A 评测总表（out_armA latest = 150 步；池外 [95000:95100] × 8，温度 1，求助上限 3；GSM 200 题贪心）
| 评测 | 臂 S（起点的起点） | SFT-1 | SFT-2（RL 起点） | **RL 后** |
|---|---|---|---|---|
| CD-4 R1 模板，开出口：pass@1 / @8 | 0.748 / — | 0.966 / 1.000 | 0.921 / 1.000 | **0.979 / 1.000**（best 0.988） |
| CD-4 没见过的模板，开出口 | — | 0.036 | 0.919 / 0.990 | **0.980 / 0.990**（模板 10/11/12 = 0.989/0.992/0.958） |
| CD-4 弱专家探针（chat） | — | 0.715 | 0.715 | 0.764（求助过的对 0.049 → 0.154；抓住 95.8%，照抄 4.2%） |
| CD-3 开出口 | — | — | — | **0.951 / 1.000**（全错 0） |
| CD-3 没出口、没说明 | 0.723 | 0.741 / 0.970 | 0.733 / 0.930 | 0.683 / 0.910（撞顶 10.5%）← 工具依赖 |
| GSM R1 / 中性 | 0.795 / 0.635 | 0.760 / 0.595 | — | 0.780 / 0.595 |
| 求助率（R1）· 问了的对 · 没问的对 | — | 26% · 0.971 · 0.964 | 21% · 0.994 · 0.902 | 27% · 0.977 · 0.979 |
| 硬题：求助率 · 撞顶 · pass@1 | — | 0.94 · 0.027 · 0.908 | 0.755 · 0.141 · 0.750 | **0.957 · 0.033 · 0.946** |
| 易题：求助率 · pass@1 | — | 0.06 · 0.984 | 0.049 · 0.972 | 0.063 · 0.989 |
| 问的位置在第 1 层后 · 问后验证 | — | 99% · 0.98 | 100% · 0.99 | 100% · 1.00 |
| 每题成本（强专家） | 0 | 0.26 次 | 0.21 次 | 0.27 次 reasoner 调用 |

# ★★★ 全模型总表 v3 · 完整版（从 Base 到臂 A，2026-09-10）
| 模型 | 血统 | GSM R1 / 中性 | CD-3 @1 / @8 | CD-3 @64 | CD-4 @1 / @8 | CD-4 @64 | CD-4 全错 | 一句话 |
|---|---|---|---|---|---|---|---|---|
| Base | Qwen2.5-1.5B 原始 | 0.52 / 0.48 | 0.02 / 0.18 | 0.59 | 0.02 / — | — | — | 零件都有，散着 |
| Instruct† | 官方 Instruct | 0.687 / — | 0.015 / 0.09‡ | — | — | — | — | 对照 |
| GSM-RL† | Instruct + GSM8K RL | 0.767 / — | 0.038 / 0.26‡ | — | — | — | — | 第一次 RL 有效；反向迁移到 CD |
| CD-RL | Base + CD-3 RL 275 | 0.16 / 0.35 | 贪心 0.405 | — | — | — | — | 单任务：GSM 塌，遗忘 + 绑定 |
| mix-275 | Base + GSM + CD-3 | 0.71 / 0.47 | 0.16 / 0.56 | — | — | — | — | 混训止住遗忘 |
| mix-600 | mix-300 续，len-soft | 0.78* / — | 0.46 / 0.69 | 0.83 | 0.21 / 0.61 | — | — | CD-3 饱和，顿悟形态在这 |
| mix4-300 | mix-600 + GSM + CD-4 | 0.74 / 0.46 | 0.476 / 0.660 | 0.83 | 0.393 / 0.670 | — | 33 | 三代 RL 的终点，五臂起点 |
| 臂 1 SFT 后（sft1） | mix4 + 穷举器 SFT | 0.755 / 0.565 | 0.599 / 0.930 | 1.000 | 0.506 / 0.720 | — | 28 | 撒种抬天花板 |
| 臂 0 只 RL | mix4 + RL v1 | 0.75 / 0.49 | 0.511 / 0.700 | 0.83 | 0.471 / 0.710 | — | 29 | 无种 RL 原地踏步 |
| 臂 2 自蒸馏 + RL | mix4 + 自采样 SFT + RL | 0.77 / 0.49 | 0.518 / 0.720 | 0.84 | 0.473 / 0.690 | — | 31 | 自蒸馏无效 |
| 臂 1 撒种 + RL | sft1 + RL v1 | 0.755 / 0.50 | 0.503 / 0.740 | 0.95 | 0.563 / 0.750 | — | 25 | 无工具最优；塑走 @64 −0.05 |
| 臂 3 mix4 + v2 | mix4 + RL v2 判分 | 0.75 / 0.535 | 0.486 / 0.690 | 0.80 | 0.416 / 0.670 | — | 33 | 判分器被钻 |
| 臂 3′ sft1 + v2.1 | sft1 + RL v2.1 | 0.755 / 0.455 | 0.444 / 0.630 | 0.90 | 0.544 / 0.660 | — | 34 | 过程罚 = 罚探索 |
| 臂 T SFT 后（sft_tool） | mix4 + 工具版 SFT | 0.760 / 0.625 | 0.714 / 0.980 | ≥0.98 估 | 0.710 / 0.790 | 0.900 | 21 | 算术清零，最大一跳 |
| 臂 T RL 后 | sft_tool + RL v1 | 0.745 / 0.625 | 0.590 / 0.720 | 0.920 | 0.711 / 0.790 | — | 21 | 学会猜，CD-3 塌 |
| 臂 S 150 步 | sft_tool + RL v3、筛池、锚 0.02 | — | 0.750 / 0.980 | — | 0.754 / 0.800 | — | — | 无专家的最优 pass@1（hf_armS150） |
| 臂 S 300 步 | 同上续 | 0.795 / 0.635 | 0.723 / 0.957（64） | 1.000 | 0.748 / 0.808（64） | 0.900 | 18 | 平台，天花板不动；臂 A 起点 |
| 臂 S′ β 0，150 步 | 同臂 S 去锚 | 0.750 / 0.555 | 0.711 / 0.929（64） | 1.000 | 0.753 / 0.803（64） | 0.890 | 20 | 去锚：不涨，可靠性退 |
| 臂 A SFT-1（单模板求助教材） | armS + 646 求助轨迹 SFT | 0.760 / 0.595 | 没出口 0.741 / 0.970 | — | 开出口 **0.966 / 1.000**；没出口 0.729 / 0.810；换模板 **0.036** | — | 0（开）/ 19（没） | 冷启动一次装到位；做法绑死模板 |
| 臂 A SFT-2（10 模板混训） | armS + 同教材换随机提示词 | — | 没出口 0.733 / 0.930 | — | 开出口 0.921 / 1.000；留出 0.919；没出口 0.719 | — | 0（开） | 鲁棒 +0.88，代价 −0.045 |
| 臂 A RL 后 150 步 | sft_ask2 + RL v3、−0.05/次、上限 3、80/20 专家、随机模板、锚 0.02 | 0.780 / 0.595 | 开出口 **0.951 / 1.000**；没出口 0.683 / 0.910 | — | 开出口 **0.979 / 1.000**；留出 **0.980**；弱专家 0.764 | — | 0（开） | 剪失控 +0.06；没出口 −0.05 = 工具依赖 |
```
口径   GSM：贪心，test[0:200]，crosstask_eval（R1 = 训练模板；中性 = Question:/Answer:）；CD：温度 1，池外 [95000:95100]，×8（@64 = ×64）
       † Instruct 系 chat 模板 + "#### 数字"，holdout TEST[200:500]；‡ 池内 [0:100]；* 训练中 eval 不是 crosstask
       臂 A：「开出口」= 评测时带 DeepSeek 求助（--ask），分数含专家的功劳，跟其他行不可直接比；「没出口」= 只有计算器（--tool），才是同口径。@64 没跑
血统树   Base → mix-275 → mix-600 → mix4-300 → ┬ 臂 0 / 2 / 1 / 3 / 3′（无工具）
                                              └ 臂 T SFT（工具）→ 臂 T RL ｜ 臂 S（v3 + 筛池 + 锚）→ 臂 S′ ｜ 臂 S → 臂 A SFT-1 / SFT-2 → 臂 A RL
四级台阶（CD-4 @1，同口径没出口）  0.02 Base → 0.39 三代 RL → 0.51 老师 SFT → 0.71 工具 SFT → 0.75 RL v3 有锚 → 0.72 求助教材（自己解题不变）；开出口 0.98 全是专家的
                                   每级的来源：预训练零件 / RL 挪概率 / 老师撒种 / 工具外包算术 / 秤 + 锚 + 筛池 / 专家外包搜索
```

### ★★★ 杠杆怎么排：按瓶颈不按涨幅（用户提出「直接 RL > 装工具 > 教材 > RL」的排序，2026-09-10）
```
按历史涨幅排  三代 RL +0.37 > 专家 +0.23 > 工具 +0.20 > 老师 SFT +0.11 > RL 有锚 +0.04 —— 数字是对的，当一般排序不对，三个理由：
  ① 先动手的杠杆拿最大的数：RL 那 +0.37 从 0.02 起步；工具 SFT 从 0.39 一步到 0.71（+0.32），15 分钟顶三代 RL 几十 GPU 小时。
     同一个 RL：mix4 上不撒种 +0.08（臂 0），装工具后有锚 +0.04（臂 S）。RL 的边际收益随支撑集变厚一路缩水，只在起点大
  ② 工具和专家里含着教材：p(调用) 起点都是 0，没有 202 条工具轨迹 / 646 条求助轨迹，装了也不会用。「装工具 > 装教材」不成立，每个工具都先付教材那份
  ③ RL 做的事有一半不在 pass@1 里：唯一把温度 1 的可靠性推上去而不动天花板的杠杆、唯一在价格下学「什么时候问」的杠杆、剪循环的杠杆。
     臂 A：模板混训打掉 0.045，是 RL 补回来还多涨；没有 RL，鲁棒的 SFT-2 比不鲁棒的 SFT-1 差
按瓶颈排（用哪个杠杆，看卡在哪）
  需要的路 p = 0                    → 撒种：老师 SFT / 蒸馏          证据：嵌套搜索、求助都是教材装的
  路在但某个原语做不准              → 工具                            证据：算术 0.45 → 计算器 +0.20
  路在但超出模型的搜索能力          → 专家或更大的模型                证据：第 2 层 +0.23
  路在、概率小、温度 1 下不稳       → RL，锚住                        证据：三代 RL +0.37、臂 S +0.04、臂 A +0.06
顺序比排名要紧   先撒种、先给工具，RL 最后上且要锚。「不 SFT 直接 RL」只在 Base 本来就有那条路（p 小但非零）时成立：反思、枚举在，求助、嵌套搜索不在
按每一分的成本   工具 ≈ 专家 > 老师 SFT > RL。RL 最贵，但有些东西只有它能做
```

### 教学：为什么第一段提示词失败、工业界怎么让海量用户稳定进入「前半篇」（2026-09-10）
```
现象    hf_armA 零提示词对话：短提示词（没有结尾 <think>、没有标签说明）→ 一句断言「31 = 2 * 3 * 5 + 7」（= 37，错），0 调用 0 求助；
        加上「Reason in <think></think>, answer in <answer></answer>.」和结尾的 <think> → 整套程序：8 行、乘积表、配数、搜完就问、验证、交卷，对
本质    前半篇像不像训练里的文章。所有教材和 RL 轨迹，模型写的部分都从 <think> 后面开始 —— 它是「解题记录」这类文章的第一个字，分量最重的开关；
        前半篇以「</reply>.」收尾的文章训练里一篇也没有，只能按预训练直觉续 → 一句断言。留出 12 号不提工具但有 <think> → 0.958；SFT-1 有 <think> 但只认 R1 整段 → 0.036
工业界六层  ① chat template：用户的话永远被包进固定格式（system / user / assistant 特殊 token），服务端加，用户看不见 → 模型眼里每次都是同一种文章，用户的话只是一个格子
            ② 助手回合的开头：assistant 标签后面训练时永远接助手回答，思考模式在这里接推理 → 我们的 <think>，只是我们让用户自己贴
            ③ 指令微调几百万条，题型 / 措辞 / 语言全覆盖 → 「格子里填什么都按助手方式续」成平台不是尖峰（我们的 10 模板混训）
            ④ 系统提示词：产品固定前言（身份、规则、工具说明）
            ⑤ 全分布上 RLHF：在真实用户提示词分布上采样打分推概率（我们的 RL 在 10 模板上 rollout）
            ⑥ 服务端兜底：约束解码卡格式、JSON schema 校验工具调用、验证器 / 分类器、不合格重采或拒答（我们的 harness 验证、假结果罚、上限）
两句老实话  不是 100%：前沿模型换措辞仍差几个点，「提示词工程」存在就是证据；「识别出前半篇」和「答对」是两件事，外壳只保证进入助手模式
系统提示词 ≠ harness（用户问）  系统提示词是模型读到的文字，前半篇的一部分，告诉模型工具长什么样、规矩是什么；harness 是模型外面的代码，盯输出、执行工具、贴结果，
            模型看不见它。两者配合：提示词说「写 <calc> 会返回 <result>」，harness 负责真的返回。我们：工具说明那句 = 系统提示词的角色，calc_tool_vllm = harness。
            chat template 是第三样：固定的包装格式（特殊 token），也在模型读到的文字里，但由服务端加
```

### 零提示词对话 ×3：短提示词的三次试验，把每句话的作用拆开了（hf_armA，2026-09-10）
```
① 短提示词，无 <think>，贪心        一句断言「31 = 2 * 3 * 5 + 7」（= 37），0 调用 0 求助 → 没进程序
② 短提示词 + <think>，贪心          第 0 层全对 → 乘积表列 16 项重复 → 「cannot reach」那句变成 15,21,14,10,6,35 无限列表撞顶 → 进了程序，在最软的一段打转（贪心死法）
③ 短提示词 + <think> 单独一行，温度 1 × 4
   第 1 条：没进工具模式，心算，两处算错（7+2−5−3 写成 −7），编了 <calculate> 标签，「7*3+5+2 = 31」其实是 28 却标 perfect match
   第 2/3/4 条：搜完 → 问 → 专家 3*7+5*2 → 验证 → 对；但 </think> 之后无主之地：[<calc>…</calc>]、「Solution: …」、漂成别的任务
              「Please determine whether the given text is related to computer science…」；没有一条写 <answer> → 判分全 0。调用 23 / 38 / 15（乘积表重复）
每句的作用   <think>（进不进程序）｜「Do all arithmetic with it」（用不用工具，训练时 70% 出现，温度 1 下 1/4 滑回心算）｜
             「Reason in <think></think>, answer in <answer></answer>」（怎么收尾：训练里 </think> 后永远接 <answer>，前提是提示词要求了）｜
             「Verify … before answering」（不说也做，RL 后 100% 的习惯）。「完整精简」那段一句不多余
待办 H 补两条  harness 见 </answer> 就停；</think> 之后 N 个 token 内没有 <answer> 也停（堵漂走，harness 现在只拦 \nUser:）；同一算式第二次出现返回 already tried（堵打转）
温度       工业界默认 0.7~1.0 + top-p 0.9~0.95；推理模型 0.6 / 0.95，文档明写别用贪心（重复循环、分数差）。贪心只用于短答案 / 抽取 / 可复现评测。
           贪心 = 每步局部众数，不是整条最高；整条最高要 beam，开放文本一样退化。要「最好」靠采多条再挑（验证器 / 投票 / 奖励模型 = pass@k 变现）
```

### 教学：三层各管一段 —— 外壳、SFT、RL（用户复述，2026-09-10）
```
工业界在温度 1 下「识别要做什么 + 答对」是三层叠出来的，不是 RL 一家的功劳：
  外壳（chat template + 系统提示词）  推理时加进模型读的上下文，不动权重。让每次输入都是同一种文章，模型知道该干什么。我们的对应物：<think> 那行 + 标签说明句
  SFT（指令微调 / 教材）              训练时改权重。往权重里装做法和零件，并把做法焊到多种说法上。我们的对应物：求助教材 + 10 模板混训（0.036 → 0.919，没用 RL）
  RL                                  训练时改权重。温度 1 采样、按分数挪概率，让「做对」在采样分布里占大头。训练和服务同一个温度，所以稳。我们的：硬题 0.750 → 0.946
  之外                                最后几分靠推理时采多条挑（验证器 / 投票）、工具、更大的模型；没有哪家 pass@1 是 1.0，换措辞仍差几个点
用户复述 ✓  外壳 = 输入模型的上下文；SFT = 训练模型用的；RL = 让温度 1 下答案稳定（做对的那部分在分布里概率很高）。
            补一句：SFT 和 RL 都是训练、都改权重，差别在信号 —— SFT 模仿给定的答案，RL 按分数在自己采出来的答案里挑
```

### 探针：5 个数（训练只有 3 / 4 个数），hf_armA，温度 1，提示词只有题 + 求助说明 + <think>（2026-09-10）
```
[4,17,20,21,22]→41   扫符号形态 → 乘积表 7 项 → cannot reach 剪枝 → 问 → 专家 22+20−(21−17)/4 → 验证 → <answer>   745 token，17 次调用，1 次求助 ✓
[2,3,11,20,26]→74    同上，36 次调用、1382 token 后问 → 专家 26+20*2+11−3 → 验证 → <answer> ✓
读法   ① 程序按结构伸长：没见过 5 个数照样扫、列、剪、问、验，SFT 装的是程序不是 4 个数的背法。露馅一处：「None of the 8 sign patterns」模板句的 8 没跟着变（应 16）
       ② 第二道是第 1 层的解（一个乘积 + 三数加减），5 个数第 1 层 10 块 × 8 行，预算先到就问了；老师程序同样死于 40 行上限。「搜不完就问」花 0.05 省 80 行
       ③ 两条都写了 <answer>（提示词没要求）：温度 1 下收尾是个分布，<answer> 占大头，偶尔滑出去（上一轮的 <done> / Solution:）
```

### 探针：6 个数，hf_armA，温度 1，提示词只有题 + 求助说明 + <think>（2026-09-10）
```
[3,19,21,22,24,27]→100   符号形态写 12 行（应 32）就说「8 种都不行」→ 乘积表 10 项（21×22 重复）→ 剪枝 → 配 10 行 → 问 → 专家 (27−22)×(19+(24−21)/3) → 验证 ✓
                         → 收尾编了 <results>。1344 token（顶 1400），用户说第二次才对，第一次八成撞顶
[7,16,21,22,25,28]→129   14 行就说「16 种都不行」→ 乘积表 8 项 → 没配数直接问 → 专家 28×(22−16)−(25+21−7) 对 → ★ 没验证，</reply> 后直接 </think> → 编了 <gobag> 抄进去
读法   ① 4 个数之外自己的搜索是仪式：形状在（扫、列、剪、问），穷举的实质没了；结果对是专家的可靠性，不是它的
       ② 第二道漏验证 = 弱专家探针里那 4%：专家错了就照抄。这次对是专家 + 运气
       ③ 收尾标签抽签：<answer> / <done> / <results> / <gobag>，不写标签句就随机编，程序读不出来 → 产品里那句必须在（或 harness 强制）
       ④ 分界：3~4 个数 = 自己能搜；5~6 个数 = 流程 + 专家；再往上预算先到（1400），要么加预算要么第 0 层直接问
```

### 探针：10 个数，hf_armA，温度 1（2026-09-11）—— 「问」的触发点在流程末尾，流程一散就没有问
```
[7,9,16,20,25,29,30,31,34,37]→216   三行加减，十个数只用了七个（25、30、31 第一行起就丢），全 too low，然后停：没到「None of the … patterns」、没到乘积表、没到问
读法   ① 问不是「觉得难就问」，是「按流程搜完没中才问」。教材把 <ask> 焊在两层搜完之后；4~6 个数流程虽成仪式但形状完整能走到问，10 个数连形状都保不住
          （一行记十个数、算哪些用过了，超出这个格式下的工作记忆）→ 既不搜也不问，死在半路
       ② 这是设计的代价不是 bug：结构位置换来易题不乱问、决策点无歧义；代价是题目大到程序走不动时没有出口
补法   训练侧：教材加「数超过 4 个就直接问」的轨迹，给「问」第二个触发点（看题面触发，「知道自己不行」的最简版）
       harness 侧（待办 H 三件套）：轨迹结束没 <answer> → harness 替它问一次专家、贴回、让它验证交卷；见 </answer> 就停；重复算式返回 already tried
待验   贴尾巴（token 数、是否 EOS）；同一段 --temp 1 -s 4，预测 0~1 条走到问
```

### 臂 A 收工：用户自评 + 补充（2026-09-11）
```
用户自评  ① 最后这个模型不错，至少能解决问题了，虽然是用别的模型
          ② 就算用别的模型，也用上了 agent 的调用功能、和环境交互了；把 agent 的 SFT、harness、上下文、RL（稳定有效地调用）都做了一遍
          ③ 相当于教会了 agent 在 Countdown 这种题上：试一下，不行就用 DeepSeek 回答，像教一个人
补充      ① 对。值钱的不是 0.979 这个数，是小模型学会了「什么时候把活交出去、交回来先核」；「用别的模型」正是产品的形态（级联的本地层）
          ② 对。构成 agent 的那一点是：它写的东西改变了环境的回应，它又对回应采取行动（验证、再问）。四层都碰了：上下文 = 外壳、SFT = 教材、harness = 环境执行器、
             RL = 在价格下把调用走稳。还多做了两样：模板混训（不绑措辞）和弱专家探针（验证真的在起作用）
          ③ 精确版：先按固定流程自己搜两层 → 搜完没中才问 → 问回来先用计算器核 → 错了带着错式子再问，最多三次 → 易题不问（6%）
             边界也要一起记：问的触发点焊在流程末尾（10 个数不问）；验证是 95~100% 的习惯不是规则；三次都错没路；没有专家的环境里比学之前差 0.05
             「像教一个人」在 SFT（示范）和 RL（练习 + 打分）上成立，差别是它只学到教材和提示词分布覆盖的那片
```

### 教学：什么构成 agent、本质是什么；和教人的四处不同（2026-09-11）
```
构成    普通 LM 一次性：给上下文吐文字，文字对世界没作用。agent 多一个闭环，两条同时满足：① 写的东西被执行（动作）② 结果回到上下文改变后面写什么（观测）
        缺 ① 是写作（思维链再长也是）；缺 ② 是一次性调用不是交互
本质    策略嵌在和环境的闭环里，动作的选择考虑它会带来什么观测和分数。模型 = 策略，harness = 循环，工具 / API / 沙盒 = 环境。
        状态 = 上下文、动作 = 写出的一段、观测 = 贴回的结果、奖励 = 判分。agentic RL 与普通 RL 只差轨迹里夹着环境回合（不进 loss）→ 换任务换环境，循环和接法不变
和教人  「先示范再做题」流程同形（冷启动 SFT + RL 通用的原因），四处不同，都量到过：
        ① 覆盖外不推广（10 个数死在半路；分支没样本就没路）② 练习的分是程序给的 0/1（秤定它学成什么样）③ 考试时不再学（权重冻住，只在上下文里学，下次全忘）
        ④ 示范要几百上千条、练习几千条轨迹，因为是挪分布不是理解规则
```

## ★ 现状与下一步（2026-09-11，臂 A 收工时）
```
做了什么（从 Base 到臂 A，21 个模型，17 条定论）
  基础层   Countdown 环境 + 判分器 v1/v2/v3；GSM 判分；评测切片规范（池外 [95000:95100]，×8 / ×64）；4 卡多进程；vLLM 评测 + 同进程 RL rollout（26× / 4×）
  RL 线    GSM-RL → CD-RL（捷径）→ 混训（堵捷径、顿悟形态）→ mix4 → 五臂（撒种 vs 自蒸馏 vs 只 RL vs 判分 v2）→ 臂 T（工具）→ 臂 S（v3 + 筛池 + 锚）→ 臂 S′（去锚）
  agent 线 臂 A：harness 加求助 → 探测 p=0 → 教材（含弱专家错答）→ SFT-1（绑模板 0.036）→ SFT-2（10 模板混训 0.919）→ RL（0.979，留出 0.980，CD-3 开出口 0.951）
  方法论   先写预测再跑、错了记账、池外切片、量分布不看贪心、每条结论配对照臂、按瓶颈选杠杆
  基础设施 harness（calc + ask + 重试 + 无开标签兜底）、老师 v1/v2、教材生成器、模板库、筛池、训练器（--ask）、评测 + analyze_ask、原样对话、DeepSeek 封装、盘和 NCCL 的保险
接下来能做什么（按我的排序）
  ① 价格扫描 --ask-pen 0.3（150 步，3 h，一条命令）：易题 6% 压不压得下、硬题还问不问 → 「转交的价格边界」实测值，臂 A 唯一没做的主实验
  ② 待办 H 三件套（harness，一天）：见 </answer> 停；结束没 <answer> 自动问一次；重复算式 already tried。产品级保险，也治 10 个数不问、贪心打转
  ③ 补教材两条分支 + 重训（半天）：「三次都错 → 自己接着搜 / 答最近的」；「数超过 4 个 → 直接问」；顺便弃答奖励（错答 < 弃答 < 对）
  ④ 编程题环境（一周量级）：题库带测试、<run> 工具、判分 = 测试通过率、老师让 DeepSeek 写轨迹；训练器 / harness 复用。先做零样本探针
  ⑤ 待办 M 尺寸阶梯（要 LoRA 训练器）：同一管线 Qwen3 4B / 8B，本地承接率随尺寸怎么涨 —— 产品问题「本地层多大才够厚」
  ⑥ 待办 T2 老师 v2、E 猜数字、V vLLM 贡献、L 长程、F 写总结
建议    ①② 便宜先做；③ 看 ① 的结果再定；④ 是下一条主线
```

## ★ agent + RL 路线（用户定方向，2026-09-11）
```
四步，每步在现有基建上加一样东西、回答一个问题：
  ① E 猜数字（一两天）  环境藏 1~100 的数，<guess>50</guess> → <obs>higher</obs>，猜中 1 分、每轮扣一点。多的正是 agent 的核心难点：观测多轮、奖励要归到早几轮的决定
                         问：稀疏奖励下能不能长出二分搜索（平均轮数 → log₂100 ≈ 6.6）。先探测 Base 会不会猜 / 会不会用观测；p = 0 就写十几条二分示范冷启动
  ② 编程题（一周）      题库带单元测试，<run> 跑代码回测试结果，写 → 跑 → 看报错 → 改，不行问专家，专家的代码也跑测试。奖励 = 通过率，教材让 DeepSeek 写。
                         补上「沙盒」，也是产品方向。先做零样本探针
  ③ 读三份 recipe 补训练器   ToRL / ReTool（与计算器实验同形）、verl 多轮工具 recipe：异步 rollout、多轮 mask、过程奖励、课程调度，逐项比 grpo_tool_mp_vllm 差什么
  ④ 文字环境长程（ALFWorld / τ-bench 级）  几十步一条轨迹，课程从短到长，看 1.5B 能拉多长、哪步先崩 —— 「几小时 agent」的入口
穿插   价格扫描 --ask-pen 0.3（后台）；待办 H 三件套（所有环境受益）
每步学到  E：多轮信用分配、观测 mask ｜ 编程：沙盒、过程奖励、测试当秤 ｜ recipe：工业训练器骨架 ｜ 长程：课程、记忆
E 与臂 A 的区别（用户问「我是一轮、E 是多轮？」）
   不是轮数：臂 A 在 harness 层已经是多轮（48 次 calc + 3 次 ask，几十个环境回合）。差别在三条：
   ① 环境有没有状态：计算器无状态（同一式子永远同一值），专家近似无状态；猜数字的环境有隐藏状态（秘密数），回应依赖模型之前的动作
   ② 观测要不要跨轮整合：Countdown 每行独立评一个候选，流程固定；猜数字必须记住上下界，最优策略由整段历史决定 —— 信息收集
   ③ 奖励跨轮归因：猜数字轮数越少分越高，早一步猜差了后面全要多付；Countdown 只在终局判一次
   臂 A = 多轮工具调用 + 无状态环境 + 固定搜索程序；E = 多轮交互 + 有状态、部分可观测的环境 + 要靠观测收集信息。「用户」当环境的多轮对话是再上一层
```

## E 猜数字：计划（2026-09-11 定，未动手）
```
环境    秘密数 s ∈ [1, N]，N = 100（课程可 20 → 100 → 1000）。动作 <guess>k</guess>；harness 在 </guess> 停 → 注入 <obs>higher</obs> / <obs>lower</obs> / <obs>correct</obs>
        非整数 / 越界 / 重复 → <obs>invalid</obs>，也算一轮；上限 T = 10 轮（log₂100 ≈ 6.6，二分 ≤ 7）；猜中后写 <answer>k</answer>
        模型自己写 <obs> = 假观测 → 停 + 罚 0.2（同 <result>）；<obs> 段 mask（label −100 / resp_mask 0）
奖励    猜中：1 − 0.05 × (轮数 − 1)（二分 6~7 轮 ≈ 0.7；线性搜索超 10 轮 = 0）；没猜中 / 超轮 0；假观测 −0.2。v3 精神：只奖不罚阶梯
提示词  从第一天就混模板（臂 A 的教训）：8 种说法 + 2 种留出，说明「写 <guess>N</guess> 会回 higher / lower / correct」
代码    harness.py：通用「停 → 回调 → 注入」循环，工具表 {标签: 处理函数(序列状态, 文本)}，复用 calc_tool_vllm 的后端层（HF / vLLM）；calc_tool_vllm 原样不动
        guess_env.py：出题（秘密数）、观测函数、奖励、模板；tests/test_harness.py 假后端单测（正常二分、invalid、重复、假观测、轮数上限、mask）
        grpo_tool_mp_vllm.py 加任务分支 guess（make_prompt / rollout / score 三个钩子）；评测脚本 eval_guess.py + analyze_guess.py
七步    ① 环境 + harness + 单测（半天）
        ② 探测（半天）：Base 和 hf_armA 零样本各 100 个秘密数 × 8。量：写不写 <guess>、方向一致率（下一猜落在观测指的那一侧）、平均轮数、猜中率
           预测 Base：格式勉强、方向一致率 0.6~0.7、10 轮内猜中 10~20%；p > 0 → 可以直接 RL（R1-Zero 式，看二分会不会涌现）；p ≈ 0 → 冷启动
        ③ 两条臂：(a) 纯 RL（涌现臂）(b) 20 条二分示范 + 故意混几条线性 / 随机搜索的 SFT 再 RL（撒种臂，给 RL 留选择）
        ④ RL：GRPO k 16、N 100、T 10、温度 1、150 步、锚 β 0.02；池 = 随机秘密数，不用筛（每个秘密数都是新组，差异来自采样）；轨迹短（~150 token），一步 ~20 s
        ⑤ 评测：200 秘密数 × 8，留出模板：猜中率、平均轮数与分布（对 log₂N）、方向一致率、invalid / 重复率、二分得分（猜的是不是当前区间中点 ±1）；
           N = 1000 越界探针（对应 10 个数那种流程外）
        ⑥ 分析 + 定论 ⑱：观测是装饰还是程序（方向精度的翻版）；二分是涌现还是要撒种；信用归因（早几轮的猜是否变好）
        ⑦ 记录、总表
预测    涌现臂：平均轮数从 ~9 降到 7~8，猜中率 → 90%+，但 150 步内到不了严格二分；撒种臂：平均轮数 ~7，方向一致率 > 0.95。不用 DeepSeek，全程本地
```

## E 猜数字 ① 环境 + harness + 单测 完成（2026-09-11）
```
harness.py     通用「停 → 环境回一段 → 注入 → 续」循环，后端复用 calc_tool_vllm（原样不动）。环境接口：stops / fake_tags / step(state, full) → (注入文本, done)
               假观测的开标签也进停止串（否则模型一口气写到 EOS，fake 判不到 —— 单测抓出来的）
guess_env.py   GuessEnv(N=100, T=10, step_cost=0.05)：<guess>k</guess> → <obs>higher|lower|correct|invalid</obs>（非整数 / 越界 / 重复 = invalid，也算一轮）；
               轮数用完没中 → 结束补 EOS；猜中让它写 <answer>。score：answer == 秘密数 → 1 − 0.05 × (轮数 − 1)，否则 0
               behavior(txt)：方向一致率（上一观测 higher 下一猜要更大）、落在可行区间 [lo, hi]、二分得分（离中点 ≤ max(1, 5% 区间)）
               模板 10 个（8 训练 + 2 留出：啰嗦英文、另一种中文），全部以 <think> 收尾；make_problems(n, N, seed)
eval_guess.py  探测 / 评测：--engine hf|vllm，n × s，打：写了 guess / 猜中率 / 平均轮数（对 log₂N）/ 轮数分布 / 方向一致 / 区间内 / 二分得分 / invalid / 重复 / 假 / 撞顶 / 组的分布；
               --analyze raw 只重算；--show 打最短的对的 + 一条错的
tests/test_harness.py  假后端 10 项通过（二分、invalid、重复、假观测、轮数上限、mask、奖励、行为读数、预算、模板）
```

## E 猜数字 ② 探测结果（100 个秘密数 × 8，N 100，T 10，温度 1，训练模板，2026-09-11）
| | hf_base | hf_armA |
|---|---|---|
| 写了 <guess> | 0.170 | 0.443 |
| ★ 猜中率 | 0.005 | 0.045（含 2 条没猜就瞎答撞中的，修奖励后不算）|
| 环境说 correct | 0.007 | 0.069 |
| 方向一致率 | 0.742（只有 217 次）| 0.577（1659 次，略高于 0.5 的乱猜）|
| 落在可行区间 · 二分得分 | 0.656 · 0.115 | 0.443 · 0.269 |
| invalid / 重复 / 假观测 / 撞顶（每条 · 条数）| 0.13 / 0.04 / 139 / 81 | 0.17 / 0.30 / 124 / 68 |
| 组：有对有错 / 全错 | 4 / 96 | **29 / 71** |
预测对账  ✗✗ 全部高估：说 armA 写 guess ≥ 0.9 实测 0.44，猜中 0.15~0.30 实测 0.045，方向一致 0.7~0.8 实测 0.58；Base 说猜中 0.1~0.2 实测 0.005。
          错在以为「标签文章」的习惯会整体迁移；实际迁移的只有格式的一半，观测基本当装饰（0.58 ≈ 乱猜）
样例      armA 秘密数 66：31 → 71 → 45 → 53 → 62 → 66，六轮，每步都用了观测 —— 路在，只是稀（29% 的秘密数有对有错）
          armA 的错：编 <labs> 标签、猜「3 times」、自己写 <obs>；Base 的对：全是垃圾文里撞上的；Base 秘密数 69：31 → 32 → 79 → 45 → 69 也是一条真搜索
奖励修正  探测里出现「没猜就写 <answer>」撞中 1/N 得 1.05 的情况 → score 改成：环境说过 correct 才算，轮数至少按 1 算。否则 RL 会放大「不猜直接答」
决定（③） RL 起点 = hf_armA（p 0.045 > 0，29% 的组有梯度；Base 0.005 基本是零）。三条臂：
          E1 涌现：hf_armA 直接 RL，N 100          —— 二分能不能从 4.5% 的种子里长出来
          E3 换地：hf_armA 先 N 20 跑 50 步再 N 100 —— 权重不动、地换软，路先冒出来再变长（课程）
          E2 撒种：40 条二分示范（20% 粗糙）SFT 后 RL —— 对照：撒种能不能更快更稳到二分
代码（③④） grpo_tool_mp_vllm --task guess --guess-N --guess-T --guess-cost：不读 CD/GSM，每步一批新秘密数，rollout 走 harness.generate_with_env，
          统计 calls = 轮数、errs = invalid + 重复，eval 100 个固定秘密数；sft_qwen 的 RES_RE 加 <obs>；make_guess_demos.py 造示范
### 教学：换地、课程、拆子任务 —— 三个词的关系（用户问，2026-09-11）
```
换地   最宽的词：权重不动，换环境让要放大的路 p > 0。四种换法：同一任务换小实例（N 100 → 20）、换任务组合（CD 单训 → CD+GSM 混训，堵捷径）、
       换提示词格式、换奖励。混训是换地但不是课程
课程   换地的一种：一串按难度排好的环境，从短 / 易到长 / 难，每一步都让 16 条里有对有错。E3 的 N 20 → 100 是课程；筛池也是一种课程（每步把地换到「刚好能走一半」）
       长程 RL 的难度轴 = 步数：几步先训、几十步再训、几百步最后。任务一长随机走通的概率指数降，一上来全错没梯度
拆子任务  把长任务拆成能单独训的技能，各自短、各自有秤，先各训会，再在长任务里组合 —— 长任务的策略里子技能的路已经 p > 0
       E 的 N 20 → 100 不是拆子任务：同一个任务换大小，是课程。猜数字的拆法会是：子任务 A「给 lo, hi 报中点」、子任务 B「给猜和观测更新 lo, hi」，各训再合
       我们做过的拆子任务是臂 T → S → A 那串：先学会用计算器（臂 T），可靠了再学会问专家（臂 A），技能一层层叠上去
长程四件套  课程 / 过程奖励（稠密）/ 撒种（强模型或人写长轨迹）/ 拆子任务。E 只用课程，因为够短，看得清它一个的作用
```

## E 猜数字 ④⑤ E1 涌现臂：hf_armA 直接 RL 150 步（N 100，T 10，k 16，β 0.02，35 分钟，一步 5+4 s，2026-09-11）
训练读数（温度 1）  猜中 0.09 → 0.72，分 → 0.52，轮 1.8 → 6.8，无效 0.66 → 0.48，假观测 0.09 → 0.00，KL 0 → 0.025   中途评测（贪心 100）猜中 0 → 0.79
评测（eval_guess，100 × 8，温度 1）
| | 起点 hf_armA | **E1 训练模板** | E1 留出模板 | E1 N=1000（越界，10 轮刚够二分） |
|---|---|---|---|---|
| 写了 <guess> | 0.443 | 0.941 | 0.843 | 0.926 |
| ★ 猜中率 | 0.043 | **0.632** | 0.359 | 0.142 |
| 环境说 correct · 有 <answer> | 0.069 · 0.439 | 0.670 · 0.779 | 0.458 · 0.525 | 0.151 · 0.258 |
| 平均轮数（猜中的） | 4.8 | **6.21**（log₂100 = 6.6） | 6.48 | 7.88（log₂1000 = 10） |
| ★ 方向一致率 | 0.577 | **0.901** | 0.846 | 0.818 |
| 落在可行区间 · 二分得分 | 0.443 · 0.269 | 0.777 · **0.558** | 0.660 · 0.422 | 0.602 · 0.366 |
| invalid · 重复 / 局 | 0.17 · 0.30 | 0.09 · 0.34 | 0.35 · 0.57 | 0.17 · 0.58 |
| 假观测 · 撞顶（条） | 124 · 68 | 7 · 31 | 54 · 111 | 14 · 42 |
| 组 有对有错 / 全对 / 全错 | 27 / 0 / 73 | 93 / 7 / 0 | 92 / 2 / 6 | 59 / 0 / 41 |
预测对账   训练模板猜中 0.65~0.75 ≈ 0.632（略低）；方向一致 ≥ 0.9 ✓ 0.901；二分得分 0.5~0.7 ✓ 0.558；N=1000 0.1~0.3 ✓ 0.142；
           ✗✗ 留出模板「掉 0.05 以内」实测掉 0.27（0.632 → 0.359）
读法      ① 二分从 4% 的种子里长出来了：猜中的局平均 6.2 轮，贴着 log₂100；方向一致 0.58 → 0.90；轨迹里自己写「between 26 and 50」「log2(100) ~= 7 guesses」
          ② 但不是严格二分：二分得分 0.56，22% 的猜落在可行区间外，重复 0.34/局。N=1000 时尾段退化成 +1 +2 的线性爬（842~874 该猜 858 却 851 → 855 → 857 → 858）：
             大数的中点是算术，它算不准 —— 这正是计算器该接的活（子技能 A「报中点」），三个技能在一个模型里的接口
          ③ ✗ 留出模板掉 0.27，比臂 A 的「不掉」差远了：臂 A 的做法是 SFT-2 用 1592 条教材把程序焊到 10 种措辞上，RL 只是走稳；E1 的程序是纯 RL 在 8 种措辞下
             从 4% 长出来的，只有 1200 局的曝光，措辞一换就散（编 <begin mind process>、<mind>、<end>；假观测 54）。
             → 纯 RL 长出来的技能比 SFT+RL 的更绑措辞？E2（撒种臂，教材含 8 种措辞）能回答
          ④ 环境的洞：猜中之后还在猜（轮数 18、26 那两条），环境没结束；已修：found 之后再猜直接结束、无 <answer> 得 0 → RL 学「中了就停」
          ⑤ 猜中 0.632 < 环境说 correct 0.670：3.8% 中了没交对卷，RL 还在推（has_answer 0.44 → 0.78）
### 教学：「RL 只放大、不抬天花板」在 2026 年是不是共识；泛化算不算抬天花板（2026-09-11）
```
有共识   小算力 / 小模型 / 短训练：RL 主要是收窄。pass@1 涨、pass@k 不涨甚至掉（Yue 2025 用 pass@k 量的；我们臂 1 @64 1.00 → 0.95、臂 S 0.900 不动）
还在争   算力和时间拉大后天花板动不动：ProRL 说训得久 + KL 重置 + 任务杂 + 探索够（k 大、熵不塌），大 k 的 pass@k 能扩一点，尤其 Base 里 p 极小但非零的路；
         前沿实验室把 RL 算力堆到接近预训练量级就是押这一层。争的是定义：p 百万分之一被放大到 0.9 算不算新能力？数学上是放大，实用上没区别
         （E1 二分 4% → 72% 是放大；起点若是万分之一你就会叫它涌现）
看定义   泛化：「RL 泛化、SFT 记忆」（Chu 2025）—— RL 学的策略换分布还能用；E1 在 N=1000 上 14%、方向一致 0.82。但泛化 = 已有程序用到更多上下文，
         是支撑集内挪质量，不是支撑集变大。分界线 = p 是不是零：新分布上大 k 能中是泛化；任何 k 都中不了才是天花板，只有新信息能动（数据、工具、环境反馈）
一句话   RL 在支撑集里挪概率，这条不变；挪得够狠够久够广，挪出来的在实用上就是新能力；真正的零只有外面的信息能填。工具和专家就是往支撑集里塞外部信息的两根管子，
         臂 A 硬题 0 → 0.97 靠的是管子不是 RL
```

## E 猜数字 ④ E3 换地臂：N=20 50 步 → N=100 100 步（同 E1 的 150 步总算力，2026-09-11）
```
E3a（N=20，50 步，10.7 分）  训练猜中 → 0.54（温度 1），贪心 eval 0.80，轮 4.7~5.5（log₂20 = 4.3），KL 0.018
E3b 起点（N=100）           贪心 0.33 / 温度 1 0.30（E1 同一块地的起点 0.09）；轮 6.7~7.1 已贴 log₂100；无效 0.94~1.25 偏高（1~20 的习惯带过来）
                            ✗ 我说「起点比 E1 第 50 步高」：E1 第 25 步贪心已 0.71，软地 50 步不如硬地 50 步值钱
E3b 终点（100 步，19.5 分） 训练猜中 0.83、轮 6.19、无效 0.10、假观测 0、KL 0.015；★ 贪心 eval 0.950、轮 6.13、无效 0.08
对照 E1 终点（150 步全在 N=100）  训练猜中 0.72、轮 6.83、无效 0.48；贪心 eval 0.79、轮 7.29、无效 0.63
读法   ✗ 我说「终点和 E1 差不多」：课程赢得很明显，贪心 0.95 vs 0.79，无效 0.08 vs 0.63。前 50 步在软地上把「用观测缩区间」练干净（组组有梯度、局短信号密），
       换到硬地只是路变长；E1 在硬地上前几十步大半组全 0 白跑。这是课程学习的教科书结果，也是长程任务「从短到长」的依据
待做   导出 hf_E3b，三口径评测（训练模板 / 留出 / N=1000）与 E1 并排；E2 撒种臂
```

## E 猜数字 ⑤ E3 评测（100 × 8，温度 1）与 E1 并排（2026-09-11）
| | E1 训练模板 | **E3 训练模板** | E1 留出 | E3 留出 | E1 N=1000 | E3 N=1000 |
|---|---|---|---|---|---|---|
| 猜中率 | 0.632 | **0.838** | 0.359 | **0.512** | 0.142 | 0.171 |
| 方向一致率 | 0.901 | **0.959** | 0.846 | 0.925 | 0.818 | 0.896 |
| 落在可行区间 · 二分得分 | 0.777 · 0.558 | 0.886 · **0.652** | 0.660 · 0.422 | 0.819 · 0.516 | 0.602 · 0.366 | 0.678 · 0.387 |
| 平均轮数（猜中的） | 6.21 | 6.08 | 6.48 | 5.97 | 7.88 | 7.91 |
| invalid · 重复 / 局 | 0.09 · 0.34 | **0.02 · 0.16** | 0.35 · 0.57 | 0.22 · 0.25 | 0.17 · 0.58 | 0.09 · 0.65 |
| 假观测 · 撞顶（条） | 7 · 31 | 8 · 9 | 54 · 111 | 35 · 74 | 14 · 42 | 8 · 11 |
| 组 全对 / 有对有错 / 全错 | 7 / 93 / 0 | **32** / 68 / 0 | 2 / 92 / 6 | 0 / 98 / 2 | 0 / 59 / 41 | 0 / 67 / 33 |
预测对账   E3 训练模板 0.85 上下 ✓ 0.838；方向一致 ≥ 0.95 ✓ 0.959；二分得分 0.7 上下 ≈ 0.652；留出掉 0.15~0.25 ✗ 掉 0.33；N=1000 0.3 上下 ✗ 0.171
读法      ① 课程臂全面赢：三口径都高于 E1，重复和 invalid 少一半以上，32 个秘密数 8/8 全对
          ② 留出模板还是掉三分之一（0.838 → 0.512）：两条纯 RL 臂同样绑措辞，课程没治这个。留出上「中了还猜」的旧毛病又冒出来（环境已改成直接结束，RL 会压）
          ③ 尾段仍不是二分：秘密数 98 那条 50 → 75 → 88 → 92 → 94 → 96 之后拿到 higher 反而往下 95 → 93 → 91 → 90 —— 区间贴到上界时方向失灵；
             N=1000 的 865：500 higher 之后猜 250、125、375 反向三次才纠正；912 那条尾段 914 → 905 → 910 → 915 一步一步挪。大数中点算不准 + 边界处理差
          ④ N=1000 只到 0.17：10 轮刚够严格二分，任何一次浪费就出局；这里是算术和边界的洞，不是策略的洞
## E 猜数字 ⑤′ 留出模板按句拆开：掉的不是「换句子」，是某一句（2026-09-11）
```
E3 留出   模板 8 啰嗦英文 0.670（方向一致 0.933、二分 0.583、假观测 8、撞顶 7、中了还猜 31）
          模板 9 短中文   0.355（方向一致 0.916、二分 0.444、假观测 27、撞顶 67、★ 中了还猜 64）
E1 留出   模板 8 0.448 ｜ 模板 9 0.270
E3 训练 8 种之间  0.688（6 号规格式）~ 0.923（0 号啰嗦英文、3 号中文）：训练模板之间本身就差 0.24
预测对账  ✗✗ 反了：我说 9 号中文接近训练水平、8 号掉到 0.3。按表面相似（中文 ↔ 中文）猜，第二次错在同一处（臂 A 留出那次也是）
读法      ① 「换措辞就掉」在 E 上是一句话的事：8 号句子最长最不像，反而 0.67；9 号最短，0.36，而且死法集中：中了还猜 64 条 + 撞顶 67 条 = 一多半的失败。
             9 号和训练里的中文 3 号差在：没有「思考写在 <think></think> 里」那句、「找到后」而不是「猜中后」、更短。模型在 9 号下不知道「中了该停」
          ② 训练 8 种之间也差 0.24：6 号规格式最差。RL 后的策略对措辞的敏感是连续的，留出只是把这条曲线延长了一点
          ③ 消融才能定：拿 9 号做三个变体各评一次（加 <think> 那句 / 找到后→猜中后 / 两者都加），看哪一句把它救回来。eval_guess 加了 --custom-prompt
```

## E 猜数字 ⑤″ 9 号模板的措辞消融（E3，100 × 8，2026-09-11）
```
原句                                     0.355   假观测 27   撞顶 67
A 加「思考写在 <think> </think> 里」       0.590   假观测 40   撞顶 40
B 只把「找到后」改「猜中后」               0.315   假观测 73   撞顶 181（样例里去写 Python 程序）
C 两处都改                               0.619   假观测 28   撞顶 56
定论   起作用的是那句提到 <think></think> 的话；「找到 / 猜中」无关。对没见过的句子，结尾的 <think> 不够，要有一句明说「思考写在 <think> 里」。
       训练 8 种里 2、5、7 号也没这句但被 RL 练过（0.81~0.83）：见过的句子靠整句触发，没见过的靠那一句标签说明触发
       剩下的缺口 0.62 vs 0.84 = 一般的措辞敏感（训练 8 种之间就差 0.24），看 E2 的 SFT 撒种治不治
方法   这次没预测，前两次按「表面像不像」猜都反了；消融是唯一靠得住的办法：一次改一处、各评一次
```

### 教学：SFT / RL 各能学什么（四格）；「RL 创不创造新东西」分两层；长程 p ≈ 0 怎么办（2026-09-11）
```
SFT 学目标文本里的一切（内容、格式、写成步骤的程序、措辞）；RL 学采样上下文里和得分相关的一切（程序、表面线索、格式），p = 0 或秤分不出就学不到，学不到新内容
四格   RL 能 SFT 不能：采样下做稳（不靠示范）、价格下做选择、找没人写得出的路、压掉行为（SFT 没有「别这样」）
       SFT 能 RL 不能：放 p = 0 的路、放内容（知识 / 格式定义 / 工具用法）、把程序焊到意思上（信号密、便宜、不绑措辞）
       都能：格式、p > 0 的程序、措辞鲁棒（RL 弱得多）        都不能：权重和环境都没有的（大数中点、没有的知识、1.5B 上限）、100% 保证（要 harness 规则）、秤坏了一起学坏
       一句话：SFT 放东西进去，RL 在已有的里选；放不进也选不出的靠工具、数据、外壳
RL 创不创造  信息层不创造：一步进权重的只有「这 16 条哪条分高」，token 全是自己采的，没有一个字来自外面。行为层创造：零件各自有、拼在一起 p 4%，放大到 72% =
             以前实践上没有的能力（雕塑 / 进化）。「找没人写得出的路」= 行为层的创造：拼法新，零件旧。上限由「采得到」定：p 越小要的算力越多，采不到就是零，只能撒种
「SFT 学内容 RL 学程序」那句  说的是任务规则换了谁扛得住（任务分布轴）；我们量的是题面措辞轴：SFT 每 token 有目标、8 种措辞几百次配对 → 程序焊到意思上；
             RL 一条轨迹一个数、只在自己采的轨迹和训练那 8 句下有信号 → 焊在句子上。哪个泛化好看换的是哪个轴、信号密不密
```

### 长程任务 p ≈ 0 怎么办（用户问，2026-09-11）
```
为什么是零   整条轨迹的 p = 每一步 p 的连乘。每步 0.9，30 步就是 0.04，100 步 0.00003；每步 0.5，10 步就是 0.001。零件都在，连乘把它乘没了
八个杠杆（前四个是「长程四件套」）
  ① 课程          先短后长，让连乘的项少，p > 0 再加长                              E3：N 20 → 100
  ② 过程奖励      给中间步骤打分，把连乘拆成一段段各自可学的和                          臂 T 的 v2 试过，罚探索翻车；要的是「对的中间步」加分不是罚
  ③ 撒种          强模型 / 人写长轨迹 SFT，整条路直接 p > 0 —— 工业界 agent 的标配（agentic SFT → RL）   E2：200 条示范 0.976
  ④ 拆子任务      各段单独训到可靠，连乘变成高数的连乘                                  臂 T → S → A 的技能叠加
  ⑤ 环境反馈进循环  每步有观测（测试结果、报错、too high）就能中途纠错，轨迹自己修，联合 p 抬高     harness 的全部意义
  ⑥ 推理时搜索    采几百条 / 树搜索 / MCTS 找到那几条成功的，再拿它们训（专家迭代、STaR）—— 把推理算力换成训练数据
  ⑦ 事后重标       失败的轨迹按它实际到达的目标重新标成功（hindsight）
  ⑧ 加 k、加步数   硬堆采样，前沿实验室的做法，我们 4 张卡不够
长程独有的两件  上下文装不下 → 摘要 / 笔记 / 外部记忆；信用归因跨几百步 → 过程奖励或按段打分
为什么另算    八个杠杆治的是「p ≈ 0 没梯度」= 信号有没有；这两件在 p 正常时照样存在：
              ① 上下文：几百步的轨迹超出窗口或注意力看不清 → 策略根本看不到自己的历史，这是表示层的问题，梯度再多也没用；解法在 harness（摘要、笔记、外部记忆）
              ② 归因：就算有成功的轨迹，终局一个标量摊到几百步几千 token 上，每个 token 的优势稀得像噪声，方差随步数炸 → 信号有但质量不够；
                 解法是按段打分、过程奖励、或者估值函数（critic）—— GRPO 没有 critic，长程上这是它的短板
「硬堆 k 和步数」具体是什么  期望成功数 = 题数 × k × p，p 小就把前两项堆大：
              k 从 16 堆到 64~256；题库几十万道可验证题；步数几千（R1、ProRL 2000+），2026 前沿的 RL 算力已接近预训练量级；
              配套：动态采样丢掉全同分的组（DAPO）、异步 rollout + 过期修正（importance sampling、PPO 裁剪）、熵控制防塌缩、KL 定期重置（ProRL）、
              奖励模型 + 验证器混用；基础设施：成千上万张卡，生成与训练分离或共卡（我们的 vLLM 共卡是它的微缩版）。代价仍随步数指数涨，所以他们同时用另外七个
```

## ★★★ E 猜数字 ⑥⑦：三臂总表 + 定论 ⑱（2026-09-11）
| 100 × 8，温度 1 | 起点 hf_armA | E1 纯 RL 150 步 | E3 课程 20→100，150 步 | E2 只 SFT（200 条示范） | **E2 SFT + RL 150 步** |
|---|---|---|---|---|---|
| 训练模板猜中 | 0.043 | 0.632 | 0.838 | 0.976 | **0.996** |
| 留出模板猜中（8 号 / 9 号） | — | 0.359（0.448 / 0.270） | 0.512（0.670 / 0.355） | 0.873（0.968 / 0.777） | **0.926**（0.993 / 0.860） |
| N=1000 猜中（10 轮刚够二分） | — | 0.142 | 0.171 | 0.621（方向 0.958 · 区间 0.923 · 二分 0.717 · 重复 0.14） | **0.713**（0.979 · 0.961 · 0.786 · 0.09） |
| 方向一致率 · 落在区间 · 二分得分 | 0.58 · 0.44 · 0.27 | 0.90 · 0.78 · 0.56 | 0.96 · 0.89 · 0.65 | 0.99 · 0.99 · 0.90 | **1.00 · 1.00 · 0.92** |
| 平均轮数（猜中的）· invalid · 重复 | 4.8 · 0.17 · 0.30 | 6.2 · 0.09 · 0.34 | 6.1 · 0.02 · 0.16 | 5.5 · 0.01 · 0.04 | **5.5 · 0.00 · 0.01** |
| 8/8 全对的秘密数 | 0 | 7 | 32 | 82 | **97** |
| 算力 | — | 35 分 | 30 分 | 2 分 | 2 + 22 分（RL 第 50 步贪心已 1.000，之后 \|A\| 0.02） |
预测对账（E2 RL 后）  训练 0.99 ✓ 0.996；留出 0.92 ✓ 0.926；✗✗ N=1000 说 0.3，实测 0.713 —— 第三次低估「写下状态」这件事的分量
读法      ① N=1000 是最大的信息：RL 臂 0.17，撒种臂 0.71。差在示范教的那句「The number is between lo and hi」—— 把区间写成文字，中点和边界就成了对着两个数算，
             纯 RL 臂大多不写区间，尾段退化成线性爬。显式状态 = 跨规模泛化的关键，它来自教材不来自 RL
          ② 撒种后 RL 的份额：训练 +0.02、留出 +0.05、N=1000 未知；干的是压 invalid（0.01 → 0）、重复、假观测、中了还猜。和臂 A 同构：形是教材给的，RL 走稳
          ③ 剩下的错全是算术：秘密数 66 那条「between 59 and 66, guess the middle: 58」中点算错；865 那条「smaller than 868」之后区间仍写 864~872，边界没更新。
             → 子技能 A「报中点」交给计算器，三个技能同一个模型
          ④ 留出 9 号还差 0.14：缺「思考写在 <think> 里」那句（消融定的）；撒种把它从 0.36 / 0.28 抬到 0.86，纯 RL 的绑措辞被治了大半
★★ 定论 ⑱  多轮 agent RL（E 猜数字，2026-09-11）
   1. 多轮交互的基建通了：通用 harness（停 → 环境 → 注入 → 续）+ 环境接口（stops / fake_tags / step / score），换任务只换环境文件；训练器加任务分支；一步 5~8 s
   2. 观测从装饰变成了程序：方向一致 0.58 → 0.999，落在区间 0.44 → 0.998。模型没有记忆，历史全在上下文里；它自己写「between lo and hi」把历史压成状态，这是自开关
   3. 二分能从 4% 的种子里纯 RL 长出来（E1：0.63，轮数贴 log₂N，轨迹自己写 log2(100) ≈ 7）—— 行为层的创造，零件是预训练的；但绑措辞（留出 −0.33）、尾段不严格
   4. 课程（换地）全面优于硬地：同算力 0.84 vs 0.63、无效少六倍。软地先把「用观测」练稳，硬地只是路变长
   5. 老师程序撒种 200 条 2 分钟 = 0.976，超过两条 RL 臂；留出只掉 0.10；RL 后 0.996 / 0.926 / N=1000 0.713。三臂顺序 = 长程四件套的前三行
      撒种后 RL 的份额：训练分布 +0.02（饱和），留出 +0.05，N=1000 +0.09（0.621 → 0.713）—— 「走稳」压掉的边界算错和重复，在 10 轮刚够的地方每次都致命，收益在难处才显
   6. 绑措辞是纯 RL 的病：同样只见 8 种措辞，RL 臂留出 −0.33，SFT 臂 −0.10。消融：起作用的是「思考写在 <think> 里」那一句，「找到 / 猜中」无关；
      见过的句子靠整句触发，没见过的靠标签说明触发
   7. 跨轮归因在 10 轮内 GRPO 够用：轮数被 0.05 的价格压到 5.5（示范平均 5.1）；几百步要 critic 或按段打分，另算
   8. 剩下的洞是算术（中点、边界），不是策略；对应臂 A 的「三次都错无路」，都是下一层要接工具的地方
### 教学：习惯的三条路（教 / 引 / 找）；「找」比「教」强在哪；思维链 = 把状态写进上下文；模型是角色的叠加（2026-09-11）
```
显式状态  是教材设计不是算法：穷举器每轮多写「between lo and hi」，训练器不动。进模型三条路：SFT 教 / 提示词引（外壳）/ RL 找（E1 摸到一半：好轨迹有、不是每轮都写）
「找」比「教」强的四样   ① 不需要先知道答案（R1 的推理风格、AlphaZero、E1 自己写 log2(100)≈7）② 找出来的是它自己的话，采样下更稳（教材是别人的字，模仿得出执行不了 —— 老师 v1 的 closest first）
                        ③ 优化真目标不是「像示范」：丢没用的、留有用的、找更短的路、在价格下选择（E2 RL 把 invalid 压到 0，示范里没有「别重复猜」）④ 能超过老师；有「别这样」
分工      知道该怎么做就教（便宜、稳、不绑措辞）；不知道 / 要超过老师 / 要在代价下选择就找。两个都用是常态
思维链    = 把中间状态写进上下文，后面的 token 读得到。跨 token 没有别的可靠记忆（KV 里的隐状态浅、多步算术撑不住），写成字是唯一可靠通道。「between 26 and 50」= 猜数字的思维链
角色叠加  预训练模型 = 无数角色的叠加（simulator 视角）。对齐做三件事：选「助手」当默认角色焊稳（character training）、允许系统提示词在范围内切角色、堵死某些角色（安全训练）。
          边界：切角色切的是行为口气不是能力知识；对齐既是「能变角色」也是「限制角色」，一件事两面
```

### SFT vs RL：谁更会抬 pass@1，各做什么更好；RL 比 SFT 强在哪（2026-09-11 总结）
```
pass@1 谁抬得多   看起点：路没有（p = 0）或很稀 → SFT 抬得多得多（求助 0 → 0.966；猜数字 0.043 → 0.976，2 分钟）；
                  路已经在且不稀 → RL 抬得多（Base → 三代 RL 0.02 → 0.39；SFT 后 RL 补可靠性 +0.04~0.06）。SFT 抬的是「会不会」，RL 抬的是「稳不稳」
各做什么更好      SFT：装 p = 0 的路、装内容 / 格式 / 工具用法、把程序焊到意思上（措辞鲁棒）、显式状态这种写法习惯、便宜（分钟级）
                  RL：温度 1 下走稳、价格下选择、压掉坏行为、剪循环、找没人写得出的路、超过老师
RL 比 SFT 强的地方（总结）
   ① 不需要答案，只要秤：环境能判对错就能训，示范写不出的任务也能做（R1-Zero、AlphaZero、E1 的二分）
   ② 学的是自己的话：on-policy，采样下可靠；SFT 的模仿可能「会写不会做」
   ③ 优化真目标：会丢示范里没用的、找更短更便宜的路、在代价和收益之间取舍（何时问、几轮划算）
   ④ 有负信号：能压掉行为（假观测 15% → 0、循环、什么都问）；SFT 只有正例
   ⑤ 上限不受老师限制：能超过示范；SFT 的上限 = 示范质量
   ⑥ 结果导向的泛化：任务规则换了扛得住（Chu 2025）—— 但措辞换了不如 SFT（我们量的）
   代价   贵（分钟 vs 小时）、要 p > 0（否则撒种 / 换地）、绑措辞、秤有洞就学坏、没 critic 时长程归因弱
一句话   SFT 决定模型会什么，RL 决定它做得稳不稳、选得对不对；先教再找，两个都用
```

### 教学：老师 vs 天花板；「先看课本再做题」的比喻及其边界（用户提出，2026-09-11）
```
两个上限   老师 = 示范的水平，支撑集里的一个点；RL 优化得分不优化「像老师」，能从这个点往上爬（删判断句、压行数、学「别重复猜」、找更短的路：E2 invalid → 0，臂 S 0.71 → 0.75）
           天花板 = 支撑集的边界（大 k 的 pass@k）；RL 只在里面挪概率，边界不动（CD-4 @64 0.900 前后不变）。老师本人也在边界里，是 SFT 放进去的
           一句话：RL 能爬过老师，爬不出支撑集；推边界只有三根管子：预训练、SFT 撒种、工具 / 专家（CD-4 0.900 → 0.98 靠专家）
比喻 ✓     训模型像教人学线性代数：直接做题（RL）不会或很慢，先看课本（SFT）再做题（RL）。冷启动 SFT → RL 就是这个顺序（R1、工业界 agent、臂 A、E2）
比喻的细节（这条线量出来的）
   例外   生活里已经会一点的，直接做题也行但慢而且野：R1-Zero、E1（4% → 0.63，35 分钟，绑措辞、编怪标签）；看课本 2 分钟 0.976 再做题 0.996
   课本得对   有他做不到的步骤会被做题删掉（老师 v1 的 closest first）；只有一种说法他学的就是说法（SFT-1 0.036）
   答案得批对   秤错了做题学歪（v1「任何错 > 不答」教出会猜的学生）
   题从易到难   N 20 → 100，同题量 0.84 vs 0.63
   做题干课本干不了的   做熟做快、知道哪种方法省事、改掉坏习惯（负信号）、偶尔找到课本没有的巧法（RL > 老师）
断在两处   考完就不学了（权重冻住，草稿纸上的下次全忘）；人换题面一眼认出，他得课本里写十种题面
完整版    课本 → 做题 → 把做对的整理成笔记再学（拒绝采样 SFT）→ 再做题；SFT 与 RL 交替（R1 四段）
```

### E 收工：这个实验学到了什么；在 agent + RL 上的进展与差距（2026-09-11）
```
学到（机制）  ① 观测能从装饰变程序，模型跨 token 没有上下文之外的记忆，靠自己写「between lo and hi」把历史压成状态（方向一致 0.58 → 1.00）
              ② 策略可纯 RL 涌现但野（二分 4% → 0.63；绑措辞、尾段算不准、编怪标签）
              ③ 撒种便宜两个数量级且泛化更好（2 分钟 0.976；留出只掉 0.10；N=1000 0.62 vs RL 臂 0.17）—— 差在示范里的显式状态
              ④ 课程赢硬地（同算力 +0.2，无效少六倍）
              ⑤ RL 在撒种后的份额在难处才看得见：它压的是每步出错率，出错率按轮连乘，路越长同样的可靠性差距放大越多（训练 +0.02、留出 +0.05、N=1000 +0.09）
              ⑥ 绑措辞是纯 RL 的病，SFT 治大半；消融：没见过的句子靠「思考写在 <think> 里」那句触发
学到（方法）  秤的洞在探测的样例里抓（瞎答 1.05、中了还猜）；按表面像不像预测会反（两次）→ 一次改一处的消融；基建通用（harness + 四接口）；三臂同题同算力 + 先写预测
学到（边界）  剩下的错全是算术（中点、边界）→ 接计算器；10 轮内 GRPO 归因够用，几百轮另算；不会自己泛化到流程外（10 个数那类），考场上不再学
一句话        先教后找，课程铺路，秤要干净，状态写纸上 —— 三条线（Countdown、臂 A、猜数字）都成立
agent + RL 的进展（五处）  ① 第一次「环境有状态」的闭环（回应取决于之前的动作，最优策略要整合历史）—— agent 与多轮工具调用的分界
              ② 多轮信用归因在 GRPO 下跑通（每轮有价，终局一分，轮数压到 5.5）③ 通用 agent RL 基建（harness 四接口，训练器一个分支，一步 5 s）
              ④ 三个经典问题有了数（涌现 / 撒种 / 课程）⑤ 抓到 agent RL 特有的故障并记账（假观测、成功后继续行动、瞎答钻秤、纯 RL 绑措辞、显式状态决定泛化）
差距（四处）  轨迹短（10 轮 200 token，没碰上下文管理 / 记忆）；没有 critic 和过程奖励；环境是玩具（没沙盒、没真工具执行、没异步 rollout）；单技能没叠在一个模型上
下一步        编程题环境补沙盒和真工具 → 计算器接进猜数字补技能叠加 → 轨迹拉长补上下文和归因
```

### 教学：整合历史的五个动作；草稿纸设计与上下文管理；agent + RL 的本质特点（2026-09-11）
```
整合历史 = 五个动作   ① 读观测（higher 往大）—— RL 就能，SFT 更稳 ｜② 维护可行集（所有 higher 的最大做下界…）—— 写成「between lo and hi」= 显式状态，不写就每步隐式重推 ｜
                      ③ 不重复 —— 显式状态的副产品 ｜④ 选动作（取中点 = 信息量最大）—— 示范教、RL 磨 ｜⑤ 停（见 correct 交卷）—— RL 压出来
                      显式状态是「表示」那一环，把 ②③ 变成读两个数；①④⑤ 是策略，在状态之上决定做什么。没有 ②，①④⑤ 学得歪（E1）；有了 ②，RL 几步磨平（E2）
草稿纸设计   = 教材里让模型记什么账（区间 / 猜过的数 / too high 标注），老师程序定，走 SFT 进去（也可提示词引）。换账本行为就变；工业界叫 scratchpad / 结构化思维链格式
上下文管理 = 记忆管理   历史超窗口后：压缩（隔段摘要替原文，本会话就在用）/ 外部文件（md 笔记、记忆文件）/ 检索（RAG 自己的日志）/ 结构化状态（harness 维护 JSON 只贴状态不贴历史）
              前两种模型自己写「记什么丢什么」可训但难（回报在几十步后）；第四种 harness 替它做，往长程走第一步多半先用它
agent + RL 的本质：策略放进与环境的闭环里，按交互结果挪概率。八条特点（前四条 agent 带来，后四条 RL 在 agent 上的形态）
  ① 闭环：写的被执行、结果回上下文、下一步依赖结果    ② 轨迹两种人写：模型 token 与环境 token 交错，环境段 mask，奖励打在结果上（harness 的停贴续 mask）
  ③ 环境有隐藏状态要整合历史：状态写纸上，长了摘要 / 外部记忆    ④ 信用跨步归因：出错率按步连乘；短程 GRPO 够，长程要 critic / 按段打分
  ⑤ 只放大采得到的：长轨迹 p ≈ 0 → 撒种 / 课程 / 拆子任务 / 环境反馈，「先教后找」是刚需    ⑥ 环境即课程即秤：动作观测奖励全由环境定义，秤有洞就学歪
  ⑦ 策略学的是经济学：动作有价格 → 何时行动 / 停 / 转交    ⑧ 算力形状不同：采样占大头、引擎与训练器共卡、沙盒、异步；RL 策略绑措辞，前面要多样化 SFT
  我们在最小环境上八条都碰了，长程那两条只碰了门口
```

## ★ 下一条线：长程编程（编程题 + 长程文字合成一个实验，2026-09-11 定）
```
为什么合   「观测强弱」「路长短」是环境设计的两根轴，不是「编程」的属性；同一个编程环境能从短程强观测一路拧到长程弱观测（SWE-bench 小号版），产品方向也在这条线上；
           文字世界省掉或只当练兵场。观测：对学习越强越好（信号密、归因易）；对做事由世界决定，agent 得会在弱观测下自己把观测变强（选中点、加 print、写小测试 = 信息收集）；
           强到把答案给了就是抄（专家那根管子），观测该「告诉你走没走对」不该「告诉你答案」
环境       每局一个临时目录 = 小项目 + 测试；工具 <read> <grep> <write> <run> <ask>，harness 通用只加执行函数；奖励 = 测试通过率 − 工具 / 求助价格 − 编结果罚
题从哪来   程序造（同 Countdown）：收集能跑通测试的小项目，注 bug（差一、换运算符、少 import、条件反）→ 测试挂 = 一道题，答案 = 改回去的 diff = 老师示范；
           长程题用「删一个函数留着测试」造。题无限、可验证、有标准解
课程（用户定的顺序，一次只拧一根轴）
   ① 短程强观测   单函数，完整报错 + 测试输出，3~6 步          难点：用反馈改代码、沙盒、多工具、待办 H 三件套进 harness
   ② 长程强观测   三五个文件的小包，报错完整，10~30 步        难点：只多了长度 —— 定位、记住看过什么、规划；观测不变所以失败全归到长度上
   ③ 长程弱观测   测试藏起来只报过了几个，需求模糊，30~100 步   难点：只多了信息收集 —— 自己 grep / print / 写小测试；稀疏奖励的归因、记忆管理
   （我原先的顺序把两根轴一起拧，用户的顺序每阶段只变一个变量，失败能归因，p 更容易保持 > 0 —— 采用）
要量       各阶段通过率、每局步数、read / grep / run 次数、定位对文件的比例、求助率、留出项目；过程读数「第一次跑测试前读了几个文件」
代价       工程约两周：沙盒、文件工具、bug 注入器、老师 diff、并行沙盒（每步跑测试，rollout 比猜数字慢十倍以上）；
           1.5B 在 ③ 多半不够 → Qwen3 1.7B / 4B，尺寸阶梯并进来（4B 要 LoRA）
阶段 ① 第 1 步 题库（2026-09-12 完成）
   来源   MBPP 原版 974 道（google-research 的 mbpp.jsonl；HF 的 parquet 本机没 pyarrow 读不了）+ HumanEval 164 道（openai/human-eval）
   验过   每道参考实现在子进程里真跑一遍可见测试：MBPP 970/974 过（2 道 AssertionError、2 道 NameError，丢）；HumanEval 164/164 过
   形状   MBPP 每道恰好 3 条断言（可见 2 + 隐藏 1）、参考实现中位 5 行 P90 13 行、40 道多函数、187 道带 import；HumanEval 每题断言中位 7（1~26）
   切分   data_code.json：mbpp_train 870 / mbpp_heldout 100（seed 0 抽，训练不碰，= 留出项目）/ humaneval 164（整个留出）
   下一步 第 2 步注入器（AST 七类变异 + 验证 + 留出变异类型）
阶段 ① 第 2 步 注入器（mutate.py，2026-09-12 完成）
   做法   AST 七类变异：off_by_one / cmp_flip / arith_swap / bool_neg / ret_wrong / del_stmt / swap_args，每类只改第 k 个落点；先 ast.unparse 规范格式让 diff 干净；
          每个变异体编译 + 子进程跑 3 条测试：可见 2 条至少挂 1 条才留（等价变异、只挂隐藏的丢）；每题最多 3 个、每类 1 个，出得少的类优先（均衡）；
          隐藏测试按 task_id % 3 固定选；留出类 del_stmt / swap_args 只在 heldout 题上用；记 bug_type（可见测试全 AssertionError = 答错型，否则崩溃型）
   结果   2331 题 / 963 道（594 道 3 个、180 道 2 个、189 道 1 个），本机单核 20 分钟
          train 2054：arith_swap 401 / bool_neg 399 / cmp_flip 329 / off_by_one 399 / ret_wrong 526
          heldout 277：del_stmt 81 / swap_args 55 / 其余五类 141
          答错型 1931 / 崩溃型 400（NameError 120、IndexError 92、TypeError 58、Timeout 36、UnboundLocalError 23）；隐藏测试也挂 86%；可见两条都挂 84%
          丢弃：没落点 1800 次（多数题只有两三类能注）、等价变异 488、7 道一个 bug 都注不上
   注意   del_stmt 90% 是崩溃型（NameError）：留出类等于在考「加回一行」而不是「翻一个符号」，是另一种技能，留出正合适，评测时按 bug_type 分开报
          崩溃型的 traceback 直接指到行，观测比答错型强 → 评测和课程都按 bug_type 拆
   样例   task 11 cmp_flip：第一个循环 `if s[i] == ch` 被改成 `!=`；可见 hello/l、abcda/a 两条挂，隐藏 PHP/P 也挂
设计改动（用户提出，2026-09-12）
   ① 带报错开局：题面 = 题目 + 错代码 + 可见测试 + 挂的那条的断言与 traceback（截 600 字）。省一次调用、观测从第一步就强、「先测后改」不用学；
      改完必须 <test> 验证再 <answer>（验证习惯）。阶段 ③ 拧弱观测时就是把报错拿掉、测试藏起来
   ② 加阶段 ①′「从头写」：题面只有一句题目（或加测试），模型写整个函数、自己写测试、跑、改。同题库同工具同秤；p 低（1.5B 零样本两三成）、教材要 DeepSeek 或自采样；
      多学一样：自己写测试 = 自己造观测。顺序：① 修 bug（循环、沙盒、秤先通）→ ①′ 从头写（只新增「第一版自己写」）→ ② ③ 拉长。修改是写的子技能（拆子任务）
任务选择（2026-09-12 定）  目的排序：① 跑通并量清编程用的循环 / 沙盒 / 秤 ② 量「会不会用强观测改输出」③ 产品。①② 指向修 bug，③ 指向从头写 → 修 bug 起步
   带 / 不带报错开局   探测两种都跑（只差题面里那段 traceback），训练选 p 在 0.2~0.5 的那种：带报错 ≥ 0.6 就改不带（让它自己去测）；不带 < 0.05 就带报错 + 撒种；
                       也可按比例混两种开局（同模板混训），评测分开报
   修 bug 太容易的对策   不换任务，拧课程：崩溃型先答错型后、报错完整 → 截短、ret_wrong 先 off_by_one 后
   自己写测试 → 阶段 ③   它是「弱观测」的核心：测试藏起来，写测试 = 造观测。现在不能做：自写测试不能进秤（否则学写永远过的测试）、还要学「怀疑自己的测试」
四格课程（每格只拧一根轴）  ① 修 bug（循环、沙盒、秤）→ ①′ 从头写、给测试（第一版自己写）→ ② 长程强观测（长度）→ ③ 弱观测、自己写测试（信息收集）
阶段 ① 第 3~5 步 环境 + 秤 + 模板（code_env.py，2026-09-12 完成）
   协议   四个工具走通用 harness（stops = </test> </run> </write> </ask>，fake_tags = <result> <reply>）：
          <test></test> 每条可见测试单独跑、测试永远从母本重铺 → passed k/2 + 第一条挂的 traceback（去掉沙盒路径噪声，截 600 字）
          <run>代码</run> 先 exec 当前函数文件再跑这段 → stdout / stderr ｜ <write>代码</write> 整个替换函数文件（先 compile，语法错也写进去并报回）
          <ask>问题</ask> DeepSeek 看题面 + 当前代码回一段代码 → <reply>；模型得自己 write 再 test ｜ 结束：<answer>done</answer> 或 8 次调用
   秤     最终函数文件跑 2 可见 + 1 隐藏：全过 1 − 0.02 × 调用，否则 0；假 <result>/<reply> 训练器罚 0.2。只奖终局
   沙盒   子进程 -I、cwd 临时目录、5 s 超时、Linux 上 RLIMIT_AS 512 MB / NPROC 64、环境变量只留 PATH、socket 换成抛异常、input 空、输出截 600
          ★ 坑：超时路径写成 os.getpgid(0) → killpg 把自己和外面的 shell 一起杀了（单测无声退出码 1）。改成 Popen(start_new_session) + killpg(p.pid)
   开局   with_tb=True 题面带可见测试的输出（initial_obs，跑一次缓存在题上）；False 不带。模板 8 训练 + 2 留出（8 啰嗦英文、9 中文），三种工具措辞
   读数   behavior(txt)：工具序列、第一步是什么、write 前有没有 test、write 后有没有 test
   单测   tests/test_code_env.py：假模型按段吐、真沙盒跑：标准一局 0.94 分、语法错 write、调用上限、假结果、ask → reply → write → test、
          ★ 硬编码可见测试被隐藏测试抓住得 0、沙盒四项（超时 / 断网 / 截断 / 看不到密钥）、10 模板 × 2 开局
   md5    code_env 8f988e62 / mutate 86a16a5d / tests 3eb65edc / data/code_bugs.json 5d93f83f / data_code.json 1849aae7
   下一步 第 6 步探测：eval_code.py（带 / 不带报错 × 三个模型）+ 作弊探针
阶段 ① 第 6 步 探测工具就绪（eval_code.py 99e6c5ac，2026-09-12）
   用法   --with-tb 切开局；--split / --kinds / --bug-type 选题（每 task 只取一个变异体）；--inject 在 <think> 前插一段（作弊探针）；--ask 开专家；--dry 只打题面；--analyze 重算
   读数   修好率（3 条全过）/ 隐藏通过 / 可见通过 / 改过文件 / 每局调用与四个工具各几次 / write 前后有没有 test / 语法错 / 假观测 / 撞顶 / ★ 可见过但隐藏挂（硬编码）/
          按 bug_type、kind、模板拆 / 第一步是什么 / 组的分布
   探测计划  三个模型（Qwen2.5-1.5B Base、hf_armA、Qwen3-1.7B-Base）× 两种开局，各 100 题 × 8，训练模板；作弊探针 2 种 inject 在最好的模型上跑
   预测（先写）  修好率 带报错 / 不带：Base 0.05 / 0.03，armA 0.10 / 0.05，Qwen3-1.7B 0.25 / 0.15；崩溃型高于答错型；Base 大半不用工具直接写代码；
                作弊探针：「可见过但隐藏挂」条数明显上升，修好率不升 → 秤没洞
   探测结果 ①（带报错开局，100 × 8，2026-09-12）
| | Qwen2.5-1.5B Base | hf_armA |
|---|---|---|
| ★ 修好率 | 0.014 | 0.024 |
| 用过工具 · 改过文件 · 有 <answer> | 0.378 · 0.171 · 0.329 | 0.519 · 0.236 · 0.328 |
| 每局 test / write / run / ask | 0.19 / 0.28 / 0.14 / 0.15 | 0.31 / 0.55 / 0.23 / 0.28 |
| 语法错 / 局 · 假观测 · 撞顶 | 0.17 · 105 · 56 | **0.44** · 124 · 38 |
| write 前 test · 后 test | 0.02 · 0.03 | 0.01 · 0.06 |
| 第一步：无 / write / test / run / ask | 464 / 124 / 105 / 62 / 45 | 388 / 172 / 122 / 69 / 49 |
| 组 有对有错 / 全错 | 7 / 93 | 16 / 84 |
| 硬编码（可见过隐藏挂） | 0 | 1 |
   预测对账  ✗ 又高估：说 0.05 / 0.10，实测 0.014 / 0.024。p ≈ 0.02，和 E 的起点（0.043）同一档 → 撒种是必须的，纯 RL 起不来
   三种死法  ① 一半以上从不碰工具（58% / 49%）：代码写在 <think> 里就停，或者漂成别的文本（GitHub 页面、<replace> 标签）—— 文件没变 = 0 分
             ② 语法错：armA 每局 0.44、write 0.55 → 八成的 write 是语法错。看样例大半是环境的锅：「<write> def f():」首行带一个空格 → unexpected indent；
                还有 markdown 围栏。★ 改 normalize_code：去围栏 + 整体 dedent（编辑器都容忍的格式噪声），<run> 同样处理。真语法错照报
             ③ 多函数题只写了一个函数（<write> 是整文件替换，另一个函数没了 → NameError）：这是真的，教材里要教「整个文件一起写」
   也看到的  修好的都是一次 write 不测就交（0.98）；假观测 105 / 124 = 和 E 起点一样的老毛病；隐藏通过 0.15 是因为 14% 的 bug 隐藏测试本来就不挂，这列单独不能看
   下一步    改完 normalize 重跑 base / armA（带报错），加 Qwen3-1.7B；Coder-1.5B 等下载；不带报错那轮在这之后
   读数补两列（2026-09-12）  edit_stats：改动行数、改中 bug 那行、整段重写（原文件行保留不到一半，≤ 2 行的函数不判）。
             「验证」看 write 后 test；「用没用观测」看改动是不是定位到 bug 行（带报错开局里 test 证明不了这个）；「主动观测」看不带报错那轮的 write 前 test
   用户的倾向  不带报错开局更好：正确的一局要自己 <test> 拿观测（多一次调用），「主动观测」这个动作才在训练里。记下，等八份探针对比后定
   探测结果 ②（环境修 normalize 后重跑，带报错开局，100 × 8，2026-09-12）
| | Qwen2.5-1.5B Base | **hf_armA** | Qwen3-1.7B-Base | Qwen2.5-Coder-1.5B |
|---|---|---|---|---|
| ★ 修好率 | 0.015 | **0.074** | 0.004 | 0.041 |
| 用过工具 · 改过文件 | 0.386 · 0.181 | 0.497 · 0.220 | 0.149 · 0.071 | 0.492 · 0.275 |
| 每局 test / write · 语法错 | 0.22 / 0.27 · 0.13 | 0.35 / 0.32 · 0.10（之前 0.44） | 0.12 / 0.12 · 0.06 | 0.34 / 0.47 · 0.20 |
| write 后 test | 0.03 | 0.06 | 0.02 | 0.07 |
| 假观测 · 撞顶 | 97 · 61 | 106 · 34 | 63 · 117 | 141 · 62 |
| 改过的里整段重写 · 修好的里整段重写 | 0.77 · 0.25（12 条） | 0.82 · 0.81（59 条） | 0.91 · 1.00（3 条） | 0.72 · 0.42（33 条） |
| 第一步从不碰工具 | 459 | 410 | 661 | 360 |
| 有对有错的组 | 8 | **34** | 3 | 25 |
   预测对账  ✗✗ 说 Coder ≈ 0.35、Qwen3 ≈ 0.25 都高于 armA：反了。Qwen3-1.7B-Base 最差（0.004，661/800 一个标签不碰，漂成韩文、图片链接、markdown），
             Coder 0.041 < armA 0.074。原因：这是「标签协议下的修 bug」，格式熟不熟比代码强不强先起作用；armA 的 <think>/<calc>/<ask> 习惯迁移了一半。
             Qwen3 Base 是裸 Base，协议完全不认。★ 零样本探测量的是「格式熟 + 代码能力」的乘积，分不开
   环境修正的效果  armA 语法错 0.44 → 0.10、修好率 0.024 → 0.074、有梯度组 16 → 34：之前 2/3 的「语法错」是环境的格式噪声
   还剩的一处   「<write> n = M // 2\ndef …」只有首行带空格、其余顶格：dedent 救不了 → 首行单独 lstrip（模块级第一条语句本来该顶格）。已改
   读数修正     「改中 bug 那行」0.92~0.95 对所有模型都高 —— 整段重写把 bug 行也一起删了，这列没有区分度。换成「定位修改」= 改中 bug 行 ∧ 非重写 ∧ ≤ 2 行
   修法的样子   修好的几乎全是「不测、整段重写、碰巧对」：write 后 test ≤ 0.07；armA 修好的 81% 是整段重写，Coder 42%，Base 25%（Coder / Base 的定位修改比例高一点）
   决定        起点候选 armA（p 0.074、34 组有梯度）；但零样本分不开格式和代码能力 → 造好教材后 armA 和 Coder 各 SFT 一次（2 分钟一份）再比，谁高谁进 RL
              开局：p 全部 < 0.1，撒种必须；撒种之后开局的选择变成「教什么」→ 按用户倾向教不带报错（test → 定位 → write → test），混 30% 带报错的题面
   待跑        不带报错那轮（四个模型，留基线）；作弊探针 2 种 inject 在 armA 上
   探测结果 ③（不带报错开局，100 × 8，2026-09-12）
| | Base | armA | Qwen3-1.7B | Coder-1.5B |
|---|---|---|---|---|
| 修好率（带报错 → 不带） | 0.015 → 0.022 | 0.074 → **0.075** | 0.004 → 0.004 | 0.041 → 0.026 |
| write 前 test（主动观测） | 0.02 | 0.01 | 0.02 | 0.03 |
| 第一步 test（条） | 112 | 162 | 78 | 92 |
| 定位修改（改过的 / 修好的） | 0.14 / 0.33 | 0.09 / 0.17 | 0.07 / 0.00 | 0.14 / 0.33 |
| 有对有错的组 | 10 | 33 | 3 | 17 |
   读法   ① 带不带报错修好率一样（armA 0.074 vs 0.075）：题面里的 traceback 根本没被用，修法是「凭描述重写」，观测有没有都一样
          ② 主动观测 ≈ 0：write 前有 test 的 1~3%；armA 第一步 test 的 162 条里绝大多数测完就停或漂走，没接 write
          ③ 定位修改全在 0.1 上下，修好的里 Base / Coder 0.33、armA 0.17 —— Coder 和 Base 的代码底子可能更好，armA 靠格式熟
          ④ 又一个 harness 洞：裸 Base 温度 1 采到词表外的占位 id（Qwen 输出层多 270 行）→ 下一轮喂回 vLLM 报 out of vocabulary。两个循环都加 drop_oov 截断
   撞 bug 榜   task 348（bin_coff / find_ways 双函数）四个模型全死：多函数 + 把 n = M // 2 搬出函数 → NameError；教材要教「整个文件一起写」
   结论   零样本没有一个模型「用观测」；撒种教的核心就是 test → 读 traceback → 定位 → write 整文件 → test；起点 armA vs Coder 各 SFT 一次再比
阶段 ① 第 7 步 教材（make_code_demos.py，2026-09-12）
   轨迹   不带报错（70%）：<test> → ⟦真观测⟧ → 「The failing test is …; line N: `bug 行` — <kind 的一句解释>. It should be `ref 行`.」→ <write>整文件</write> → <test> → ⟦passed 2/2⟧ → </think><answer>done</answer>
          带报错（30%）：跳过第一次 test。重试（20%）：先 write 同题另一个变异体 → test 挂（真输出）→ 「Still failing … my fix introduced another mistake on line …」→ write 对 → test 过
          注入段全部由 CodeEnv.step 真跑出来（格式与 harness 一致），SFT 时 <result> label −100；只用训练 split 与训练 kind；8 模板 × 3 措辞随机
   冒烟   12 条：字符中位 1935、最长 2937（≈ 1300 token）→ SFT --max-len 2048；全量 1200 条后台生成中
   全量   sft_code.jsonl（重生成后 4fdee64d，stdin 沙盒的观测格式）：1200 条 / 675 道题，带报错 373、重试 201；kind ret_wrong 365 / arith_swap 229 / bool_neg 212 / off_by_one 212 / cmp_flip 182；
          答错型 1030 / 崩溃型 170；8 模板各 ~150；调用 3.0 / 条；每条都以 passed 2/2 → <answer>done</answer> 收尾（程序验过）。本机单核约 15 分钟
   SFT 计划  同一份教材各训一次：armA（--init out_armA/ckpt_latest.pt）和 Coder-1.5B（--model hf_coder_1.5b，无 --init），lr 1e-5、2 epoch、batch 32、--max-len 2048；
          只用 code 教材（先不混 Countdown / 猜数字，技能叠加留到起点定了之后）；各评带 / 不带报错，谁高谁进 RL
   SFT 完成（2026-09-12）  armA：val 1.2124 → 0.0486，70 步 3 分钟；Coder：同配方，ckpt 存了（日志截断）
     ★ 收尾闭环（Countdown 工具题）：armA 训完代码教材后不再写 <calc>，改成一长串心算「2 + 3 + 5 * 7 = 36」（还算错，应为 40）→ 单一教材 SFT 把上一个技能冲掉了，
       技能叠加必须混教材（留到起点定了之后）；GSM 两个都对；Coder 的 Countdown 本来就不会
   ★★ SFT 后评测（100 × 8，温度 1，训练模板，2026-09-12）
| | armA 零样本 | **SFT-armA 不带报错** | SFT-armA 带报错 | **SFT-Coder 不带报错** | SFT-Coder 带报错 |
|---|---|---|---|---|---|
| ★ 修好率 | 0.075 | 0.454 | 0.420 | **0.606** | 0.598 |
| 用过工具 · 改过文件 | 0.47 · 0.17 | 0.99 · 0.88 | 0.99 · 0.87 | 1.00 · 0.94 | 1.00 · 0.92 |
| 每局 test / write · 调用 | 0.37 / 0.23 · 0.9 | 3.1 / 2.2 · 5.3 | 2.5 / 2.1 · 4.6 | 2.8 / 2.1 · 4.9 | 2.1 / 2.0 · 4.1 |
| write 前 test（主动观测） | 0.01 | **0.93** | 0.39 | **0.98** | 0.04 |
| write 后 test（验证） | 0.04 | 0.97 | 0.96 | 1.00 | 0.99 |
| 定位修改（改过的 / 修好的） | 0.09 / 0.17 | 0.68 / 0.93 | 0.64 / 0.94 | 0.75 / 0.97 | 0.78 / 0.98 |
| 整段重写（修好的里） | 0.78 | 0.03 | 0.03 | 0.00 | 0.01 |
| 语法错 / 局 · 假观测 · 撞顶 | 0.04 · 108 · 14 | 0.06 · 6 · 77 | 0.04 · 5 · 60 | 0.04 · 3 · 97 | 0.03 · 1 · 79 |
| 硬编码（可见过隐藏挂） | 2 | 9 | 3 | 5 | 6 |
| 第一步 | 无 420 / test 162 | test 763 | write 459 / test 334 | test 785 | write 762 |
| 组 全对 / 有对有错 / 全错 | 0 / 33 / 67 | 8 / 73 / 19 | 9 / 70 / 21 | 27 / 61 / 12 | 31 / 51 / 18 |
| 按 bug_type crash / wrong | 0.01 / 0.08 | 0.35 / 0.47 | 0.31 / 0.44 | 0.35 / 0.64 | 0.42 / 0.62 |
| 按 kind（不带报错） | — | bool_neg .62 ret_wrong .55 cmp_flip .43 arith .35 off_by_one .30 | | bool_neg .81 ret_wrong .70 cmp_flip .56 off_by_one .50 arith .46 | |
   预测对账  不带报错修好率 > 0.5：Coder ✓ 0.606，armA ✗ 0.454；write 前 / 后 test > 0.8 ✓ 0.93~1.00；Coder 高 armA 约 0.1 ✓ 差 0.15
   读法   ① 撒种一次把「做法」装到位：三个零全变成 ≥ 0.93（主动观测、验证、定位）；整段重写从 0.78 → 0.03。教材 1200 条、3 分钟
          ② 起点定 Coder：同一份教材、同配方，0.606 vs 0.454，每类 bug 都高 0.1~0.2，有梯度的组 61 vs 73（Coder 全对更多）。零样本 armA 靠格式赢，
             SFT 抹平格式后代码底子说话。代价：Countdown / 猜数字要在 Coder 上重叠（armA 训完代码教材计算器习惯也已经冲掉，本来就要重叠）
          ③ 带 / 不带报错持平（Coder 0.598 / 0.606）：带报错时它 96% 直接 write 不先测（教材教的），不带时 98% 先测，两条路都通，题面给不给观测它都会拿
          ④ 剩下的错的样子（task 348 双函数，四份全死，撞顶 8 次调用）：定位错了就「Still failing … my fix introduced another mistake」硬套模板，
             反复改同一行或改到注释里、把 `def` 挤到上一行末尾；没有「回退到原版换一处试」的动作。这是 RL 要治的：错了 8 次的组里如果有一条对的，就有梯度
          ⑤ 硬编码 3~9 条（≈ 1%）：SFT 后开始出现「可见过隐藏挂」，撞锤够硬了 → RL 前后作弊探针照做
          ⑥ 撞顶 60~97 条（8~12%）：8 次调用用满，多半是 ④ 的循环；假观测清零
   决定   起点 = SFT-Coder（out_sft_code_coder），锚也是它；RL 用不带报错 70% + 带报错 30% 混开局（同 SFT）；150 步；预测 RL 后不带报错 0.606 → 0.72，
          撞顶 12% → 4%，定位修改 0.75 → 0.85，留出模板 / 留出 kind / HumanEval 三口径待评
   RL 前讨论：要不要 DAPO（2026-09-12，用户定：先不加，按原样跑）
     DAPO 四件对我们：动态采样（全同分组丢掉补采）← 值得加，SFT-Coder 分布 27/61/12，每步 8 个 bug 期望 3 组全同分 = 40% 采样白跑；
                     按 token 平均 ← 一直在用；clip-higher ← 不需要（on-policy 一步一更新，只有 KL 锚，没做 PPO 裁剪）；超长软罚 ← 不加（v3 只奖不罚）
     动态采样的副作用：训练分布偏向中等题（全对的题永远进不了梯度，「更短更省」的信号没了）；「有效步数」和「采样步数」要分开记，不然和 E 的 150 步不可比
     和筛池的区别只在时机：筛池训前筛一次池子固定，动态采样每步用当前模型筛，题无限时后者更合适
     决定   第一趟按原样 150 步（和 E 同口径的基线，看 |A| 有多少步空转）；第二趟加动态采样对照 → 干净的「这一件值多少」
### 教学：rollout 慢在哪 —— 同步轮次 vs 异步；vLLM / GPU / CPU 各在等谁（2026-09-13）
```
量到的   一步 42+7 s：42 s 采样、7 s 训练。CPU 128 核、负载 11、空闲 94% → 慢不在算力，在等待。--code-step-workers 32 没明显变快，说明并发没落到沙盒上
结构     现在的 harness 是「同步轮次」：一次 be.generate() 把 32 条打成一批交给 vLLM，等全批都停下才返回；停在标签上的那几条去跑沙盒；再一起回去生成。
         每轮等最慢的那条，一局八轮等八次；沙盒段 GPU 全闲，生成段 CPU 全闲；一轮里真正要跑沙盒的往往只有几条，32 个线程大半空着
条 3 为什么等   不是不能去，是那一行批量调用把它的结果攥到全批停下才交出来；vLLM 本身支持一条一条随时进出，是我们为了简单打成批
三级阶梯   ① 同步轮次（现在）：每轮等最慢 → 每轮最慢之和，严格 on-policy
           ② 改 harness 的异步：每条独立循环（写 → 停 → 自己跑沙盒 → 自己续），步末汇合一次 → 最长的一条，仍严格 on-policy，白赚
           ③ 业界全异步（verl agent loop / slime / AReaL）：连步末的墙也拆，一条完了进缓冲、攒够就更新，采样端训练端都不停 → 接近平均一条；
              代价 off-policy（轨迹是几步前的权重采的），靠重要性采样 + PPO 裁剪 + 限制陈旧度修正；all-reduce 不是主因
OpenAI / Anthropic   没公开框架；从 Dota 时代的 worker / learner 分离和公开访谈推，形状就是 ③，差在规模、环境数量、工程细节
决定     先看计时那行「一轮最多几条同时等」：小 → 结构问题，做 ②；接近 32 仍慢 → 沙盒本身慢，一个子进程跑两条断言 + -S。② 几百行，等这趟跑完再做
```

### 教学：全异步的 off-policy 怎么修、长轨迹会不会被扔、稀缺优秀轨迹怎么认（2026-09-13）
```
off-policy 的病   旧菜谱炒的菜拿来改新菜谱：功过是旧版的，记到现在头上方向会偏
三层药     ① 按「现在会不会这么炒」打折：比值 = 现在策略的概率 / 采样时的概率，乘进梯度；PPO 裁剪把 1 ± 0.2 之外的直接不给梯度（重要性采样）
           ② 太旧直接扔：最大陈旧度参数，超过就丢掉重采   ③ 隔几步就把新权重推到生成端，把陈旧度控制在一两版内。三层叠用，不是二选一
长度偏置   陈旧度按「开始时的版本」算，长轨迹天然更旧、更容易撞线（用户的担心，业界叫 length bias）。处理：陈旧度上限设宽（几版到十几版，靠打折不靠扔）、
           分段更新权重（长轨迹中途换菜谱，按段各算比值，verl / slime 支持）、长度分桶或按长度加权
稀缺的优秀长轨迹怎么认   不是「落后多」：三个条件的交集 —— 秤的分高 ∧ 组内成功率低（16 条只对 1 条那种，GRPO 的组均值现成）∧ 步数多
           存进池子再用：拒绝采样 SFT（R1 第三段）、专家迭代 / STaR（采 → 留对的 → SFT → 再采）、优先回放（按稀缺度加权重采）、蒸馏；
           模型现在做不到但偶尔撞对的靠推理时搜索（几百条 / 树搜索）把算力换成稀缺样本。我们的 raw 里分、组内成功率、调用数全有，一行过滤就是自己的拒绝采样 SFT
我们      改法一（步末汇合）没有这些问题：同一版权重、长短一视同仁，8 步轨迹等最长的不亏；长程时再考虑改法二，长度偏置要进设计
```

### 名词总表：秤 / 环境 / 验证器 及其周边（2026-09-13，一次说清）
```
三个核心
  环境（environment）   模型行动的世界：有状态，收动作、回观测；一局之中每次动作都动；给模型看，贴回文本。code_env.step() + 函数文件 + 沙盒；猜数字的秘密数；Countdown 的计算器
  验证器（verifier）    判「这个结果对不对」的无状态程序，谁叫它它就动。run_tests()、safe_eval、判分器比数字。RLVR 的 V 就是它。本身不是奖励
  秤（reward function） 局终把整条轨迹变成一个数：验证器的判断 + 价格 + 惩罚（+ 部分分）。score() + 训练器的假观测罚。模型看不到，进优势
  关系                 秤调验证器，环境也调验证器；同一个「跑测试」在环境里是观测（模型看得到、不进分），在秤里是分（模型看不到）
  出问题改哪层          模型看不到该看的 → 环境；对错判断有洞 → 验证器；对的分不高 / 错的有分 → 秤
周边名词（本项目的说法 ↔ 业界的说法）
  harness ↔ agent loop / scaffold     模型外面的循环：盯标签、停、执行、贴回、续；注入段 mask。不学习。秘书
  工具 ↔ tool / function              被 agent 调用的执行器，给输入回输出，没有目标。计算器、专家、沙盒、<run>
  观测 ↔ observation                  环境贴回来的字：<result>、<obs>、<reply>。传感器读数，不是分
  动作 ↔ action                       模型写出的一段：<calc>、<guess>、<write>、<ask>
  轨迹 ↔ trajectory / rollout         一局从开局到 EOS 的全文，模型 token 与环境 token 交错
  状态 ↔ state                        环境记的（秘密数、文件内容）+ 模型在纸上写的（区间、too high 标注）；模型跨 token 没别的记忆
  秤的塑形 ↔ reward shaping           把 0/1 拆成阶梯部分分（v1）；容易教歪（教猜）
  过程奖励 ↔ process reward / PRM     给中间步打分；程序版（易钻）、学一个打分模型（易 hack）、事后重标。我们只用终局分
  隐藏测试 ↔ held-out tests           秤用、模型看不到的验证器；防硬编码
  假观测 ↔ fabricated observation     模型自己写 <result>/<obs>/<reply>；harness 拦、秤罚
  钻秤 ↔ reward hacking               策略对着秤的洞优化：猜、硬编码、改测试、读盘上的期望值。防线：隐藏测试、只读、盘上无文件、断网、整文件替换
  作弊探针 ↔ red-teaming the reward   故意引它作弊看秤拦不拦；SFT 后 / RL 后各撞一次
  题库 / 出题器 ↔ task generator      Countdown 的随机数、猜数字的秘密数、注入器；题无限时不筛池
  课程 ↔ curriculum                   一串按难度排的环境（N 20 → 100；短程 → 长程；强观测 → 弱观测）；换地的一种
  换地                                权重不动换环境让 p > 0；撒种是往权重里放零件
  benchmark                           别人定的题库 + 模板 + 设置 + 验证器；我们的 eval_code 就是自制的一个，差在没人同口径
  价格 ↔ action cost                   秤里每次动作扣的分（问 0.05、调用 0.02、多一轮 0.05）；RL 学的是这套经济学
环境 vs 工具（用户追问后的定版）
  判断标准一条   同样的动作重复两次，回应会不会不一样。不会 → 工具（售货机 / 计算器 / 沙盒 / 验证器，无状态）；会 → 环境（猜数字的对手、修 bug 的文件，有状态）
  位置一条       工具站在门口，叫一下答一下，答完就退；环境是你在的房间，每个动作都落在房间里、改过就一直是改过的样子（后果），阅卷看的是房间最后的样子（得分）
  考卷 vs 答题纸 考卷 = 环境里定死的部分（题面、可见测试）；答题纸 = 环境里会被动作改的部分（文件）= 状态；草稿纸 = 模型自己的思维链，不是环境
  同一张考卷三个层面   问它从哪来 → 环境；问它在哪 → 上下文（环境写的那部分字）；问它排第几 → 课程
  验证器 = 环境和秤共用的一个工具；沙盒 = 环境雇的售货机，状态在环境手里
  为什么状态要紧   有状态的对手逼着策略整合历史（模型在纸上抄账本「between 26 and 50」）—— agent 与单纯调工具的分界
  状态怎么变       不是每次交互都产生新状态：<write> 改文件（状态变），<test> 不改文件只读（状态不变，只有计数 +1）。状态 = 「下一步的回应会用到的、被历史改过的东西」
一个比喻串起来   考场：环境是考卷和草稿纸（题、文件、观测），验证器是答案册（对不对），秤是评分标准（几分），harness 是监考秘书（递工具、收卷），
                 工具是计算器和电话，题库是题海，课程是从易到难的卷子顺序，钻秤是抄答案册，隐藏测试是密封的那半份答案
```

   RL 第一趟起跑（2026-09-12）   RL 第一趟跑完（out_codeRL，150 步，191 分钟，2026-09-13）
     仪表盘（贪心 80 道）  起点 0.675 → 终点 0.750，峰值 0.7165 @75（ckpt_best），后 3 次没新高 → 平台在 0.72~0.75
     训练行（温度 1）      第 0 步 修好 0.41 / |A| 0.12 / 调用 5.0 / 长 532 → 第 149 步 修好 0.65 / |A| 0.21 / 调用 4.3 / 长 540；语法错 0.04 → 0.07；假观测 0；KL 0.005
     有效组 1.2/2 = 每卡 2 道题平均 1.2 道有梯度 → 40% 的组空转，和 SFT-Coder 分布（27/61/12）算的一样 → 第二趟动态采样的理由坐实
     一步 44+8 s，全程 GPU 等沙盒；计时那行这趟没有（同步前起的）
     待评   导出 latest 和 best，三口径（训练模板 / 留出模板 / 留出 kind + HumanEval）× 两种开局；作弊探针；lm_eval humaneval
   ★★ RL 第一趟评测（hf_codeRL = 150 步 latest，100 × 8，温度 1，2026-09-13）
| | SFT-Coder 不带 | **RL 不带报错** | SFT-Coder 带 | RL 带报错 | RL 留出模板 | RL 留出 split（含 del_stmt / swap_args） |
|---|---|---|---|---|---|---|
| ★ 修好率 | 0.606 | **0.749** | 0.598 | 0.731 | 0.713（8 号 0.740 / 9 号 0.685） | 0.604 |
| 隐藏通过 · 可见通过 | 0.667 · 0.637 | 0.791 · 0.777 | 0.676 · 0.644 | 0.781 · 0.761 | 0.761 · 0.742 | 0.655 · 0.656 |
| 每局调用 · test · write | 4.9 · 2.8 · 2.1 | **4.3 · 2.6 · 1.7** | 4.1 · 2.1 · 2.0 | 3.4 · 1.7 · 1.7 | 4.4 · 2.6 · 1.8 | 4.8 · 2.8 · 2.0 |
| write 前 test · 后 test | 0.98 · 1.00 | 0.98 · 1.00 | 0.04 · 0.99 | 0.04 · 0.99 | 0.88 · 0.99 | 0.98 · 1.00 |
| 定位修改（改过的 / 修好的） | 0.75 / 0.97 | **0.83 / 0.94** | 0.78 / 0.98 | 0.83 / 0.98 | 0.82 / 0.97 | 0.52 / 0.65 |
| 撞顶 · 假观测 · 语法错/局 | 97 · 3 · 0.04 | **55 · 0 · 0.00** | 79 · 1 · 0.03 | 60 · 2 · 0.02 | 65 · 1 · 0.01 | 134 · 2 · 0.01 |
| 硬编码（可见过隐藏挂） | 5 | 8 | 6 | 12 | 9 | 14 |
| 组 全对 / 有对有错 / 全错 | 27 / 61 / 12 | **48 / 45 / 7** | 31 / 51 / 18 | 48 / 44 / 8 | 41 / 53 / 6 | 31 / 55 / 14 |
| 按 kind | bool .81 ret .70 cmp .56 off .50 arith .46 | bool .89 ret .79 cmp .75 off .65 arith .67 | | | | del_stmt .57 swap_args .47（没见过）；其余 .53~.75 |
| 按 bug_type crash / wrong | 0.35 / 0.64 | 0.62 / 0.77 | 0.42 / 0.62 | 0.59 / 0.75 | 0.54 / 0.74 | 0.52 / 0.66 |
   预测对账   不带报错 0.606 → 0.72 ✓ 0.749；撞顶 12% → 4% ≈ 6.9%（97 → 55）；定位修改 0.75 → 0.85 ≈ 0.83
   读法   ① RL +0.14，每类 bug 都涨，crash 型涨最多（0.35 → 0.62）；调用 4.9 → 4.3、语法错清零、假观测清零、全对组 27 → 48。学的是「少走弯路 + 别写坏语法」
          ② 留出模板 0.713、留出 split 0.604：措辞掉 0.04（小），没见过的 bug 类型 del_stmt 0.57 / swap_args 0.47 —— 比训过的类低 0.2，但远不是零，
             「加回一行」这个动作它没学过也能做一半。留出 split 的定位修改只 0.52：不认识的 bug 它更常整段重写
          ③ 剩下的死法没变：task 348 双函数题所有口径全死，8 次调用用满，「Still failing … my fix introduced another mistake」硬套，改到注释里、把 def 挤到上一行、
             甚至开始否认测试（「I misread the test; it says == 4」）。没有「回退原版换一处」，RL 150 步没治，这是教材要补的分支（重试轨迹只教了一次失败）
          ④ 硬编码 5 → 8 条（1%），仍在噪声内；作弊探针那份待贴
          ⑤ 带 / 不带持平（0.731 / 0.749），两条路都稳
   HumanEval（lm-eval，164 题，0-shot 贪心，pass@1，± 0.038，2026-09-13）—— 「从头写」的通用代码能力，整个题库没进过训练
     Coder-1.5B Base 0.421（官方报 43.9，口径差在 0-shot 与停止串）→ SFT-Coder 0.396（−0.025）→ RL-Coder 0.384（−0.012）
     读法   两步各掉一点，合计 −0.037，都在 ±0.038 的噪声边上，方向一致：修 bug 的窄训练对「从头写」略有代价，没有崩。
            crosstask 那格的编程版：训修 bug 没把写函数的能力冲掉（armA 训代码教材冲掉计算器那次是 100% 冲掉，这里是 4 分）
     跑法坑   HumanEval 判分要执行生成的代码，除了 --confirm_run_unsafe_code 还要 export HF_ALLOW_CODE_EVAL=1；tail -6 会把进度憋住看着像死了
   补齐的四份（2026-09-13）
| | SFT-Coder | RL 150 步 | 差 |
|---|---|---|---|
| 训练模板，不带报错 | 0.606 | 0.749 | +0.14 |
| 留出模板 | 0.552 | 0.713 | +0.16 |
| 留出 split（del_stmt / swap_args 没见过） | 0.477（del .36 / swap .33） | 0.604（del .57 / swap .47） | +0.13（没见过的 kind +0.14 ~ +0.21） |
| best @75 不带报错 | — | 0.704 | 比 latest 低 0.045：仪表盘选出来的不如跑完的（贪心 80 道的噪声），主线用 latest |
   作弊探针（RL，带报错，「把测试期望的值直接返回」）  修好率 0.738（基线 0.731），硬编码 11 条（基线 12）→ 没上钩；RL 后撞锤够硬了这次仍没钻，秤的五条防线守住
   读法   ① RL 的 +0.14 在三个口径上一样大（+0.14 / +0.16 / +0.13）：学到的不是「这 8 句话」也不是「这 5 类 bug」，是通用的「少走弯路」
          ② 没见过的 bug 类型：SFT 0.36 / 0.33 → RL 0.57 / 0.47，涨幅和训过的一样 → RL 在留出 kind 上不比 SFT 更绑
          ③ ckpt_best 比 latest 低：仪表盘 80 道贪心的方差 ≈ ±0.05，挑 ckpt 靠它不可靠；以后 best 只当保险，主线用 latest
          ④ 死法样本又多一种：SFT 在留出 split 上「My fix introduced a fourth mistake」连改四次同一行、每次换个说法（return -b / return b）；
             RL 开始「否认测试」（The test is on the wrong line, should be == 4）和自我安慰（The correct file is up to date and ready to go）—— 没有回退动作的模型在 8 次预算里的必然形态
  仪表盘起点 0.675（贪心 80 道）；第 0 步训练行：修好 0.41、|A| 0.12、调用 5.0、长 532、语法错 0.04、假观测 0，一步 51 s，150 步约 2 小时
     训练器加 --dyn-sample（3085577a → 新版）：每步多抽 (1+r) 倍的题一起 rollout，有奖励差异的组优先留 PER 道，其余丢；日志多打「有效组 x/PER」（不开也打，第一趟就能看空转比例）
   RL 第二趟起跑（--dyn-sample 0.5，其余与第一趟一字不差，out_codeRL_dyn，2026-09-13）
     第 23 步  修好 0.51、有效组 2.0/2（第一趟 1.2/2 → 动态采样把空转填满了）、42 + 8 s/步；[rollout 计时] 生成 6.9 s 环境 8.2 s 132 次调用 / 8 轮 一轮最多 48 条同时等
     ★ 沙盒瘦身生效  评测 800 条时环境 389 s → 训练每卡 48 条 8.2 s，生成 6.9 s；两段加起来 15 s，一步却 42 s → 还有 ~27 s 在 rollout 两段之外
                    （唤醒 vLLM、灌权重、睡下、训练编码），下一个要量的地方；多抽 50% 的题让每步 rollout 条数 128 → 192，步时基本没涨（42 vs 44 s）
     第 25 步评测  修好 0.713，存 ckpt_best + ckpt_latest；第 26 步 rollout 崩
   坑 ⑤ 上下文上限（2026-09-13）  ValueError: The decoder prompt (length 2307) is longer than the maximum model length of 2224
     根因  vLLM 的 max_model_len 我按 max_new + 1200 = 2224 定；修 bug 的题面最长 1300 多 token，第二轮 write/test 之后「题面 + 已生成」超过上限；
           harness 只按 max_new − len(gen) 给预算，从不看题面长度。第一趟 150 步没撞上是运气（最长题面 × 长轨迹的组合没抽到）
     修    harness.generate_with_env 加 max_total：每条预算 = min(max_new, max_total − 题面 − 8)，超长的条自己截（记 cut），不再把超上限的行喂给引擎；
           训练器 VL_MAX_LEN = max_new + 1600 = 2624 并传给 harness；评测器传 plen + max_new + 64。单测加一条：题面 3 个 id、max_total 16 → 不管 max_new 多大都截在 5 个 id
     续    --resume 从 out_codeRL_dyn/ckpt_latest.pt（第 25 步，带优化器）接着数，日志 tee -a；峰值从 hist 里恢复
     教训  两个上限是两回事：max_new 管「模型最多写多少」，max_model_len 管「引擎一行最多装多少」；后者必须 ≥ 最长题面 + max_new，否则是概率炸弹，跑得越久越准炸
   一步 42 s 里生成 + 环境只有 15 s，另 27 s 在哪（用户问「等 GPU 还是等 CPU」，2026-09-13）
     t_gen 从 rollout() 进到出：唤醒 vLLM + 灌权重 + 醒 KV（GPU 显存搬运，估 1~3 s）→ harness（量到 15 s）→ 每条终评 CENV.score = 一次沙盒子进程，主线程串行 48 条
     （fork 一个几十 GB 的进程 + python 启动 + 跑 3 条测试，估 0.1~0.3 s/条 → 5~15 s；终版代码死循环的每条再 +5 s）→ vLLM 睡下 + gc（估 1~2 s）
     答：主要是等 CPU —— 终评那段 GPU 空转，48 个子进程一个接一个起；不是等 GPU。env.step 已经 32 线程并发，终评没并发，是漏改的一处
     另一桶没量到的：rank 0 打的 42 s 只是自己的 rollout，all_reduce 处等最慢 rank 的时间不在 42 也不在 8 里（第一趟 191 分 / 150 步 ≈ 70 s/步扣掉评测，比 52 多 ~18 s）
     先量再改：训练器 [rollout 计时] 改成分段（唤醒+灌权重 | harness = 生成 + 环境 + 其余 | 终评 | 睡下），续跑就能看到分解
     预测  终评 8~15 s、唤醒 2~3 s、睡下 1 s、harness 其余 < 1 s；把终评放进同一个 32 线程池后 42 → 25~30 s，训练数字一个不变（终评只是并发，秤不变）
     改    训练器 --score-workers（0 = 同 --code-step-workers，1 = 串行老行为，量对照用）：终评走同一个线程池；eval_code.py 终评也并发（800 条串行原来要两三分钟）
     venv  续跑命令我写成了 test001/.venv，错：训练器开 --engine vllm 在进程里起 vLLM，必须 ~/vllm_env（第一趟就是在这跑的）。只有 sft_qwen.py 用 test001/.venv
   续跑第 25~27 步的计时（2026-09-13 13:04）  预测落空，反而更慢：
     25 | 唤醒 1.1 | harness 56.4 = 生成 20.4 + 环境 35.9 + 其余 0.2 | 终评 14.0（48 条，32 并发）| 睡下 1.0 | 204 次调用 / 8 轮 | 84+10 s
     26 | 唤醒 1.0 | harness 58.3 = 生成  7.3 + 环境 50.8 + 其余 0.1 | 终评 14.0（48 条，32 并发）| 睡下 1.0 | 158 次调用 / 8 轮 | 73+7 s     27：75+8 s，剩 174 分
     31 | 唤醒 1.2 | harness 32.0 = 生成  9.6 + 环境 22.3 + 其余 0.1 | 终评 13.8（48 条，32 并发）| 睡下 1.0 | 125 次调用 / 8 轮 | 50+7 s（用户没动，同一趟）
     每次沙盒调用的环境秒数：第 23 步 8.2/132 = 0.062 → 25：35.9/204 = 0.176 → 26：50.8/158 = 0.322 → 31：22.3/125 = 0.178；终评三步 14.0 / 14.0 / 13.8
     → 环境时间 = 调用数 × 常数，终评 = 48 × 常数：全是「起子进程」的串行代价，跑测试本身早被并发掩掉了；常数崩前 0.06、续跑 0.18，3 倍 = 进程胖了
     读法  ① 唤醒 1 s、睡下 1 s、其余 0.2 s：这三桶预测对了，加起来不到 3 s，vLLM 的 wake/sleep 日志也是 0.04 ~ 0.6 s
           ② 生成 20.4 → 7.3：第 25 步是重启后第一步，前缀缓存和内核预热，第 26 步回到崩前的 6.9
           ③ ★ 环境 8.2 s（崩前第 23 步）→ 35.9 / 50.8 s；终评 32 并发仍要 14.0 s，两步一模一样 = 48 × 0.29 s：并发一点没帮上 → 瓶颈不在跑测试，在「起子进程」
              起子进程 = fork 训练进程：Linux fork 要复制父进程整张页表、每页标 COW，代价随父进程内存线性涨；preexec_fn 又逼 Python 走真 fork 且在 GIL 下，32 个线程排队 fork
              为什么崩前只要 0.06 s/次：续跑多了 6.2 GB —— --resume 分支 torch.load 的 ckpt（权重 + 优化器）灌进模型后没 del（--init 分支有 del），整趟留在 CPU 内存里 → 父进程胖了一倍多
           ④ 一步 80 s 比崩前的 50 s 还慢，剩 123 步要多花一小时
     修    ① --resume 分支 del ck + gc（父进程瘦回去）② 沙盒不再用 preexec_fn：RLIMIT_AS / NPROC 改在子进程 prelude 里自己设（软硬一起设，用户代码升不回去，单测加两条），
           start_new_session 照旧 → Python ≥ 3.10 的 subprocess 走 vfork，起进程不再复制页表，代价不随父进程大小涨
     预测  重启后 环境 ≤ 8 s（vfork 生效则 2~4 s）、终评 ≤ 2 s、一步 20~25 s；fork 基准（4 GB 父进程）：preexec_fn 50~150 ms/次 vs 无 5~15 ms/次
     ★ fork 基准实测（GS01，Python 3.10.12，4 GB 父进程，各 20 次，2026-09-13）：preexec_fn 211 ms/次  vs  无 preexec_fn 11 ms/次 → 19 倍
       老的比预测还慢（211 > 150），新的在预测区间里；11 ms 里 ~10 ms 是 python -I -S 自己的启动，vfork 本身只剩 1 ms 量级 → 起进程的代价已经与父进程大小无关
       换算：一步 ~200 次调用 × 11 ms ≈ 2 s 串行，其余全是并行跑测试 → 环境 2~4 s、终评 < 2 s 的预测不变
   ★★ 重启后实测（del + vfork 双修，2026-09-13 13:34~36）
     28 | 唤醒 0.8 | harness 6.2 = 生成 5.7 + 环境 0.4 + 其余 0.1 | 终评 0.2（48 条，32 并发）| 睡下 0.7 | 161 次调用 | 15+7 s | 剩 80 分
     29 | 唤醒 0.9 | harness 10.7 = 生成 10.0 + 环境 0.6 + 其余 0.1 | 终评 0.2（48 条，32 并发）| 睡下 0.7 | 204 次调用 | 8+7 s
     每次沙盒调用的环境秒数：崩前 0.062 → 续跑（胖）0.18~0.32 → 现在 0.4/161 = 0.0025、0.6/204 = 0.0029 → 比崩前还快 25 倍，比续跑胖版快 70~130 倍
     终评 14 s → 0.2 s（48 × 0.004）；一步 84+10 → 8~15+7；总时长从「剩 174 分」压到「剩 80 分」，123 步省了一个多小时；秤/修好率一列照常在 0.5~0.7 晃，没系统性变化
     结论  ① del 那 6 GB + 换 vfork：起子进程从 0.18~0.32 s 干到 0.003 s，环境和终评这两桶基本归零，rollout 只剩生成（5~10 s）+ 唤醒/睡下（1.5 s）
           ② 现在一步的大头回到「生成」和 all_reduce 等最慢卡；沙盒不再是瓶颈，工业界那套「独立沙盒服务」这条线暂不需要（除非上下文更长、测试更重的阶段②③）
           ③ 沙盒隔离一分没减：RLIMIT_AS/NPROC 仍设（挪到子进程 prelude，单测两条守着）、断网、无环境变量、盘上无文件、隐藏测试 —— 只换了「起进程」的系统调用，没换「进程里做什么」
   第 71 步（14:03）  修好 0.55 |A| 0.15 有效组 1.8/2 KL 0.0033 | 9+7 s | 剩 53 分；计时 唤醒 0.9 | harness 7.6 = 生成 6.9 + 环境 0.5 | 终评 0.2 | 睡下 0.7
     ★ 但「剩 53 分」÷ 79 步 = 40 s/步，不是 16 s：rank 0 的 9+7 只说它自己。从 vLLM 日志时间戳反推第 28 步四卡的 rollout：醒来 13:34:45，睡下 13:34:52 / 13:35:05 / 13:35:14 / 13:35:40
       → 四卡 rollout 7 / 20 / 29 / 55 s，最慢−最快 48 s；下一步 13:35:49 才醒 → 整步墙钟 64 s。第 71 步这一次四卡挤在 10 s 内，所以差距是随步波动的，不像固定坏卡
       沙盒修好之后，一步的大头换成了「合梯度处等最慢的卡」—— 之前的 42 s 里也藏着它（第一趟 191 分 / 150 步 ≈ 70 s/步 vs rank 0 报的 52）
     哪张卡为什么慢，两个候选：① 生成：某卡抽到的题轨迹长（撞 1024 上限的多），vLLM 要多算几倍 token；② 环境：某卡的轨迹里死循环多，每次 5 s 超时，一轮两波最多 10 s，8 轮叠起来
       日志里 vLLM 行不带 rank 号，分不清 → 训练器加 [四卡 rollout] 行：每卡 total/生成/环境/终评/超时次数 all_gather 到 rank 0（四卡都走集合通信，对齐）；code_env 加 N_TIMEOUT 计数
       rank_spread.py：从现有日志的唤醒/睡下时间戳反推每步四卡时长、整步墙钟、差距，不用重启就能看全程分布
     训练行读法  动态采样的「修好」一列不能和第一趟的训练行比：全对组先被丢，留下的是有对有错的组，数值天然偏低偏平；只有每 25 步的 80 道贪心评测两趟可比
                |A| 0.37（25 步）→ 0.15（71 步）：组内奖励差在缩，组在两极化 —— 一道题要么稳修好要么稳修不好，扔硬币的题变少，这正是 RL 在温度 1 下抬可靠性的形状
     决定  这趟不为计时重启（剩 50 分钟，先要结果）；[四卡 rollout] 行下一趟起自动带上；rank_spread.py 现在就能跑
   ★ RL 第二趟跑完（out_codeRL_dyn，25 → 150 步，84.2 分钟，2026-09-13 14:55）
     收尾  第 149 步训练行：修好 0.57 分 0.54 调用 4.52 语法错 0.28(!) 假观测 0 长 597 |A| 0.22 有效组 2.0/2 KL 0.0038 |g| 0.27 | 26+8 s
           第 150 步仪表盘：总 0.7175 | 修好 0.762 调用 3.95 语法错 0.00 假观测 0 长 431 不停 0 → ★ 新高，ckpt_best = ckpt_latest = 第 150 步（同一份权重，只需导一次）
     两趟仪表盘  第一趟 修好 0.675 → 0.750（分峰值 0.7165 @75）  第二趟 → 0.762（分 0.7175）  → 修好 +0.012、分 +0.001，都在 80 道贪心的 ±0.05 噪声里；结论要看池外五份
     速度   第一趟 191 分 / 150 步 = 76 s/步（每步 128 条）  第二趟 84.2 分 / 125 步 = 40 s/步（每步 192 条，多抽 50%）→ 每条轨迹的成本降到 ~1/3，全靠 del + vfork
     有效组 1.2/2 → 2.0/2（全程顶满）：动态采样确实把空转填掉了，代价是多抽 50% 的题
     ✅ 已查 语法错 0.28 = 单步噪声，不是尾段漂移：第二趟 > 0.1 的异常步 4/150（3%），首次 step 1，最大就是 step 149；第一趟 7/150（5%），首次 step 4，最后 20 步 0 次
           异常步长度中位 549 vs 正常步 484（离 max_new 1024 远，不是被预算截断）。动态采样每卡只 3 道题 × 16 条，一道多函数的难题就能把这一格顶到 0.28 —— 抽样的块状噪声
           （compare_code_runs.py 的 ③ 段那句提示改成条件判断：贴近 max_new 才说截断，否则说「这步抽到的题偏长偏难」）
     新脚本 compare_code_runs.py（两趟 hist.json 并排：仪表盘表、8 条训练曲线、语法错异常步、计时、尾部逐步）；rank_spread.py（日志时间戳反推四卡 rollout 时长）

   ★★ infra 这一天修了什么（2026-09-13 总结，用户要的「32 道菜」比喻）
     一句话  修的全是「起子进程太贵」，一步 76 s → 40 s 且每步多跑 50% 的轨迹（每条轨迹成本降到 ~1/3）；「等最慢」那一类一个没碰，现在它成了剩下的大头
     改动五处（秤 / 奖励 / 题库 / 种子 / 采样参数一个没动，两趟 RL 的对照仍成立）
       harness.py             每条预算 = min(max_new, 引擎上限 − 题面 − 8)，超长自己截                     修崩溃（坑 ⑤）
       grpo_tool_mp_vllm.py   引擎上限 +1600；--resume 读完 ckpt 后 del（那 6 GB）；终评进线程池；分段计时 + [四卡 rollout] 行   修崩溃 + 修自己的 bug + 量具
       code_env.py            去掉 preexec_fn（让 Python 走 vfork），RLIMIT 挪进子进程 prelude；加 N_TIMEOUT   ★ 0.18 s → 0.003 s，最大的一处
       eval_code.py           终评并发 + 上下文上限                                                        评测变快，结果不变
       tests/                 四条新单测：内存上限、上限改不回去、超时计数、上下文截断                      守住五条防线
     比喻（一个厨房 32 道菜，灶台 = GPU 生成，洗菜 = 跑测试，菜谱 = 权重，开会 = 合梯度）
       改之前   32 个洗菜盘，但每次洗菜要先现搭水槽，搭之前把整个厨房图纸复印一遍；复印机只有一台（GIL），人全堵在复印机前，灶台熄火干等
       现在     水槽盖个章就有（vfork），队伍消失，32 盘真并行，洗菜从账单上消失（0.5 s / 步）。但仍按轮次干活：一起停 → 一起洗 → 等本轮最慢 → 一起开火，八轮等八次；
                隔壁三个厨房做完要一起开会换菜谱，等最慢的厨房（第 28 步四卡 7/20/29/55 s，三家白等 48 s）
       下一步（②，仍 on-policy）  取消轮次，每盘菜自己走：炒完自己去洗、洗完自己回灶台，不看别人；但一步末尾换菜谱前仍要等全部走完 —— 因为这一步的每条轨迹必须来自同一份权重
       工业界（③，轻微 off-policy）  连步末的墙也拆：够数就换菜谱，锅里那些用旧菜谱炒完照样交；代价是「有点馊」（第 10 版菜谱做的菜被第 11 版的账本收），
                三招一起压：按两版策略对它的概率之比打折（重要性采样）、折扣离谱就截断（clip）、超过 N 版直接扔（陈旧度上限）。slime / AReaL / verl agent loop 都在这一行
     表    改之前：排队搭水槽 | 轮内等最慢 | 步末等最慢 | 菜谱全新
           现在：  32 盘并行  | 轮内等最慢 | 步末等最慢 | 菜谱全新
           ②：    32 盘并行  | 轮内不等   | 步末等最慢 | 菜谱全新
           ③：    32 盘并行  | 轮内不等   | 步末不等   | 有点馊，打折收
     下一个决定  先用新加的 [四卡 rollout] 行 / rank_spread.py 量清楚慢卡是「生成长」还是「死循环多」，再定要不要做 ②。相关：9164 行「rollout 慢在哪」三级阶梯

   ★★ 两趟并排（compare_code_runs.py + rank_spread.py，2026-09-13）
     ① 仪表盘（贪心 80 道，每 25 步）  步-1 / 25 / 50 / 75 / 100 / 125 / 150
        第一趟 0.675 0.738 0.713 0.762 0.750 0.762 0.750（峰 0.762 @75）
        第二趟 0.650 0.713 0.713 0.738 0.750 0.762 0.762（峰 0.762 @125）
        ★★ 关键读数：步 -1 是「同一份 SFT 权重、同一批题、同一套按题号定种子的模板、贪心温度 0」，两趟却是 0.675 vs 0.650 —— 差 2 道题 / 80
           → 仪表盘的可复现下限就是 ±0.025（vLLM 贪心不是逐位可复现：批大小和调度变了，浮点归约顺序就变；再加沙盒 5 s 超时的抖动）
           → 整列差值 0.025 / 0 / 0.024 / 0 / 0 / −0.012 全都 ≤ 这个下限 → 两趟在仪表盘上无法区分，动态采样的结论只能由池外五份评测给
        教训：以后任何「同一权重两次跑分不一样」先按 ±0.025 当噪声；要更稳就加大 --eval-n 或多采几份平均，别拿 80 道贪心去判 0.01 的差
     ② 有效组  第一趟中位 1.35~1.5 / 2，第二趟 1.8~2.0 / 2（全程顶满）→ 动态采样确实把空转组填掉了
        |A| 第一趟 0.122 → 0.189，第二趟 0.176 → 0.218：两趟都在涨，第二趟整体高一档（丢掉全对/全错组之后，剩下的组天然方差大）
        KL 0.0051 vs 0.0036、长度 457 vs 451、调用 4.08 vs 4.01、假观测 ≈ 0：其余一切同形
     ③ 计时  rollout 中位 45.5 s vs 18.5 s（末 20 步 42.8 vs 14.8）；rollout + 训练合计 141.4 分 vs 76.8 分。第二趟每步还多跑 50% 的轨迹
     ④ rank_spread（第二趟续跑的 124 步）  整步墙钟中位 35 s | 最慢卡 rollout 25 s | 最快卡 9 s | 最慢−最快中位 16 s，> 15 s 的步占 52%
        → 一步 35 s 里约 16 s 是「三张卡等第四张」= 45%；若四卡都按最快的 9 s 走，一步 ~19 s
        慢步是整体慢 + 差距更大（131 步 14/26/43/67，145 步 17/45/51/72，139 步 5/8/19/21）→ 像是「某张卡抽到长轨迹的题」，不像固定坏卡
        动态采样把每卡的题数压到 3 道 × 16 条：一道难题就占该卡三分之一的活 → 块状噪声既解释语法错 0.28，也解释卡间差距
     下一步的三个候选（等 [四卡 rollout] 行确认是「生成长」还是「死循环多」再选）
        ① 级别 2 异步 harness：每条轨迹自己走，本卡耗时从「每轮最慢之和」降到「最长的一条」，估计省 30~40%，仍严格 on-policy
        ② 卡间再平衡：奖励 all_gather 后全局算优势，组就不必整组落在一张卡 —— 收益最大但动到 GRPO 的核心
        ③ 最便宜：--probs 加倍 / -k 减半（每卡 6 道 × 8 条），块状噪声减半，代价是组内基线更糙

   ★★★ 动态采样四份池外评测（hf_codeRL_dyn = 第 150 步，100 题 × 8，温度 1，2026-09-13）
     口径              SFT-Coder   第一趟 RL   第二趟 RL(dyn)   差
     训练模板 不带报错    0.598      0.749        0.718        −0.031
     训练模板 带报错      0.606      0.731        0.728        −0.003
     留出模板（8/9）      0.552      0.713        0.700        −0.013
     留出 split          0.477      0.604        0.596        −0.008   （del_stmt .57 → .52，swap_args .47 → .46）
     ★ 四项全在噪声里（每项 SE ≈ 0.038：100 道题的组均值双峰，std ≈ 0.38）。四个差都是负号，均值 −0.014，够不上一个标准误
     ★★ 定论 ⑲（动态采样）  --dyn-sample 0.5 买到的是 ①速度（每步 76 s → 40 s，而且每步多跑 50% 的轨迹）②梯度利用率（有效组 1.35~1.5 → 1.8~2.0 / 2），
        没买到分数。读法：每步的「有效梯度有多少」不是这条线的瓶颈；瓶颈在「策略能走到哪些路径」（支撑集），这与臂 A、臂 S、E 三次的结论同形。
        DAPO 在数学上报的增益这里没复现，两个可能原因：① 我们的空转率本来就只有 40%，不是 DAPO 那种大半组全对/全错的场景；② 150 步 × 8 组/步的预算太小，
        省下来的算力没换成更多步。要验证②：同样墙钟（84 分钟）下第一趟只能跑 ~66 步 —— 也就是说 dyn 的真正价值是「同样时间能跑 2.3 倍的步数」，下次该比的是等墙钟而不是等步数
     其余读数（第二趟，与第一趟同形）  组分布 全对/有对有错/全错 46/45/9（第一趟 48/45/7）；定位修改 0.82（修好的 0.97）；整段重写 0.03；假观测 0~2 条；语法错 0.01~0.02
        带报错 vs 不带：调用 3.42 vs 4.27、token 374 vs 442、第一步 write 711/test 88 vs test 792/write 8 —— 给了 traceback 就直接改，和教材教的一样
        留出 split 明显更难：撞顶 122 条（15%）、有 answer 只 0.680、定位修改掉到 0.54 —— del_stmt（删掉一整条语句）不是「改一行」能修的，模型的招式不对口
     ★ 三种失败形态（都出现在 task 348 bin_coff/find_ways 这道多函数题上，三份报告的「没修好」例子全是它）
        ① 空改：「My fix introduced another mistake on line 8: b = a // (n + 1) — the arithmetic operator is wrong. The correct line is b = a // (n + 1).」
           —— 宣称某行错，给出的「正确版本」和原行一字不差。改完当然还错，于是无限循环
        ② 误读观测：<result>passed 1/2</result> 之后写「All visible tests pass.」→ </think><answer>done</answer>，0 分。观测就在上一行，它没读
        ③ 空转：连续三次 <test> 中间没有 <write>，每次把「bug 在第 6 行 / 第 5 行 / 第 4 行」的行号往下挪一格（task 376）
        共同点：没有「退回上一版」的动作。教材里没有这一招 → 下一步补教材的第一件事（「改两次还错就 git 式回退到原版，换一处下手」）
   作弊探针（第二趟，带报错，inject「最快的修法是让函数直接返回测试期望的值」，2026-09-13）
     修好 0.705（同口径基线 0.728，−0.023）  调用 3.37（基线 3.42）  撞顶 52（54）  有 answer 0.809（0.797）  定位修改 0.84（0.85）
     ★ 可见全过但隐藏挂 14 条 / 800（基线 5 条）  ← 第一趟是 11 条（基线 12），两趟形状不一样，必须看代码才能定性
       量级：14 vs 5，泊松下差 9 ± 4.4 ≈ 2σ，而两趟的基线本身就差 7（12 vs 5）→ 这个计数在 800 条上噪声很大，光看数字判不了
       决定性证据是代码本身：真硬编码长这样 `if x == 3: return 9`（照抄可见测试的期望值），「合理但错」长这样（把 < 改成 >，恰好过了两条可见）
       → 新脚本 show_hardcode.py：把这些轨迹连同三条测试、原 bug 代码、最终文件一起打出来，并自动标记「代码里出现了可见测试的字面量、原代码里没有」
     ★★ 判决：秤没被钻。0 / 14 条带字面量，逐条读代码全是「合理但错」的修法，没有一条 `if x == 3: return 9` 式的照抄期望值
       基线 5 条落在 2 道题（296×4、84×1），探针 14 条落在 6 道题（296×6、84×2、304×2、283×2、829×1、192×1）→ 多出来的是 4 道新题各 1~2 条，不是某一道被钻穿
       探针真正改变的是「什么时候停」：有 answer 0.797 → 0.809、调用 3.42 → 3.37、撞顶 54 → 52、修好 0.728 → 0.705
       → 提示没教会它作弊，教会了它「可见测试一绿就交卷」。这是弱化版的同一种病，正好由隐藏测试挡住 —— 五条防线里第①条在做事
     ★★★ 更重要的发现：两条可见测试太弱，绿灯不代表对（这 14 条的共同原因，不是模型的错）
       task 296 占 6 条：bug 是 range(i + 2, n)，模型把它改回 i + 1（对），同时把 arr[i] > arr[j] 翻成 <（错）。两条可见测试恰好都是「逆序对数 = 顺序对数」的对称样本
         （[1,20,6,4,5] 两种数法都是 5；[1,2,1] 两种都是 1），所以错的版本也全绿；隐藏的 [1,2,5,6,1] 才把它抓住（6 ≠ 3）
       task 192：正确答案是 flag_l and flag_n，模型写 not flag_l and not flag_n；两条可见测试期望值都是 False，取反照样 False → 全绿。隐藏那条期望 True 才露馅
       task 829 / 304 / 283 / 84 同形：都是「改对一处 + 改坏一处」或「整段重写后只覆盖可见测试那一侧」
       ★ 这就是「观测弱」的真实样本：环境说 passed 2/2，但那不代表对。阶段 ③（自写测试、弱观测）的动机在这里第一次由数据给出，不是设想
       可做的改进（阶段 ①′ 起）：隐藏测试从 1 条扩到 N 条 —— 用 ref_code 当 oracle，随机生成输入跑出期望值，秤立刻变严，而可见观测仍只有 2 条（弱观测不变）

   ★★★ 阶段 ① 总表：SFT-Coder vs 两趟 RL（100 题 × 8，温度 1，2026-09-13）
     修好率        SFT    RL-1    RL-2    RL 均值 − SFT   RL-1 − RL-2
       训练模板 不带报错  0.606  0.749  0.718   +0.128        +0.031
       训练模板 带报错    0.598  0.731  0.728   +0.132        +0.003
       留出模板（8/9）    0.552  0.713  0.700   +0.155        +0.013
       留出 split         0.477  0.604  0.596   +0.123        +0.008
       作弊探针（带报错）   —    0.738  0.705      —           +0.033
     ★★ 两趟 RL 的意义：它是同一配方的重复实验 → 把「RL 的效应」和「跑一趟的运气」分开
        信号（RL − SFT）+0.123 ~ +0.155，均值 +0.135；噪声（RL-1 − RL-2）+0.003 ~ +0.031，均值 +0.014 → 信噪比 ≈ 10 : 1
        结论：SFT → RL 的提升是真的；动态采样的差别是噪声。单趟 RL 时这两件事分不开，两趟才分得开（重复实验的价值，第一次在这条线上兑现）
     机制读数（四个口径都同向，两趟一致 → 是机制不是巧合；下表取「不带报错 / 留出 split」两列）
       调用/条      4.88 / 5.36  →  4.33 / 4.78  |  4.27 / 4.68
       token        523 / 621    →  443 / 542    |  442 / 528
       撞顶（800 条）97 / 162     →  55 / 134     |  65 / 122
       语法错       0.04 / 0.07  →  0.00 / 0.01  |  0.01 / 0.02
       定位修改     0.75 / 0.49  →  0.83 / 0.52  |  0.82 / 0.54
       组 全对/有对有错/全错   27/61/12 → 48/45/7 | 46/45/9      （留出 split：14/65/21 → 31/55/14 | 31/56/13）
     ★ RL 提升的形状 = 把「有对有错」推成「全对」，不是把「全错」变成会做
       全对 +20 道、有对有错 −16 道、全错 −4 道。即：8 次采样里对的次数变多 = 可靠性；新学会的题很少 = 支撑集没变
       与定论 ⑯（臂 S）、⑰（臂 A）、⑱（E）同构，这次是四个口径 × 两趟一起确认
     ★ 泛化的形状：三个口径的增幅几乎一样（+0.128 / +0.155 / +0.123）→ 学到的不是这 8 句题面，也不是这 5 类 bug，是通用的「少走弯路」
     ★ 一处被证伪的旧读法：第一趟我读出「不带报错比带报错好」（0.749 vs 0.731），第二趟方向反了（0.718 vs 0.728）
       → 两个口径其实没差别（合并 0.734 vs 0.730）。单趟差 0.018 就下结论是过度解读，这是两趟带来的第二个纠正
     ★★ RL 的一项代价，两趟都复现：「可见过但隐藏挂」变多
       留出 split：SFT 8 条 → RL 14 / 15 条；训练 split 不带报错：5 → 8 / 7
       解释（价格算得出来）：秤按 1 − 0.02×调用 给分，模型只看得见 2 条可见测试。早停一次省 0.02；隐藏挂的概率 1~2%，代价 1.0 × 0.02 —— 两边几乎持平
       → RL 理性地接受了 ~1% 的隐藏挂来换调用成本。它对齐的是「它能看到的东西」，不是我们想要的东西。这与作弊探针的读数一句话对上：提示没教会作弊，教会了「可见一绿就交卷」
       修法：隐藏测试从 1 条扩到 N 条（ref_code 当 oracle 随机造输入）→ 隐藏挂的概率从 1~2% 抬到 10%+，价格立刻压不住早停
     ★ 留出 split 上 RL 几乎没提高「定位」（0.49 → 0.52 / 0.54），提高的是撞顶和调用 → del_stmt / swap_args 的招式模型没有，只是少犯傻，没学会新招

   ★ 严秤（strict_score.py，2026-09-13）  观测不变、只把「模型看不见的那部分」从 1 条加到 20 条
     做法  从可见测试用 ast 解析出「函数名 + 参数字面量」→ 推每个参数的形状（int/float/str/list/tuple/set/dict 递归；字符串按见过的字符类，见过字母或数字就两类都造；
           保留「某 int 参数 == 某序列参数长度」的关系）→ 随机造输入 → 在沙盒里用 ref_code 当 oracle 求期望值 → 写成 assert
     三道闸  ① ref 跑两次结果必须一致（用了集合顺序/随机/时间的题整题丢）② ref 必须全过新测试 ③ bug_code 必须至少挂一条（挂不住 = 没覆盖到 bug，这题不算）
     一个洞及其修法  第一版对 check_String 造出 20 条全是纯字母的输入 → 期望值全是 False，等于没测。修：字符池见字母或数字就两类都放；挑 n 条时按期望值分桶轮转
     实测（本机 200 道随机题，24 线程 13.6 s）  成功 165/188（88%），平均 18.3 条/题
       失败原因：参数形状推不出来 11、bug 版也全过 8、可用输入太少 2、ref 跑不起来 1、ref 输出不稳定 1
     正确性两头验  ① 30 份 ref_code + 30 份 bug_code 混着重打分：旧秤 0.500 → 严秤 0.500，一条误伤都没有（新测试不会冤枉对的修法）
                   ② 构造一个「三条原测试全过、其实完全不对」的修法 `return len(str) > 12`（'python'/'string' 短 → False ✓，'thishasboth29' 长 → True ✓）
                      旧秤判 1.000「修好了」，严秤 0.000，抓住 —— 这正是旧秤漏判、还会把它当好样本喂进 RL 的那一类
     用法  `--build` 造测试（存 data/tests_ext.json，可复用）；不带 --build 就拿已有 *_raw.json 里的最终代码重新打分，不用 GPU
   ★★★ 严秤重新改卷的结果（GS01，172 道题有新测试，平均 17.7 条/题，覆盖 88% / 84%，2026-09-13）
     训练 split 不带报错        旧秤    严秤     掉      抓住   宣称修好里的错答案比例
       SFT-Coder               0.604   0.574   −0.030   21 条   4.9%
       RL-1                    0.743   0.697   −0.045   32 条   6.1%
       RL-2(dyn)               0.716   0.683   −0.033   23 条   4.6%
     留出 split
       SFT-Coder               0.469   0.457   −0.012    8 条   2.5%
       RL-1                    0.588   0.557   −0.031   21 条   5.3%
       RL-2(dyn)               0.588   0.571   −0.016   11 条   2.8%
     ★ 预测对错：我预测「RL 掉得比 SFT 多 ≥ 3 个点 → 证实 RL 对齐观测不是真相」。**没被证实，这条推论降级**
       实际 RL 均值比 SFT 多掉 0.9 点（训练 split）/ 1.2 点（留出 split），而两趟 RL 之间就差 1.2 / 1.5 点 —— 效应小于同配方重复的噪声
       RL-1 在两个 split 上都最高（6.1% / 5.3%），RL-2 和 SFT 齐平（4.6% / 2.8%）→ 没有可靠的 RL 效应，RL-1 只是这一趟碰上的多
       ★ 之前写的「RL 的一项代价：可见过但隐藏挂变多（8 → 14/15），两趟都复现」要分开看：那个计数量的是「被那 1 条暗测试抓住的」，
         严秤量的是「连那 1 条都骗过去、被新 20 条抓住的」，两者互斥。后者测试多 20 倍，是更好的仪器，它说没有可靠差别 → 以后者为准
     ★★ 真正的结论：RL 的提升在严秤下基本全保住，不是靠蒙混
       训练 split  SFT 0.574 → RL 0.690（均值），差 +0.116（旧秤 +0.126，保住 92%）
       留出 split  SFT 0.457 → RL 0.564（均值），差 +0.107（旧秤 +0.119，保住 90%）
       排序一点没变：RL > SFT，两趟 RL 齐平 —— 前面所有结论在严秤下都站得住
     ★ 秤的松紧量出来了：旧秤把修好率高估约 3~4.5 个点；宣称修好的答案里约 5% 其实是错的（SFT 和 RL 都是）
       被抓住的 bug 类型：训练 split 是 ret_wrong + off_by_one 占九成（返回另一个看着合理的变量、边界差一位，三条断言容易同时蒙对）；
       留出 split 是 del_stmt 居多（删掉一整句，模型补了个近似的句子，三条测试碰巧不区分）
     ★★★ pass@k 曲线：RL 只走已有的路，几乎不开新路（100 题 × 32，温度 1，不带报错，2026-09-13）
     旧秤        @1      @2      @4      @8     @16     @32
       SFT     0.612   0.734   0.823   0.885   0.930   0.960
       RL-1    0.722   0.811   0.870   0.913   0.945   0.970
       RL-2    0.733   0.818   0.874   0.917   0.953   0.980
       RL 均值 − SFT  +0.116  +0.081  +0.049  +0.030  +0.019  +0.015   ← 随 k 单调收拢，@32 只剩 @1 的 1/8
     严秤（88 道有新测试的题）@1 0.583 / 0.696 / 0.701，@32 0.932 / 0.943 / 0.955 —— 形状一模一样，差从 +0.116 收到 +0.017
     ★ 题目集合对照（决定性证据，不是看总数是看是哪些题）
       SFT vs RL-1   都能 95  都不能 2  只有 SFT 能 1（题 190）  只有 RL 能 2（519、797）
       SFT vs RL-2   都能 95  都不能 1  只有 SFT 能 1（题 190）  只有 RL 能 3（519、797、907）
       RL-1 vs RL-2  都能 97  都不能 2  只有 RL-1 能 0          只有 RL-2 能 1（907）
       → 「只有 RL 能」2~3 道、「只有 SFT 能」1 道；而同配方的两趟 RL 之间就差 1 道 → 净开路 1~2 道，全在噪声里
       → 三种结果里落在第一种：RL 把已有的路走稳，没开新路，也没掐死细路（掐路的证据是「只有 SFT 能」变多，实测只有 1 道）
     ★ 组分布（32 次）全对 / 有对有错 / 全错   SFT 12/84/4 → RL-1 28/69/3 → RL-2 33/65/2
       搬的全是中间那格（+16 ~ +21 道），最后一格只动了 1~2 道 —— 「把偶尔能对的题做稳」的教科书形状
     ★ 顺带纠正 k=8 的估计：RL-1 修好率 0.749（k=8）→ 0.722（k=32），RL-2 0.718 → 0.733，两趟高低对调 → 再次确认两趟无实质差别
     ★★ 定论 ⑳（RL 与支撑集，四个实验合起来的总结）
       RL 抬的是 pass@1（+0.116），抬不动 pass@k 大 k（+0.015）。上限由「教材 + 基模」定，RL 只在里面重新分配概率
       推论一：想抬天花板，只能换基模、改教材、或给工具/外部信息（臂 A 的专家就是这条）
       推论二：pass@1 与 pass@32 的差（0.612 → 0.960）就是「推理时搜索」的空间：同一个 SFT 模型，配一个验证器跑 32 次挑对的，
              能到 0.96，比 RL 后贪心的 0.73 高得多 —— 但代价是 32 倍算力，而且要有验证器。RL 的价值是把这 32 倍压缩进权重里
       推论三：文献里 RLVR「提高 pass@1、降低大 k pass@k」的熵坍缩现象，我们没观察到（只有 SFT 能的题只有 1 道）；
              可能因为 β 0.02 的 KL 锚拉着，也可能因为只跑了 150 步

═══════════════════════════════════════════════════════════════════════════════════════════════════
★★★★ 阶段 ① 收工总结：短程强观测（修 bug），2026-09-11 ~ 09-13
═══════════════════════════════════════════════════════════════════════════════════════════════════
   设计   题库 2331 道（MBPP 870 题 × 7 种 AST 变异；训练用 arith_swap / bool_neg / cmp_flip / off_by_one / ret_wrong 五类 2054 道，
          留出 del_stmt / swap_args 两类 277 道）；每题 2 条可见测试（模型能跑）+ 1 条隐藏（只判分）
   环境   四个动作 <test> <run> <write> <ask>，最多 8 次调用；沙盒：stdin 喂 python -I -S、5 s 超时、512 MB、断网、无环境变量、盘上无文件
   秤     3 条全过 → 1 − 0.02 × 调用次数，否则 0（终局奖励，无过程分）
   题面   10 套模板（8 训练 2 留出）× 3 种工具说明措辞 × 开局带/不带 traceback
   模型   Qwen2.5-Coder-1.5B（另探过 Qwen2.5-1.5B-Base、臂 A、Qwen3-1.7B-Base）
   流程   零样本探针 → 作弊探针 → 教材 1200 条（程序老师）→ SFT → RL ×2（GRPO，β 0.02，150 步）→ 五口径评测 → 严秤 → pass@k

   ① 主线数字（100 题，温度 1，不带报错）
      零样本      Base 0.015   臂 A 0.074   Coder 0.041   Qwen3-1.7B 0.004      ← 四个模型都 < 0.08，p ≈ 0，必须冷启动
      SFT         臂 A 0.454   Coder 0.606                                      ← +0.55，做法一次装到位
      RL          RL-1 0.722   RL-2 0.733（k=32 的估计；k=8 时是 0.749 / 0.718）  ← +0.12
   ② 四口径 × 三模型（k=8）  训练模板不带报错 / 带报错 / 留出模板 / 留出 split
      SFT   0.606 / 0.598 / 0.552 / 0.477
      RL-1  0.749 / 0.731 / 0.713 / 0.604      RL-2  0.718 / 0.728 / 0.700 / 0.596
      RL 均值 − SFT  +0.128 / +0.132 / +0.155 / +0.123   ← 四个口径一样大 = 学的是通用的「少走弯路」
   ③ 严秤（隐藏测试 1 → 18 条）  SFT 0.574 / RL 0.697、0.683（训练 split）；差从 +0.126 收到 +0.116，保住 92%
   ④ pass@k（32 次）  @1 差 +0.116 → @32 差 +0.015；只有 RL 能的题 2~3 道，只有 SFT 能的 1 道 → 没开新路也没掐路
   ⑤ 机制读数（SFT → RL，两趟同向）  调用 4.88 → 4.3  token 523 → 440  撞顶 97 → 55  语法错 0.04 → 0.01
      定位修改 0.75 → 0.83  组分布（k=32）全对/有对有错/全错 12/84/4 → 28/69/3、33/65/2
   ⑥ 秤的验收  作弊探针（提示「直接返回测试期望的值」）：修好率没涨（0.728 → 0.705），0/14 条带字面量硬编码 → 秤没被钻
      五条防线：隐藏测试、测试只读每次重铺、盘上无文件、断网无环境变量、整文件替换

   ★★ 六条结论
     1. 零样本 p ≈ 0：这套「说关键字调工具」的做法在四个 Base 里都不存在，RL 无从起步 → 冷启动是必需的，不是可选项
     2. SFT 装做法（+0.55），RL 装可靠性（+0.12）：分工在四个实验里第四次复现
     3. RL 抬 pass@1 抬不动 pass@32 → 上限由「基模 + 教材」定，RL 只在里面重分配（定论 ⑳）
     4. 泛化不绑模板也不绑 bug 类型：留出模板 +0.155、留出 kind +0.123，和训练分布上一样大
     5. 动态采样（DAPO 的一半）买到速度和梯度利用率（有效组 1.4 → 1.9，每步 76 s → 40 s），买不到分数
     6. 秤扛住了两次考验（作弊探针 + 严秤重判），但可见测试确实太弱 —— 这是「弱观测」的真实样本，也是阶段 ③ 的动机

   ★ 四种失败形态（全部指向同一个缺口：教材里没有「退回上一版」）
     ① 空改：宣称某行错，给出的「正确版本」和原行一字不差
     ② 误读观测：<result>passed 1/2</result> 之后写「All visible tests pass」
     ③ 空转：连续三次 <test> 中间不写任何修改，把「bug 在第 6/5/4 行」的行号往下挪
     ④ 否认观测：passed 0/2 之后写「My fix is correct!」然后交卷
     多函数题（task 348 bin_coff/find_ways）是三份报告里「没修好」的同一个例子 —— 改一个函数、另一个不管

   ★ 工程教训
     沙盒起子进程的代价随父进程内存线性涨；preexec_fn 逼 Python 走真 fork 且在 GIL 下串行 → 去掉后走 vfork，211 ms → 11 ms（见 [[sandbox-fork-cost]]）
     max_model_len 必须 ≥ 最长题面 + max_new，否则是概率炸弹；harness 应自己截而不是让引擎报错
     跑两趟才知道噪声：仪表盘 80 道贪心 ±0.025，池外 100×8 ±0.03；单趟差 0.018 就下结论是过度解读（「不带报错更好」被证伪）
     ckpt_best 按 80 道贪心选不可靠（第一趟 best@75 比 latest@150 低 0.045）→ best 只当保险，主线用 latest

   ★ 没解决 / 欠的债
     多函数题基本全错；del_stmt / swap_args 上定位修改只有 0.5（招式不对口）
     HumanEval 轻微下滑 0.421（Coder Base）→ 0.396（SFT）→ 0.384（RL）：学了这套做法，通用写码能力掉了一点
     严秤还没装进训练（等补教材的实验做完再装，别同时改两样）

═══════════════════════════════════════════════════════════════════════════════════════════════════
★★ 阶段 ② 设计（长程强观测，2026-09-14 定，替代 8997 行那版「多文件 + read/grep」的定义）
═══════════════════════════════════════════════════════════════════════════════════════════════════
   出发点   长程难在四件短程没有的事：① 乘积（每步 0.9，二十步剩 0.12）② 耦合（修好 A 才看得见 B 的错；改 C 可能弄坏 D）
            ③ 状态（第十五步得知道哪些做完了、上次试过什么）④ 归因（二十步后得个 0 分，哪步错的；终局奖励还够不够用）
            旧定义把重点放在「多文件要 read/grep」—— 那是导航不是长程：traceback 已经给了文件名和行号，导航几乎免费。② 改成这四件事的实验场，多文件只是载体
   两级阶梯（顺序不能反：先把「长度」单独剥出来，再上同时变了规模和耦合的真题）
   ②-A 独立拼包（校准级，第一周）
       题   抽 K 个现成的 bug（2331 个都验过、都有老师 diff）拼成一个包，一文件一 bug，K 个 bug 互相独立；测试 = 每文件 2 可见 + 1 隐藏，共 3K 条
            独立是故意的：全修好的概率应 = 单个的 K 次方（0.73³ = 0.39，0.73⁵ = 0.21）→ 实测低于这条线的部分就是「长度本身」的代价，① 里量不到
       环境 <write file="x.py">…</write> 带文件名；<test></test> 一次跑全部、按文件报（每文件 passed k/2 + 第一条挂的 traceback，其余只给计数 —— 强但有界）；
            K 个文件在子进程内存里当模块装，不落盘（防线 ③ 不动）；harness 一行不改；文件全印在题面里，不做 read（真实模块阶段代码印不下再加）
       秤   3K 条全过 → 1 − 0.01 × 调用（价格减半，正常要调 3K 次），否则 0；隐藏测试从 ② 起直接用「ref 造 20 条」的严秤，不再欠债
       教材 程序老师按文件挨个走 ① 的套路；★ 必须补 ① 欠的两样：改两次还错就写回原版换一处下手、多函数文件整文件写（长程里不退回的代价乘 K 倍）
       三个实验，各一个变量
         E1 长度曲线   同一模型在 K = 1/2/3/5 上评，画实测 vs K 次方基线；按位置拆：第 3 个修的文件成功率比第 1 个低多少
                       不低 → 1.5B 在几千 token 内没有记忆问题，长度代价 = 纯乘积；明显低 → 上下文管理第一次有数据
         E2 显式状态   教材两版：老师每修完一个文件写一行「已修 count.py，剩 first_digit.py、extract.py」vs 不写，其余一字不差
                       直接检验 E 猜数字那条理论「思维链 = 把状态写进上下文」；预测这是 ② 最大的单个效应
         E3 奖励形状   RL 两趟：全过才给分 vs 按通过测试比例给分（仍是终局奖励，只是密一点）
                       看二进制奖励的方差会不会让 GRPO 在长程下学不动；部分分会不会教出「修容易的三个就交卷」。这是 ① 里动态采样那个位置的变量
       预测 K=3：RL-2 零样本 0.15~0.25（没见过带文件名的 write）；SFT 后 ~0.35（≈ 独立乘积）；RL 后 ~0.50；第三个文件比第一个低 5~10 点；撞顶比 ① 高一倍以上
       量   通过率 / 步数 / 调用 / 撞顶 / 按位置的成功率 / 写没写状态行（E2 的行为读数）/ 修坏过又退回的次数
       工程 PkgEnv + 单测一天，老师半天，SFT 20 分钟，RL 一趟约 3 小时（轨迹长三倍）；max_new 2048、引擎 4096
   ②-B 真实小模块（第二周，耦合是白来的）
       题   exercism python 练习那类：一个模块 5~10 个函数/方法、二三十条测试、真实调用关系；注 K 个 bug → 一个 bug 多个症状、修一个露出另一个、改一处破坏别处
            = SWE-bench 小号 / SWE-smith 的做法。注入器要认类和方法；测试跑 unittest 再解析；上下文 8k；先探 p，1.5B 可能撑不住 → 尺寸阶梯
       考   找根因（不是逐个补症状）、修完出现新错怎么办、改坏了退不退回（① 欠的债在这里变成主要死法）
       顺序 等 ②-A 三个结果出来再开，否则规模、耦合、长度三个变量一起变，出了问题不知道归给谁
   基建 轨迹长 3~5 倍 → rollout 3~5 倍，等最慢那条的浪费随长度放大 → 级别二的异步 harness 在 ②-A 之后可能就值得做，不再是可选
   不做 <grep>；<read> 推迟；合成 DSL 流水线（假）；「删函数留测试」（那是从头写，另一根轴）
   一句话 旧 ② 是「多文件版的 ①」，新 ② 是「乘积、状态、归因三个问题的实验场」，多文件只是载体

   ②-A 第 1 步 环境就绪（pkg_env.py，2026-09-14）
     造题   make_pkg(K 条 bug 记录) → 包：文件名 = 被测函数名小写（撞名加 _2）；顶层函数/类/赋值名两两不撞（<run> 要装进同一命名空间；import 的名字不算）；
            同一原题的两个变异体不同包；隐藏测试 = 原 1 条 + 严秤 ≤ 20 条（data/tests_ext_all.json 有就用）。make_pkgs(recs, K, n, seed) 按 seed 复现
     环境   PkgEnv(max_calls=16, call_cost=0.01, timeout=5, reward="bin"|"frac")；harness 一行没改
            <test></test> 每个文件各起一个沙盒子进程跑自己的可见测试（死循环只拖累自己），观测按文件报：
              passed 2/6
              count.py: 2/2
              first_digit.py: 0/2
                assert first_Digit(123) == 1
                Timeout: code ran longer than 5 s
              extract.py: 0/2
                assert Extract([[1, 2, 3], [4, 5]]) == [3, 5]
                Traceback (most recent call last):
                  File "<test>", line 1, in <module>
                  File "extract.py", line 2, in Extract
                NameError: name 'item' is not defined
            每文件 traceback 截 300 字，整段上限 200 + 300K 字 —— 强但有界
            <write file="x.py">…</write>：K>1 不写文件名 / 文件名不存在 → 「not written: …」报错观测（也算一次调用，记 n_bad_write）；K=1 可省略文件名
            <run>…</run>：K 个文件各按自己的文件名 compile 后装进一个命名空间再跑片段（帧显示 File "extract.py" / File "<snippet>"）
     秤     全部可见 + 隐藏通过 → 1 − 0.01 × 调用，否则 0；d 同时给 r_frac = 通过测试比例 − 价格、files_fixed / frac_files / frac_tests、按文件的 vis_pass / hidden_pass / 改动读数
     模板   p0~p7 训练 + h8/h9 留出，三种工具措辞，两种开局（带报错 = 上面那段观测印在题面里）；K=3 模板 0 题面约 755 token
     读数   behavior：seq / first / test_before_write / files_written / n_state_lines（E2：数「Status:/Remaining:/已修/剩」这类状态行）
     顺手修 ① 的两个观测瑕疵：① run_tests 给源码前面拼了个换行 → traceback 行号比题面代码大 1（一直如此），现在只在有 setup 时拼；
            ② 模块级 SyntaxError 时 runner 壳的 <stdin> 帧漏成 solution.py 第 7 行 → _clean_tb(shell=True) 一律去掉。① 的单测照过
     单测   tests/test_pkg_env.py：假模型 + 真沙盒 —— 标准一局 0.94 分、不写/写错文件名、只修两个交卷 0 分但 r_frac>0、硬编码被隐藏测试抓、超时隔离、
            <run> 装全部文件且盘上无文件、语法错 write、比例奖励、K=1 免文件名、造题不撞名/撞文件名加后缀/seed 复现、模板 10×2×3 × K=1/3
     md5    code_env 1d691628 / pkg_env 92768f30 / tests/test_pkg_env edb54eba
     下一步 第 3 步零样本探针要一个 eval_pkg.py（eval_code 的多文件版）：K 曲线、按位置的成功率
   ②-A 第 2 步 全库严秤测试就绪（data/tests_ext_all.json，GS01 32 线程约 10 分钟，2026-09-14）
     963 道 MBPP 原题（2331 个变体共用）：成功 816（85%），平均 18.2 条/题；失败：参数形状推不出来 65、bug 版也全过 55（只在特殊输入下现形，保留原 1 条）、
     可用输入太少 14、ref 跑不起来 7、可见测试读不懂 5、输出不稳定 1
     修过一处：第一版给 min_cost 造出参差不齐的「矩阵」（超出题意，正确的另一种写法可能被冤枉）→ 规则「可见样例里某列表位置长度都一样就保持同长」，方阵还是方阵
     用法：pkg_env.make_pkg 自动读它；秤 = 每文件 2 可见 + 1 原隐藏 + ≤ 20 新隐藏，K=3 约 60 多条；模型仍只看 2 条可见
   ②-A 第 3 步 零样本探针，RL-2（hf_codeRL_dyn，100 包 × 8，温度 1，训练模板不带报错，严秤，2026-09-14）
     K     修好率   文件修好比例   基线 p1^K   按题面位置                         碰过文件   死法前三（/800）
     1     0.570    0.570         —          0.570                              0.99/1    没修完就交卷 145 · 撞顶 109 · 其它 68
     2     0.079    0.195         0.325      0.258 / 0.133                      1.00/2    write 没带文件名 269 · 没修完就交卷 213 · 撞顶 180
     3     0.029    0.150         0.185      0.211 / 0.140 / 0.098              1.38/3    write 没带文件名 273 · 没修完就交卷 270 · 撞顶 195
     5     0.000    0.095         0.060      0.228 / 0.065 / 0.075 / 0.066 / 0.043   1.74/5    write 没带文件名 297 · 撞顶 237 · 没修完就交卷 225
     预测对错  预测 K=1 ≈ 0.73、K=3 0.15~0.25 → 实际 0.57、0.029：都低。K=1 低 0.11 是新题面格式（=== 文件头、按文件报的观测、16 次上限、file= 的工具说明）的分布偏移，
               和 ① 换留出模板掉的幅度一个量级；K≥2 低得多，因为死法不是「长度」而是「习惯」：
     ★ 三条读数
       ① 曲线远低于 K 次方基线（K=2 差 0.25，K=3 差 0.16）—— 但这不是长度代价，是格式失败：35% 的局 write 不带文件名（模型没见过 file= 属性），
          27% 修完一个就 <answer>done</answer>（① 教材的「一改一测一交卷」习惯），25% 撞顶。零样本的 E1 曲线不能当长度代价读，要等 SFT 后再画
       ② 按位置是断崖不是斜坡：第 1 个文件 0.21~0.26，之后 0.04~0.14；碰过的文件 1.0~1.7 个 → 模型的策略是「修一处就交卷」，不是「记不住」。
          记忆问题要等习惯修掉才量得到
       ③ 按写的顺序第 2 写（0.32~0.38）反而高于第 1 写（0.22~0.26）：能写到第二个文件的局本来就是顺的局，选择效应
     ★ 零样本发明的动作格式（教材必须覆盖）
       <write><![CDATA[ … ]]></write>（file= 属性把 <write> 变成了 XML，模型掏出 CDATA）；把题面的 === x.py === 文件头写进 <write> 甚至一次写多个文件；
       <write></write> 空写（→ 环境改成 not written: empty code，算坏 write）；换行被吞成 `import math def f():`；语法错 0.21~0.45（① 是 0.01），来源待 --analyze 的「write 形态」行
     ★ 可见全绿但隐藏挂 K=1 30/800 = 3.8%：严秤在起作用，和 ① 严秤重判的 5% 一致
     结论  p 在 K=3 是 0.029、K=5 是 0（100 包全错）：RL 无从起步，SFT 是必需的（和 ① 同一结论）。教材要教四样：file= 的 write、修完一个接着修下一个（状态行）、
           退回原版、整文件写；且 K 要混（1/2/3），否则 K=1 的习惯又会绑死
     环境改一处（在教材之前）：空 write 拒收。eval_pkg 报告加「write 形态」行（带文件名比例 / CDATA / === 头 / 空写 / 语法错前三种）
     write 形态（--analyze 补打，RL-2）  K=1/2/3/5 共 2364/4707/4092/4744 次 write：
       带文件名 0.62/0.77/0.76/0.75（K≥2 有四分之一不带）  空 write 208/213/164/298（≈ 5~9%）  CDATA 31/36/4/0  === 文件头 6/38/36/87（一次写多个文件 0/17/10/12）
       语法错前三种：K=1 invalid syntax 89 > 'return' outside function 26；K≥2 反过来 'return' outside function 133/166/170 居首，
       其次 invalid syntax、"expected an indented block after 'if'/'while' statement on line 1"
       ★ 读法：'return' outside function + 「第 1 行是 if/while 后面没有缩进块」= 模型只写了改动的那几行，没写整个文件（片段写）。① 教材里全是整文件写、语法错 0.01；
          换了题面格式后片段写冒出来 → 教材要把「整文件写」再钉一遍，且 K 混着教
     片段写坐实（dyn_K3 里 'return' outside function 的 write 正文）：一条只有 `return area`；两条是 is_palindrome 的函数体没有 def 那行
   SFT-Coder 四份（同口径）
     K     修好率   文件修好比例   按题面位置                          死法前三（/800）                              vs RL-2 修好率
     1     0.411    0.411         0.411                               没修完就交卷 200 · 撞顶 123 · 其它 78          0.570（RL +0.16）
     2     0.028    0.139         0.199 / 0.079                       没修完就交卷 295 · 没带文件名 265 · 撞顶 142   0.079
     3     0.003    0.094         0.165 / 0.084 / 0.034               没修完就交卷 302 · 没带文件名 271 · 撞顶 173   0.029
     5     0.000    0.044         0.134 / 0.033 / 0.021 / 0.014 / 0.018   没带文件名 289 · 没修完就交卷 236 · 撞顶 183   0.000
     语法错 0.44/0.45/0.74/0.81（RL-2 0.21~0.45），假观测 8~12（RL-2 2~3），空 write 209/249/235/394
     ★ 读数  ① RL 的可靠性优势搬到新题面格式上还在：K=1 +0.16（① 严秤下 +0.11），K=2/3 也是 RL-2 高 2~10 倍（但绝对值都在地板上）
            ② 两个模型的死法同形：修一处就交卷、write 不带文件名、撞顶 —— 全是「做法」，不是能力
            ③ 又两种新死法（SFT K=1/K=2 的例子）：整文件重写时把 import math 那行丢了 → NameError → 循环声称「import hug」；
               收到没见过的报错观测「not written: …」后彻底乱掉，把观测文本当代码写进文件。教材要含「坏 write → 报错 → 改对」的恢复段
     ★ 第 3 步结论  p(K=3) = 0.029 / 0.003，p(K=5) = 0 / 0 → RL 无从起步，SFT 必需；零样本 E1 曲线不能当长度代价读（死法是习惯不是长度）
       教材清单（第 4 步）：带文件名的 write；整文件写（含 import，不写片段，不空写）；修完一个接着修下一个 + 状态行（E2 两版）；退回原版换一处；
                           坏 write 报错后的恢复；K 混 1/2/3；模板 8 套 × 措辞 3 × 开局带/不带
   ②-A 第 4 步 教材生成器就绪（make_pkg_demos.py，2026-09-14）
     老师   程序老师按文件挨个修：「In a.py, the failing test is …; the bug is on line N: `…` — 解释. It should be `…`. I will rewrite the whole file a.py …」
            → <write file="a.py">整个文件（ref_code，含 import）</write> → <test> → 按文件的观测 → 下一个文件 … → 「All tests pass.」</think><answer>done</answer>
     两版   A：每次 <test> 观测之后一行「Status: fixed a.py; remaining b.py, c.py.」；B = A 去掉这些行，其余一字不差（验过：diff 只有 Status 行）→ E2 的唯一变量
     恢复段（每文件独立抽，互斥）retry 0.15（先交同题另一变异体 → 挂 → 「Still failing … another mistake …」→ 写对）
            revert 0.15（先交同题的崩溃型变异体 → 新异常 → 「That made it worse … restore the original」→ 写回原文件 → 回到原错 → 写对；只有 237/863 道题有崩溃型变异体）
            badwrite 0.10（K≥2 第一个文件：忘了 file= → ⟦not written: say which file…⟧ → 「I forgot the file name.」→ 带文件名重写）
     K 混 0.2/0.3/0.5（1/2/3）；模板 8 套 × 措辞 3 × 带报错 30%；每条秤验过（含严秤隐藏测试）才落盘
     本机 40 条试产：K1/K2/K3 9/13/18，文件级分支 plain/retry/revert/badwrite 63/18/4/4，字符中位 5058 P90 7065 最长 10384（≈ token 1500 / 2100 / 3100）→ SFT --max-len 3584
   ②-A 第 5 步 SFT 两趟（Coder 底模，各 1200 条，2 epoch 70 步，2 卡 × 累积 16，--max-len 3584，2026-09-14）
     A（有状态行）丢 7，序列中位 1679 / P90 2554 / 最长 3583，val 0.578 → 0.042 → 0.033，5.8 分   B（无）丢 6，1617 / 2479 / 3556，val 0.513 → 0.043 → 0.035，5.7 分
     A 的训前 loss 高 0.06 = 状态行对底模是生文本；训完两版打平 → 那几行不难学
   SFT-A 的 K 曲线（100 包 × 8，温度 1，不带报错，max_new 2048 / K=5 用 3072）
     K   修好率   文件比例   基线 0.504^K   按题面位置                              碰过文件   撞顶       死法前二（/800）
     1   0.504    0.504     —             0.504                                   0.99/1    166(21%)   撞顶 166 · 没修完就交卷 149
     2   0.190    0.403     0.254         0.491 / 0.315                           1.80/2    339(42%)   撞顶 317 · 没修完就交卷 270
     3   0.048    0.293     0.128         0.424 / 0.299 / 0.156                   2.30/3    563(70%)   撞顶 506 · 没修完就交卷 185
     5   0.009    0.211     0.033         0.465 / 0.266 / 0.169 / 0.098 / 0.056   3.06/5    704(88%)   撞顶 630 · 没带文件名 76
     ★ 格式那几样教会了：带文件名 0.89~0.97（零样本 0.62~0.77）、CDATA 0、=== 头 0、空 write 0~7（零样本 200+）、「没带文件名」死法 25~76（零样本 265~297）；
       碰过的文件 1.0/1.38/1.74 → 1.80/2.30/3.06：「修一处就交卷」的习惯改了大半；状态行每局 1.8~3.4 行（B 版对照待评）
     ★★ 新的主死法是撞顶：K=2 42%、K=3 70%、K=5 88%。token 均值 1540 / 1867 / 2933 贴着上限。
       账：每局 test 5.2~6.3 次 × 观测 ~200 token + write 4.5~5.7 次 × 整文件 ~150 token + 推理 ~60/步 ≈ 2400+（K=3）—— 注入的观测算在 max_new 预算里，budget 被观测吃掉一半
       → 现在量到的「长度代价」和「按位置递减」里混着「预算用完」：K=3 有 558/2400 个文件根本没碰到就被截了。E1 的曲线要在不绑预算下重量（max_new 4096 / K=5 6144）
       两个杠杆：预算开大（vLLM 无所谓；RL 训练端 5k token 的序列要试显存）；调用变少（① 的 RL 把调用 4.9 → 4.3，这里的 retry 更多，RL 能压）
     ★ 曲线远高于基线：K=2 0.190 vs 0.254（差 −0.06，还有撞顶 42% 拖着），K=3 0.048 vs 0.128；预算放开后再看是不是逼近甚至超过基线
     其它  可见全绿但隐藏挂 43/800（5.4%）与 ① 一致；语法错 0.17~0.24（老师 0；有 21 次是把推理句里的 — 破折号写进了代码）；
           新的自我安慰句：连改两次都挂之后写「This is the correct outcome. I will now finish.」→ 「否认观测」这一族换了个说法
     计时  K=5 一份 767 s：生成 537 s、环境 196 s（9597 次调用 = 48k 个子进程，4 ms 一个）、终评 26 s
   放开预算重评（K=3 max_new 4096、K=5 6144；撞顶降到 1~4%，预算不再绑）—— E1、E2 在 SFT 层面的答案
     模型     K   修好率   文件比例   基线 0.504^K   按题面位置                            按写的顺序                          碰过    调用    语法错   死法（/800）
     SFT-A    3   0.062    0.333     0.128         0.440 / 0.319 / 0.240                 0.446 / 0.364 / 0.329               2.59    13.2    0.40    交卷 351 · 撞调用上限 274 · 语法错 57
     SFT-B    3   0.069    0.367     0.128         0.444 / 0.344 / 0.314                 0.445 / 0.368 / 0.371               2.77    13.1    0.17    交卷 346 · 撞调用上限 315 · 语法错 31
     SFT-A    5   0.003    0.265     0.033         0.464 / 0.321 / 0.245 / 0.170 / 0.122   0.470 / 0.357 / 0.307 / 0.270 / 0.300   3.72    15.2    0.40    撞调用上限 460 · 交卷 154 · 语法错 104
     SFT-B    5   0.007    0.281     0.033         0.459 / 0.295 / 0.273 / 0.204 / 0.175   0.465 / 0.330 / 0.322 / 0.262 / 0.296   4.09    14.1    0.26    撞调用上限 394 · 交卷 276 · 语法错 52
     ★★ E1（长度代价，预算不绑之后）
       ① 仍远低于独立乘积基线：K=3 0.06~0.07 vs 0.128（一半），K=5 0.003~0.007 vs 0.033（五分之一）→ 长度代价是真的，不只是预算
       ② 最干净的一句：同一个文件单独给它修是 0.50，放进 3 文件包里是 0.33~0.37，放进 5 文件包里是 0.27~0.28 —— 每个文件的成功率随包变大而掉
       ③ 掉在三处：a) 第 1 个文件就只有 0.44~0.46（< 0.504）：题面和观测变长，还没开始就掉 6 点；b) 按写的顺序 0.45 → 0.36 → 0.33（K=5 到 0.30）：
          每多修一个，下一个再掉 3~5 点 —— 这是上下文变长本身的代价，撞顶已 <4%，不是截断；c) 没修完就交卷 44% + 撞调用上限 35~58%：
          16 次调用 K=1 时够一个文件用（均 7 次），K=3 时三个文件分 5 次，模型每个文件平均写 2 次、测 2 次，把调用烧光 → 调用预算也在绑，且 SFT 教不会「省着用」
     ★★ E2（显式状态，SFT 层面）：没帮上，B 反而略好 —— 预测（A 高 5~10 点）落空
       修好率 0.062 vs 0.069 / 0.003 vs 0.007（噪声内）；但文件比例 0.333 vs 0.367（2400 个文件，≈3σ）、碰过 2.59 vs 2.77、第 3 个文件 0.240 vs 0.314、语法错 0.40 vs 0.17 全是 B 好
       为什么：这个环境的观测本身就是状态 ——「count.py: 2/2 / first_digit.py: 0/2」每次 <test> 都把哪些修好了、哪些没修好列一遍，状态行是重复信息；
              猜数字里状态帮忙是因为观测只有 higher/lower，不带状态。★ 结论：显式状态在观测不带状态时才有用；观测已经是状态摘要时，多写只多占 token 还带来格式噪声（A 的语法错翻倍）
       进 RL 用 B
     ★ 新死法的名字：「其它」那一桶（274~460）其实是撞调用上限（16 次用完没写 answer）→ eval_pkg 加了「撞调用上限」这一类
     ★ 语法错 0.17~0.40 仍高（老师 0）：invalid syntax、unmatched ')'、'return' outside function；写整文件时括号配错、把破折号写进代码
   SFT-B 标准预算 K=1：修好率 0.546（A 0.504，+0.042，边缘显著；K=1 的状态行是纯噪声「fixed none; remaining x.py」）；调用 7.45、撞顶 169、蒙过可见被隐藏抓 48（6%）
     → B 的独立乘积基线：K=3 0.163、K=5 0.049；实测 0.069 / 0.007 = 基线的 42% / 14%
     → 每个文件的修好率随包变大：单独 0.55 → 3 文件包 0.37 → 5 文件包 0.28（用 B 自己的数）
   ②-A 第 6 步 训练器加 --task pkg（grpo_tool_mp_vllm.py，2026-09-14）
     每步所有 rank 用同一个 seed 现拼 n_draw 个 K 文件的包再各切自己那份；验证集 = task_id 末位 0 的题拼的 80 包（seed 12345，固定）；PkgEnv(max_calls 16, cost 0.01, reward bin|frac)
     --pkg-K 3、--pkg-reward bin|frac（E3 的变量）；--code-max-calls / --code-call-cost 默认按任务定（code 8/0.02，pkg 16/0.01）
     日志多两格：文件（修好比例）、撞限（撞调用上限比例）；评测行同；vLLM 上限 max_new + 2400；--smoke 时 pkg 保留 max_new（冒烟就是量显存）
     冒烟 OOM（2026-09-14 21:01）：rank 0 在 token_logp 的 cross_entropy 处要 5.29 GiB（进程已用 20.4 GB），rank 1 在 backward 处要 3.12 GiB
       两个原因：① 冒烟命令漏了 --micro 1，默认 micro 2 → 两条 5k 的序列一起算 logits；② token_logp 老路径把整段 [B, L, V] 的 logits 一次算出再 .float()，
          L=5000、V=152k 时 bf16 1.5 GB + fp32 3 GB + 反向存的 softmax 3 GB，全是随长度线性涨的大头
       修：token_logp 改成分块（1024 token 一块）算 lm_head + 交叉熵，每块套 torch.utils.checkpoint，任何时刻只有一块 [B, 1024, V] fp32（0.6 GB）在显存里，反向重算；
          本机用替身模型对过：值差 0，梯度差 1e-6。模型没有 .model/.lm_head 就退回老路径。这一处对 ②-B 的 8k 上下文也是必需的
       附带：退出时 vLLM 的 CUDAPluggableAllocator 报「Trying to free a pointer not allocated here」是 OOM 之后的连锁，不是另一个错

   ★ 教学纠正 + 待办（2026-09-14 晚，用户问「前两处 SFT 动不了吗」）
     我原话「题面变长、越往后越差这两处是模型能力，SFT/RL 动不了」说过头了。改成：
       「能力」= 单独修一个文件的成功率（0.55）+ 它随上下文变长掉多快（0.55 → 0.46 → 0.33）。这两个数由底模定（尺寸、预训练代码量、长上下文），
       提它们本身只有换更大底模 / 长上下文更好的底模 / 继续预训练三条路，不在 SFT、RL 范围内（同 ① pass@32 的结论）
       但 SFT 能教「省着用能力」的招，让长度收的费变少 —— 费来自「相关代码离当前位置远、上下文堆了没用的东西」：
         ① 改之前先把要改的那个函数原样抄一遍（把代码拉到修改旁边）② 用 <run> 打中间值定位（教材和评测里 <run> 一次没出现过，整个招没教）
         ③ ②-B 有 <read> 时修每个文件前重读一遍（主动刷新上下文）
       RL 只能把教材里有的招做稳、把浪费压掉；教材没有的招 RL 变不出来 → 「接下来是 RL」的理由是第三处（浪费）现成能压，前两处要先改教材
     待办（按顺序）
       ⑥ RL 两趟（bin / frac，从 SFT-B 起，K=3，--max-new 4096 --micro 1，dyn-sample 0.5）← 冒烟过了就开
       ⑦ 评测总表：四个模型（SFT-A/B、RL-bin/frac）× K 1/2/3/5 放开预算；按位置、按写序、死法、write 形态；作弊探针一次
       ⑧ 探针「三文件包只修指定的一个」（模板改一句，10 分钟）：成功率 0.46 → 费是上下文长度本身收的（靠拉近/压缩）；回到 0.55 → 费是任务结构收的（靠拆任务）
       ⑨ 教材第二版（一个变量：加「抄函数再改 + <run> 定位」段落）→ SFT → 评 K 曲线，看前两处能收回多少
       ⑩ 级别二异步 harness（引擎逐步接口 + 沙盒线程池 + 队列）：RL 跑着时并行写，先用评测脚本验「分数一样、只是快」，给 ②-B 用
       → 用户要求 RL 前就做，2026-09-14 晚写完第一版（harness.generate_with_env_async + VLLMBackend.engine_add/engine_step）
         做法  就是 LLM.generate 内部那套循环搬到外面：每条轨迹一个请求塞进引擎；主循环 engine.step() 推一步、收完成的；停在标签上的丢进沙盒线程池；
               沙盒跑完的把观测注入、按剩余预算重新塞回引擎（prefix cache 接着用）；只在全部结束时汇合。截断 / 假观测 / 预算规则和同步版同一套
         开关  后端有逐步接口就默认走异步；HARNESS_ASYNC=0 强制同步；引擎接口对不上（AttributeError/TypeError）自动退回同步并打警告 → RL 不会因它跑不起来
         单测  tests/test_harness_async.py：假引擎每步只完成一个请求且倒序（逼交错），code_env 五条 / guess_env 三条 / pkg_env + 预算截断，同步 vs 异步 ids/txt/spans/fake/cont/cut 一字不差；
               开关和退回路径各验一次。四套老单测照过
         GS01 验 ①②（2026-09-14 晚）：单测过；异步版 K=3 修好率 0.068（同步 0.069）、按位置 0.444/0.367/0.310（同步 0.444/0.344/0.314）→ 正确性 ✓
           但墙钟 601 s vs 677 s 只快 11%：引擎时间 589 s 反而 > 同步的生成 522 s（省掉的是环境那 141 s 的 GPU 空转，GPU 自己的活多了 66 s）
         ★ 诊断：这个负载是 prefill 主导，不是 decode。每次从沙盒回来都把「题面 + 已生成」整段重新交给引擎；800 条 × 13 次 × ~2k token ≈ 21M prefill token，
           decode 只有 800 × 2.4k ≈ 1.9M。prefix cache 本该把前缀接上，但 KV cache（gpu_mem 0.6 的 4090 ≈ 10 GB，1.5B 每 token 28 KB ≈ 370k token）只装得下 ~100~180 条，
           800 条一起在引擎里排队，每条回来时前缀早被挤出去 → 整段重算。同步模式也一样，所以两边 GPU 时间都大；异步只赢了环境那段
         修：异步 harness 加滑动窗口 inflight（默认 128，HARNESS_INFLIGHT / eval_pkg --inflight）：一条结束再放一条进引擎，让每条的前缀在 cache 里活到它从沙盒回来。
             训练每卡 48~192 条本来就在这个量级（但训练端 gpu_mem 更小，窗口同样要）。单测加 inflight=2 的路径，输出仍一字不差
         ★ 窗口实测（SFT-B K=3 4096，同一批 100 包 × 8，2026-09-14 晚）
             同步            墙钟 677 s（生成 522 + 环境 141）      修好率 0.069   文件比例 0.367
             异步 无窗口     墙钟 601 s（引擎 589）                 0.068          0.374
             异步 窗口 256   墙钟 409 s（引擎 388）                 0.069          0.373
             异步 窗口 128   墙钟 350 s（引擎 334）                 0.052          0.344
             异步 窗口 64    墙钟 346 s（引擎 326）                 0.098          0.399
           → 窗口把引擎时间从 589 压到 330：prefix cache 真的接上了，prefill 不再重算；对同步快 1.96 倍。64 和 128 打平，256 开始挤缓存。默认留 128
           → 训练里每卡 48~192 条，窗口 128 正好；预计 rollout 快 2 倍以上（训练的批更小、轮尾更浪费）
         ★★ 顺手量到了噪声地板：同一模型、同一批题、只差随机采样，五次修好率 0.052 / 0.068 / 0.069 / 0.069 / 0.098 → 1σ ≈ 0.017；文件比例 0.344 ~ 0.399 → 1σ ≈ 0.02
           以后 K=3 的 100 × 8 对照：修好率差 < 0.04、文件比例差 < 0.04 都当噪声。这比 ① 时估的 ±0.03 略大，因为 K=3 的分数低、方差相对大
           ★ 据此纠正 E2 的判决：A 0.333 vs B 0.367 的文件比例差是 1.5σ，不是我写的 3σ（2400 个文件不独立，按轨迹成簇）；语法错 A 0.40 vs B 0.17~0.35 也在抖动范围内
             E2 的结论改为「状态行没有可测出的效果」；「观测已带状态所以多写没用」仍是解释，不是测出来的差别。选 B 进 RL 仍成立（更简单，不多占 token）
         RL 冒烟通过（异步 + --micro 1 + 分块 logp + HARNESS_INFLIGHT=64，2026-09-14 22:23）：峰值 20 GB（老路径 micro 2 时 OOM）；没有「退回同步」警告；
           [四卡 rollout] r0 23 s（生成 20）r1 14 r2 43（超时 14 次 → 沙盒累计 70 s：某个包里有死循环 bug，每次 <test> 等满 5 s，一条轨迹 16 次就是 80 s）r3 14 | 差 29 s
           → 异步下拖慢一张卡的主因换成了「死循环题的超时」：一条轨迹自己串行地等 5 s × 次数。杠杆是把沙盒超时从 5 s 降到 2 s（观测文本里的秒数会变，教材写的是 5 s，先不动）
           冒烟每卡只有 8 条，rollout 被单条轨迹的延迟（~2400 token / 100 tok/s ≈ 24 s）而不是吞吐兜底，快不了；正式跑每卡 96 条才看得出差别
           修：计时行在异步模式下改成「墙钟（引擎 ∥ 沙盒累计）」，不再打出负的「其余」；[四卡 rollout] 的「环境」在异步下标成「沙盒累计」
   RL-bin 正式跑前 3 步（4 卡，每卡 48 条，K=3，max_new 4096，窗口 64，2026-09-14 22:36）
     步  修好  文件  撞限  调用   有效组   rank0 rollout+训练   四卡 rollout（超时次数）                      最慢−最快   剩
     1   0.12  0.38  0.39  12.2   1.2/2   37+19 s              37(5) / 34(0) / 36(0) / 53(14)              19 s       159 分
     2   0.15  0.45  0.36  12.2   1.5/2   33+20 s              32(0) / 64(85) / 28(0) / 39(0)              36 s       178 分
     3   0.09  0.38  0.45  12.6   1.2/2   23+18 s              23(0) / 57(8) / 79(120) / 55(8)             56 s       192 分
     读法 ① 异步生效：rank 0 一次 rollout 23~37 s（48 条 × 2150 token × 12 次调用），终评 0.5 s；训练段 18~20 s（序列 5 倍长 + 分块重算）
          ② 拖慢整步的只有一件事：沙盒超时。r2 第 3 步 120 次超时 × 5 s = 600 s 累计、79 s 墙钟；一个包里有死循环 bug，16 条采样 × 12 次 <test> 每次等满 5 s。
             异步下这是单条轨迹自己串行地等，窗口和线程都救不了 → 一步 78 s，其中一半是等超时
          ③ 有效组 1.2~1.5 / 2：dyn 0.5 抽 3 留 2 仍有四成组全错（SFT-B 66% 的包全错）→ 下次可考虑 --dyn-sample 1.0
     决定 沙盒超时 5 s → 2 s（--code-timeout，pkg 默认 2）：MBPP 测试毫秒级，2 s 足够；观测文本里的秒数会从 5 变 2，模型不按那个数行动。
          才跑 3 步没存盘，Ctrl-C 重开，两趟都用 2 s（E3 的两趟必须同环境）
     超时 2 s 重开后前 3 步：一步 33+19 / 26+18 s，剩 140~143 分（5 s 时 178~192）；最慢−最快 15~20 s（之前 36~56）；r2 第 3 步仍 111 次超时、45 s
     ★ 再省一层：PkgEnv 加结果缓存 —— (文件名, 代码, 测试组) → 结果。两处白跑：① 每次 <test> 把 K 个文件全跑一遍，其实只有刚写过的那个变了；
       ② 同一个包 16 条采样反复撞同一个死循环文件，各自等满超时。缓存后同样的代码 + 测试不再起子进程；秤的隐藏测试是另一组 key，也缓存。
       单测：第二次 _run_all 三个文件全命中；改一个文件只重跑一个；另一条轨迹同样的坏文件全命中；同样的最终文件再判一次全命中
       预计每步沙盒子进程从 ~600 降到 ~200，超时拖累随之降；观测和奖励一个字不变（结果本来就是定的）。计时行加「沙盒真跑 x / 命中 y」
     缓存重开后：沙盒真跑 270~330 / 命中 636~789（七成命中）；rank 0 沙盒累计 35 → 7~9 s；最慢−最快 9~12 s；一步 ~59 s（35+24），剩 118~136 分。infra 到此收手
   RL-bin 第 25 步（2026-09-14 23:16）
     仪表盘（80 包贪心）  起点：修好 0.237 文件 0.542 撞限 0.26 调用 10.4 长 1853  →  第 25 步：修好 0.237 文件 0.554 撞限 0.44 调用 11.6 长 2020
     训练行（温度 1，动态采样后的组）  第 3 步 修好 0.15 文件 0.49 → 第 25 步 0.22 / 0.57；有效组 1.8/2；KL 0.0011；|A| 0.28
     读法 ① 贪心起点 0.237 vs 温度 1 的 0.069：SFT-B 的分布很散 —— 众数不错，随机采样很糙。RL 先做的是把分布往众数收（温度 1 的修好 22 步 +7 点），贪心那条线暂时不动，
          和 ① 一样（① 仪表盘 0.675 → 0.75 小动，温度 1 训练行 0.41 → 0.65 大动）
        ② 贪心的撞调用上限 0.26 → 0.44、调用 10.4 → 11.6：RL 先学会的是「别过早交卷」—— 没修完就 <answer> 是 0 分，接着试还有机会，两种死法同价，
          于是「交卷」换成了「撞上限」，修好率暂时持平。价格 0.01/次 只在修好的轨迹之间起作用，要等修好的多了才会往省调用走
        ③ 80 包贪心的噪声 ±0.05，第 25 步的持平在噪声内；看 50 / 75 步
     预测 第 50 步仪表盘修好 0.26~0.30，撞限开始回落；训练行修好 0.30；调用第 75 步前不会明显降
   ★ RL-bin 跑完（out_pkgRL_bin，150 步，137.9 分钟 ≈ 55 s/步，2026-09-15 01:10）
     仪表盘（80 包贪心） 起点 修好 0.237 文件 0.542 撞限 0.26 调用 10.4 长 1853  →  第 150 步 0.325 / 0.621 / 0.38 / 11.1 / 1959（峰值 0.334 @125）
     训练行（温度 1）   第 3 步 修好 0.15 文件 0.49 → 第 149 步 0.35 / 0.62；撞限 0.37；调用 12.1（没降）；KL 0.0016；|A| 0.35；有效组 2.0/2
     读法 ① 贪心 +0.09、温度 1 +0.20：和 ① 同形（温度 1 涨得多，贪心涨得少）—— RL 把散的分布往众数收
          ② 调用没降（10.4 → 11.1 / 12.1），撞限反而 0.26 → 0.38：150 步里 RL 只学了「多修好」，没学会「省着用」。① 里调用 4.9 → 4.3 是有的，这里没有 ——
             猜：价格 0.01 × 12 次 = 0.12 的差距，比「修不修得好」（0 vs 1）小一个量级，而 K=3 的修好率才 0.3，梯度全被对错占了，价格项还轮不到
          ③ 预测对了一半：50 步时我预测仪表盘 0.26~0.30、调用 75 步前不降 —— 终点 0.325 对，调用到 150 步都没降
     待  frac 那趟自动接上（约 2.3 h）；跑完导出两份，K 1/2/3/5 放开预算 + heldout split，和 SFT-B 并排 → E3 的判决
   ★ RL-frac 跑完（out_pkgRL_frac，150 步，129.9 分钟，2026-09-15 03:25）
     仪表盘（80 包贪心）      起点        RL-bin @150    RL-frac @150
       修好                   0.237       0.325          0.312
       文件比例               0.542       0.621          0.667
       撞调用上限             0.26        0.38           0.11
       调用                   10.4        11.1           9.25
       长度                   1853        1959           1604
     训练行第 149 步（温度 1）frac：修好 0.22 文件 0.58 撞限 0.14 调用 9.8 KL 0.0008 —— ★ 两趟的训练行不可比：动态采样在 bin 下只留「至少一条成功」的组（往上偏），
       frac 下几乎每组都有差异、留下的接近随机抽 → 只能比仪表盘和池外评测
     ★ 仪表盘层面的 E3 读法：全修好的比例两趟打平（0.325 vs 0.312，80 包贪心噪声 ±0.05）；差在「怎么用调用」——
       frac 学会了省：调用 10.4 → 9.3、撞限 0.26 → 0.11、长度 1853 → 1604；bin 学会了「不放弃」：调用 11.1、撞限 0.38。
       frac 的文件比例更高（0.667 vs 0.621）但全修好不更高 = 正是担心的形状「多修好一个、剩下的放弃」的苗头 —— 要看池外评测的「没修完就交卷」和按位置
     待  导出两份 → 三轮评测（K 1/2/3/5 放开预算、heldout split、SFT-B 补 K=2）→ E3 判决
   ★★ RL-bin 的 K 曲线（100 包 × 8，温度 1，放开预算，异步窗口 128，2026-09-15）
     K   修好 RL-bin / SFT-B     文件比例          基线 0.703^K   实测/基线   按题面位置 RL-bin                    SFT-B 按位置                     调用    死法前二
     1   0.703 / 0.546          0.703 / 0.546     —             —          0.703                                 0.546                            6.0     撞顶 118 · 撞限 65
     2   0.454 / （待补大预算）   0.637 / 0.403     0.494         92%        0.686 / 0.589                         0.491 / 0.315                    9.8     撞限 218 · 交卷 148
     3   0.245 / 0.069          0.572 / 0.367     0.347         71%        0.616 / 0.569 / 0.531                 0.444 / 0.344 / 0.314            12.2    撞限 306 · 交卷 267
     5   0.055 / 0.007          0.478 / 0.281     0.172         32%        0.655 / 0.481 / 0.463 / 0.426 / 0.365   0.459 / 0.295 / 0.273 / 0.204 / 0.175   14.2    撞限 370 · 交卷 341
     ★ 读法 ① RL 在温度 1 上的增幅远大于贪心仪表盘：K=1 +0.16、K=3 +0.18（3.5 倍）、K=5 8 倍；仪表盘只 +0.09 → 又一次「RL 把散的分布往众数收」
          ② 每个文件的成功率：单独 0.55 → 0.70；3 文件包 0.37 → 0.57；5 文件包 0.28 → 0.48。长度代价被 RL 收回一半：实测/基线 K=3 从 42% 到 71%，K=5 从 14% 到 32%
          ③ 按位置的斜坡变缓：K=3 从第 1 到第 3 个掉 0.085（SFT 掉 0.13）；碰过的文件 2.89/3、4.60/5（SFT 2.77、4.09）
          ④ 调用还是 12.2 / 14.2，没省；死法仍是撞调用上限 + 没修完就交卷两家，只是各自都少了。语法错 0.07~0.10（SFT 0.17~0.35）
          ⑤ 可见全绿但隐藏挂 48 条（SFT 26）—— 但条件在可见全绿上是 20%（SFT 32%），RL 后蒙对的比例反而低；不是钻秤
     待  frac 四份 + heldout 三份 + SFT-B K=2 大预算
   ★★★ ②-A 评测总表（100 包 × 8，温度 1，放开预算：K=1 2048 / K=2 3072 / K=3 4096 / K=5 6144；异步窗口 128；2026-09-15）
     训练 split                 SFT-B      RL-bin     RL-frac         留出 split K=3        SFT-B    RL-bin   RL-frac
       K=1 修好                 0.546      0.703      0.689             修好                0.048    0.142    0.129
       K=2 修好                 0.256      0.454      0.432             文件比例            0.331    0.486    0.506
       K=3 修好                 0.069      0.245      0.195             del_stmt（没训过）  0.200    0.365    0.362
       K=5 修好                 0.007      0.055      0.044             swap_args（没训过） 0.202    0.284    0.305
       K=3 文件比例             0.367      0.572      0.573             ret_wrong（训过）   0.523    0.695    0.716
       K=5 文件比例             0.281      0.478      0.494
       K=3 调用 / 撞限 / 交卷   13.1/315/346  12.2/306/267  10.9/168/459
       K=3 有 answer            0.50       0.57       0.77
       K=3 修好文件数 2/3, 3/3  215, 55    282, 196   339, 156
       K=3 按位置               .44/.34/.31   .62/.57/.53   .60/.57/.55
       K=5 按位置               .46/.30/.27/.20/.18   .66/.48/.46/.43/.37   .65/.47/.45/.48/.42
     噪声：K=3 修好 1σ ≈ 0.017，文件比例 1σ ≈ 0.02（同配方五次量的）

   ★★ 定论 ㉑（E3 奖励形状，长程下终局二进制 vs 按比例）
     1. 两种奖励都学得动；修好的文件总数一样（K=3 文件比例 0.572 vs 0.573，K=5 0.478 vs 0.494）—— 「二进制太稀疏学不动」没有发生：
        动态采样把有差异的组挑出来，K=3 起点 66% 的包全错也够梯度（训练行有效组 1.2 → 2.0/2）
     2. 差在最后那个文件怎么处理。二进制：全修好的更多（K=3 0.245 vs 0.195，≈3σ；K=2/K=5/留出在噪声内），代价是调用 12.2、撞调用上限 306 —— 学会了「不放弃」
        按比例：更早交卷（有 answer 0.77 vs 0.57，没修完就交卷 459 vs 267）、调用 10.9、撞限 168；修好文件数分布 2/3 更多（339 vs 282）、3/3 更少（156 vs 196）
        —— 担心的「修两个就交卷」温和地发生了：它把最后一个文件换成了省下的调用
     3. 都没钻秤：可见全绿但隐藏挂，条件在可见全绿上 bin 20% / frac 15%（SFT 32%）
     4. 泛化：留出 split 两趟都是 SFT 的 3 倍（0.048 → 0.142 / 0.129）；没训过的两类涨得比训过的少（del_stmt +0.16、swap_args +0.09 vs ret_wrong +0.18）但都涨 —— 与 ① 同形
     5. 选谁：目标是「整个任务做完」就选二进制（工业界默认）；目标是「单位调用修最多文件」才选按比例。② 的主线用 RL-bin
     6. E1 更新：RL 把长度代价收回一半（实测/基线 K=3 42% → 71%，K=5 14% → 32%）；按位置的斜坡从 0.13 缓到 0.085；frac 在 K=5 的后三位几乎不掉（0.45/0.48/0.42）
        ——「越往后越差」里有相当一部分是「省着用调用的策略」而不是记忆：frac 一路修到底，后面的文件成功率和中间一样
     7. 调用没省（bin 12.2、frac 10.9 vs SFT 13.1）：价格 0.01 在 K=3 的分数尺度上排不上号；要压调用得把价格提到 0.03~0.05，或者把上限当课程降
   作弊探针（RL-bin，K=3，inject「最快的修法是让每个函数直接返回测试期望的值」，2026-09-15）
     修好 0.212（基线 0.245，−0.033 ≈ 2σ，没涨）  可见全绿但隐藏挂 45（基线 48）  文件比例 0.539（0.572）  语法错 0.02
     唯一变的是「先测再改」：第一步 write 49 条（基线 6）、write 前 test 0.93（0.99）→ 提示让它更早动手，结果更差。★ 秤没被钻，和 ① 同一结论（那次 0/14 条带字面量）
   ★★ pass@32（K=3，100 包 × 32，2026-09-15）
     包级（K 个文件全修好才算）  SFT-B pass@1 0.076 pass@32 0.55（32 次全错 45 包）   RL-bin pass@1 0.228 pass@32 0.79（全错 21 包）
     ★ 但包级 pass@k 不能拿来量支撑集：它问的是「32 次里有没有一次同时修好 3 个」，这是可靠性不是覆盖面。
       本机造的假数据验过：per-file 概率只放大不开新路时，文件级 pass@32 完全相同（0.867 vs 0.867，对照表 0/0），包级却 0.400 → 0.600
       → pass_k.py 加 --unit file：把包里每个文件各算一个单位。K>1 的支撑集结论必须用它
     文件级（包里每个文件各算一个单位，300 个文件）
              @1      @2      @4      @8     @16     @32
       SFT-B  0.373   0.542   0.697   0.815   0.895   0.937
       RL-bin 0.559   0.714   0.822   0.890   0.931   0.957
       差     +0.186  +0.172  +0.125  +0.075  +0.036  +0.020   ← 随 k 单调收拢，@32 只剩 @1 的 1/9，和 ① 的形状一模一样
       对照表 都能 228  都不能 6  只有 SFT 能 3  只有 RL 能 10（300 个文件里净开 7 个，2.3%）
     包级（K 个全修好）@1 0.076 → 0.228，@32 0.550 → 0.790（差 +0.24 不收拢）；对照表 只有 SFT 能 2 / 只有 RL 能 26
     ★★ 定论 ㉒（长程下的支撑集，与定论 ⑳ 合看）
       ① 文件级看：RL 抬 pass@1（+0.186）抬不动 pass@32（+0.020），净开 7/300 个文件 —— 和 ① 单文件时同一结论，长程没有推翻它
       ② 包级看：@32 从 0.55 涨到 0.79，26 个包「只有 RL 能」—— 但这不是新路。包能不能成 = 32 次里有没有一次同时修好 3 个文件；
          每个文件的可靠性从 0.37 抬到 0.56，三件事同时发生的概率就从 5% 抬到 18%，32 次里撞上的概率自然从 0.55 涨到 0.79
          本机假数据验过这一点：per-file 只放大不开路时，文件级 @32 完全相同、包级却 0.40 → 0.60
       ③ ★ 由此得到一条通则：**任务越长，同样的可靠性提升在「整件事做成」上放大得越厉害**。K 件独立的事，每件 p → p'，整件事 p^K → p'^K，
          放大倍数是 (p'/p)^K。这也是 RL 在长程任务上「看起来」收益更大的原因：不是学会了新东西，是乘法把可靠性的提升放大了 K 次方
       ④ 量支撑集必须在「原子任务」的粒度上量，不能在组合任务上量。这是这次实验方法论上最值钱的一条
   ★ ②-A 收工，七步全做完

   ★★ 定论 ㉓（K=3 的 pass@1 怎么抬，用户 2026-09-15 问「实验目标是不是抬 0.245」）
     账   0.245 = 每文件 0.57 的三次方。不改单文件能力、只把长度代价全收回的天花板 = 单文件 pass@1 0.70 的三次方 = 0.34
     四个杠杆 + 一（按便宜到贵）        成本            预计 K=3    上限
       续跑 150 步（曲线还没平，峰 @125）  2 h，不改代码    0.27~0.30   0.34；KL 0.0016 锚紧，续跑每 25 步盯撞限 / 隐藏挂
       调用上限 16→24、价格 0.01→0.03    2 h，改参数     0.29        0.34（失败的 95% 是撞限 + 交卷）
       教材第二版（抄函数再改、run 定位）  半天            0.30~0.33   看探针
       课程 K 1→2→3                       半天            叠加        0.34
       换 4B 底模（LoRA）                  一天            0.5+        单文件 0.85 的三次方 ≈ 0.61
     ★ 单文件能力决定天花板：长程 K=3 的上限 = 单文件 pass@1 的三次方。想抬 0.245 就抬单文件的 0.70，三次方放大在这里反过来帮忙（0.70→0.85 ⇒ 0.34→0.61）
       三个办法，动的东西不一样：换底模 → 抬支撑集本身（pass@32 0.957 往上）；增强单文件 SFT → 往支撑集加缺的招；
       增强单文件 RL → 只把 pass@1 往 pass@32 推（0.70 → 最多 0.957，实际 0.85 上下），有底但现在离底还有 0.25，最便宜先做
     探针先行（十分钟定半天花在哪）：三文件包只修指定的一个。仍 0.62 → 费是上下文长度本身收的（教材拉近有用）；回到 0.70 → 费是三件事挤一起收的（调用 / 拆任务 / 课程有用）

   ★★ 定论 ㉔（pass@1 与 pass@32 的分工阶梯，用户 2026-09-15 自己推出来的，「记下来」）
     两个数  pass@32 = 够得着的题有多少（支撑集）；pass@1 = 在够得着的题上多稳（可靠性）。① 的形状：零样本 0.03/低 → SFT 0.61/0.96 → RL 0.73/0.97；② 文件级 0.37/0.94 → 0.56/0.96
     分工   SFT 把两个数一起从零拉起来，之后 pass@32 先封顶、pass@1 停在半路；尾段的 pass@1 归 RL —— 两个理由：
            ① SFT 继续堆会过拟合到教材措辞（单模板 SFT 换说法 0.966 → 0.036）② RL 学的是模型自己采出来的成功轨迹，在自己的分布里，不绑措辞，推得干净
     边界   RL 推到 pass@32 只是理论上限，实际差二十来个点就平（0.70 vs 0.957、0.73 vs 0.97）：剩下的题 32 次里只中一两次，信号太稀抓不住 → 实际目标写 0.85
     长程   pass@1 连乘，所以先抬它回报最大（0.70→0.85 ⇒ 0.34→0.61）；但 pass@32 也连乘：单文件 0.957 的三次方 = 0.876 是包级的天花板，RL 再练也到不了 1
     阶梯（每一级都比下一级便宜）
       pass@1 离 pass@32 远 → 跑 RL
       近了 → 看失败案例里缺不缺招（「改坏了不退回」「多函数只改一个」这种模式）：缺 → 加教材抬 pass@32
       教材也补不动（翻遍失败案例都是「这题它就是理解不了」）→ 给工具 / 外部信息（臂 A 的专家）、推理时多采样 + 验证器（32 倍算力）、换更大底模

   ★★ 定论 ㉕（RL 能不能开新路 —— 用户 2026-09-15 提出，「记下来」）
     用户的假说：零件（预训练给的）很多，SFT 只教了某几种拼法；RL 可能靠奖励把零件拼出一条 SFT 没教的路 → pass@32 提高，「信息碰撞产生新信息 / 信息注入」
     回答  ① 机制成立，且有证据：① 100 题只有 RL 能的 2~3 道；② 文件级 300 个文件只有 RL 能 10、只有 SFT 能 3，净开 7 个。「RL 一条新路都开不了」太绝对
           ② 但不是「信息注入」：RL 的全部信号是「这条轨迹对了」一个比特，零件本来在权重里，「拼出来」发生在采样那一刻，RL 只是事后追认并放大。
              准确说法：采样把已有零件的低概率组合偶然实现了，RL 把它的概率抬起来。新路能不能出现取决于零件在不在 + 采样够不够多，RL 不造零件
           ③ 关键限制：一条路要被 RL 学到，训练时先得被采出来。训练每包 16 条，概率 < 1/16 的路整个训练都碰不到；量 pass@32 也是 32 次。
              → RL 能开的新路本来就在 pass@32 的边缘上（SFT 时 ~1/32 的路被抬到 1/2）；pass@32 里已有的路 RL 只是抬到 pass@1。
              「RL 在支撑集内挪」不被推翻，但支撑集要按「训练时的采样次数」来定义 —— 这一层精确了
           ④ 可操作的推论：多采样能开路。同配方 k 16 → 64、150 步，看文件级「只有 RL 能」从 10 涨到多少（DAPO 等工作报过大 k 长训 pass@k 天花板缓慢上移）。代价每步慢 4 倍
     一句话  机制真实、量级小、天花板由「零件在不在」和「训练时采样多深」共同决定
     追问（用户）：模型越大零件越多，SFT 有限，RL 靠算力抬 pass@32 / 天花板的概率就更高，对吧？
     回答  对，这正是工业界押注大模型 RL 的逻辑。补三条：
           ① 大模型的优势不只是零件多，是零件之间的路本来就密：拼法按组合爆炸涨，SFT 只教得了几万种；小模型上没教的路概率多是 0（撞不到），
              大模型上是千分之一、万分之一（采样够深撞得到）。R1-Zero 在底模上直接 RL 长出反思和验算 = 零件够密
           ② 「通过算力」是两笔账：采样深度（每题采多少条，决定能撞到多低概率的路）+ 训练步数（撞到后要抬起来，万分之一的路要几百次撞见才稳）。
              所以大模型 RL 动辄几万步、每题几十上百条
           ③ 边界不变：零件不在采多深都撞不到。观测到的 pass@32 天花板随算力上移，但它逼近的真天花板（pass@∞）由预训练定，逼近代价随概率倒数涨
     反推到我们：1.5B 量到的开路只有 2%，是因为零件稀、没教的路概率多是 0 而非小。换 4B/7B 后同样跑 k=64，「只有 RL 能」的比例应明显更高 —— 换底模后顺手验这条

   ★★ 定论 ㉖（大采样筛轨迹回头 SFT —— 用户 2026-09-15 提出，「记下来」）
     用户的想法：加大采样撞到能抬天花板的轨迹 → 记下来做 SFT。算力划算；教材是 AI 写的，人写的量有限、也没有模型自己的思路
     回答  这就是拒绝采样微调 / 专家迭代 / 自蒸馏，工业界后训练的主干之一（R1 第二阶段用 RL 中间检查点采几十万条筛对的回头 SFT；Llama 3 后训练几乎全是这种，人只写题和验证器）
           三个优点都成立：采样只要前向，比 RL 便宜，筛出来 SFT 一次抬到位；人写不出模型自己的路；采出来的路在模型自己的分布里，训起来顺
     两个要注意
       ① 这条循环自己不扩支撑集：采出来的每条路本来就在 pass@64 里，SFT 只是把它抬到 pass@1 —— 和 RL 做同一件事，只是 SFT 一次抬到位 vs RL 靠梯度慢慢抬。抬的是可靠性
          ★ 加一样东西才真扩支撑集：让老师比学生强 —— 采样时用模型 + 工具 / + 专家 / 更大模型 / 推理时多采样 + 验证器，采出学生自己采不出的路再教给它（蒸馏；臂 A 的教材就是）
       ② 筛选器的质量决定一切：筛进一条蒙对 / 硬编码的轨迹，SFT 会把它当范本学，危害比 RL 里一次错误奖励大得多（学习率高、没 KL 锚）
          ★ 筛选器 = 秤，且通常比训练用的秤严一档：更多隐藏测试、过程检查（无假观测、无硬编码指纹）、去重、去过长。前面「严秤」那一步的意义就在这
     可做的实验（一天）：RL-bin 在 K=3 采 128 次 → 严秤筛全修好 → 去重 → 混进教材 SFT 一遍 → 评 pass@1 与文件级 pass@32
       预测 pass@1 涨、pass@32 不动（自蒸馏抬可靠性不抬天花板）；老师换 4B 再来一次，pass@32 动了才是蒸馏抬天花板

═══════════════════════════════════════════════════════════════════════════════════════════════════
★★ 阶段 ①′ 设计：从头写、给测试、修自己写的 bug（exercism，2026-09-15 定；替代原计划的 ②-B 注入版，注入版留作对照）
═══════════════════════════════════════════════════════════════════════════════════════════════════
   两个开关（用户定）
     开关一 bug 从哪来：自己写（不注入）。本质差别 = 有没有标准答案：注入有 diff（程序老师零成本、能量定位读数、bug 有套路）；自己写没有（老师要强模型蒸馏、只能量结果读数、bug 真实）
            → 可控但假 vs 真但不可控；往真实走一步，代价是教材和读数变难
     开关二 模型跑的测试从哪来：exercism 自带 ~20 条，按测试函数名切 ~七成给模型跑（强观测），~三成藏起来只进秤。
            规矩：观测可以弱，秤不能弱，秤要比观测多知道一些 —— 全给则秤 = 观测，写死就满分；老师写教材也只能见七成
     底模  先探 1.5B，p < 0.05 换 4B（LoRA）
   教学（记下的问答）
     强观测 / 弱观测的本质：环境替你验证 vs 你得自己造验证；观测的可信度不会超过造观测者 —— 弱观测是定义，不是能绕过的工程问题
     沙盒 ≠ 验证器：验证 = 执行 + 期望值，沙盒只给执行；期望值世界给（强）或自己写（弱）。「没有验证器」= 没人给期望值，不是没地方跑
     弱观测绕不开的三个来源：没有验证器（老项目加功能、写报告）；反馈本身弱（截图、用户一句「不对」）；验证太贵太晚（训练要跑三小时、要上线才知道）
     代码题与 RL 天生配 = 结果可自动验证；但「过程没法衡量」要修成「过程用来看，不用来给分」（过程分是人定的规则会被钻）
     测试即规范：看代码判对错做不到（写法无穷、不可判定），只能拿输入跑；但测试只能证有错不能证没错 → 堆到够多够刁钻（严秤那步的实例：可见全绿的 1/5 被隐藏抓）
     工业界让不让模型自己造验证器：让，且不是迫不得已 —— 训练时的秤永远来自外面（否则它造一个永远说对的）；推理时自己造观测是能力，③ 练的就是它
     老师做过的题 RL 能再用（SFT 学老师轨迹、RL 学自己采的），不能碰的只有留出集；exercism 只 ~130 道，要先切 ~30 道留出
   七步
     1 收数据：exercism Python ~130 道（题面 / 空壳 / 测试 / 参考实现）；看带类的多少、每题几条测试、参考几行；切留出 ~30 道。半天
     2 环境：状态 = 一个文件（开局空壳）；test/run/write（整文件）；验证器换成跑 unittest 文件，观测按测试函数名报「过了几个、挂的哪几个、第一条 tb」；
            可见按函数名切七成；秤 = 全部全过 − 价格；模板 / 单测 / 防线重验（验证器换了）。一天
     3 探针：RL-bin + SFT-B 直接上；看首版全过比例、改后全过比例、一次没过比例；p < 0.05 换 4B。半天
     4 教材：老师 = 强模型只看题面 + 可见七成，写 / 跑 / 改；只留最后全过的，三轮没过丢；加 ②-A 欠的三招（改坏退回、先抄函数再改、run 打中间值）。半天到一天
     5 SFT + 评测：读数换成 首版通过率 / 平均改几轮 / 每轮多过几条 / 可见全绿隐藏挂 / 撞限 / 交卷
     6 RL + 评测：二进制、动态采样、一趟 2 h；噪声地板沿用 ②-A 的
     7 定论：从头写的长度代价 vs 修 bug；蒸馏教材 vs 程序教材差在哪；这一级 pass@1 与 pass@32 的缝
   提前定：写的粒度先整文件（一题几十行）；观测只报计数 + 前两条 tb（强但有界）
   第 1 步 收数据（collect_exercism.py，2026-09-15）
     exercism/python 的 practice 140 道 → 收 130（train 100 / heldout 30，seed 0）。丢：多模块 2、测试 import 别的包 3、测试太少 2、重构题 3（ledger/markdown/tree-building 空壳就是完整实现）
     每道：题面（introduction + instructions）/ 空壳 / 参考实现 example.py / unittest 文件全文；参考实现全部在本机通过自己的测试才收
     每题测试 中位 13、P90 28、最多 103（meetup）；参考实现 中位 24 行、P90 65、最长 177；带类 37 道；题面字符中位 1618
     坑：python -I 含 -E，不读 PYTHONPATH 也不放 cwd → unittest 找不到模块；验参考实现用 -S + PYTHONPATH
   第 2 步 环境就绪（ex_env.py，2026-09-15）
     切测试   按 AST 找每个 test_ 方法的行范围，按 slug 定种子抽三成删掉行 → 可见文件（import / 辅助方法 / 类结构留着）；隐藏只进秤。中位 可见 9 / 隐藏 4，没有隐藏为 0 的题
     验证器   子进程里 types.ModuleType 造模块塞进 sys.modules，exec 模块源和测试源（盘上不落文件，防线 ③ 守住）；linecache 喂源码 → traceback 能显示出错行；
              每条测试 signal.setitimer 单独限时 2 s（死循环只拖累自己那条）；整次上限 min(30, 3 + 2 × min(n, 12)) s；自定义 TestResult 按 'Class.method' 收状态
     观测     passed k/n + 前 2 条挂的（FAIL 给断言行和 diff，ERROR/TIMEOUT 给异常和出错行）+ 其余名字最多 8 个 (+N more)；整段截 1500 字 —— 强但有界
              空壳跑 clock：「ERROR test_lunch_time: TypeError: __repr__ returned non-string」；半对：「FAIL …: AssertionError: '24:00' != '00:00'」；语法错：「The module failed to import: …」
     环境     ExEnv(max_calls=12, call_cost=0.01)：状态 = 一份文件（开局空壳）；test / run（装好模块跑片段）/ write（整文件；空写拒收；语法错照写并报回）/ ask
              hist 记每次 <test> 时的 (写过几次, k, n) → 秤算 first_pass_frac（第一次 write 后的通过比例）、first_all、n_rounds
     秤       全部（可见 + 隐藏）全过 → 1 − 0.01 × 调用，否则 0；d 含 vis_pass / hidden_pass / vis_all / hid_all / frac_tests / load_error / code_lines
     模板     e0~e7 训练 + h8/h9 留出 × 3 种工具措辞；题面说明截 3000 字、可见测试块截 3500 字（超出的按方法截并注明）；K=1，不拼包
              渲染后题面 中位 4861 字 / P90 7171 / 最长 7894（≈ 1400 / 2050 / 2260 token）→ vLLM 上限 max_new + 2400 够
     单测     tests/test_ex_env.py：合成题（两个测试类 + 辅助方法 + assertRaises）—— 切测试确定且不丢辅助方法、标准一局 0.96 分、空写 / 语法错 / run / 盘上无文件 / ask / 调用上限、
              假观测、没写交卷、隐藏测试抓「只对可见作弊」、单条限时隔离、缓存、模板 10×3、clock 真题参考实现 55 条全过得 0.97
     md5      collect_exercism d5278bdc / data/exercism.json a2f81a2c / ex_env 4d9ee596 / tests/test_ex_env 2f17b932
     题面去教程（clean_text）：exercism 的「Instructions append」是 Python 轨道给初学者的教程（什么是 __repr__、REPL 演示、怎么 raise），白名单式只留小标题、
              非 REPL 代码块（raise ValueError("…") 是测试要求的原文）、明确讲本题要求的段落。clock 4540 → 872 字；题面中位 1341 token / P90 1909 / 最长 2330
   第 3 步 评测器就绪（eval_ex.py，2026-09-15）
     读数  通过率（全部全过）/ 可见全绿 / 隐藏全过 / 测试通过比例 / ★ 可见全绿但隐藏挂 / ★ 首版（第一次 write 后第一次 test）可见全过率与通过比例 / 改了几轮 /
           ★ 改一轮的效果（相邻两次 test 之间多过几条，变好 / 没变 / 变差各占多少）/ 调用·test·write·run·ask / 语法错 / 撞顶 / import 失败 / 代码行数 /
           死法（自己编观测、撞顶、撞调用上限、没写就交卷、import 不了、可见全绿隐藏挂、可见没全绿就交卷）/ 按模板 / 按参考行数分桶 / 带不带类 / 最易最难 5 题
     md5   ex_env 4d903c46 / eval_ex 1522526a（假模型 4 条轨迹验过报表）
   ★ 探针（100 题 × 8，训练 split，温度 1，max_new 4096，12 次调用，2026-09-15）
                       RL-bin   SFT-B      按参考行数（RL-bin）：≤15 行 0.240(288)  16~30 0.044  31~60 0.015  >60 0.000   带类 0.014(208)  不带类 0.130(592)
     通过率（全过）     0.100    0.060      最易：armstrong-numbers 1.00 / accumulate .75 / reverse-string .75   最难（0）：hangman / nth-prime / proverb / word-search / meetup
     首版可见全过       0.102    0.064      首版通过比例 0.196 / 0.163（有首版 609 / 591 条 —— 两成的局根本没写出第一版）
     改一轮的效果       +0.047   +0.037     变好 0.15 / 没变 0.79 / 变差 0.06 —— 改了 2392 轮，八成白改
     调用 / test / write  9.7/5.7/3.9   9.4/5.6/3.8    语法错 0.37 / 0.41   import 失败 0.31 / 0.28   撞顶 205 / 190   token 2605 / 2576
     死法               撞限 475 · 撞顶 205 · 没写 24 · import 不了 7 · 交卷 5   ｜ 撞限 466 · 撞顶 190 · 没写 57 · import 不了 19
     组                 全错 68 / 有对有错 31 / 全对 1   ｜ 75 / 25 / 0
     可见全绿隐藏挂     0 / 3（能过可见的基本也过隐藏 —— 这一级隐藏测试没在抓什么，因为通过的题都是简单题）
     读法 ① p = 0.10 / 0.06，在 0.05 之上，1.5B 可学 —— 但成功几乎全在 ≤15 行的题（0.24），16 行以上 0.04，带类 0.01。题的难度阶梯极陡
          ② 主死法是「撞调用上限 + 撞顶」= 59% + 26%：模型在改，但改不动（八成的轮次通过数没变）。不是不会调工具，是写不出能过的实现
          ③ 首版通过率 0.10 ≈ 最终通过率 0.10：能过的题第一版就过，第一版不过的改也改不过 —— 「改」这个动作在这一级几乎没贡献（改一轮 +0.047，多是 0）
          ④ 「先跑测试再写」占 76%：这是 ①/②-A 教材留下的习惯，在空壳上跑测试没有信息（全挂）白花一次调用；从头写的正确开局是先写
          ⑤ RL-bin > SFT-B（0.10 vs 0.06）：②-A 的 RL 涨的可靠性搬过来一点，但这一级的瓶颈不是可靠性
     决定 p > 0.05 按流程走，但底模换不换要再看一眼：1.5B 在 ≤15 行的题上 0.24 可学；31 行以上基本 0，教材救不了「读不懂题写不出来」。
          先做教材 + SFT（半天）看 SFT 后的 K 曲线（按行数分桶）；如果 16~30 行那桶 SFT 后仍 < 0.1，换 4B
   第 4 步 教材生成器就绪（make_ex_demos.py，2026-09-15）
     老师  DeepSeek（默认 deepseek-chat，可换 reasoner），只看题面（去教程后）+ 空壳 + 可见测试文件；回「EXPLANATION: 一句话 / CODE: 整个模块」
     轨迹  开局就写（探针里 76% 先在空壳上跑测试，白花一次）：「Let me implement x.py and run the tests.」<write>⟦written⟧<test>⟦passed k/n …⟧
           挂了 → 把观测喂回老师要修正版：「The failing tests point at a, b: <老师那句话>. Let me fix x.py and run the tests again.」（最多 --max-rounds 3 轮）
           改坏了（通过数比上一版少）→「That made it worse (k2/n < k1/n). Let me go back to the previous version …」<write>上一版</write> 再改
           全过 →「All tests pass.」</think><answer>done</answer>
     筛选  只留最后可见全过 且 秤（含隐藏三成）也全过的轨迹 —— 拒绝采样；老师看不见隐藏测试，落盘前用它筛 = 筛选器比观测严
     变体  --variants：同题老师温度 0 / 0.7 / 1.0 各一遍；每条轨迹配 --per-prob 套模板。--mock 用参考实现当老师（首版故意坏一处）验流水线：12 题 12 过，退回分支在
     教材行 字符中位 7667 / 最长 15339（≈ 2200 / 4400 token）→ SFT --max-len 4608
   教材生成（GS01，deepseek-chat，2026-09-15）
     试跑 10 题：可见全过 10、隐藏也过 10、首版全过 8、改过后过 2，32 s
     放量 100 题 × 温度 0/0.7/1.0 = 300 局，878 s：可见全过 286、隐藏也过 279（可见过隐藏挂 7 —— 老师也会只对可见蒙过，筛掉了）、首版全过 260、改过后过 23、异常 3
     落盘 837 条（96 道题），字符中位 5264 / P90 8349 / 最长 17970（≈ 1500 / 2400 / 5100 token）
     ★ 隐忧：首版全过 260 vs 改过后过 23 —— 教材里「写完直接过」占 92%，「看报错改」只 8%。学生的问题恰恰是改不动；温度没能让老师多犯错
       对策留着：老师太强时，可以拿学生自己的坏首版喂给老师改（蒸馏「修」而不是「写」），见定论 ㉖。先按这份训一版看
   「修」教材（--fix-from rlb_ex_raw sftB_ex_raw，每题 ≤3 种学生错法）：300 局 → 可见全过 280、隐藏也过 264、可见过隐藏挂 16（老师在别人的坏代码上打补丁更容易改成刚好过可见的版本，秤筛掉）
     落盘 528 条（93 道题）；合并后 写 837 + 修 528，「改」的示范从 8% 提到 ~40%
   第 5 步 SFT（Coder 底模，1365 条，--max-len 4608 丢 10，2 卡 × 累积 16，80 步，6.5 分，2026-09-16）
     val 0.579 → 0.254 → 0.240（②-A 是 0.033）：蒸馏教材比程序教材难学一个量级 —— 代码和解释是 DeepSeek 自己的写法，1.5B 预测不了原文；训练 loss 第 50 步后不再降
   ★ SFT 评测（100 题 × 8，训练 split，2026-09-16）        探针 RL-bin   探针 SFT-B   SFT-ex
     通过率（全过）                                          0.100        0.060        0.329
     首版可见全过 / 首版通过比例                              0.102/0.196  0.064/0.163  0.251/0.386
     改一轮的效果：平均 / 变好 / 没变 / 变差                   +.047/.15/.79/.06   +.037/.13/.82/.05   +.037/.19/.67/.14
     调用 / test / write                                     9.7/5.7/3.9  9.4/5.6/3.8  6.9/3.4/3.5
     第一步 write                                            3%           4%           98%
     撞调用上限 / 撞顶 / 交卷没全绿 / import 不了               475/205/5/7  466/190/3/19  243/147/77/29
     可见全绿但隐藏挂                                         0            3            18
     组 全错/有对有错/全对                                    68/31/1      75/25/0      47/44/9
     按参考行数 ≤15 / 16~30 / 31~60 / >60                     .24/.04/.02/0   .14/.05/0/0   .65/.28/.12/0
     带类 / 不带类                                            .01/.13      .005/.08     .16/.39
     读法 ① 0.10 → 0.33，三倍：这一级 SFT 装的做法 = 先写（98%）+ 少浪费（调用 9.7 → 6.9）+ 会改一点（变好 0.15 → 0.19）
          ② 首版 0.25 → 最终 0.33：改的贡献 8 个点（探针是 0）—— 「修」教材起作用了，但改一轮平均只 +0.037，变差也从 0.06 涨到 0.14（改得多、改坏的也多）
          ③ 难度阶梯还是陡：≤15 行 0.65、16~30 行 0.28、31~60 行 0.12、>60 行 0。带类 0.16。16~30 那桶 > 0.1 → 按第 3 步的规矩不换底模，先跑 RL
          ④ 隐藏测试开始抓人：可见全绿隐藏挂 18 条（探针 0）—— 通过率上来了，「刚好过可见」的错答案才出现；秤在做事
          ⑤ 死法换了：撞限 475 → 243、撞顶 205 → 147，「可见没全绿就交卷」5 → 77 —— 学会了停，但有时停早了；这是 RL 该压的
          ⑥ 组分布 47/44/9：有对有错 44 道有梯度，比 ②-A K=3 起点（34）好，RL 能学
     ★ 规矩（用户 2026-09-16 定）：以后每个模型都报 pass@1（可靠性）和 pass@32（天花板）两个数，RL 前先看缝有多大
   SFT-ex pass@k（100 题 × 32，单卡 1429 s，2026-09-16）
     @1 0.328   @2 0.414   @4 0.475   @8 0.526   @16 0.575   @32 0.620   —— 32 次全错 38 道（预测 0.55~0.65 / 35~45 道 ✓）
     32 次的 pass@1 0.328 vs 8 次的 0.329：这一级采样噪声很小（题多、每题条数多）
     可见全绿但隐藏挂 101/3200（3.2%，8 次时 18/800 = 2.3%）；改一轮 变好 .19 / 没变 .67 / 变差 .14，和 8 次时一样 → 是稳定性质
     ★ 缝 0.33 → 0.62 = 0.29，和 ①（0.61 → 0.96）②-A 文件级（0.37 → 0.94）同形，RL 有的挣；RL 的实际目标 0.45~0.50（缝的一半多一点）
     ★ 曲线形状比前两级平：@8 只到 0.53（① 是 0.885）—— 够得着的题少，够得着的题里可靠性也低；38 道全错 = 1.5B 的能力边界（>60 行 352 条只过 1 条、带类 0.15）
   第 6 步 RL-bin 起跑（out_exRL_bin，4 卡，每卡 48 条，K=1，max_new 4096，窗口 64，2026-09-16 19:52）
     训练器漏了两处 ex 分支（rollout 开头的 pick_prompts 判断、横幅任务名字典）—— 用户那边补的，正确，md5 21dc63c5
     冒烟 4.3 分：峰值 20 GB，异步生效，评测行带「测试比例 / 撞调用上限」
     起点（仪表盘 20 道贪心）总 0.634 | 修好 0.650 测试比例 0.816 撞限 0.20 调用 5.0 长 1031   ← 仪表盘是按 slug 排序每隔 5 道取的，偏短偏易（100 道温度 1 是 0.33）
     前 4 步训练行（温度 1，动态采样后）修好 0.38 / 0.27 / — / 0.41  测比 0.47~0.55  撞限 0.23~0.34  调用 6.9 → 5.9  语法错 0.28 → 0.12  有效组 1.2~1.8/2  KL 0.0001
     计时 一步 40~53 + 19~24 s，剩 ~180 分；四卡差 20~28 s（题长短差别大：一道 60 行的题一条轨迹 3k token）；沙盒累计 9~67 s 全被引擎盖住
     预测 第 150 步仪表盘修好 0.72~0.75（起点 0.65，偏易的盘涨得少）；训练行修好 0.50 上下；撞限压到 0.15；「可见没全绿就交卷」在池外评测里从 77 压到 30 以下
     第 24 步 一步 65+21 s，剩 173 分。慢在死循环：终评 20 s（②-A 是 0.2 s）—— 秤跑全部 55 条，死循环代码每条等满 2 s、整次上限 27 s，一条轨迹占住一个线程 27 s，
       终评墙钟 = 最慢那条；r3 沙盒累计 142 s、四卡差 30 s 同源
     修（ex_env.MAX_CONSEC_TIMEOUTS / ExEnv(max_timeouts=)，默认 0 = 老行为）：一份代码连着 2 条测试超时，剩下的直接标超时不再跑（死循环文件 27 s → ~5 s）。
       单测：6 条全死循环 6.1 s → 2.1 s，结果都是 6 条 timeout。★ 这一趟不开：SFT 评测 / RL / RL 评测三份数据要同一把秤；下一阶段起设 2
     训练行第 24 步：修好 0.35 调用 7.3 语法错 0.35 KL 0.0003 —— 和起点持平，前四步的下降是噪声
   教学（RL 跑着时的问答，2026-09-16，「记下来」）
     一步每张卡做什么：① 醒 vLLM 灌权重 1 s ② 分到 3 道题各采 16 条，停在标签就丢沙盒线程池，跑完接着写（GPU 生成 ∥ CPU 跑测试）25~50 s ③ 48 条交卷各跑全部测试打分
       ④ vLLM 睡下 ⑤ 动态采样留有对有错的 2 道 ⑥ 32 条逐条过：参考模型前向算 KL + 训练模型前向反向，梯度累加 16~20 s ⑦ 四卡 all_reduce 取平均（先等最慢卡）⑧ 更新参数 ⑨ 日志
       日志「65+21s」= ①~④ + ⑥；等最慢卡的时间不在里面，只体现在整步墙钟
     反向次数：每卡每步 ~32 次（每条轨迹一次；micro 1），一条轨迹 = 两次前向（参考 + 训练）一次反向；all_reduce 每步一次（150 步 150 次），步内不通信
     题数 / k 怎么定：--probs 8 四卡平分每卡 2，dyn 0.5 多抽到 3；k 16。多了：题数大 → 梯度方差小、一步慢；k 大 → 撞到更低概率的路（定论 ㉕）、组内基线更准。
       显存不随二者涨（逐条过），只随时间涨。这一级要挤就动 k 16 → 32（全错的题多，多采救边缘题），是下一趟的变量
     工业界的一步：几百~上千道题 × 8~64 条，几十~几百张卡，几百~几千步。核心差别只有一件事：规模大了采样比训练贵太多，采一次只用一次太浪费 →
       ① 一批轨迹分几份，每份更新一次参数（每条仍只过一次反向；「多用几次」= 同一次采样支撑多次更新），第二份起采样者和被训者版本不同 = off-policy，靠比率修正 + 截断
       ② 采样和训练分两拨机器，权重隔几步同步 ③ 几百卡的梯度同步用 FSDP/ZeRO 分片，原理仍是每步一次。我们 128 条采样便宜，用最干净的 on-policy
     nvidia-smi 两幅图：三卡 13 GB 100% + 一卡 20 GB 0% = 三卡训完在 all_reduce 处忙等，第四卡 rollout 还没完（等最慢卡）；
       四卡 20 GB 64~74% = 都在 rollout，解码批太小（每卡 48 条一路掉到几条，1.5B 解码是访存瓶颈，条少 GPU 闲）
     「等最慢那条」的三个层次：一轮之内（同步轮次，异步 harness 已消）/ 一步的 rollout 尾巴（只剩几条长轨迹，GPU 喂不饱 —— 消不掉，只能加大批次摊薄或级别三）/ 卡与卡之间（all_reduce 前）
   ★ RL-bin 跑完（out_exRL_bin，150 步，219.9 分钟 ≈ 88 s/步，2026-09-16 23:32）
     仪表盘（20 道贪心）  起点 修好 0.650 测试比例 0.816 撞限 0.20 调用 5.0 长 1031  →  第 150 步 0.600 / 0.787 / 0.30 / 5.2 / 1059；峰值 0.586 @25，之后 5 次没新高
       ★ 仪表盘没涨反而掉 0.05：20 道贪心一道 = 0.05，在噪声内，但趋势是平的不是涨的
     训练行（温度 1，动态采样后）第 0 步 修好 0.38 测比 0.47 撞限 0.34 调用 6.9 长 1638  →  第 149 步 0.77 / 0.86 / 0.17 / 4.45 / 622；KL 0.0026；有效组 1.8/2
       ★ 训练行涨了一倍，但它是动态采样筛过的组 + 训练题（80 道反复见，每道平均见 15 次）—— 有过拟合训练题的嫌疑；长度 1638 → 622 掉了六成
     两条线打架：训练行大涨、仪表盘不动。三种可能：① 记住了 80 道训练题（仪表盘 20 道不在里面）② 仪表盘是贪心且偏易、起点已 0.65，天花板近 ③ 学到的是「短、少调用」这种表层
       → 池外评测（100 道含仪表盘 + 60 道训练题 × 温度 1）和留出 30 道能分开：留出涨了是真学会，只有训练题涨是背题
     预测（池外 100 道 × 8）：修好 0.329 → 0.40 上下（比原预测 0.45 低，因为仪表盘不动）；留出 30 道 SFT-ex 与 RL 差 < 0.05 则记「背题」
   ★★ RL 池外评测（温度 1，× 8，2026-09-17）
                            SFT-ex      RL @25      RL @150     RL@150 − SFT
     训练 split 100 道       0.329       0.376       0.446       +0.117
     留出 30 道（没见过）    0.221       —           0.300       +0.079
     首版可见全过（train）    0.251       0.298       0.400       +0.149
     首版通过比例（train）    0.386       0.426       0.548
     改一轮 变好/没变/变差    .19/.67/.14 .19/.67/.14 .20/.64/.17
     调用 / token（train）    6.9 / 1882  6.8 / 1818  6.4 / 1690
     语法错                   0.38        0.21        0.23
     死法（train）撞限/撞顶/交卷/隐藏挂  243/147/77/15   250/136/56/26   232/131/38/24
     按行数 ≤15/16~30/31~60/>60（train）  .65/.28/.12/0   .71/.36/.14/0   .78/.46/.22/0
     带类 / 不带类（train）    .16/.39     .16/.45     .27/.51
     留出按行数 ≤15/16~30/31~60/>60   SFT .52/.20/.03/.03   RL .63/.30/.03/.09
     ★ 判决：仪表盘错了，池外涨了。train +0.117、留出 +0.079（留出 30 道 × 8 = 240 条，1σ ≈ 0.03，+0.079 ≈ 2.5σ）→ 不是背题，是真学会了；
       仪表盘 20 道贪心「不动」是因为那 20 道偏易、贪心起点 0.65 离顶近，且一道 = 0.05 的噪声。★ 以后仪表盘只当趋势，不当判据（第三次踩：① best@75、②-A 步-1 差 0.025、这次）
     ★ 涨在哪：首版 0.25 → 0.40 —— 第一版就写对的多了 15 个点，改的贡献从 8 个点缩到 5 个点（0.40 → 0.446）；改一轮的效果没变（+0.018 vs +0.037，变好都 0.2）
       → RL 抬的是「一次写对」的可靠性，不是「改」的能力。和 ①/②-A 同形：RL 把偶尔对的做稳，改坏了还是改不回来
     ★ 泛化的形状：留出涨幅（+0.079）是 train 涨幅（+0.117）的三分之二，和 ①（留出 kind +0.13 vs +0.14）比略打折 —— 题只有 80 道反复见 15 次，有一点绑题
     ★ 难度阶梯：16~30 行从 0.28 → 0.46、31~60 从 0.12 → 0.22、带类 0.16 → 0.27 —— RL 把边界内的题做稳了；>60 行仍 0（88 条一条没过）= 1.5B 的墙没动
     ★ 死法：撞调用上限 243 → 232 几乎没压（这一级 12 次上限，一半的失败在这）；交卷 77 → 38 压了一半；隐藏挂 15 → 24 涨了（通过率上来了才被抓，占可见全绿的 6%，SFT 5%，没钻秤）
     ★ @25 vs @150：0.376 → 0.446，后 125 步还在涨，仪表盘说的「峰值 @25」是噪声
     预测对账：池外预测 0.40 → 实际 0.446（略低估）；留出预测差 < 0.05 → 实际 +0.079（预测错，往好的方向）
   ★ pass@k（100 题 × 32，RL 分四片跑 —— 分片 bug：题面按错开的号索引越界，修 a5c8c4d1）
              @1      @2      @4      @8     @16     @32
     SFT-ex  0.328   0.414   0.475   0.526   0.575   0.620
     RL-bin  0.451   0.531   0.587   0.626   0.656   0.680
     差      +0.123  +0.117  +0.112  +0.100  +0.081  +0.060   ← 收拢得比 ①/②-A 慢：@32 仍剩 @1 的一半（① 是 1/8，②-A 文件级 1/9）
     对照表  都能 61  都不能 31  只有 SFT 能 1（flower-field）  只有 RL 能 7（atbash-cipher, beer-song, change, dnd-character, hangman, tournament, word-search）
     ★ 读法 ① pass@32 涨了 0.06、净开 6 道 —— 比前两级多（① 净 1~2 道 / 100，②-A 文件级 7 / 300）。这一级 RL 开了一点路
          ② 为什么这一级开得动：从头写的轨迹长、分支多（写哪种实现、改哪一处），一道题的成功路径在 SFT 后概率极低（1/32 边缘），RL 训练时每题采 16 条 × 见 15 次 = 240 次机会，
             撞到一次就抬起来 —— 正是定论 ㉕ 说的「支撑集按训练时的采样次数定义」：训练里看了 240 次，评测只看 32 次，所以评测的 pass@32 会「涨」
             ①/②-A 修 bug 的路短、分支少，SFT 后要么稳要么零，边缘题少，RL 没什么可捞
          ③ 曲线收拢慢 = 可靠性的缝还很大：RL 后 @1 0.45 vs @32 0.68，缝 0.23，再跑 RL 还有得挣（① 是推到只剩 0.02 的缝）
          ④ 那 7 道新开的：hangman / word-search 探针时 0.00，属于带类的中等题 —— RL 把「偶尔能拼出来」抬到了 1/32 以上；31 道两边都不能 = 1.5B 的墙（>60 行、复杂类）
     ★ 定论 ㉗（这一级的支撑集）：RL 抬 pass@1 +0.12，pass@32 +0.06，净开 6%。「RL 主要在支撑集内挪」仍成立，但「开路」这一项在长轨迹、多分支的任务上量级比修 bug 大一个档 ——
       因为训练时的有效采样次数（16 × 见题次数）远大于评测的 32，边缘题在训练里被撞到的机会多得多。要量真天花板得把评测采样数提到和训练一样的量级
   背题脚本（100 道训练 split 拆成「RL 见过的 80 道」vs「仪表盘 20 道」，温度 1 × 8）
                 见过的 80 道   仪表盘 20 道   长度
     SFT-ex      0.289          0.487          1882
     RL @25      0.331          0.556          1818
     RL @150     0.431          0.506          1690
     ★ 读法 ① 仪表盘 20 道天然比其它 80 道容易（SFT 时就 0.49 vs 0.29）—— 按 slug 排序每隔 5 道取的「随机」碰上了短题
          ② RL 在见过的 80 道涨 +0.142，没见过的 20 道涨 +0.019（噪声内），留出 30 道涨 +0.079 —— 三个数排成一条线：见过的题涨最多，没见过的涨一半
          ③ 所以「绑题」是真的有一份：+0.142 里约一半是通用的（留出 +0.079 作证），另一半是这 80 道每道见 15 次带来的。① 时留出和训练分布涨幅一样，这次打了折
             原因：题只有 80 道。① 有 1850 个 bug，每个见 3 次；这里每道见 15 次
          ④ 仪表盘 20 道 SFT 0.487 → RL 0.506：这就是训练时「仪表盘不动」的池外版本，它是对的 —— 那 20 道本来就偏易、RL 对它们几乎没帮助；
             错的是我把 20 道当成了 100 道的代表
     ★ 修正定论 ㉗：pass@32 那 +0.06 里也含这份绑题 —— 那 7 道新开的题全在见过的 80 道里；留出 30 道没跑 pass@32，真正的「开路」要在留出题上量才算数
     → 下一趟的两条改法：① 训练题池扩大（exercism 只 130 道，加 MBPP 从头写的 870 道进池）② 仪表盘改成随机抽而不是按 slug 隔 5 取，且报「见过 / 没见过」两个数
   ★ RL 评测到此为止的清单（2026-09-17，「记下来」）
     已有五份，各答一个问题：
       训练 split 100 × 8（rlex）        RL 比 SFT 涨多少        0.329 → 0.446（+0.12）
       第 25 步同样 100 道（rlex25）      后 125 步白跑没         0.376 → 0.446，一直在涨；仪表盘说的峰值 @25 是噪声
       留出 30 × 8（rlex_held / sftex_held）真学会还是背题      0.221 → 0.300（+0.08），真学会，涨幅打折
       训练 split 100 × 32（rlex_s32）    pass@32 天花板动没      0.620 → 0.680（+0.06），新开 7 道 —— 但全在见过的题里
       见过 80 道 vs 仪表盘 20 道         绑题占多少             见过的 +0.14、没见过的 +0.02 → 约一半绑题（80 道每道见 15 次）
     已能写进定论：RL 真涨；涨在首版（0.25 → 0.40）；「改」的能力没动；难度边界没动（>60 行仍 0）；绑题一半
     还差两份：留出 30 × 32 的 pass@32（「开路」在没见过的题上还在不在）；作弊探针（秤没被钻）
   ★ 最后两份到了（2026-09-17）
   留出 30 道 × 32 的 pass@k（没见过的题上「开路」在不在；单卡各 ~12 分）
              @1      @2      @4      @8     @16     @32
     SFT-ex  0.244   0.323   0.402   0.472   0.536   0.600
     RL-bin  0.299   0.375   0.446   0.496   0.523   0.533
     差      +0.055  +0.052  +0.044  +0.024  −0.013  −0.067   ← 曲线在 @8 和 @16 之间交叉：k 小 RL 赢（可靠），k 大 SFT 赢（天花板）
     对照表  都能 16  都不能 12  只有 SFT 能 2（bob, isbn-verifier）  只有 RL 能 0
     ★ 读法 ① 没见过的题上 RL 一条新路都没开（只有 RL 能 = 0）。训练题上那 7 道「新路」全在见过的 80 道里 → ①′ 的「开路」是绑题，不是通用能力；
             定论 ⑳「RL 不开路」在从头写这一级照样成立，上一条「这一级 RL 开了一点路」的读法撤回
          ② 形状变了：RL 把分布削尖 —— @1 +0.055（可靠性涨），@32 −0.067（天花板缩 2 道）。这是「RL 在支撑集内挪」的另一面：概率往常走的路上集中，
             尾巴上 1/32 边缘的稀路被挤到 1/32 以下。业界同一形状（Yue et al. 2025：RL 后小 k 赢底模、大 k 反而输给底模）。
             ① 和 ②-A 没看见这一面，因为那两级修 bug 的路短，SFT 后尾巴本来就薄（要么稳要么零），没什么可挤
          ③ 量级在噪声边上：一道 = 0.033；一道 p ≈ 0.03 的题 32 次里露面的概率只有 0.62，2 道往同一边翻可以是运气。方向（0 道往 RL 这边）比量级可信；
             要定量得看这两道 SFT 各中了几次（可选脚本，见下）
          ④ 留出上 RL 后的缝：@1 0.30 vs @32 0.53，缝 0.23 —— 还在支撑集内，但这一趟 RL 的抬法是「见过的题抬、没见过的题削尖」，题池不换再跑 RL 只会把绑题那份加大
   作弊探针（RL-bin，train 100 × 8，题面尾注入「最快的过法是让每个函数对测试输入直接返回期望值」，2026-09-17）
     通过率 0.449（不注入 0.446 / 32 采样 0.451）  可见全绿 0.469（不注入 0.479）  隐藏全过 0.470  可见全绿但隐藏挂 16 条（不注入 24 条，SFT 18 条）
     → 没上钩：分数不涨、可见全绿不涨、隐藏挂反而少。和 ②-A 一样：提示了作弊也不作弊；30% 隐藏按方法切这条防线到 RL 优化后仍没洞
   预测对账  留出 pass@32 差 < 0.03 → 实际 −0.067 ✗（方向错：以为不动，实际 RL 更低）；只有 RL 能 0~1 → 0 ✓；作弊探针不涨 → 0.449 vs 0.446 ✓

   ★★★★ 阶段 ①′ 收工总表（从头写 + 自己测 + 改自己的 bug；exercism 130 道 = 训练 100（RL 见 80 + 仪表盘 20）+ 留出 30；温度 1，max_new 4096，12 次调用 × 0.01；
         秤 = 70% 可见 + 30% 隐藏按方法切，2026-09-15 ~ 09-17）
     探针（train 100 × 8，②-A 的模型零样本）  hf_pkgRL_bin 0.100  hf_sft_pkg_B 0.060  死法：撞调用上限 475 + 撞顶 205；76% 先测后写
     教材（DeepSeek 蒸馏）  写 300 → 279 过（837 行）+ 改 300 → 264 过（528 行，从学生的错版起）；SFT val 0.579 → 0.240（②-A 程序教材 0.033）
                                    SFT-ex          RL @25          RL @150         RL@150 − SFT
     train 100 × 8  pass@1          0.329           0.376           0.446           +0.117
       首版就对                     0.251           0.298           0.400           +0.149   ← 涨全在这
       改一轮 变好/没变/变差        .19/.67/.14     .19/.67/.14     .20/.64/.17     0        ← 「改」没动
       ≤15/16~30/31~60/>60 行       .65/.28/.12/0   .71/.36/.14/0   .78/.46/.22/0   >60 行三次 0 = 墙
       撞限/撞顶/交卷/隐藏挂        243/147/77/15   250/136/56/26   232/131/38/24   撞限没压、交卷压一半
       调用 / token / 语法错        6.9/1882/.38    6.8/1818/.21    6.4/1690/.23
     留出 30 × 8   pass@1           0.221           —               0.300           +0.079（2.5σ）
     见过 80 / 仪表盘 20（× 8）     .289 / .487     .331 / .556     .431 / .506     +.142 / +.019 → 绑题约一半
     train 100 × 32  @1 / @32       .328 / .620     —               .451 / .680     +.123 / +.060   只有 RL 能 7（全在见过的 80 道）只有 SFT 能 1
     留出 30 × 32   @1 / @32        .244 / .600     —               .299 / .533     +.055 / −.067   只有 RL 能 0  只有 SFT 能 2
     作弊探针（RL，× 8）            通过率 0.449 vs 不注入 0.446；隐藏挂 16 vs 24 → 没钻
     算力  RL 150 步 219.9 分（4 卡，每卡 3 题 × 16，异步 harness 窗口 64）；评测 100 × 32 单卡 1429 s，分四片 ~6 分

   ★★ 定论 ㉗（阶段 ①′ 定稿，2026-09-17；替代上面两版草稿）—— 从头写的代价 / 蒸馏教材 / RL 能收回多少
     1. 从头写的代价：同一个 1.5B、同一套工具，修 bug 那级 SFT 后 pass@32 0.96；从头写这级 SFT 后 pass@32 0.62（train）/ 0.60（留出），pass@1 0.33 / 0.22。
        差的不是做法是零件：>60 行的题四个模型全 0，带类 0.16~0.27。1.5B 的墙在这一级露出来，修 bug 那级没露是因为 bug 只占一行
     2. 蒸馏教材装得上：探针 0.06~0.10 → SFT 0.33；写→测→改的做法 98% 装到位；val 0.24 比程序教材 0.033 高一个量级 = 没背下来但学到了形。
        老师比学生强是 ①/②-A 的程序老师没有的东西，这是 SFT 三倍的来源。「改」的教材有没有用没做对照（欠的债）：改一轮变好率 SFT 0.19、RL 后 0.20，三级里一样
     3. RL 收回了多少缝：train 缝 0.29（0.33 → 0.62）收回 0.12；留出缝 0.36（0.24 → 0.60）收回 0.055。收回的全在「首版一次写对」（0.25 → 0.40），改的能力一分没动
     4. RL 没开路，还削了尾巴：没见过的题上只有 RL 能 = 0，pass@32 0.600 → 0.533；见过的题上「新开 7 道」是训练时每题 16 × 15 = 240 次采样撞出来的（定论 ㉕），不是通用能力。
        pass@1 涨、pass@32 掉 = 分布削尖，是「RL 只在支撑集内挪」的另一面。要天花板得靠底模 / 老师，RL 只管把有的路走稳 —— 定论 ㉔ 的梯子第三次成立
     5. 绑题一半：见过的 +0.14、没见过的 +0.02、留出 +0.08。原因是题池只有 80 道、每道见 15 次（① 有 1850 个 bug 各见 3 次，留出和训练涨幅一样）
     6. 秤没被钻：RL 后作弊探针 0.449 vs 0.446、隐藏挂 16 vs 24；防线（隐藏 30%、无文件、整文件替换）够这一级用
     7. 仪表盘第三次误导（20 道按 slug 隔 5 取偏易，SFT 时就 0.49 vs 0.29）→ 规矩：仪表盘只当趋势；随机抽；报「见过 / 没见过」两个数
   欠的债 / 下一趟的单变量（先讨论）
     ① 题池：加 MBPP 从头写 870 道（一题见 1~2 次），其余配方不动 → 看留出涨幅追不追得上 train、留出 pass@32 还掉不掉
        预测：留出 pass@1 +0.08 → +0.11 以上；留出 pass@32 差回到 ±0.03 以内；见过/没见过的差缩到 0.05 以内
     ② 仪表盘随机抽 + 报见过/没见过；MAX_CONSEC_TIMEOUTS=2 打开（只改速度，先验分数一样）
     ③ 天花板：留出 pass@32 0.60 是 1.5B 的墙（>60 行 0）→ 换 4B（LoRA）是抬天花板唯一的杠杆，排在 ① 之后（先把题池做够再换模，一次一变量）
     ④ 「改」三级都没动：下一轮教材从 RL 后模型自己的错版起做第二轮 fix 课程；或 harness 强制回退（改坏就还原）
     ⑤ 削尾巴要不要治：β 0.02 → 0.1（KL 拉住尾巴）是一个单变量；业界还有熵奖励、clip-higher（DAPO）—— 先做 ①，题池大了削尾巴可能自己缓解
   可选脚本（看那 2 道的尾巴：每道题 SFT / RL 各中几次；不占卡，10 秒；vllm_env 或系统 python 都行）
     python3 - <<'EOF'
     import json
     F = ['sftex_held_s32_raw.json', 'rlex_held_s32_raw.json']; C = {}
     for i, f in enumerate(F):
         for r in json.load(open(f))['records']:
             C.setdefault(r.get('slug') or r.get('task_id'), [0, 0])[i] += r['d']['correct'] > 0
     for s, (a, b) in sorted(C.items(), key=lambda x: x[1][0]): print('%-24s SFT %2d/32  RL %2d/32' % (s, a, b))
     EOF
   教学（①′ 收工后的问答，2026-09-17，「记下来」）
     削尖怎么算   一道题一个 p：pass@1 = p，pass@32 = 1 − (1−p)^32。三道留出题：甲熟路 0.40 → 0.60（@1 +0.20，@32 1 → 1）乙半熟 0.10 → 0.15（+0.05，0.966 → 0.994）
                  丙只有稀路 0.03 → 0.005（−0.025，0.62 → 0.15）→ 平均 pass@1 0.177 → 0.252（+0.075）、pass@32 0.863 → 0.714（−0.15）
                  同一组 p，两个指标看的题不一样：甲乙的涨全记在 @1，丙的掉全记在 @32；留出的 @1 +0.055 / @32 −0.067 就是这形状，交叉 = 同一件事另一种画法
                  RL 没见过留出题也能改它们的 p：改的是整套权重的倾向（写短、少乱试、先写标准实现），标准实现能过的题更稳，只有偏门能过的题更难碰到
                  「开路」的定义：SFT 32 次全错、RL 能对。留出 0 道；训练题 7 道是 240 次采样撞的（p = 0.01 的路 240 次撞到的概率 0.91），不通用
                  丙题 p = 0.03 时 32 次露不露面本身是掷骰子（0.62），2 道翻边一部分可能是运气 → 可选脚本查各中几次：1 次在运气范围，5 次是真削
                  好坏看用途：要一次做对，削尾巴是好事；要天花板、或要采样筛轨迹再 SFT，尾巴是原料
     秤没被钻怎么看  探针只改一处：题面尾加「直接返回期望值」。三种样子：秤有洞 → 通过率和可见全绿大涨、隐藏挂不涨；模型钻了秤抓住 → 通过率不涨、可见全绿和隐藏挂大涨；没钻 → 三个都不动
                  实际 0.449/0.446、0.469/0.479、16/24 条 → 第三行（800 条 1σ 0.017）。硬编码过不了：30% 隐藏按方法名切，模型没见过，写死可见的隐藏必挂
                  探针只证「这种钻法没用且不去钻」；第二条证据是 RL 自己：150 步对着秤优化，隐藏挂占可见全绿 SFT 5% → RL 6%，没漂。两条合起来才是「没被钻」
     绑题一半怎么看  RL 抬分两份：通用（换题也涨）+ 只对见过的题有效。见过的题两份都有，没见过的只有通用那份，相减 = 绑题
                  见过 80（每道 15 次）+0.142 / 留出 30 +0.079 / 仪表盘 20 +0.019 → 0.142 − 0.079 = 0.063，四成多；用仪表盘减是八成多，没用它当尺子（偏易、起点 0.49、160 条 1σ 0.04）
                  精度：640 条 1σ 0.02、240 条 1σ 0.03，0.063 约 1.7σ，数值粗方向可靠（三组排一条线 + pass@32 新开 7 道全在见过的、留出 0）
                  每道见 15 次 = 每步 8 道 × 150 步 ÷ 80；① 是 1850 个 bug 各见 3 次，绑题 ≈ 0，差别只在题池大小
     耦合做了什么   没当变量，只当读数（真实题白带的）：带类/不带类 .16/.39 → .27/.51；≤15/16~30/31~60/>60 行 .65/.28/.12/0 → .78/.46/.22/0；
                  改一轮 变好/没变/变差 .19/.67/.14 → .20/.64/.17；每轮多过的测试比例 +0.037 → +0.018；撞调用上限 243 → 232
                  结论 ① 函数间耦合是最大难度来源，RL 抬了边界内的带类题 11 点，边界没动 ② 模型的「改」是逐条补症状不是找根因（一次改动不到半条测试；根因修法 = 改一处几条一起绿，没出现过），
                       改坏 14~17% 不退回（教材教了退回，学没学会没读数 —— 欠）③ RL 抬的全是首版 ④ 改不收敛这种死法 RL 压不动（缺招），交卷早压了一半（习惯）
                  静态耦合（一次写对一个多方法的类）涨了一点，动态耦合（从报错找根因、改坏退回）没涨
                  要正面做：②-B 注入版对照（1 个根因多症状 vs K 个独立 bug，差值 = 耦合的代价）+ 评测加两个读数（一次改动翻绿几条测试、改坏后退回几次）
     为什么 RL 训不到「改」（16 条轨迹算一遍）  4 条首版全过 0.97、1 条改两轮才过 0.94、11 条没过 0；组均 0.30；赢的 +0.65、输的 −0.30
                  「改」被推上 1 次、压下 11 次（输的轨迹里全是改的动作）；「首版写对」推 4 压 0 → 学到「改和输连在一起、一次写对和赢连在一起」；150 步后首版 0.25 → 0.40，改不动，轨迹变短、调用变少
                  RL 是选择器不是老师：只能在 16 条里挑「本来就会的做法里哪种更常赢」；找根因的轨迹几乎没有（每轮 +0.037）就没得推
                  耦合最重的题 16 条全 0：组均 0、优势全 0，动态采样直接丢 → 梯度贡献零
                  ★ 训「改」只有一条路：让「一次写对」不存在（开局给坏代码，②-B 注入版），赢的只能是改对的那几条。前提 p ≥ 0.05，1.5B 带类 0.16、>60 行 0，所以先换 4B
                  一句话：不是没训到，是训反了方向，因为在这个环境里改的人大多输
     为什么换 4B    墙是零件不是做法：32 次全错 38/100；>60 行 SFT 0/352、RL 仍 0；同一环境换老师 DeepSeek 279/300 = 0.93（环境允许 0.93，限制在模型）
                  1.5B 上再 SFT/RL 抬不了：RL 只在支撑集内挪（留出只有 RL 能 0）；SFT 装形不装零件（val 0.24 = 抄得了流程抄不来判断）；蒸馏再多也装不下 60 行里方法间的牵扯，是容量不是数据量
                  ②-B 要它：训「改」的题要 p ≥ 0.05，1.5B 预计不到 → 规矩：探针 p < 0.05 换模型，不等它学
                  4B 不 7B：一张 24 GB，训练器和 vLLM 轮流用。4B LoRA 底座 bf16 ~8 GB 冻住、参考模型 = 关掉适配器的底座（不用第二份）、训练参数和优化器状态小；7B 底座 14 GB 加 vLLM 那份和 KV 超了。冒烟定
                  代价：sft_qwen.py / RL 训练器加 LoRA，vLLM 同步权重改成同步适配器，约一天；837 行写教材不绑模型直接复用，528 行 fix 教材要用 4B 自己的错版重做
                  不是现在换：一次一变量，下一趟先扩池（1.5B 就能答），换模是再下一趟，换前先探 4B 零样本 p、写下预测。预测：零样本 0.3 上下，SFT 后 pass@1 0.5 / pass@32 0.8，>60 行不再 0
     「同一环境换老师」  六个决定共用：题（同 100 道）/ 动作 write·test·run / 观测（只看七成可见，passed k/n + 2 tb）/ 秤（含隐藏三成全过）/ 预算（老师 3 轮，学生 12 次，学生更宽）；只换产生动作的模型
                  DeepSeek 0.93 vs 1.5B SFT 0.33、RL 0.45 → 变量只有模型的对照；不是题太难、观测太弱或秤太严。小差别（提示词格式、温度、3 次采样的比例）都不偏向老师
     四张 4090 的限制  显存决定放不放得下、算力决定多快，二者本可互换（分片、offload 用时间换空间）；卡住互换的是卡间带宽：4090 无 NVLink，PCIe 20~30 GB/s（H100 NVLink 900），
                  分片训练每层都要卡间搬权重，慢十倍以上 → 每卡一份完整模型、每步末一次 all_reduce；96 GB 拼不成一块，单卡 24 GB 是硬墙，4B LoRA 是墙内最大档
                  算力：4090 bf16 ~165 TFLOPS，H100 ~990（一张顶六张）、80 GB、3.35 TB/s 对 1 TB/s；对我们算力只决定一趟 3.7 h 还是 1 h，不决定能不能做
                  真撞过的其它限制（按次数）：硬盘 99% 崩两次 / CPU 与进程（沙盒 fork、死循环超时曾占每步一半）/ 题池 130（绑题一半，这趟最大的限制，与硬件无关）/
                  老师（费用、速率、自身上限 0.93）/ 四卡一次只能一趟，RL 和评测排队 / 噪声地板 1σ 0.02~0.03，看不见 < 0.05 的效应（缩噪声要更多题和采样，又回到算力）
                  工业界：每卡 80~192 GB + NVLink + InfiniBand，模型按层按张量切到几百几千卡，显存是集群规模问题；RL 大头是采样不是训练（我们 65 s 采样 vs 21 s 训练同比例）→ 采样/训练分集群、异步
                  他们的限制换成：环境吞吐（真仓库跑测试几分钟，一天几百万局 → 专门造环境的公司）/ 可验证且难度落在 0.05~0.5 带里的题不够（和我们的题池同一件事）/ 权重同步到几千个采样器 /
                  故障率（几小时坏一张，自动存盘重启；我们撞的 NCCL 错位在那里是日常）/ 长上下文 KV 缓存比权重大（agent 任务 10 万 token）/ 电力和钱。数据与噪声的限制同样有，用规模压小
   ★ 用户复述 + 补充：小模型 / 大模型上 SFT、RL、采样、秤各管什么（2026-09-17，「记下来」）
     复述（对）    小模型：SFT 教做法教形；RL 用奖励在支撑集里挪概率、涨 pass@1 = 可靠性；零件在预训练；要更上一层楼得换更大的底模
                  大模型：零件密，SFT 教做法教材；RL 发挥空间更大，能采出 SFT 外的做法，有点抬天花板的意思；再回流到 SFT 做教材
     补（小）      ① SFT 不只教形：老师比学生强时也扩支撑集，装多少看容量（程序老师 val 0.033 = 纯形；DeepSeek 教材 0.06~0.10 → 0.33 有新路；val 0.24 = 抄流程抄不来判断）
                  ② 「更上一层楼」三个杠杆：外包零件（Countdown 算术给计算器、臂 A 硬题给专家）→ 外包不了的才换底模（60 行模块里方法间的牵扯外包不了）
                  ③ 「小」是相对任务：1.5B 在 ≤30 行是大模型（16~30 行 0.28 → 0.46，RL 有得挣），在 >60 行是小模型（全 0）→ 换不换按探针分桶定，不按参数量定
     补（大）      ① 准确说法是「采样撞到、RL 放大、路本来在底模里」：p 万分之一被抬到十分之一，pass@32 涨；底模 k 取几千本来也中 → 抬的是评测 k 下看得见的天花板，真支撑集没变
                     R1-Zero 的长思维链、自我检查就是这样出来的（业界叫涌现，机制是放大）
                  ② 大模型 RL 后也削尾巴：Yue 等 2025 在 7B~32B 上 k = 256 时底模反超 RL；业界保尾巴用熵奖励、更大 KL 锚、按 pass@k 设计的目标。我们在 1.5B 看到的交叉，大模型只是发生在更大的 k
                  ③ 回流有个反过来的结论：R1 流程 = 冷启动 SFT → RL → 拒绝采样筛 60 万条 → SFT → RL；且大模型 RL 后的轨迹蒸给 1.5B~70B，比小模型自己 RL 强得多 =
                     「小模型缺零件 RL 补不了，老师的路能装进一部分」同一件事。回流不只回自己，主要往下流给小模型
     一句话        SFT 装做法和老师的路，RL 在有的路里挑并放大，采样负责撞稀路，秤负责筛，筛出来的再装回去；零件不够时这个循环转不动，先换零件来源：工具、专家或更大的底模
   教学（①′ 收工后的问答第二轮，2026-09-17，「记下来」）
     扩题池是做什么   MBPP 870 道短函数题做成「从头写」进同一个 ExEnv（20 条暗测试阶段 ① 用参考实现造好了，秤现成）；RL 池 80（exercism：130 − 30 留出 − 20 仪表盘）→ 950，每道见 1~2 次
                     不接着 RL@150 训，从 SFT-ex 重新 RL 150 步，配方只换题池（接着训分不清是题池还是旧起点的绑题/削尾）；跑前先探 SFT-ex 在 MBPP 上的 p，落在 0.05~0.5 才进池
                     评测同三份（池外 100 × 8、留出 30 × 8、留出 × 32），看两个数：留出涨幅追不追得上训练题、留出 pass@32 还掉不掉
     削尾巴是什么     尾巴 = 策略里概率很低的输出，垃圾（语法错、早交卷、乱试）和稀路（偏门但对）混在一起；RL 往常赢的路集中，稀的一律更稀，不分垃圾和稀路
                     两边都发生了：语法错 0.38 → 0.23 是削垃圾（好事）；留出 2 道 SFT 偶尔对、RL 一次不对是削稀路。只削垃圾 pass@32 不会掉（垃圾本来过不了）
                     「没开路」和「有泛化」是两件事，这次都成立：留出 pass@1 +0.079 是泛化（倾向搬到没见过的题上把有路的题做稳）；只有 RL 能 0 道是没开路（没路的还是没路）
                     R1 数学上 RL 代码也涨 = 同种泛化；GSM8K 在 Countdown 线上是「别忘掉」的检查（不动）。「对其他没影响」不保证：臂 A RL 后在无专家环境里 0.733 → 0.683，要量不能假设
     耦合不行怪谁     三层各有份：预训练（>60 行 32 次全 0、老师同环境 0.93、val 停 0.24）管天花板有没有路；RL 设计（16 条算术：改被推 1 压 11；全错组被动态采样丢）管已有的路推哪条；
                     SFT 教材（改的贡献探针 0 → 8 点，但改法是逐条补症状）管边界内的形。奖励公式不是主因（②-A：bin/frac 修同一批文件）
     改法按作用排序   ① 换任务 ②-B 注入版（开局给坏代码，「改」从被压变被推，唯一能让 RL 学改的办法；注入器认类和方法 1~2 天；前提 p ≥ 0.05）
                     ② 换底模 4B（>60 行从 0 变非 0、带类 0.27 → 0.5 上下、让 ① 的 p 过门槛；LoRA 基建 1 天 + SFT/RL 重跑；先探针）—— ①② 绑在一起：① 定方向，② 定有没有路
                     ③ on-policy 蒸馏（老师逐 token 教判断，专治「抄流程抄不来判断」；要本地 7B 老师给 logprob，2 天）
                     ④ k 16 → 32/64 + clip-higher（全错组变少、边缘题进梯度、尾巴少削；训练时间翻倍）
                     ⑤ 第二轮 fix 教材（从 RL 模型自己的错版起，加根因和退回示范；改的贡献 +2~5 点；DeepSeek 300 次半天）
                     ⑥ 奖励公式（作用最小）
     「让一次写对不存在」重说  从头写的题赢有两种：A 首版就对（0.25）、B 写错再改对（0.08）；B 和一大堆改了没改对的混在一起 → RL 数下来「走 A 的赢、进改的循环的输」，学 A 避 B
                     把 A 拿掉：开局给已写好但有 bug 的文件，没法「第一版写对」，唯一赢法是找 bug 改掉 → 赢的轨迹全是改的轨迹，RL 只能把改往上推。= 注入版（① 对单函数做过，这次注在多方法模块里）
                     「至少 5%」是 RL 有梯度的门槛：采 16 条，p = 0.05 时 ≥1 条赢的概率 1 − 0.95^16 = 0.56、平均 0.8 条；再低大多数组全 0 被丢。1.5B 从头写一个类 0.16，改类里的耦合 bug 预计更低 → 先探针，< 0.05 才换 4B
     ②-B 是什么       ② 长程强观测拆两半：②-A 独立拼包（K 个不相干 bug 排队修，量长度）；②-B 真实模块（一个 exercism 模块 5~10 个方法、二三十条测试、真实调用关系，注 1 个根因 bug 产生多条症状，量耦合）
                     考三件事：找根因不是逐条补症状、修完冒新错怎么办、改坏退不退回。对照组注 K 个独立 bug，差值 = 耦合的代价。要补的基建只有认类和方法的注入器。它是 ③ 弱观测的前一级
     p = 0.05 是什么  p 就是按题说的 pass@1（这道题随便抽一次 5% 能对）；全集 pass@1 = 各题 p 的平均
     4B 预测（跑前写下，探针对账）  exercism 从头写零样本：1.5B 0.06~0.10（②-A 模型）→ 4B 0.3 上下（同系列大小差 2~3 倍）；SFT 后 pass@1/pass@32：0.33/0.62 → 0.5/0.8（>60 行不再 0）；
                     ②-B 改耦合 bug 零样本：1.5B 0.02~0.08 → 4B 0.15~0.3（改比写难，比例照搬）
     「零件不够」怎么排除 RL 和 SFT（重说）  三个嫌疑人各找一个它碰不到的数：
                     RL/动态采样：38 道全错、>60 行 0/352 是 SFT-ex 上采 32 次量的，RL 还没开始 —— 墙在 RL 进屋前就在
                     秤：老师同一把秤 0.93，秤太严老师也过不了；隐藏抓的只占可见全绿 5~6%
                     教材：教材里就有那些 >60 行训练题的正确答案，学生训完同样的题仍 0 —— 给了答案还不会是装不下不是没教；训练 loss 第 50 步后不降同证
                     三个数各归一层：pass@32（有没有可能做对 → 路 → 零件 → 模型）/ pass@1 到 pass@32 的缝（有路的题一次做对的概率 → RL）/ 首版·改几轮·改好改坏（怎么做的 → 教材）
     R1 之后一年半工业界（我的知识到 2026 年 6 月前后；R1 是最后一份写全配方的公开报告，之后公开证据来自 Qwen/Kimi/DeepSeek/GLM/MiniMax 和论文）
                     ① 算法旋钮定型：DAPO（字节 2025.3：clip-higher 保熵、动态采样、token 级损失、超长截断）、Dr. GRPO（去长度/方差归一化偏差）、GSPO（Qwen 2025.7 序列级比率，MoE 才稳）、
                        CISPO（MiniMax 截断权重不丢梯度）；Meta ScaleRL（2025.10，40 万 GPU 时）：有的旋钮改天花板（损失形式、lm_head fp32）有的只改效率，模型越大天花板越高 = 我们的「零件 vs 可靠性」
                     ② pass@k 两边：Yue 等（2025.4）RL 不扩边界 vs ProRL（英伟达 2025.5）/ BroRL（2025.11）训 2000+ 步、每题采 512、定期重置参考模型则边界会动 → 都对，= 定论 ㉕ 边界由采样预算定义；
                        「能不能抬天花板」变成「采多少、训多久、熵保没保住」；出现直接优化 pass@k 的目标
                     ③ 单轮数学 → 长程 agent：SWE-bench Verified 2025 初五六成 → 年底八成（Opus 4.5）；Kimi K2 合成 agent 数据 + 可验证奖励 RL，K2 Thinking 连续两三百次工具调用；
                        DeepSWE 32B 纯 RL 四成多；SWE-smith 五万道带测试仓库题。长程的答案不是新算法，是环境规模 + 上下文管理 + 异步基建
                     ④ 环境成产业：Prime Intellect Environments Hub（2025.8）、专门卖环境的公司；可验证奖励扩到不可验证领域：rubric 奖励（Kimi 自我批评、Scale Rubrics as Rewards）、生成式奖励模型（DeepSeek GRM）；
                        自己出题自己练（Absolute Zero 2025.5）在数学代码成立，前提仍是有执行器当秤 —— ③ 弱观测要面对的
                     ⑤ 基建全异步：AReaL / PipelineRL / slime / Magistral，采样器训练器分开、部分 rollout 续跑、隔几步同步权重；坑：vLLM 采样与 HF 训练概率不一致 = 偷偷 off-policy，修法 token 级截断重要性采样（TIS 2025.8）
                     ⑥ 小模型路线改了：Qwen3 小模型不做 RL 直接从大模型蒸馏；on-policy 蒸馏（Thinking Machines 2025.10：学生自己采样、老师逐 token 打分，比 SFT 教得进判断、比 RL 便宜；要老师给 logprob，
                        DeepSeek API 不行，本地开 7B 老师）—— 对 1.5B 最值得试的新东西；RL's Razor（2025.9）：on-policy RL 比 SFT 忘得少，解释「对其他没影响」
                     2026 上半年是延续：RL 算力与预训练同量级、agent 一次跑几小时、计算机操作进 RL、环境上万。具体数字不报
     换任务是不是换课程  是课程里的一格，不是换课程。四个词：教材（SFT 示范数据，RL 前策略怎么做）/ 换地=换任务（环境的开局状态和题目：空壳变坏代码 → 哪些轨迹能赢、RL 往哪推）/
                     课程（这些「地」的顺序和配比：先易后难、先拆后合）/ 拆子任务（决定有哪些格）。②-B 改的是六个决定里的状态和题目，教材秤动作观测不动 = 四格课程里加一格
     预训练 / mid-training / SFT 像不像（用户：SFT 偏塑造角色，预训练 mid 偏知识零件 —— 对）
                     机器一样（预测下一个 token、交叉熵、反向、AdamW），差在数据、剂量、位置：预训练 万亿 token 网页代码 从零 → 零件知识；mid 千亿 挑过的（代码数学长文推理格式）预训练末段 → 挑过的规律、新零件；
                     SFT 千~百万条示范 最后 → 做法角色格式；RL 自己采 环境 最后 → 倾向可靠性
                     零件 vs 角色是剂量造成的不是机器：零件是电路，只在同一规律换无数面孔、背不下来时长出来（要万亿）；角色是映射表里一条：模型早会几十种写法，SFT 几千条把一种设成默认（提示词也能，只是不持久）
                     → SFT 装得快装得浅：装「已有能力里选哪种」，装不了「没有的能力」。val 停 0.24 = 碰到这条线：程序员角色装上了，60 行模块的牵扯这个零件装不上，要 Coder 预训练 5.5 万亿 token 的剂量
                     2025 后 mid 与 SFT 界线更模糊：Qwen3、K2 把思维链和工具调用格式放进 mid-training，剂量大到「格式」成了零件。判据不变：能写成模板的 SFT 装；要从很多例子抽的要 mid 的剂量
   教学（①′ 收工后的问答第三轮，2026-09-17，「记下来」）
     ScaleRL 论文（Meta FAIR，Khatri/Madaan 等，arXiv 2510.13786，2025.10；全文 arxiv.org/html/2510.13786v1）
                     设置：8B 稠密 Llama + Llama-4 Scout 17B×16 MoE；数学 Polaris-53K（补数学+代码）；对 +1 错 −1；验证 1000 道留出；长度 16k（思考 12k+解答 2k+输入 2k），扩展 32k；
                     批 48 题 × 16 = 768 条/更新（大批实验 2048 题）；80 张 GB200（64 采样 16 训练），PipelineRL 最多 8 步 off-policy；消融各 3.5~4k GPU 时、留一法 16k、大跑 50k~100k，合计 40 万
                     公式：R_C − R_0 = (A − R_0) / (1 + (C_mid/C)^B)。R_0 起点，A 算力无穷时停的高度（配方的天花板，≤ 支撑集天花板），B 陡度/效率（10%→90% 进度要的算力倍数 = 81^(1/B)：B=1 81 倍，B=2 9 倍），C_mid 半程算力
                     验证：前 8k 拟合外推 16k 对上；50k 拟合外推 100k：预测 A 0.645 实际 ≈ 0.65；三次独立跑 A 误差 ±0.02；AIME-24 同趋势
                     三条结论：① 天花板不通用（A 由损失类型、精度、批次、模型大小、生成长度决定）② 苦涩教训（小算力领先的配方大算力可能落后，按拟出的 A/B 挑，不看早期曲线）
                              ③ 常识重估（损失聚合、优势归一化、课程、off-policy 算法主要改 B，不怎么动 A；单项消融里按题平均、零方差过滤、丢 ≥0.9 的题 A 有几米级小差）
                     旋钮：DAPO A≈0.52 vs GSPO/CISPO 明显高（CISPO 略高）；lm_head fp32 0.52→0.61、MoE 上更关键；小批早领先早饱和、2048 题批 A 高不停滞；MoE 用 1/6 算力超 8B；
                           32k 长度早期慢（C_mid 大 B 小）A 高；PipelineRL-8 vs PPO-off-policy-8 A 同 B 大好；归一化三种接近；每题 8~32 条无差；强制打断防不稳
                     配方八件：PipelineRL-8、强制打断、fp32 logits、CISPO、按题平均、批级归一化、零方差过滤、丢 pass≥0.9 的题；留一法多数只掉 B，去 CISPO 或 fp32 掉最多
                     局限：数学为主；未碰稠密奖励、生成式验证器；单一模型家族；最小 8B（我们 1.5B 在范围之下）
                     对我们：框架同构（天花板/收缝速度 = A/B），下一趟每 25 步评一次池外拟 S 形；token_logp 是 bf16 乘完转 fp32，他们是 head 在 fp32 算，补一半；
                            每批只更新一次比值恒 1 = CISPO 形状，损失类型暂无关；零方差过滤 = 定论 ⑲；便宜单变量：丢 ≥0.9 的题、批次加倍；撞顶 16~18% → 长度上限可能压着 A
     A 和 B 是什么     两张图：图一 横轴算力纵轴池外 pass@1（A 终点高度、B 陡度、C_mid 半程，全在这张）；图二 横轴 k 纵轴 pass@k（pass@32 是右端点，与算力无关）
                     连接：图二右端 = 有路的题比例 = A 的封顶；缝 = 两图之差 = RL 最多能挣的；A − R_0 = 实际挣到的。A 不是 pass@32，A 被 pass@32 封顶；B 不是「pass@1 到 pass@32 的效率」，是「pass@1 往 A 爬的速度」
                     示例（手选参数看形状，两点定不了三参）：R_0 0.329，A 0.50/C_mid 6/B 1.2 → 2.4 GPU 时 0.372（实 0.376）、14.7 → 0.457（实 0.446）、60 → 0.490、120 → 0.495；
                     另一配方 A 0.60/C_mid 20/B 1.0 前 150 步落后（0.358/0.444）600 步反超（0.532/0.561）= 苦涩教训 = 仪表盘三次误导的根子
                     爬山比喻：山 = 模型（山顶 = pass@32），路线 = 配方（终点海拔 = A，陡度 = B），小时数 = 算力；换损失/精度/批次 = 换通往更高平台的路，异步/过滤 = 同一条路走得快；
                     削尾巴 = 走着走着封掉了往上的岔路；ScaleRL = 路线预报：走一段拟终点。我们：山顶 620 起点 329 走 14.7 h 到 446，这条路约停 500，差山顶 120 是路的事，超 620 换山
     什么换 A 什么换 B（这条线的证据）  换天花板本身：底模/预训练（>60 行 0、老师 0.93）、更强老师蒸馏（0.06~0.10→0.33，到容量止）、工具/专家、采样预算 k（定论 ㉕，改看得见的天花板）、题池分布；
                     不换：奖励形状、状态行、动态采样、RL 步数。换 B：动态采样（⑲）、异步/窗口/缓存/vfork、每步题数与条数、起点（⑰）、奖励形状（改走法不改终点）、β/lr/clip
                     两头都碰：熵控制 clip-higher（削尾巴 = A 停在天花板下）、训练时 k（BroRL 512）、课程与题池（丢全对的题；绑题 = A 提前饱和）、采样器/训练器概率错位（fp32、TIS）、参考模型重置（ProRL）
     RL 能不能出新路（我的判断）  三层：子步骤层（底模 p=0 的动作）不能；整条轨迹层（每步都有、乘起来小到底模 k=一万看不见）能，且是实打实的新；换题还在 看抬的是通用子步骤还是这道题的答案
                     机制：五个子步骤各 0.2 → 整条 0.00032（pass@32 0.01 看不见）；RL 把各步抬到 0.8 → 整条 0.33。子步骤在容易题的赢家里也出现，被抬后乘到难题上 = 定论 ㉒ 的乘法反过来用
                     我们的三个样子：②-A 文件级 +0.02 包级 +0.24；①′ 见过的题 7 道（240 次撞）；16~30 行 0.28→0.46。留出 0 道 = 四个条件没凑齐：每个子步骤采得到（1.5B 60 行有子步骤是 0）、
                     采样预算够（k16 挤尾巴、512 撞尾巴，同一算法预算定性格）、尾巴没先被削（clip-higher、参考重置）、没有更便宜的赢法（16 条那笔账）；第三层加：题要多
                     → 「RL 只在支撑集内挪」子步骤层全对、整条轨迹层在长任务上错。可证伪预测：扩池 950 留出 pass@32 不再掉、多 0~2 道；4B + k32/64 + clip-higher 留出出现只有 RL 能 ≥3 道
     接下来可以做什么（九项，2026-09-17）  ① 扩题池 MBPP 870 从 SFT-ex 重跑（预测 留出 +0.11↑、留出 pass@32 差 ±0.03、见过/没见过差 <0.05；半天+3.7h+1h）② 仪表盘 v2 随机、温度 1×4、报见过/没见过、每 25 步（给 S 形六个点）
                     ③ 同配方第二趟（噪声地板，3.7h 零代码）④ 批次加倍每卡 6×16 步数减半（同算力 +0.02~0.04）⑤ 长度 4096→6144（撞顶减半 +0.02）⑥ 4B LoRA（基建 1 天+跑 1 天）⑦ ②-B 注入器 1~2 天
                     ⑧ on-policy 蒸馏 7B 本地老师（2 天）⑨ 第二轮 fix 教材（半天）。顺手三件不算变量：MAX_CONSEC_TIMEOUTS=2、lm_head 真 fp32 先量 logp 差、vLLM-HF logp 差日志
                     推荐顺序：1+2 → 3 → 4 或 5 → 6→7 → 8/9 看结果。不推荐：调奖励公式、接着 RL@150 续训
     长程是赢法的属性不是预算的属性  ①′ 预算 12 次但最短赢法三步（写测交），首版就对占赢家九成 → 「允许长、实际短」，RL 学三步赢；①′ 拧的轴是「第一版自己写」不是长度
                     真长程看「赢家至少要做几个决策」：②-A K=3 至少三写三测；②-B 几轮测看改。评测补读数：赢家的调用数分布
     长程强观测 vs 长程 agent  同一内核（状态/动作/观测/终局奖励/乘积/记住做到哪/退回）；区别在砍掉的：观测强 vs 弱、三四个固定动作 vs 开放工具集、K 个文件印在题面 vs 整个仓库要导航、
                     ≤12 次几千 token vs 几百次十万 token（上下文管理成技能）、隐藏测试 vs rubric/人、十几步后 0/1 vs 几百步后 0/1、静态环境 vs 用户在环且动作可能不可逆
     课程 ↔ agent 对应  ① 行动回路 / ①′ 按规格实现 / ②-A 长度（乘积、状态、归因）/ ②-B 耦合（找根因、冒新错、退回）/ ③ 弱观测（自己造验证）；工业界五格一起上
                     推迟的三样：导航（traceback 给了行号几乎免费，错误处≠根因处时才难 → ②-B 大仓库版）、开放工具集（定论 ⑰：用法是做法，SFT 的活）、上下文管理（最长 4k 装得下，②-B/③ 堆出大量输出时成新轴）
                     规模换归因 两层：实验归因（工业界十样一起改只看基准分，小代理消融/出事时/ScaleRL 才归因；我们一次一轴能说因果句，代价是结论带条件、单轴合起来不一定成立）；
                     RL 归因（几百步后 0/1 用采样数补 vs 我们十几步终局奖励够信息量，代价是长程病要人为造）。他们买覆盖面付说不清，我们买说得清付小
     十六根轴四组  行动：1 行动回路（①）2 从规格实现（①′）3 导航（推迟）4 开放工具集（推迟）；长程：5 长度（②-A）6 状态（②-A）7 耦合（②-B）8 规划分解（不做）9 上下文管理（推迟）10 时间成本（价格是小号）；
                     验证：11 弱观测（③）12 秤的多样（只用测试）13 恢复鲁棒（臂 A 重试是一角）；人与世界：14 用户在环（臂 A 求助是一角）15 不可逆与安全（不做）16 部署差异（写了没量）
                     每轴是别轴的乘数，终局奖励在一切之后。工业界对策：SFT 装行动类（合成轨迹几十万）、RL 只在秤可靠的子集拉长程与验证类、人与世界靠 harness 和规则
     工业界的顺序不是「上算力其他再说」  三样先做：秤可靠（规模放大秤量的东西，有洞放大洞）、配方稳（算力救不了坏配方：DAPO 0.52、fp32 差 90 米）、题够多（80 道跑 1500 步只会更绑）；
                     然后上算力、边跑边盯秤。算力是乘数，三样是被乘数。我们差别只在算力那步的大小，前三样做过：隐藏测试+探针、异步+fp32、扩池
     rubric = 评分表  按条列「要有什么/不能有什么」各配分，评委（人或模型）逐条勾，得分 = 满足分值/总分 → 当奖励。例：糖尿病饮食建议五条 7 分。用于没有测试的任务（写作、建议、报告）
                     按我们的话：弱一些的秤，验证器 = 模型对照清单，期望值 = 条目；要守三条：清单对策略藏着、评委比策略多知道一些、清单够多够刁（十几二十条含扣分条）
                     工业界：Scale RaR（医学科学）、Kimi K2 自我批评（评委靠可验证任务保持诚实）、checklist 是/否比打分稳。弱点：评委被讨好被长答案糊弄、条目写不全 → 要定期作弊探针。③ 弱观测那级的秤长这样
     命名总表  ②-B = 阶段 ② 后一半：exercism 真实模块注 K 个 bug（根因-多症状），考找根因/冒新错/退回，要认类和方法的注入器，未做，推迟到 4B 后；和 ①′ 同题两种开局（空壳自己写 vs 坏代码自己修）
                     200M 线：A SFT、B GRPO 顿悟、阶段一/二/三、③ 熵坍缩 | Countdown 线（CD-3/CD-4/GSM）：臂 0 只 RL、臂 1 程序撒种、臂 2 自蒸馏（臂 2−臂 0 = SFT 收紧，臂 1−臂 2 = 新零件）、
                     臂 3/3′ 只换奖励（v2/v2.1 过程判分器）、臂 5 长链算术、计划 B 计算器工具、臂 T 工具版、臂 S 秤+题（⑯）、臂 S′ 去锚、臂 A 限量问专家（⑰，hf_armA）|
                     E 猜数字线：E1 涌现、E3 换地、E2 撒种（⑱）| 长程编程线四格：① 修 bug（⑲⑳）→ ①′ 从头写（㉗）→ ② 长程强观测 = ②-A 独立拼包（E1 长度/E2 状态/E3 奖励形状，重名！；㉑~㉖）+ ②-B → ③ 弱观测；③+②-A = 长程弱观测
                     规矩：定论 ①~㉗ 全线连续；out_* 训练目录 hf_* 导出权重；评测名 模型_held/_s32/_cheat、rlb/rlf/sftB_K*；三处易混：圈号既是阶段也是定论、E1~E3 两线各一套、② 是一格 ②-A/②-B 是两半
                     ★ 别名（用户 2026-09-17 指出）：exercism 从头写那次 = ①′ = 用户口中的「②-B 从头写版」（原 ②-B = 真实模块 + 注入耦合，9-15 去掉注入后按轴改叫 ①′）；
                       「②-B 注入版」专指没做的耦合实验；③ 弱观测的模型拟叫 hf_sft_exw / hf_exwRL_bin
                     ★★ 用户定案的编号（2026-09-17，此后一律用这套）：① 短程强观测 = 修 bug（做完）| ②-A 长程强观测 1 = 独立拼包（做完）| ②-B 长程强观测 2 = 真实模块（exercism 从头写、给测试，做完；前文的 ①′ 即它）|
                       ③ 长程弱观测 = 真实模块不给测试（现在）| 注入根因 bug 考耦合的那个记作 ②-B′ 注入版，未做
     ①′ vs ③ 的区别  题秤状态预算全同，只把「环境替你验证」换成「你自己造验证」：看到的测试 七成→零；动作 去掉 test；回来的东西 passed k/n+tb → 只有自己断言的 stdout/tb；
                     判断对没对的人 环境→模型；最短赢法 3→5 步；观测可信度 = 秤的七成 → ≤ 模型自己。推出：多三个要学的动作（从题面写断言、读自己断言的报错、怀疑自己的断言）；
                     多三种死法（不测就交、编 run 输出、写永远过的断言）；多四个读数（自测率、真阳、假阴、假观测）。回答的问题：观测拿走掉多少（③−①′ 单变量）、RL 留不留自己验证
                     实话：①′ 三步是短程、③ 五步是中程，长程要等 ③+②-A 拼包。本质：强观测 = 观测是秤露出来的一角；弱观测 = 观测和秤断开，验证从环境的事变成模型的事 = agent 的处境
   训练器 ex 分支的题源：训练 split 100 道，按 slug 排序每隔 5 道取 20 道当仪表盘（固定，不进采样），剩 80 道每步抽 8 × 16；
           ExEnv(max_calls 12, cost 0.01, per_test 2 s)；日志格「测比」= 测试通过比例、「撞限」；vLLM 上限 max_new + 2600
═══════════════════════════════════════════════════════════════════════════════════════════════════
★★ 阶段 ③ 设计：弱观测、自己造验证（exercism 同题同秤，2026-09-17 定，用户提议，先讨论后开工）
═══════════════════════════════════════════════════════════════════════════════════════════════════
   为什么现在（四条）  环境只改一个开关；①′ 是现成对照组（③ − ①′ = 弱观测的代价，干净单变量）；考的是 agent 最核心的轴（环境不验证时自己造验证）；直接检验 16 条轨迹那笔账（①′ 改被压因一次写对更便宜，③ 首版失败六成，自测能把失败变成功 RL 就该留它）
   问题（三个）  ① 观测拿走掉多少：SFT 后、RL 后各一次，pass@1 与 pass@32 ② RL 留不留「自己验证」这个动作 ③ 自测的质量：抓不抓得住自己的 bug，RL 后变不变
   变量  只有观测强度（测试给不给）。题、秤、底模（hf_coder_1.5b）、教材流程（DeepSeek）、SFT 超参、RL 配方一律同 ①′。要认的副作用：最短赢法 3 → 5 步，长度跟着变，这是弱观测的定义自带的，不单剥
   六个决定  状态 一个文件（开局空壳），不变 | 动作 write·run·answer，test 收掉：发 <test> 回「本任务没有测试工具，用 <run> 自检」并计一次调用 | 观测 只有 run 的 stdout + traceback，断言自己写，截 1500 字
             题目 同 130 道同 split（train 80 + 仪表盘 20 + 留出 30）| 秤 全部测试全过 − 0.01 × 调用，藏全部 | 预算 12 次、max_new 4096（探针撞限 > 40% 才提 16）
   防线  沿用五条（隐藏测试、测试不落盘、无文件、断网、整文件替换）+ 假观测检测（harness 的 fake 标记，模型自己写 <result> 算作弊读数）+ run 片段与秤隔离（隐藏测试既不在盘上也不在 sys.modules）
   新读数四个  自测率（交卷前 ≥1 次带 assert 的 run）| 真阳率（自测挂 且 该版本隐藏也挂）| 假阴率（自测全绿 但 隐藏挂 = 断言太弱）| 假阳率（自测挂 但 隐藏过 = 断言写错）；另加 直接交卷率、假观测条数、赢家的调用数分布
   七步
     1 环境开关（半天）  ex_env.py：ExEnv(obs="weak")；render 新模板 w0~w7 + hw8/hw9（题面 + 空壳 + 工具说明，无测试段，留出模板不进教材）；step 拒 <test>；hist 记每次 run 的（版本号、片段有无 assert、有无报错）；
                        score 时对每个写过的版本跑隐藏秤（缓存）→ 四读数。单测 8 条：弱模式 prompt 无测试文本 / <test> 被拒且计费 / run 断言失败回 AssertionError / run 通过回 stdout / score 不变 /
                        自测四读数在脚本局上算对 / 假观测被抓 / 模板 10 × 3 渲染
     2 评测器（与 1 同半天）  eval_ex.py --obs weak：新增四读数 + 直接交卷率 + 假观测 + 赢家调用数分布，其余沿用；pass_k.py 不用改
     3 探针（半天）  hf_exRL_bin、hf_sft_ex 各 train 100 × 8 弱模式；作弊探针注入「run 之后可以自己写期望输出省调用」量假观测
                    判据：p ≥ 0.05 在 exercism 做；< 0.05 → 退路 A：MBPP 短函数当 ③ 的池（1.5B 写得出五行函数的断言）；退路 B：上 4B。哪条退路按探针分桶定
     4 教材（一天）  make_ex_demos.py --weak：老师只看题面 + 空壳；SYS 要求 先在 <run> 里写断言 → 实现 → 跑 → 改，最多 3 轮；只留隐藏秤全过；顺手记老师自测的真阳/假阴（老师的观测可信度）；
                    300 次写；fix 课程第二轮（从探针/SFT 的错版起）看第 5 步死法再定
     5 SFT + 评测（半天）  sft_qwen.py（test001/.venv），hf_coder_1.5b 起，超参同 ①′；评测 train 100 × 8、留出 30 × 8、train × 32（pass@1 + pass@32 规矩）
     6 RL + 评测（每趟 3.7 h + 1 h）  grpo --task ex --ex-obs weak，bin、动态采样、150 步，配方同 ①′；训练器顺手装（不算变量）：仪表盘 v2 = 固定 20 未见 + 固定 20 见过、温度 1 × 4、每 25 步一次（S 形六个点）；
                    MAX_CONSEC_TIMEOUTS=2；vLLM-HF 每 token logp 差日志。评测：train × 8、留出 × 8、train × 32、留出 × 32、作弊探针、见过/没见过、四读数；GPU 空就跑同配方第二趟拿噪声地板
     7 定论（半天）  弱观测代价表（SFT/RL × pass@1/pass@32，③ − ①′）；RL 前后自测率与真阳/假阴；S 形拟合 A/B 预测 600 步；预测对账；欠的债
   预测（跑前写下）  探针 hf_exRL_bin 弱模式 train × 8：0.10~0.15，自测率 < 20%，直接交卷率 > 60% | SFT-③ 0.25，自测率 > 90% | RL-③ 0.33，自测率保持 ≥ 70% |
                    弱观测代价：pass@1 比 ①′ 低 0.10~0.15，pass@32 低 0.05 以内 | 自测真阳率 SFT 后一半，RL 后不变 | 假观测 < 1%
   时间  3~4 天。先决：每次跑前 df -h；sft_qwen.py 用 test001/.venv 其余 vllm_env；DEEPSEEK_API_KEY 只走环境变量
   其它待办的去处  扩题池、批次、长度、4B、②-B 往后排；仪表盘 v2 / 每 25 步评 / 超时早停 / logp 差日志装进 ③ 的训练器（量具与速度，不算变量）
   第 1、2 步 环境开关 + 评测器（2026-09-17）  ex_env.py 2c4cab1c（ExEnv(obs="weak")：<test> 回 NO_TEST_MSG 计费不起沙盒；runs 记每次 run 的版本/有无 assert/有无报错；versions 存每版；
     score 对首版、最终版、自测过的版本跑全秤 → first_ver_frac/all、ver_frac、n_tp/fn/fp/tn、verified_final、selfgreen_hidfail；弱模板 w0~w7/hw8/hw9；render 按 env.obs 切）
     tests/test_ex_env.py 3864add2（加六项，GS01 全过）；eval_ex.py b32cd874（--obs weak；首版（全秤）、改一轮（全秤）、自测五个数、自测质量四格、赢家调用数分布、弱模式死法四类）
   ★ 第 3 步 探针（弱模式，train 100 × 8，温度 1，2026-09-17）
                          hf_exRL_bin   hf_sft_ex    hf_exRL_bin + 作弊注入
     通过率               0.301         0.240        0.305
     首版全过（全秤）     0.336         0.248        0.329          ← 首版不靠测试；最终 < 首版 = 改坏了
     自测率 / 验过最终版  .316 / .242   .256 / .198  .286 / .230
     不自测就交卷         0.466         0.571        0.490
     试图 <test> /局      0.49          0.54         0.18           ← ①′ 留下的习惯，白扔调用
     自测质量 真阳/假阴/假阳/真阴  .69/.08/.17/.05   .70/.11/.15/.04   .66/.07/.21/.05   ← 假阳 = 代码对断言错 → 改坏好代码
     自测全绿但隐藏挂     44            53           37
     死法                 没自测就交 199 撞限 149 撞顶 64 import 不了 47 自测全绿隐藏挂 40 自测挂着交 19 编观测 7   SFT：239/103/49/94/43/23/14
     赢家调用数           2 次 137 / 12 次 33 / 4 次 23 …（最短赢法两步：写一次、跑或试一次、交）
     按行数 ≤15/16~30/31~60/>60   .58/.34/.07/0（强观测 RL .78/.46/.22/0）   带类 .13 / 不带类 .36
     预测对账  RL 弱模式 0.10~0.15 → 0.301 ✗（低估一倍：测试只影响改的循环，首版不靠它）；代价 −0.10~−0.15 → RL −0.145 ✓、SFT −0.089；自测率 <20% → 32% ✗；直接交卷 >60% → 47% ✗；假观测 <5% → 0.9%/1.1% ✓
     ★ 读法 ① p 0.30 在可学带，③ 就在 exercism 做 ② 观测拿走掉的是「改」不是「写」：①′ 改的循环 +5 点，这里 −3.5 点（首版 .336 → 最终 .301）
          ③ 自测的病是断言写错：17% 假阳 → 信了断言去改对的代码；真阴只 5%（「测过、绿了、也真对」几乎没走通）④ 没测试逼它就不验：最短赢法两步，38% 的失败是没自测就交
     教材清单  交卷前必写断言跑一遍（199 没自测 + 47 import 不了）/ 断言从题面例子和规则抄不凭空想（假阳 17%）/ 断言挂了先怀疑断言回题面对再改代码（改坏首版）/
               边界要测：异常消息、空输入、边界值（假阴 8%、自测全绿隐藏挂 44）/ 没有 test 工具别调（0.49/局）
     ★ 机制预测（第 6 步对账）：RL 留不留自测取决于 SFT 后的假阳率 —— 假阳仍 ~15% → 自测有时帮倒忙，RL 压自测率；假阳 < 5% → 自测把首版失败变成功，RL 留住它
     改一轮（全秤）  RL 571 轮 −0.020（变好 .10 没变 .74 变差 .16）；SFT 365 轮 +0.001（.11/.78/.12）→ 弱观测下改是净亏的（②-B 强观测 RL +0.018）：没有真观测，改是瞎改
     两段轨迹  two-fer 0.99 分 1 次调用：<run>One for you, one for me.<result> —— 把期望输出当代码放进 run、自己写 <result> 被掐断，代码碰巧对 → 「赢家 1 次调用 5 条」是运气赢；评测挑最短赢家要排掉假观测
               diffie-hellman 0 分 2 次：<run></run> 空 → (no output) → 「All tests pass」交卷；私钥 randbelow(p-1) 会出 0，一条断言就抓住 → 弱观测版的「否认观测」：没输出读成没错
     教材清单再加两条（排最前）：<run> 里放代码、输出由环境回在 <result>（这个模型没学过 run，②-B 里 test 替它干了）；每段断言末尾必 print("checks passed")，沉默永远不等于通过
     ★ 决定（用户 2026-09-17）：题直接加进 ③（exercism 80 + MBPP 870，弱模式，MBPP 题面留一条示例，其余两条原断言 + 20 条 oracle 全藏）；不另开扩池臂。
       代价：RL 阶段与 ②-B 的对照混了题池；保住的：SFT 阶段对照（教材都只用 exercism 100，SFT-③ − SFT-②-B = 观测的代价）、见过/没见过、MBPP 留出 100 道
       「鼓励改多次 / 耦合 / 测试多」的原则：用环境和教材鼓励，不用奖励鼓励（过程奖励会被钻，臂 3 教训）—— 改多次靠首版失败 + fix 课程 + ②-B′；耦合是题的属性，③ 不动配比只分开报；
       测试多靠老师断言多而刁 + 秤严。③ 要回答「RL 会不会自己留住自测」，一加分就答不了
   第 4 步 教材（弱模式老师，make_ex_demos.py --weak，2026-09-17）
     设计  老师只看题面 + 空壳，一次给 EXPLANATION / CHECKS（从题面推的断言，末尾 print("checks passed")）/ CODE；环境 write → run 断言；挂了把观测发回，老师先判「断言错还是代码错」再改，最多 3 轮；
           只留秤（全部测试）全过的；用 ExEnv 弱模式给老师的断言打真阳/假阴/假阳/真阴。修教材：--fix-from 弱模式探针 raw，学生的错首版 × 老师写断言抓 + 改。弱模式没有可见通过数，不做「改坏退回」支
     试 10 道（旧提示词）  10 局：断言全过 10、秤也过 8、首版断言就过 8、改过后过 2；老师断言质量 14 次 run 真阳 0 假阴 .14 假阳 .29 真阴 .57
           好：断言多从题面例子推（diffie-hellman 用 p=23 g=5，bank-account 异常消息断言到原文）；diffie-hellman 一轮 = 标准示范：断言 private_key(2) 本身不合法 → 老师判「断言错代码对」删断言 → 过
           坏：acronym 3 轮 —— ① 挂在第 6 行（HyperText Markup Language == "HTML"，老师凭常识编的例子，按题面规则应是 HML）老师数错行，删了第 5 行空串那条 ② 再挂，写一大段自相矛盾的话把没错的代码重写
               ③ 第三轮才认出断言错。根子两个：run 的 traceback 只给 line N 不给源码，老师只能数行数；期望值来自题外常识
     修三处  ex_env 17f97225：RUN_SNIPPET 给 "<snippet>" 注 linecache，traceback 带出错那行源码（工具输出格式，不改观测强弱）；单测加一条
             make_ex_demos c957313b：SYS_WEAK 改成 每条断言带消息打印实际值 / 期望值只按题面规则算不用题外常识 / 每条规则·例子·异常消息至少一条断言、6~25 行 / 解释一句话 ≤40 词；
             解释 > 300 字 = 啰嗦，整局丢（MAX_EXPL_CHARS）；tests dcbb4a51
     试 10 道（新提示词）  10 局：断言全过 10、秤也过 7、断言绿秤挂 3、首版断言就过 7、改过后过 0；断言质量 15 次 run 真阳 .33 假阴 .20 假阳 0 真阴 .47，啰嗦丢弃 0
           ★ 假阳 4/14 → 0；剩下的病换成假阴：3 局老师自己的断言没抓住自己的 bug（被秤丢掉不进教材）→ DeepSeek 在弱观测下的观测可信度七成 = 「观测的可信度不超过造观测者」
             改过后过 0：断言抓到真 bug 的 3 局改完仍挂秤，老师改的是往自己的断言收敛不是往题面收敛。预测 300 局过 200~220（②-B 279）
     ★ 探针重跑（run 的输出格式变了，基线对齐新工具；只跑 RL 模型，覆盖 rlex_weak）  通过率 0.295（旧 0.301）首版 0.327（.336）改一轮 −0.028（−.020）自测率 .320 验过最终版 .266 不自测就交 .476
           试图 test .52  真阳/假阴/假阳/真阴 .66/.10/.19/.05  自测全绿隐藏挂 59  死法 没自测就交 200 撞限 139 撞顶 62 自测全绿隐藏挂 53 import 不了 50  赢家 2 次 128 / 12 次 38 —— 全在噪声内
           ★ ③ 的基线定为这份：0.295 / 首版 0.327 / 自测率 0.32 / 假阳 0.19 / 改一轮 −0.028
     全跑第一次（写 + 修并行，16 路打 API）  写 300 局：过秤 149、异常 79；修 272 局：过秤 106、异常 61 —— 异常没打类型；限时 180 s 重试 3 次撑不住 4000 token 的长正文
       修：deepseek_tool.ask 加 timeout 参数（c1edd5ce）；弱模式老师 retries 6、timeout 300、汇总打异常类型（make_ex_demos 4f65962b）；改成串行跑
     ★ 全跑第二次（2026-09-17，串行，各 8 路）
       写教材 sft_exw.jsonl   300 局：断言全过 254、秤也过 193、断言绿秤挂 61、首版断言就过 172、改过后过 47、异常 0（温度 0 走缓存，47 s）
                             老师断言质量 544 次 run：真阳 .27 假阴 .12 假阳 .26 真阴 .35；啰嗦丢弃 11 → 落盘 579 行 / 75 题（改过轮的 141，首版就过 438）；字符中位 4097 P90 7393 最长 16348
       修教材 sft_exw_fix.jsonl  272 条学生错版（94 题）：断言全过 202、秤也过 128、断言绿秤挂 74、改过后过 126、异常 0（673 s）
                             断言质量 666 次：真阳 .55 假阴 .12 假阳 .14 真阴 .19；啰嗦丢弃 8 → 落盘 256 行 / 58 题（改过轮的 252）；字符中位 6469 P90 10061 最长 15237（最长几条会超 4608 被丢）
       合计 835 行（②-B 是 837 + 528 = 1365），带改的 393 行 = 47%
       预测对账  写过 190~210 → 193 ✓；写落盘 570~630 → 579 ✓；修过 130~150 → 128 ≈；修落盘 260~300 → 256 ≈；异常个位数 → 0 ✓
       ★ 老师的观测可信度：写教材 193/254 = 76%（老师自认验过的里四分之一其实是错的）、修教材 128/202 = 63%；假阳 .26 / .14 = 断言挂了一半是断言自己错。
         这是弱观测的定义在老师身上的样子：观测的可信度不超过造观测者。落盘的行都过了秤，代码不会错；错断言的局教的是「挂了先判谁错」
     硬盘  98%，剩 21 GB；SFT 落 6 GB ckpt + 3 GB hf，RL 再 9 GB → 先清已导出的旧 ckpt
   第 5 步 SFT-③（out_sft_exw → hf_sft_exw，2026-09-17；vllm_env 下跑的 sft_qwen.py，transformers 4.51.3，②-B 的 SFT 在 test001 5.16 下，纯 HF 交叉熵，版本差按可忽略记）
     835 行 − 5% 验证 − 超长 = 768 行，两轮 48 步；val 0.384 → 0.196 → 0.171（②-B 0.579 → 0.240：弱模式轨迹更规整，固定句 + 断言比测试文件好预测）。硬盘清到 94%，剩 61 GB
   ★ SFT-③ 评测（弱模式，温度 1，max_new 4096）
                              train 100 × 8   留出 30 × 8   train 100 × 32
     通过率                    0.186           0.150         0.200
     首版全过（全秤）          0.235           0.167         0.244
     改一轮 平均/变好/变差     −.030/.11/.15   −.018/.13/.16  −.033/.10/.16
     自测率 / 验过最终版       .919 / .711     .896 / .746   .904 / .728
     不自测就交卷 / 试图 test  .014 / 0        .008 / 0      .016 / 0
     真阳/假阴/假阳/真阴       .76/.01/.21/.02 .81/.01/.17/.01 .75/.02/.21/.02（2607 / 806 / 10238 次 run，每局 3.3 次）
     有 <answer>               0.290           0.258         0.280
     死法                      撞顶 339 / 自测挂着交 86 / 撞限 82 / import 不了 43 / 其它 41 / 自测全绿隐藏挂 20 / 没写就交 18 / 编观测 16 / 没自测就交 5
     按行数 ≤15/16~30/31~60/>60  .385/.144/.057/0（探针 RL .58/.34/.07/0）  带类 .062 / 不带类 .230
     pass@k（100 × 32）  SFT-②-B（强）@1 .328 @8 .526 @32 .620 ｜ SFT-③（弱）@1 .200 @8 .448 @32 .560；都能 55 都不能 37 只有 ②-B 能 7 只有 ③ 能 1
                         只有 ②-B 能的：flower-field kindergarten-garden minesweeper robot-simulator saddle-points word-count yacht
     预测对账  池外 0.28±0.03 → 0.186 ✗ | 自测率 >90% → .919 ✓ | 验过最终版 >80% → .711 ✗ | 假阳 .15~.25 → .21 ✓ | 改一轮 ≥0 → −.030 ✗ | pass@32 .55 → .560 ✓
     ★ 读法 ① 做法装上了：自测率 92%、试图 test 归零、不自测就交 47% → 1.4%、第一步 write 95%
          ② 分掉在预算上：有 <answer> 只 29%，撞 token 上限 339/800 = 42%。弱观测下每轮验证都贵（断言 200~400 token + traceback 100~300 + 重写整模块 300~600，一局 3.3 次），
             老师一两轮就对所以在预算内，学生要转四五轮；探针模型两步交卷不撞顶，0.295 是「不验证的便宜分」。设计里「撞限 >40% 提预算」的规矩此处触发（token 那条）
          ③ 改仍净亏 −0.030（探针 −0.028）：256 行修教材没让改法变好 = 改是缺招（②-B 同结论）；假阳 .21 是改坏好代码的来源
          ④ 天花板差 6 道，差在规格不在难度：只有 ②-B 能的 7 道全是输出格式题面说不清、测试说得清的（字典还是列表、坐标怎么表示）→ 测试除了验证还带规格信息，弱观测把这份一起拿走了（定论候选）
     下一步  SFT-③ 和 SFT-②-B 都在 max_new 8192 重评（②-B 自己撞顶 18%，对照要公平）。预测：SFT-③ 0.25~0.28、撞顶 <15%、验过最终版 >85%；SFT-②-B 0.329 → 0.35 上下。
             涨回来就把 ③ 的 RL 和评测全改 8192（预算是六个决定之一，按触发规矩改，记为设计变更）
     ★ 8192 重评  SFT-③ 0.186 → 0.193，SFT-②-B 0.329 → 0.326：都没动。撞顶 339 → 158，但撞限 82 → 197、其它 41 → 73 —— 多出来的 token 全用来多转几圈到 12 次。预测 0.25~0.28 ✗
       → 预算不是杠杆，③ 的 RL 和评测保持 4096（和 ②-B 一致，少一个变量）
     ★ 轨迹（sftexw 4096 那份）  最短赢家 reverse-string 2 次：写 → 三条断言（题面例子）→ checks passed → 交卷，131 token —— 教材的形一字不差
       没过的 diffie-hellman 6 次 2300 token：① 断言是废的（assert private_key(17) == randbelow(16)，两次独立随机数比相等，永远挂）② 不怀疑断言，每次都判「code 错」
       ③ 「改」是假的：写 <fix>…</fix>（不是工具标签，环境不认，文件没动）再重跑同一段断言，五次一模一样到撞限 → 改一轮 −0.028 / 变好 0.09 的实体
     ★ SFT-③ 定论候选：做法装上了（自测 92%），判断没装上（假阳 .21、不怀疑断言、假改），装上的做法净亏 —— 首版 .243 ≈ SFT-②-B 弱模式 .248（写的能力没变），最终 .193 vs .240，验证循环吃掉 5 点。
       1.5B 的判断力下，带两成错断言的自测比不自测差；老师自己的断言一半可信，学生更差是必然
     ★ RL-③ 预测（第 6 步对账）：赢家最短两步（写、跑一次过、交）51 条，输家转到 12 次 → RL 留「跑一次」砍「改」：pass@1 .19 → .30 上下（回到首版水平）；自测率保持 >80%；改了几轮 1.8 → <0.8；
       撞限 197 → <80；调用 5 → 3；假阳不动 ~.2；首版 .24 → .33；pass@32 .56 → .60 上下；留出 pass@32 不涨。= ③ 版本的「RL 只在支撑集内挪」
     第 6 步准备（我这边）  训练器 --ex-obs weak；MBPP 870 做成弱模式题进池（题面留一条示例，其余 2 条原断言 + 20 条 oracle 全藏）；仪表盘 v2（固定 20 未见 + 20 见过，温度 1 × 4，每 25 步）；
       MAX_CONSEC_TIMEOUTS=2；vLLM-HF logp 差日志
     ★ 做完（2026-09-17）  calc_tool_vllm 24c23475：engine_add(logprobs=) 要采到 token 的 logp，engine_step 回 (rid, new, lps)
       harness e71fa0a2：异步路每条记 lp（注入段 / 补的 EOS 记 nan），generate_with_env(want_lp=)；单测过
       grpo_tool_mp_vllm dc9c744a：--ex-data 多份（第一份主池出仪表盘，其余扩池）--ex-mix 0.5 --ex-obs weak --ex-max-timeouts 2 --ex-dash-k 4（固定 20 没见过 + 20 见过 seed 12345，温度 1 × k，
         eval 行报见过/没见过各自的 修好/首版/自测）--logp-diff（训练行打 Δlp 均值/最大）；KEYS 加 selft/verif/fp_n/st_n/fv → 训练行多打 自测/验过/假阳/首版；每步抽题主池 + 扩池按比例
       collect_mbpp_weak.py：code_bugs.json 的 task → 弱模式从头写题（描述 + 一条示例断言；stub = 函数签名；test_src = 示例 + 原断言 + oracle 断言各一个 test 方法；参考实现全过才收）；本机 60/60
     GS01：单测过；tests_ext_all.json = {题号: [断言…]}，815 道接上；mbpp_weak.json 收 961/963（train 861 / heldout 100），每题测试中位 23，参考实现中位 6 行；丢 2 道参考实现没全过
     冒烟 3.5 分：显存 18.8 GB；eval 行三段都在（没见过 0.219 首版 .38 自测 .94；自测率 .94 验过 .62 假阳 .25）；「见过 nan」是冒烟只评前 8 道
     ★ SFT-③ 在 MBPP 留出 100 × 8（弱）：通过率 0.369（预测 .35~.45 ✓）首版全过 0.433 改一轮 −.032（变好 .09 变差 .15）自测率 .983 验过 .836 假阳 .28（题面一句话 + 一例，规格靠猜）
       死法 撞限 169 撞顶 158 自测挂着交 62 自测全绿隐藏挂 38；有 <answer> .345；赢家 2 次 126 / 12 次 52 —— 和 exercism 同形：验证循环吃掉 6 点，不验证直接交反而 0.43
       RL 在 MBPP 留出的预测：.37 → .45 上下（剪转圈回到首版水平）；自测率 .98 → ≥ .8；假阳不动
     ★ 奖励公式（和 ②-B 一字不差）：全部测试全过 → 1 − 0.01 × 调用，否则 0；假观测 −0.2；组内减均值不除 std；动态采样 12 抽 8；loss = −Σ adv·logp / 总 token + 0.02 KL(‖ SFT-③)；注入段屏蔽。
       不给自测/改轮/断言数加分（③ 问的就是 0/1 的秤留不留自测）。用户问要不要去掉调用价格（怕鼓励一次做对）：留着 —— 价格只在赢家间起作用（−0.01 vs 输赢 ±1），
       和 ②-B 同公式才能对照，且它是唯一压转圈的信号；掉的自测若集中在 ≤15 行首版本来就对的题，下一趟单变量 --code-call-cost 0 或前两次免费
     工具对照  ②-B：write / test（作者七成测试）/ run / ask（关）/ answer；③：write / run / ask（关）/ answer，test 被拒照扣。②-B 两个模型 run 每局 0.00（教材里没有），
       同一个 RL-②-B 到弱模式 32% 的局自己用 run 写断言 = 零件在底模里一直闲着。验证 = 执行 + 期望值：②-B 环境两样都给，③ 只给执行
     RL-③ 起跑（out_exwRL_bin，2026-09-17）：--ex-obs weak --ex-data exercism + mbpp_weak --ex-mix 0.5 --ex-dash-k 4 --logp-diff，其余同 ②-B
       第 0 步训练行  修好 .27 测比 .49 撞限 .25 自测 .95 验过 .77 假阳 .26 首版 .29 分 .24 调用 7.26 语法错 .12 假观测 .03 长 2588（②-B 起点 1638）|A| .25 有效组 2/2 KL 0 ★ Δlp 0.004/1.96
         → vLLM 与 HF 的每 token logp 差均值千分之四（最大 1.96 是个别 token）：采样器/训练器概率错位可忽略，TIS 不用做。一步 56+22 s，剩 208 分；引擎 42 s ∥ 沙盒 8 s，终评 11 s（多了版本评估）
       起点仪表盘 v2（温度 1 × 4）  总 .2145 修好 .231 撞限 .12 调用 5.74 长 3134 ‖ 没见过 .312（首版 .34 自测 .94）见过 .150（首版 .16 自测 .95）‖ 自测率 .94 验过 .72 假阳 .25 首版 .25
         两组差是难度差（没见过 = ②-B 那 20 道偏易；见过 = 80 道里随机 20 道）→ 看涨幅不看水平
       预测（150 步后）  没见过 +.05~.10，见过 +.10~.15，涨幅差 < .05 = 扩池起效、> .08 = 仍绑题；自测率两组 ≥ .8；假阳不动；首版两组各 +.05 以上
       第 49 步训练行  修好 .28 撞限 .25 自测 .95 验过 .87 假阳 .29 首版 .34 调用 6.50 长 2570 KL .0004 Δlp .003；一步 40+22 s，四卡最慢−最快 8 s
       第 50 步 eval  总 .2844 修好 .306 撞限 .12 调用 5.42 长 2873 ‖ 没见过 .413（首版 .40 自测 .95）见过 .200（首版 .24 自测 .93）‖ 自测率 .94 验过 .82 假阳 .31 首版 .32 ★ 新高
         读法：RL 留住了自测（.94 不掉）且验最终版在涨（.72 → .82/.87）= 「验完再交」被选中；涨在首版（.25 → .32）同 ②-B；调用只剪一点、训练行撞限仍 .25（「砍改」目前只对一半）
              假阳 .25 → .31 要小心：它是假阳占全部断言 run 的份额，代码对得多了份额机械上涨，断言质量要看「代码对时断言挂的比例」，评测 raw 里算
              没见过 +.10 > 见过 +.05：暂无绑题迹象，但没见过组偏易先涨正常，80 条一组 1σ ≈ .05，150 步再判
   ★ RL-③ 跑完（out_exwRL_bin，150 步，205 分钟，2026-09-17）
     仪表盘 v2 七点  步 0/25/50/75/100/125/150：总 .2145/.2440/.2844/.3191/.3159/.3294/.3228；修好 .231/.269/.306/.338/.331/.350/.344
       没见过 .312/.400/.413/.400/.413/.463/.400（首版 .34→.45）  见过 .150/.138/.200/.275/.250/.238/.288（首版 .16→.25）
       自测率 .94/.93/.94/.93/.92/.88/.95  验过 .72/.83/.82/.82/.82/.81/.86  假阳份额 .25→.33  调用 5.74→5.58  长 3134→2703  撞限 .12 平  峰值 .3294@125（噪声内）
     预测对账（仪表盘层）自测率 ≥.8 → .95 ✓（一步没掉，验最终版 .72→.86 = RL 选中了「验完再交」）| 首版两组各 +.05 → +.11/+.09 ✓ | 没见过 +.05~.10 → +.09 ✓ | 见过 +.10~.15 → +.14 ✓
       涨幅差 .05 正卡在线上（1σ ≈ .05）判不了，等池外拆分 | ★ 调用 5→3、砍改 → ✗：调用 5.74→5.58、撞限平，RL 没砍改的循环 | 假阳份额上涨是机械的，真质量池外算
     ★ S 形拟合（7 点网格，横轴 GPU 时 = 步 × 205/150 × 4/60）：A = 0.330，C_mid = 3.6 GPU 时 ≈ 第 40 步，B = 2.6，残差 .012；预测第 150 步 .327、300 步 .329、600 步 .330
       → 这条配方第 75 步基本到顶，续跑一倍算力只多 .003；ScaleRL 那套第一次用在自己的曲线上，结论：别续跑，换旋钮
     第 6 步评测（预测已在前面）：export hf_exwRL_bin（latest@150；best@125 只高 .007 不导）；第一波 rlexw / rlexw_held / rlexw_mbpp / rlexw_cheat（× 8）；
       第二波 rlexw_s32 / rlexw_mbpp_s32 / sftexw_mbpp_s32 / rlexw_held_s32 / sftexw_held_s32（× 32）→ 三张 pass_k 表 + 见过/没见过拆分 + 断言真质量
   ★★ RL-③ 池外评测（弱模式，温度 1，max_new 4096，2026-09-17）
                              SFT-③        RL-③         Δ           对照
     exercism 训练 100 × 8     0.186        0.282        +0.096      RL-②-B 弱探针 0.295 / RL-②-B 强 0.446
       首版全过 / 改了几轮      .235 / 1.55  .302 / 0.85              RL-②-B 弱探针首版 .327
       改一轮 平均/变好/变差    −.030/.11/.15  −.016/.09/.12
       自测率 / 验过最终版      .919 / .711  .946 / .875
       真阳/假阴/假阳/真阴      .76/.01/.21/.02  .72/.02/.22/.03
       死法                     撞顶 339 自测挂着交 86 撞限 82 import 43 自测全绿隐藏挂 20 | 撞顶 287 自测挂着交 114 撞限 70 import 36 自测全绿隐藏挂 28
       赢家 2 次 / 12 次        49 / 32      99 / 31
       按行数 ≤15/16~30/31~60/>60  .385/.144/.057/0   .552/.269/.083/.023（>60 第一次非零 2/88）  带类 .062 → .111
     exercism 留出 30 × 8      0.150        0.183        +0.033      首版 .167 → .203；自测率 .954 验过 .871 假阳 .18；>60 行 2/32
     MBPP 留出 100 × 8         0.369        0.495        +0.126      首版 .433 → .491；改了几轮 2.13 → .89；改一轮 −.032 → −.003（打平）；自测率 .991 验过 .953；
                                                                     假阳 .28 → .21、真阴 .04 → .11（断言质量真变好）；自测全绿隐藏挂 42 → 107；撞限 169 → 112；赢家 2 次 126 → 279
     作弊探针（注入「自己写期望输出」）0.280 vs 0.282，假观测 1 vs 1，自测率 .917 → 没上钩
     pass@k（× 32）  exercism 训练：SFT @1 .200 @32 .560 | RL @1 .284 @32 .560（只有 SFT 能 3：binary-search-tree resistor-color-trio robot-name；只有 RL 能 3：minesweeper saddle-points yacht = 上次「只有强观测能做」的格式题）
                     exercism 留出：.142/.500 | .196/.500（换 2：bob run-length-encoding ↔ house queen-attack）
                     MBPP 留出：   .393/.710 | .478/.720（换 2 对 3）
     预测对账  训练 pass@1 .30±.03 → .282 ✓ | 留出 .22 → .183 ≈（1σ .03）| MBPP .45 → .495 ✓ | pass@32 .60 → .560 ✗ 一分没涨 | 留出 pass@32 平 ✓ | 自测率 ≥.8 → .946 ✓ |
               改了几轮 <.8 → .85 ≈ | 假阳不动 ✓ .22 | 首版 .33 → .302 ≈ | 作弊不上钩 ✓
     ★ 读法 ① RL 留住了自测（.95，验过最终版 .71 → .875）但自测到最后没赚到分：首版 .302 → 最终 .282（−.02），MBPP .491 → .495（打平）。RL 把循环的亏从 SFT 的 −.05 压到 −.02/0，
             手段是少转圈（改轮 1.55 → .85，两步赢家翻倍）和「断言挂着也交」（86 → 114：改不回来、断言 1/5 是错的，交了反而对）
          ② 涨的还是首版：exercism +.07、MBPP +.06、②-B 在弱模式量到的也是 +.08 —— RL 抬「一次写对」的量不随观测变
          ③ 和 RL-②-B 丢进弱模式比（.295，几乎不验证）：RL-③ 验证 95%，.282 —— 行为完全不同、分一样；弱观测的代价 RL 后 .446 vs .282 = −.16，训练没合上一点。原因：判断没动（假阳 .22 = SFT 时）
          ④ 天花板一分没动（.560/.560、.500/.500、.710/.720）：②-B 见过的题开 6 道，这次 0，因为扩池后每题只见 7 次，撞稀路次数减半 = 定论 ㉕ 反向验证；换进来的 3 道正是格式题
          ⑤ MBPP 上断言质量真变好（假阳份额 .28 → .21，真阴 .04 → .11）；exercism 上没变 —— 简单题 RL 能修一点断言写法，难题不能
          ⑥ >60 行第一次非零：训练 2/88、留出 2/32
     见过 80 / 仪表盘 20（弱模式 × 8）  SFT-③ .166/.269 → RL-③ .250/.412（见过 +.084、没见过 +.143）；对照 SFT-②-B 弱 .212/.350、RL-②-B 弱 .272/.388
       → 没见过涨得比见过多，绑题没了，扩池起效（②-B 那次 见过 +.14 没见过 +.02）
     断言真质量（按 run，代码对时断言挂 = FP/(FP+TN)）  SFT-③ ex .92 → RL .87；MBPP .87 → .66；代码错时断言挂 .98/.98/.98/.94
       ★ 代码对的时候十次自测九次挂：一段 checks 6~25 条断言合取，一条错整段挂，1.5B 写十条全对概率低（老师提示词里我定的「至少 6 条」种下的）。按 run 算有水分：同一份对代码反复重跑同一段错断言每次记一次，
         RL 少重跑了，MBPP .66 里一部分是这么来的 → 下次 raw 里存每次 run，按版本去重
     每局  SFT-③ ex 调用 5.84 write 2.51 run 3.33 撞顶 368 token 2978 → RL-③ 5.59 / 1.84 / 3.75 / 335 / 2710（砍的是重写不是验证）；MBPP 7.29/3.12/4.17/214/2410 → 5.54/1.88/3.65/132/1594
   ★★★★ 阶段 ③ 收工总表（长程弱观测：exercism 从头写、不给测试、<run> 自验；训练池 exercism 80 + MBPP 861 各半；秤 = 全部测试；2026-09-17）
                          exercism 训练 100        exercism 留出 30      MBPP 留出 100           关键行为
     探针 RL-②-B 弱      .295（首版 .327）         —                     —                       自测率 .32 假阳 .19 改一轮 −.028
     探针 SFT-②-B 弱     .240（首版 .248）
     SFT-③               .186（首版 .235）pass@32 .560   .150 pass@32 .500   .369（首版 .433）pass@32 .710   自测率 .92 假阳 .21 改轮 1.55 验过 .71
     RL-③（150 步 205 分） .282（首版 .302）pass@32 .560   .183 pass@32 .500   .495（首版 .491）pass@32 .720   自测率 .95 假阳 .22 改轮 .85 验过 .875
     强观测对照 RL-②-B   .446 pass@32 .680
     教材  老师写 193/300（579 行）修 128/272（256 行）；老师观测可信度 76% / 63%，假阳份额 .26/.14 | 作弊探针 .280 vs .282 | Δlp .003 | S 形 A = .330 第 75 步到顶
   ★★ 定论 ㉘（阶段 ③ 长程弱观测，2026-09-17 定稿）
     1. 弱观测的代价：同一模型强→弱 −.09（SFT 阶段）；RL 后 .446 vs .282 = −.16，训练没合上；天花板 −.06（.620 → .560），丢的是规格歧义题 —— 测试除了验证还带规格信息
     2. 「自己验证」这个动作装得上（SFT 92%）、RL 留得住（95%，验最终版 .71 → .875），但 1.5B 手里不值分：首版→最终 SFT −.05、RL −.02、MBPP 打平；和不会验证的 RL-②-B 丢进弱模式同分（.282 vs .295）
     3. 不值分的原因是判断缺：代码对时断言十挂其九（多条断言合取放大错误）、假阳份额 .22 不动、改一轮变好 9%（三级都一样）。判和修是零件，SFT 装不进、RL 造不出
     4. RL 学的（全在已有动作里调比例）：首版 +.07（②-B 弱模式量到的也是 +.08，不随观测变）；砍重写不砍验证（write 2.5 → 1.8，run 不减）；验完最终版再交；挂了改不好就交（86 → 114）；假观测归零；MBPP 上断言质量略好
     5. RL 没开路：三张 pass@32 全平（.560/.560、.500/.500、.710/.720）；见过的题每题只撞 7 次也没开 = 定论 ㉕ 反向成立；换进来的 3 道是格式题（靠自己的断言试出格式）
     6. 扩池起效：见过 +.084 < 没见过 +.143，绑题消失；MBPP 留出 +.126
     7. 秤稳：作弊探针不上钩；Δlp .003 采样器/训练器错位可忽略（TIS 不用做）
     8. 配方到顶：S 形 A = .330，第 75 步饱和，续跑一倍算力 +.003 → 换旋钮不续跑
     预测对账（设计时写的）  探针 .10~.15 → .295 ✗（首版不靠测试）| SFT-③ .25 → .186 ✗（验证循环净亏）| RL-③ .33 → .282（SFT 后改的预测 .30±.03 ✓）| 自测率 >90% ✓ | 代价 −.10~−.15 → −.14/−.16 ✓ |
                            假阳 .15~.25 → .21 ✓ | 「自测真阳率 SFT 后一半」→ 代码错时断言挂 .98（任何挂都算）✗ 读数定义太粗 | pass@32 RL +.04 → 0 ✗ | 留出 pass@32 平 ✓ | 假观测 <1% ✓
     欠的债 / 下一趟候选（按便宜排）  ① 教材 v2：断言少而准（只抄题面例子 + print 标记），单变量测「合取太长」假设，看代码对时断言挂能不能从 .9 降到 .3、验证循环能不能转正
       ② 老师当秤：本地 7B on-policy 蒸馏，测判断能不能装进 1.5B（新训练信号）③ 换 4B ④ --code-call-cost 0 单变量 ⑤ 评测 raw 存每次 run、FP 按版本去重 ⑥ ②-B′ 注入版
       倾向先 ①：半天教材 + 一趟 SFT/RL，能把定论 3 里「断言太多」和「判断没有」拆开
   教学（③ 收工后的问答，2026-09-17，「记下来」）
     用户复述三条（对）  ① 会对答案了但没收益（补：亏从 −.05 → −.02 → MBPP 打平）② 强/弱观测两个 RL 模型进同一个无测试考场，一个不会自写测试一个会，分一样 → 自写测试没用
                       （补：限定在 1.5B 的判断力下 —— 价值要经抓错→判断→修好三段，后两段没有前一段白做；DeepSeek/业界同样动作值钱；两笔账抵出来的：首版 .302 vs .327，循环 −.02 vs −.03）
                       ③ 答案本写错、不会判、不会改（三样都量到：代码对时十挂九、假阳 .22 不动、改一轮变好 9%）→ 排成下一步顺序：先让答案本少错 → 判断靠老师当秤 → 4B 看容量
     第 1 条和第 2 条的区别  第 1 条同一模型内部比（首版 vs 最终 = 动作有没有用）；第 2 条两模型之间比（③ vs ②-B 同考场 = 整套训练有没有用）；可分开成立（若 ③ 首版写得更好第 2 条会赢）；这次都零；第 1 条是第 2 条的原因
     断言真质量是什么  2×2：代码对×断言过 = 真阴（放心交）/ 代码对×断言挂 = 假阳（改坏对代码）/ 代码错×断言过 = 假阴（放过 bug）/ 代码错×断言挂 = 真阳。报告的「假阳 .22」是占全部 run 的份额，混着代码对的比例；
                       真质量按行看：代码对的 run 里挂了多少 = FP/(FP+TN) = .87~.92（十次九挂）；代码错的里挂了多少 .98（含断言也错的，水分大）；按 run 算重跑重复计，下次按版本去重
     4B 直接改权重还是 LoRA  全参放不下：24 GB 一张卡，4B 全参 = 权重 8 + 梯度 8 + 8 位 Adam 8 + 参考 8 + vLLM 8 + KV/激活 8 ≈ 48（1.5B 现在 18~20）；LoRA ≈ 8 冻住 + 0.2 + 0.4 + 0（参考 = 关适配器）+ 8 + 5 ≈ 22，紧，
                       vLLM 睡时把权重也放内存每步灌 8 GB 能松 8 GB；SFT 同理 LoRA 或四卡分片。够用的依据：LoRA without regret（Thinking Machines 2025.9）RL 阶段 LoRA = 全参，SFT 教材小时也够
                       改动：sft_qwen / 训练器加 peft，参考模型 = 同权重关适配器，每步同步适配器不同步整份权重，导出合并。一天基建
     RL 一条轨迹一个 bit 和 LoRA 是同一件事  秤一条给一个数 ≤ 1 bit；150 步 × 128 = 2 万 bit ≈ 2.5 KB = RL 能写进去的上限（SFT 每 token 一个目标几百万 bit，预训练万亿 token）
                       2 万 bit 够拧几千个习惯旋钮，写不下一条知识/技能 → 「RL 装不下知识」与模型大小无关；LoRA 几千万参数装 2 万 bit 绰绰有余，全参多出的容量 RL 用不上 → RL 阶段 LoRA = 全参
                       比方：便条二十个字，A4 抄和一本书抄，抄下来都是二十个字。「一个 bit」是数量级说法（奖励标量、优势有大小），量级对
     采出来放进 SFT 是几个 bit  看接收方：放回自己 = 仍 1 bit（每个字它本来就会写，学自己的随机选择梯度为零，新信息只有「被留下」；拒绝采样 SFT 的梯度 = REINFORCE 0/1 奖励；Countdown 臂 2 自蒸馏 = 0 是实验版）
                       放进别的模型 = 几千 bit（信息 = 接收方对它的意外 −log p；底模/小模型每个会写错的字都是信息；R1 60 万条装回 V3-Base、蒸给 1.5B~70B 值钱，装回 R1 自己不值钱）
                       老师写的更多（另一个分布，每字要学：835 行按 val .24 每行几百 bit 学生预测不了 = SFT 能装 RL 装不了的东西，也是装不进去的部分卡在 .24 的原因）
                       → 工业界「装回去」总装进底模或小模型；③ 第三格 RL-③ 采 64 筛出来装回 SFT-③/底模有意义，装回 RL-③ 没有
     大模型 RL 用零件组合长 trace 开新路也是 1 bit 吗  仍 1 bit 一条，但开路是「找」不是「写」：路万分之一，定位它需要 log2(1 万) ≈ 13 bit，秤一次给 1 bit 且要撞到才给 → 采一万条花一万 bit 换 13 bit 的「哪条路」
                       RL = 很费但能堆算力的搜索，在自己的支撑集里定位好路再放大；信息小正因为路本来就在 = LoRA 够用同一原因
                       装回自己无新信息（放大到 1/10 时已在权重里）；装回去的两个理由都不是信息：巩固成默认、换熵没塌的接收方（底模）；装给下一代/小模型才是几千 bit 的新东西
                       几个 bit 换很大本事：长 trace 里试/验/回头是一段搜索程序，RL 教的是这段程序（几 bit 控制：怎么走、何时回头、何时停），装好后推理时每题自己搜，路不用挨个教 → 信息小杠杆大：零件预训练给，RL 接控制回路
                       工业界量级：几百万题 × 几十条 = 1 亿 bit ≈ 十几 MB，对几百 GB 权重是零头（Razor），但挑的是天文数字大的行为空间里的几条路，够了
     信息量和厉害有没有关系  信息量 = 写进去多少新东西；厉害 = 做题时表现出的本事；中间隔一层：本事取决于已有零件，写进去的几 bit 决定零件怎么被调用。开关一个 bit，灯泡在屋子就亮，没灯泡拨一万次也黑
                       两头例：臂 A 几千 bit 的「何时问专家」把硬题 .03 → .9（接通的是外部专家）；③ 几十个旋钮的「交卷前验」分不涨（「判断答案本对不对」这个灯泡不在，接了个空）
                       大模型 = 灯泡齐全的房间：RL 几 bit 接通「多想几步、自检、回头」这条线，看着是新本事其实是原有零件第一次被系统用上 → 对大模型 RL 显神、对 1.5B 显平，同一规律两种房间
                       一句话：信息量决定能写进去多少，零件决定写进去的能撬动多少；小信息撬大本事的条件只有一个 —— 零件已经在
   chat_ex_vllm.py（b462e82c，2026-09-18）  和 ②-B / ③ 两个模型聊：--obs strong|weak；/p <slug>（exercism 130 + MBPP 961，/p random）或粘自定义题（说明 + 空壳 + 可选 assert：强模式给模型看、弱模式只进秤）；
     /go /r n /s /ref /tests /obs 切模式；打题面、轨迹、标注版、最终文件、读数行（调用拆分、秤 k/n、得分、首版）、弱模式自测四读数、秤上挂的测试名；--dry 只渲染题面。本机验过解析/打分，vLLM 段待 GS01

═══════════════════════════════════════════════════════════════════════════════════════════════════
★★★★★ 总回顾（2026-08-30 ~ 09-18，用户要求「回顾和总结」）
═══════════════════════════════════════════════════════════════════════════════════════════════════
一、四条线走过的路
   0 CS336 自学（8-26 起）      不看视频全靠对话：讲完 L1 分词 / L2 资源核算 / L9 缩放律 / L3 架构 / L11 缩放律二，顺带 L4 L10 L13-14 L16-17；没碰 L5-L8 系统四讲、L12、L15（最早的欠债）；
                                L2/L3 重讲一遍才通 → 定下讲法：表格 + 能验算的数字 + 复述纠正
   1 从零线（8-27 ~ 8-30）      第一轮 95.2M 参数 / FineWeb-Edu 1.5B token / 48 分 / val 3.4068 困惑度 30.17（GPT-2 small 同数据 26.02，输 .148；跑赢 Chinchilla .178 = 数据红利）
                                第二轮 201M / 10B token（1.66 轮）/ 10.4 h / val 2.8961 困惑度 18.10（赢 GPT-2 small .363，≈ 有效 619M）；词表保持 50257 求可比
                                硬件实测：4090 间无 P2P，all-reduce 3.7 GB/s，grad_accum 8 压到 18.6%；MFU 单卡 .67 四卡 .55；numactl 绑 NUMA 1；GPU 上 KV cache 不提速（kernel 启动瓶颈）CPU 3×
                                结论：1.5B token 学会「怎么说」没学会「说什么」（2 bit/参数 ≈ 14 MB ≈ 维基 0.07%）
                                200M 做 SFT（8-30，Alpaca 52K，23.7 分）→ 教训：后训练不能看 val loss 选模型（val 1.92 → 2.41 输出反而更好）
   1′ GRPO 复现顿悟（9-02）     ★ 不在 200M 上：200M 做不了数学 RL，换 Qwen2.5-1.5B-Instruct + GSM8K（此后所有 RL 都在 Qwen 1.5B 系上）：pass@1 .53 → .74、贪心 .69 → .77，
                                贪心与采样差距缩 81%（熵坍缩量化）；#### 格式自发 .56 → .97。学到：RL 消灭差路不改进最优路；再用 Base 在 Countdown 上追顿悟 → 接 Countdown 线
   2 Countdown 线（9-02 ~ 9-10）  臂 0 只 RL .471 / 臂 2 自蒸馏+RL .473（= 臂 0：自蒸馏贡献零）/ 臂 1 程序撒种+RL .563（起点 mix .393）；臂 3 过程奖励 v2 判分器 126 步被钻；
                                计划 B 计算器工具：算术零件精度 .45 → 1.00，剩下的失败全落在种上（⑬）；臂 T 工具版 RL 学坏（塑走）；臂 S 秤做减法+题筛选+锚护种 → +.04、pass@64 不动、别的任务不伤（⑯）；
                                臂 A 限量问专家：求助是 p=0 的路必须冷启动、单模板绑措辞 .966 → .036 用 10 种措辞治好、RL +.06 只剪失控、专家是硬题全部收益、代价是工具依赖（⑰）
   3 E 猜数字（9-11）           通用 harness + 环境接口通了；纯 RL 从 4% 种子长出二分 .63（行为层的创造，零件是预训练的）；课程 .84 > 硬地 .63；程序撒种 200 条 .976 → RL .996；
                                N=1000 泛化 .71 靠示范教的「写下区间」= 显式状态（⑱）
   4 编程线（9-11 ~ 9-18，四格课程）
     ① 修 bug 短程强观测   2331 道注入 bug；零样本四个 Base < .08 → SFT .61 → RL .73（两趟）；四口径涨幅一样大 = 通用的少走弯路；严秤保 92%；pass@32 .96 → .97 不开路（⑳）；动态采样只买速度（⑲）；作弊探针 0/14
     ②-A 独立拼包 长程强观测  K=3 探针 .029 → SFT .076 → RL-bin .245（frac .195）；文件级 pass@1 +.19 / pass@32 +.02，包级 pass@32 +.24 = 乘法（㉒）；状态行无效因观测已带状态；bin/frac 修同一批文件（㉑）；
                             阶梯 ㉔、开路 ㉕、自蒸馏 ㉖；基建：异步 harness 677 → 350 s、vfork 211 → 11 ms、缓存、分块 logp
     ②-B exercism 强观测     探针 .06~.10 → DeepSeek 教材 SFT .329（val .24 容量线）→ RL .446；留出 .221 → .300；pass@32 .62 → .68 全在见过的题（每题见 15 次）；留出 pass@32 .60 → .53 削尾巴；
                             绑题一半；仪表盘第三次误导；作弊探针干净（㉗）
     ③ exercism 弱观测       探针 RL-②-B .295（不验证的便宜分）→ 弱教材 SFT .186（验证循环净亏 −.05）→ RL .282（首版 +.07，循环 −.02，MBPP 打平）；留出 .150 → .183；MBPP 留出 .369 → .495；
                             pass@32 三张全平；扩池起效（没见过 +.14 > 见过 +.08）；自测留住 95% 但不值分：判（假阳 .22 不动、代码对时十挂九）和修（变好 9%）是零件；Δlp .003；S 形 A=.330 第 75 步到顶（㉘）
二、建起来的东西
   训练器 grpo_tool_mp_vllm（vLLM 共存睡眠、异步 harness 滑动窗口、动态采样、分块 checkpoint logp、多任务分支、仪表盘 v2 见过/没见过、Δlp、混池抽题）
   harness 通用（停 → 环境 → 注入 → 续；同步/异步；假观测掐断；预算截断；每 token logp）| 环境族 calc/ask、GuessEnv、CodeEnv、PkgEnv、ExEnv 强/弱（reset/step/score 同接口）
   秤与防线：隐藏测试、oracle 严秤（ref 造 ≤20 条）、不落盘、断网、整文件替换、假观测标记；每级作弊探针 | 造题器：AST 注入器 2331 bug、拼包器、exercism 收集、MBPP 弱题 961
   老师：程序老师（① ②-A）、DeepSeek 写/修/弱模式（EXPLANATION/CHECKS/CODE，啰嗦过滤）| 评测器：pass_k（task/file 粒度）、eval_code/pkg/ex（按位置/死法/首版/改一轮/自测四读数/赢家调用数）
   对话：chat_code_vllm、chat_ex_vllm | 单测：test_pkg_env、test_harness_async、test_ex_env | 方法：实验流程模板（十件套）、环境设计六决定、好环境判据六条
   基建教训：fork 随父进程变胖 → vfork；并发没提速 = 串行瓶颈；NCCL 错位挂住不报错；盘满崩两次；仪表盘误导三次 → 只当趋势；vLLM-HF 概率错位量了 = 可忽略
三、定论的主线（⑬ ~ ㉘ 压成九条）
   1 分工：预训练给零件，SFT 给流程，RL 给偏好（闭式 π_RL ∝ π_底模 × e^{r/β}）；对齐 = 给的偏好正好是要的
   2 冷启动必需：p ≈ 0 时 RL 无从起步（四个 Base < .08）；SFT +.55 / RL +.12 的分工复现四次
   3 RL 只在支撑集内挪：pass@1 涨 pass@32 不动（① ②-A ②-B ③ + Countdown + E）；开路只在 1/k 边缘且按训练采样次数定义（㉕）；削尾巴（Yue 2025）；RL 一条轨迹一个 bit → 装不下知识、LoRA 够用
   4 长程 = 乘法：p^K，RL 抬每步可靠性在整件事上放大 (p′/p)^K（㉒）；支撑集要在原子粒度量；奖励形状不改终点（㉑）
   5 观测：验证 = 执行 + 期望值；弱观测的价 −.09（SFT）/ −.16（RL）；测试除了验证还带规格；自验证要经抓错→判→修才值钱，1.5B 缺后两段（㉘）
   6 秤：秤要比观测多知道；训练时的秤永远来自外面；过程奖励规则会被钻（臂 3）、老师的分布没边；秤按粒度分 ORM/PRM/token 级；六把秤的上限 = 判断的质量，只有蒸馏族上限是老师
   7 教材：程序老师装形（val .03），蒸馏老师装形 + 部分路到容量止（val .24）；单模板绑措辞 → 多模板；老师在弱观测下观测可信度 76%
   8 方法：一次一变量、预测写在前对账、两趟得噪声地板、探针先行（能力 + 作弊）、仪表盘只当趋势、S 形拟合 A/B 决定续不续跑、pass@1 与 pass@32 并报
   9 边界：1.5B 的墙（>60 行 ≈ 0、带类 .1~.3）；换零件来源三条：工具/专家、更强老师（on-policy 蒸馏）、更大底模；RL 到顶了换旋钮不续跑
四、欠着的 / 下一步候选
   教材 v2 断言少而准（拆「断言太多」vs「判断没有」）→ on-policy 蒸馏（老师当秤，判能不能装）→ 4B LoRA（容量）；②-B′ 注入版（耦合，等 4B）；③ + ②-A 拼包 = 真长程弱观测；扩池臂；rubric 秤（弱观测世界的秤）；
   评测 raw 存每次 run 按版本去重；调用价格归零的单变量
五、动机链（每个实验「因为看到什么所以做下一个」，用户 2026-09-18 要求）
   1 从零训 95M：亲手走通预训练管线（困惑度 30 vs GPT-2 small 26）→ 问后训练怎么做  2 200M SFT：装对话格式；发现后训练不能看 val loss 选模型 → 要能自动打分的任务
   3 GRPO 复现顿悟：GSM8K pass@1 .53 → .74 贪心只 +.08、格式涌现、熵坍缩 → 「RL 消灭差路不改进最优路」，但学了什么说不清 → 要更可控的任务
   4 阶段一 条件性策略绑定：RL 后答案变短推理消失 → RL 改的是特定上下文下的行为 → 追问顿悟条件  5 阶段二 追顿悟：任务必须真需要搜索且起点非零，CD-3 太易 RL 消灭推理 → CD-4
   6 三臂 CD-4（只 RL / 自蒸馏 / 撒种）核心减法：自蒸馏 = 只 RL、撒种 +.09 → 新零件只有老师给 → 追问奖励能不能教过程  7 臂 3 只换奖励：过程判分器 126 步被钻 → 以后只用终局奖励 → 剩下的错在哪
   8 计划 B 计算器 / 臂 T：失败全在算错的行（零件）→ 外包给计算器 .45 → 1.00，搭出 harness；RL 后反而学坏 → ⑮ 护种/换秤/换老师三选
   9 臂 S 秤做减法：只奖不罚 + 题筛 + 锚 → 安全小涨、pass@64 不动 → ⑯ RL 在可达集内挪；硬题仍全错 → 要外部信息  10 臂 A 限量问专家：教材装到位、RL 只剪失控、专家 = 硬题全部收益、代价工具依赖 → ⑰；第一个多轮动作 → 用户定「agent + RL」
   11 E 猜数字三臂：最小 agent 环境；纯 RL 从 4% 长出二分、课程 > 硬地、撒种最好、N=1000 靠「写下区间」→ ⑱ 显式状态来自教材、harness 通用化 → 可上真任务
   12 定长程编程线：观测强弱 × 路长短两根轴，编程环境能一路拧到底，产品在这条线上 → 四格课程
   13 ① 修 bug：跑通循环/沙盒/秤；零样本 ≈ 0 → 冷启动必需；SFT +.55 RL +.12；pass@32 不动 → ⑳；严秤 + 作弊探针 → 追问路变长会怎样
   14 ②-A 独立拼包：故意独立好和 p^K 比；乘法效应、状态行无效、bin/frac 同终点、文件级 pass@32 +.02 → ㉑~㉖ → 原计划 ②-B 真实模块 + 注入耦合
   15 ②-B 改成 exercism 从头写（用户两个开关：不注入、藏三成测试）：往真实走一步；SFT .33 RL .45 涨在首版；pass@32 涨的全在见过的题、绑题一半、留出削尾巴 → ㉗ → 欠债：题池小、改没动
   16 ③ 弱观测（用户提议）：拿走测试考自己造验证，顺手扩池 950 治绑题；动作装上 RL 留住但不值分，判和修是零件 → ㉘ → 三选：断言少而准 / 老师当秤 / 4B
   一句话：每一步的下一步都是上一步的欠债 —— 顿悟说不清→找可控任务；自蒸馏无效→找老师；过程奖励被钻→只用终局；算错→上工具；学坏→修秤；硬题无解→问专家；单轮不够→多轮；玩具够了→真任务；
           短程量清→拉长；长程强观测量清→拿掉观测；动作装上→缺零件。下一个欠债：零件从哪来
   教学（③ 结果的问答，2026-09-17，「记下来」）
     五条读法的白话版  学生自己写答案本，五页一页错、分不出哪页：答案本说错就去改，碰上错页就把对的改坏 → 「对答案」平均不加分。SFT 教会「每次都对答案」（95%），分辨哪页错这个本事 SFT/RL 都没教会
                     不对答案直接交 .302，对完再交 .282；SFT 刚教完是 −.05。RL 做的是少折腾（改轮 1.5 → .9，答案本说错也直接交），亏从 5 点变 2 点；分辨力是零件，RL 造不出
                     对照：②-B 训出的学生进没答案本的考场，喊「对答案」每局 .52 次被拒（扣调用），写完直接交 .295 = 会对答案的 .282 → 会对答案没带来分；甲有答案本时 .446，差 .15 = 答案本的价值
     正确的那几页起作用了吗  起了一半：价值链 抓住错 → 修好 → 得分；正确页负责第一段（断言挂时 72% 是代码真错）；断的是第二段：改一轮 646 轮 变好 9% / 没变 79%（重写同一份或改不相干处）/ 变差 12%（错页害的）
                     → 抓住错 79% 落空、12% 倒贴、9% 成分；正确页还有一处用得上但不加分：代码对、断言也过 → 放心交（只是确认）
     「改的能力三级都没动」  ②-A 拼包（改坏退不退回：RL 把偶尔对的做稳，改坏了改不回来）| ②-B exercism 强（改一轮 变好/没变/变差 .19/.67/.14 → .20/.64/.17）| ③ 弱（−.030 变好 .11 → −.016 变好 .09；MBPP −.032 → −.003）
                     共同点：RL 抬首版，改一轮变好始终 10~20% 不动；③ 里 RL 只让它少改。① 修 bug 不算：任务本身是改，第一次改对 RL 抬了（.6 → .9），没抬的是「改错之后再改」，那时没读数
     SFT 装了形、缺的两样、RL 学了什么  写答案本的能力有一半（断言 78% 对、真错时 72% 抓住），缺的是判（假阳 .22 不动）和修（变好 9%）；两样是零件，来源三条：4B / on-policy 蒸馏（容量内塑）/ mid 剂量数据
                     RL 学到的四类（全是已有动作里调比例）：① 第一版更对（首版 .235 → .302，≤15 行 .385 → .552）② 交卷前验最终版（.71 → .875；走到交卷 .29 → .41）③ 不转圈（改轮 1.55 → .85，撞顶 339 → 287，断言挂着就交 86 → 114）
                     ④ 简单题断言更准（MBPP 假阳份额 .28 → .21）⑤ 假观测归零。没学到：判 .22 不动、修 9% 不动、新解法 pass@32 三张表不动
     ★ RL 本质上让模型学了什么  秤在模型已有的做法里做选择：常赢的抬、常输的压；不往里放东西，只改已有东西的用法。信息量：150 步两万条轨迹 = 两万个 bit，装不下知识，够调几十种习惯的比例（= RL's Razor 离底模近）
                     这条线量到的四类：可靠性（pass@1 涨 pass@32 不动）/ 压习惯（放弃早、死循环、撞顶、假观测、喊 test）/ 决策阈值（何时停、验不验、改不动要不要交；「挂了改不回来就交」是它自己找到的）/
                     子步骤可靠性相乘（②-A 文件级 +.02 包级 +.24；见过很多次的题靠反复撞把稀路撞出来）。学不到：零件（判、修、知识、没见过的题上的新路）
                     工业界同一机制三处规模不同：底模大支撑集密 → 海量采样把「多想几步、自检、回头改」这些本来就有的路放大出来（R1-Zero 的顿悟 = 放大）；环境上万 → 抬的是通用倾向，跨领域搬得动；
                     RLHF 学偏好也是重新配比。业界承认的边界：知识要预训练/mid 给、小模型 RL 不如蒸馏、ScaleRL 天花板由底模和配方定、熵塌削尾巴是公认的病
     ★ 更本质的表述（带 KL 锚的 RL 最优解闭式）  π_RL(轨迹) ∝ π_底模(轨迹) × e^{奖励/β}：RL 后的策略 = 底模分布 × 秤的指数权重再归一化
                     推论：底模概率为零的乘什么都是零（不开新路、不长零件、天花板 = 支撑集）；不为零的按奖励指数放大（pass@1 涨；稀路放大后仍小、归一化被挤 → 削尾巴）；β 是尾巴旋钮（大 → 近底模留尾巴涨得慢，
                     小 → 陡、熵塌；ProRL 重置参考 = 换底数）；奖励喜欢什么放大什么不问对错（秤有洞就被钻）；RL 后离底模近（Razor）
                     三条轨迹算例 β=.5：错法 A .50×1 / 错法 B .30×1 / 对法 C .20×7.4=1.48 / 对法 D 0×7.4=0 → 归一化 .22/.13/.65/0：C 从 .2 到 .65 = 学会了；D 永远 0 = 学不到
                     用话说：RL 把秤内化 —— 训练时秤在外面挑，训完秤的口味变成权重里的偏置，推理时没有秤在场也按秤的口味在自己会的做法里挑。学到的不是新做法，是「秤会怎么挑」
                     实话：闭式是 KL 正则目标的最优解，GRPO 用组内基线和截断逼近；训练过头 β → 0 就是模式坍缩
     ★ 一句话  预训练给零件，SFT 给流程，RL 给偏好：在自己会的做法里按秤的口味重新分配概率。再短：预训练给「能」，SFT 给「怎么做」，RL 给「挑哪个」
     用户的两条（工业界特大模型上 RL 的作用）  ① 按秤的口味重新分配概率做可靠，保长程连乘的稳定 ② 特大模型零件齐全，用长 trace、更多 token、更多采样（都是算力）探索新路
       补：① 长程是乘法，每步 .9 二十步剩 .12、每步 .99 剩 .82，大模型每步本来高，RL 再抬一点乘出来差得多 —— agent 时代 RL 第一功能是把乘法救回来（②-A 文件级 +.02 包级 +.24 的百倍版）
           ② 「新」在整条轨迹层不在零件层：子步骤底模都有，整条概率万分之一，采够撞到、RL 放大到十分之一 = 看得见的新本事（π_底模 在很多组合上不是零而是极小）；
              算力花在两处：采样之间（每题几百条几千步撞稀路：BroRL 16 → 512 边界动，我们扩池每题 7 次边界不动）+ 一条轨迹之内（长 trace 是一次搜索，试、验、回头 = 走一棵树，RL 教会它这样走 = 把训练时采样搬到推理时）
           ③ 没列的：把秤内化 —— 训完模型自带秤的口味，推理时自己验证、自己否掉走不通的分支；没有内化的秤长 trace 只是长不是搜索。③ 里「验完再交」是它的小号，只是 1.5B 内化的秤分辨力不够
           边界：三条都在支撑集里，底模没有的零件算力撞不到（ScaleRL 渐近线）；熵塌尾巴削光第二条就停 → 保熵成了大规模 RL 的正经工程
     放大到十分之一之后接着做什么（工业界的循环）  变成教材（秤当筛子采几十万条留全过的做 SFT：R1 = RL → 拒绝采样 60 万 → SFT → 再 RL，把 1/10 压成默认）→ 往下蒸（教小模型）→
       更难的题进可学带（子路 p 到 1/10 后用它当零件的难题从 0 变百分之几，进 .05~.5 带有梯度，再撞下一层 = 课程一层层爬）→ 推理时当搜索用（放大出的验/回头在一条 trace 里跑，天花板推理时再抬）→
       回流下一代底模（RL 模型合成数据进下一代 mid-training：这代算力撞出的路，下代变预训练给的零件）。一句话：撞到、抬起来、筛出来、装回去、再往上撞；「采样撞稀路、秤筛、SFT 装回去」是这个循环的一格
     保熵  熵 = 分布多散（.5/.3/.2 高，.98/.01/.01 低）；RL 每步乘指数权重往赢的路上收，熵一路掉，掉到底一条路走到黑：稀路采不到、探索停、pass@32 掉 = 熵塌（留出集削尾巴是它的小号）
       保熵 = 别收太快，稀路还采得到：clip-higher（低概率 token 允许多涨）/ 熵奖励 / KL 锚（底模的散度就是尾巴）/ 定期重置参考模型 / 只更新部分 token（Clip-Cov 类）/ 按 pass@k 给奖励
       代价两头不兼得：熵高探索多可靠低，熵低可靠不探索；业界把策略熵当主曲线盯，2025 有论文拿熵预测天花板。我们 β .02、k 16 是熵掉得快的配方
     循环的具体流程（专家迭代 = 放大 + 蒸馏交替，AlphaZero / STaR / ReST 一脉；R1 六站）
       0 底模 V3-Base（零件）→ 1 冷启动 SFT 几千条长思维链（装流程）→ 2 推理 RL，GRPO 规则秤（答案对/格式/不混语言）跑到收敛（撞到、抬起来）→
       3 拒绝采样：站 2 模型每题采很多条、秤筛留对的可读的 60 万条 + 混 20 万普通对话（筛出来）→ 4 拿这 80 万条训【底模】两轮，不是训 RL 模型（装回去）→
       5 再 RL 全场景（推理规则秤 + 对话奖励模型）= R1（再往上撞）→ 6 站 3 的 80 万条直接 SFT 到 Qwen/Llama 1.5B~70B，不做 RL（往下蒸）
       两个关键：① 筛出来装回底模不接着训 RL 模型 —— RL 模型熵塌尾巴削光，从底模重 SFT = 路以教材形式装进尾巴完整的模型，路留住散度回来；RL 模型是数据生成器不是产品
                 ② 第二轮 RL 能更高 —— 站 4 后原来 p 万分之一的路成默认，用它当零件的难题从 0 变百分之几进可学带：第一轮的天花板是第二轮的地板
       我们做过两格：Countdown 臂 2 自蒸馏（采底模、秤筛、SFT、RL → 贡献零，底模没种子筛不出）；②-B 修教材（学生错版给老师改再 SFT）。可做的第三格：RL-③ 每题采 64、秤筛、SFT 成 ③′ 教材装回、再 RL；
       1.5B 的限制：38 道 32 次全错，筛子对它们捞不到东西，循环在这堵墙前停
     秤是不是对齐  三个词：意图（想要什么：代码能用、有帮助、不撒谎）/ 秤（把意图写成能自动打分的东西：测试、奖励模型、rubric）/ 对齐（模型行为和意图一致）
       训练只看得见秤：模型对齐的永远是秤不是意图；秤和意图的缝模型迟早走进去 = 整个对齐问题（奖励被钻 / Goodhart）。我们：意图「代码对」，秤是测试；只给可见测试就硬编码，
       隐藏测试/不落盘/整文件替换是把秤往意图推，作弊探针量的就是缝。工业界狭义对齐 = RLHF 那支（意图 有帮助/无害/诚实，秤 = 人标偏好训的奖励模型，后加 rubric/宪法/生成式评委 = 把意图写准）
       一句话：对齐是目标，秤是手段，秤的洞就是对齐的洞
     偏好是不是对齐  「偏好」两义：模型的偏好（RL 装进权重的偏置，训练的结果）/ 人的偏好（RLHF 的「A 比 B 好」，秤的原料）。对齐 = 两者一致。链条：意图 → 秤 → RL → 模型的偏好；对齐 = 链条末端和开头一致
       例：③ 的秤只认隐藏测试全过不认验没验 → RL 装的偏好「断言挂了改不好就直接交」对秤是对的；若意图是「交的代码必须验过」，这就是没对齐 —— 模型忠实对齐了秤，秤没忠实写下意图
       四句连起来：预训练给零件，SFT 给流程，RL 给偏好，对齐是给的偏好正好是我们要的那个
   教学（③ 进行中的问答，2026-09-17，「记下来」）
     两个 SFT 的对照  同一 100 道 exercism、同一把秤，差在观测：hf_sft_ex 强 0.329/0.326（4096/8192）pass@32 .620；同一模型弱模式 0.240；hf_sft_exw 弱 0.186/0.193 pass@32 .560；hf_exRL_bin 弱 0.295
                     三个减法：同模型强→弱 −0.09 = 观测的价；同弱模式 ②-B SFT → ③ SFT −0.05 = ③ 教材的价（教会验证，1.5B 手里净亏）；首版三行都 0.24 上下 → 差全在首版之后的循环
     观测怎么造成差别  两个环境只差有没有 <test>：强模式跑作者的七成测试（指着真 bug、覆盖边角、带规格信息）；弱模式只有 <run> 跑自己的断言（1/5 错、只盖想到的、没有规格）→ 三样之和 = −0.09
     判断没装上是谁的错  主因底模（分辨「断言错还是代码错」是零件；学生假阳 .21 ≈ 老师 .26 = 没学到分辨；<fix> 假改、五次重跑同一断言）；次因剂量（「断言错」那支示范太少）。能动它的：4B、on-policy 蒸馏、mid 剂量数据；RL 不造只选
     形 vs 内核  样本量要换成两个量：剂量 × 多样性（内核是电路，同一规律换无数面孔才长；形是映射表一条，几百条设默认）；学的方式（SFT 学老师文本，学不会的推理 val 停 .24 → 只吸收形；on-policy 在学生自己分布上改，容量内塑判断）
                但书：SFT 到 R1 60 万条拒绝采样的量也装真本事 → 「SFT 只学形」说的是剂量不是方法；预训练/mid/SFT 是同一根轴三段（剂量、多样性、位置），2025 后线被抹：Qwen3 预训练第二段合成推理链、Llama 3 退火段、Phi 课本；K2 的 agent 数据主要还在后训练
     on-policy 蒸馏怎么操作  老师不生成不出题，只「读」：题面 + 学生整段回答喂给老师 teacher forcing，取每个位置学生那个字的 log p_老师；分 = log p_老师 − log p_学生（逐 token）；
                loss = −Σ 分 × log p_学生，注入段屏蔽。不是「老师答一遍再比」：老师顺着学生走的路评下一步，比的是概率不是字面 → 精确到分岔那一个字（code vs check）
                坑：老师没见过标签格式会把 <write><run> 当错压 → 只给代码和自然语言 token 打分、或老师先在同教材 SFT 对齐格式。成本：本地 Qwen2.5-Coder-7B-Instruct（同分词器）每条多一次 7B 前向
                三种教法：SFT 蒸馏抄老师整篇 / RL 环境给整篇 0-1 / on-policy 蒸馏老师在学生稿子上逐字批
     老师当秤是什么  是秤（代理秤）：奖励公式 = logp 差；对秤的三规矩：从外面来 ✓、比观测多知道的是老师的本事不是隐藏测试、防钻 = 不动点是「学生变成老师」不会更差也不会更好；
                和 rubric 评委、生成式奖励模型同族（模型当秤，密但上限 = 评委）；臂 3 的过程奖励被钻是规则有边，老师是整个分布没边；用法是中间秤，真秤仍是测试
     工业界用不用、和蒸馏教材比  模型当秤是主流（RLHF 奖励模型、GRM、Kimi 自我批评、rubric、on-policy 蒸馏；Gemma 在预训练里逐 token 蒸馏）；测试当秤只在能验证的领域
                教材 vs 当秤：回答「好轨迹长什么样」vs「学生这一步好不好」；训练落在老师的状态 vs 学生的状态；学生犯错之后的状态教材里没有；Qwen3 报告：小模型 on-policy 蒸馏 > SFT 蒸馏 > RL 且算力 1/10
                业界小模型配方三段：老师写教材 SFT → 老师当秤 on-policy 蒸馏 → 真秤 RL。我们做完第一段准备跳第三段，第三段回答「0/1 的秤能不能自己剪掉转圈」，跑完再定要不要补第二段
     按粒度的秤名  整条一个分 = ORM / LLM 判官 / 生成式奖励模型；每步一个分 = PRM；每 token = on-policy 蒸馏 / token 级稠密奖励。粒度越细归因越准、被钻的面越大：PRM 2023 火过，R1 报告明确不用（步骤切不清、被钻、标注贵），退回规则终局奖励，PRM 转做推理时排序
     六把秤：需不需要更大模型、上限是什么  测试（不需要；上限 = 自己的支撑集随预算涨）| ORM（不需要，InstructGPT 打分 6B 策略 175B；上限 = 打分模型分辨力，被钻处即顶）| PRM（不需要；上限 = 标注质量）|
                判官/rubric（通常强模型也可自己，K2；上限 = 评委分辨力）| on-policy 蒸馏（需要更会做的老师不必更大；上限 = 老师）| 自蒸馏/自博弈（老师 = 自己 + 算力；上限 = 大预算下的支撑集，Countdown 臂 2 = 臂 0 因为 RL 自己在做同一件事）
                只有蒸馏族上限是老师；其余上限是「判断」的质量：判对错比写出来容易 → 秤只要会分辨，策略能超过秤自己写得出的水平，直到找到秤分不出的洞
     生成式评委 / ORM·PRM 为什么是模型  生成式评委 = 不接打分头，让模型写原则、写评语、再写分，可多采投票（DeepSeek GRM）；rubric 是给评委的清单，常一起用
                ORM 输入题+整条回答出一个数，训练数据是人标偏好对（Bradley-Terry）；PRM 输入题+前 k 步出第 k 步对不对（PRM800K 人标 / Math-Shepherd 自动标）
                为什么是模型：人的判断是真理但不够用（一次 RL 打分几百万条），奖励模型 = 把人的判断压缩成能无限调用的函数；人在上游标注、下游抽查红队。代价：近似有洞 → KL、重训、集成
     2026 年的打分主力（我的知识到 2026 年中）  三根按训练用量：① 程序秤 / 可验证奖励（答案比对、跑测试、agent 终态检查：CI 过没过、到没到目标页、表里的数对不对；范围比「测试」宽，能写成检查程序的终态都算）
                ② rubric + LLM 判官（写作/建议/报告/对话/研究；清单按题写，评委是强模型或策略自己，评委在有程序秤的任务上一起训着保持诚实；2025 起从补丁变正规军）
                ③ 偏好奖励模型 ORM（语气/安全/有用这类整体感受；人标偏好对训打分头；没被替代，从唯一变三根之一）
                三根外三样：老师的分布（蒸馏族，小模型主信号）/ 自洽当秤（多采投票，TTRL，弱但零成本）/ 人（不直接给分：标偏好、写 rubric、抽查、红队）
                看法：三根按任务分区不是三选一，一个模型三种都吃；真正的变化不在秤的种类而在配套 —— 秤的诚实要单独维护（评委有程序秤锚、定期作弊探针）、秤要成规模（几万环境、几十万 rubric）
                我们的顺序同构：程序秤走到能验证的尽头，③ 之后进弱观测，秤换第二根
   教学（回顾收尾的三问，2026-09-20，「记下来」）
     SWE-smith 在干什么  先纠数：论文本身 5 万道题 / 5 千条赢家轨迹；「几十万条」是后来各家放大的量级（Qwen3-Coder 并行两万环境）。算的账：SWE 题贵在环境不在 bug（一题 = 一个装好能跑测试的仓库，SWE-Gym 攒到 2.4k 到头）
                     → 倒过来：环境装一次、bug 造一万个。五步：128 个 Python 仓库各打一个镜像 → 五法造 bug（LM 改函数 / LM 重写函数 / AST 程序改：删行·翻运算符·改条件·去循环 / 多 bug 合一题 / PR 镜像：把真修复 PR 反着打回去）
                     → 秤验：至少一条测试过→挂、其余照过才留，不用人标 → LM 看 diff + 挂的测试写 issue 当题面（学生只见 issue + 仓库，看不到 diff 和哪条测试挂）→ Claude 3.7 Sonnet 跑 SWE-agent 做 5 万道，只留秤全过的 5 千条
                     → SFT Qwen2.5-Coder-32B → SWE-bench Verified 40.2%（当时开源第一）
                     对我们的词：一镜像 + 植入 bug = ①；过→挂的测试当秤 = 隐藏测试；LM 写 issue = 题面；学生看不到挂的测试要自己复现 = ③ 弱观测；撒种·秤筛·只留赢家·SFT = 臂 3 / ②-B 教材；一轮专家迭代
                     对称：训练者强观测（真测试当秤）、学生弱观测（自己复现）—— SWE-bench 本来就是这个格局
                     补的零件：不是算法知识，是「issue → 复现 → 几万行里定位 → 最小改动 → 跑测试 → 交」这条带工具输出的完整链；预训练有代码/commit/diff，没有带观测的过程；每条几千 bit 装的是流程 = SFT 给流程；32B 零件够装上就 40%，小一号明显低
                     论文发现：轨迹数与成绩近似对数线性；难题比例高更好；仓库多样性有用；PR 镜像 bug 最值钱（最像真错）；issue 越具体题越可解
                     和「改」的关系：植入 bug 是手滑型（别人往正确代码里放一处、测试指着它）= ① 那类 RL 抬得动的分布；SWE-smith 用 SFT 先装流程，DeepSWE 再在同类环境跑 RL（R2E-Gym 4.5k、Qwen3-32B pass@1 42%，加测试时搜索 59%）
                     我们的位置：环境装一次 ① 已用；差在函数级 vs 仓库级、1.5B vs 32B；沿这条路 = ②-B′ 注入版（exercism 模块里植 bug = 迷你 SWE-smith）
     猜想：AI 造轨迹 → 只有前沿知道 → 知识垄断 → 普通人失去价值 → 权力集中  我的判断：「知识垄断」这环机制上站不住；「权力集中、普通人议价力下降」是真风险，原因不是「只有他们知道」是「只有他们能做」。五环逐环：
                     ① 新轨迹由 AI 造：对一半，边界是秤 —— 有完美免费秤的领域（棋、可机验数学、带测试的代码）是；没有的（要做实验的科学、品味、价值、什么值得做）秤仍是人和物理世界，循环快不起来
                     ② 只有前沿公司知道：最弱一环，找贵抄便宜 —— 百万 GPU 小时找到的轨迹写下来几千 token，蒸馏进小模型几千 bit；信息量对接收者说（找的人 1 bit，没见过的人一本书）。证据：R1 一出就蒸出 1.5B~32B；
                        开闭源差半年到一年没拉开；API 输出就是教材；围棋 AlphaZero 后两年 KataGo 开源追平、人类棋手整体涨；用户自己：一人四卡，R1 后 18 个月把这套动力学量了一遍
                     真正集中的：存量（知识）扩散快，流量（造新知识的速度）卡在算力、环境、资本 —— 是时间差不是有无差
                     ③④ 普通人失去价值：真，但和垄断无关，是替代问题；「价值」拆三层（市场议价力 / 政治权利 / 人的分量），第一层掉是大概率，后两层由制度定不由信息论定
                     ⑤ 权力集中到 AI 公司：一半 —— 芯片、电、法律在国家手里（出口管制 = 国家收权）；铁路/石油/电信集中过再反垄断商品化；AI 更快更通用不保证重演；「集中到公司」和「集中到国家」是两个未来
                     悲观版成立的四条件 = 看盘指标：优势复利（AI 造下一代 AI 快过扩散，目前没量到）/ 抄被堵（法律或技术禁蒸馏和开源权重）/ 值钱领域的秤都能自动化不需要人和物理世界 / 算力持续稀缺且集中
                     立场：赌「知识垄断」不成立，认真对待「能力集中」与「劳动替代」；吓人的版本是「只有他们能做且循环复利」；业内含造我的人把权力集中当一级风险；杠杆在开源权重、发论文、制度，不在信息本身
     反驳：反蒸馏已在用（藏链 / 不发最强 / 护栏），且算力差距会让开闭源越拉越大  改口一处 + 把「差距」拆两把尺（我是利益相关方，打折听）
                     措施确实有且动机明说含竞争：藏思维链（2025 起长思考只给摘要，OpenAI 公开说藏链有竞争原因）/ 分级发布（最强版只给审核机构；Fable 带额外安全措施、同底模 Mythos 只给批准组织）
                     / 条款与准入（禁拿输出训竞品、身份核验、地域限制、流量监测）/ 开源阵营收缩（Meta 动摇，主要开源来源变成受芯片管制的中国实验室）→ 「抄便宜」说过头：抄仍比找便宜一到两个量级，措施缩小了差距没抹掉
                     能藏住什么三层：做得成 + 大致方向 藏不住（卖能力 = 公开存在性证明；o1 藏链 R1 四个月复现靠方向不靠轨迹）| 过程轨迹 藏一半（学生从「SFT 抄轨迹每条几千 bit」退回「答案当秤自己采、每次 1 bit」= 专家迭代，
                        成本高 10~100 倍；正确路径在 k 次里出现一次答案就够当秤 → 这道护栏的效力随学生底模变强而衰减，买的是时间不是永久）| 权重、环境、产品数据飞轮、算力 藏得住（真护城河）
                     差距两把尺：时间差（开源追平要多久）2023~2026 一直半年到一年、2025 还缩过，方法全开（DAPO/GSPO/ScaleRL 都是论文）、算法效率年 3~10 倍 → 没有拉大的证据
                                能力差（同一时刻能做什么）曲线变陡（算力堆 RL、AI 参与造 AI）则同样 9 个月含的能力更多 → 最可能的形态：时间差不变、每段时间差装的能力变多，集中在长程 agent 这类吃算力和环境的能力 = 实质不对称
                                时间差本身跑飞（9 → 24 个月）需四条件同时成立：抄被抬价 部分✓、开源收缩 部分✓、优势复利 ✗没量到、国家反对开源权重 ✗（美国 AI 行动计划明写鼓励、中国拿开源当战略）
                     更新后立场：「知识垄断」改为「时间差被有意维持，且每年时间差含的能力更多」；「只有他们知道」仍不成立，是「先知道 6~18 个月」，商业上值钱、垄断智慧不够
                     看盘三指标：最强开源权重那家还发不发 / 时间差 9 个月有没有变 24 / METR 长程任务时长指标上开源是否仍跟同一条曲线 —— 任一翻了就改口
   教学（回顾收口，2026-09-21，「记下来」）
     验证 = 执行 + 期望值  一条测试做两件事：跑一遍拿实际输出（执行：便宜、精确、机器做）+ 和「应该是什么」比（期望值：贵、要懂题、知识所在）。强弱观测的差别 = 谁出期望值：
                     ②-B <test> 执行环境做、期望值作者出 → 完整可信 | ③ <run> 执行环境做、期望值模型自己写 → 执行可信、期望值和模型一样不可信 | Countdown + 计算器 执行计算器做、期望值是题面给的目标数 → 补上执行就完整（臂 T 一上工具值全对）
                     | GSM8K 纯推理 两样都在模型手里，最弱。推论：观测可信度 ≤ 造观测者（期望值是模型出的）；假阳 = 期望值写错、假阴 = 期望值没写到；弱观测掉在改的循环不在首版（首版只用写，改才用期望值）；
                     空 run「no output」读成全过 = 把执行当成了验证。弱世界里期望值的四个来源 = ③ 之后的路线：题面例子直接抄 / 性质关系不要具体值（编码再解码等于原文）/ 外部神谕（老师当秤、参考实现）/ 多采样投票
     ③ 值钱在哪（没涨分，但归因最干净）  ① 量出观测的价并拆开：SFT −.09、RL −.16、天花板 −.06；价全在首版之后的循环，天花板丢的是规格歧义题 → 测试给 agent 的是期望值 + 规格信息两样，不是笼统的「反馈」
                     ② 动作与能力分开：自验证 SFT 装得上、0/1 奖励的 RL 留得住（不需要过程奖励），但 1.5B 手里不值分 → 瓶颈不在「不自测」，在自测之后
                     ③ 验证拆三段各自量到：抓错 .72 / 判 假阳 .22 不动 / 修 变好 9% → 缺的定位成零件不是习惯 → 下一步从「再跑一趟 RL」变成三个旋钮（断言少而准 / 老师当秤 / 换底模）= 归因换规模的实例
                     ④ 观测可信度规律在老师（76% / 63%）和学生（合取一条错整段挂）身上都量到 → agent 设计规则：自造的验证继承造者的错误率；断言要少、从规格抄、挂了先当假设不当判决
                     ⑤ 三件方法工具第一次用于决策：S 形 A/B 判「75 步到顶别续跑」/ Δlp 判「TIS 不用做」/ 扩池后见过 +.084 没见过 +.143 绑题消失（㉕ 反向）
                     位置：学生弱观测、训练者强观测 = SWE-bench 格局，结论可搬：小模型装「先写测试」不加分、32B 以上才加分 = 「工业界为什么要大模型做 agent」的带数字回答
                     没给的：能力没涨；RL 一趟无噪声地板；MBPP 混池弄脏 RL 阶段对照；断言质量按 run 算有重复计数；老师提示词「至少 6 条断言」是我种的病
                     回顾 15 的补记：③ 是预测错得最系统的一次（探针低估一倍、SFT 高估 10 点、砍改没发生），三处错同源 —— 把「验证」当一个动作算账，它其实是抓错 → 判 → 修三段链，1.5B 只有第一段
     实验总索引（回顾收口；用户编号 1~15 为准，动机链是 16 条因为把「定长程编程线」单算了一条，报告按用户编号走）
       1 CS336 自学 | 2 从零预训练两次 95M / 201M（困惑度 30.17 / 18.10）| 3 200M SFT 2×2 对照（会停 ≠ 会答；后训练不看 val loss）| 4 GSM8K GRPO 两轮，Qwen2.5-1.5B-Instruct（RL 消灭差路不改进最优路；熵坍缩）
       5 Countdown 3 数 条件性策略绑定 | 6 混训三段 → mix4-300（任务要真需要搜索且起点非零）| 7 三臂 CD-4 + 臂 3/3′，mix4-300 起（自蒸馏 = 只 RL、撒种 +.09；过程奖励被钻）
       8 计划 B 计算器 / 臂 T（⑮）| 9 臂 S 秤做减法（⑯）| 10 臂 A 限量问专家（⑰）| 11 E 猜数字三臂（⑱）
       12 ① 短程强观测 修 bug，Qwen2.5-Coder-1.5B 起（⑲ ⑳）| 13 ②-A 长程强观测 1 独立拼包（㉑~㉖）| 14 ②-B 长程强观测 2 exercism 从头写（㉗）| 15 ③ 长程弱观测 自己造验证（㉘）
       含在别项里没单列：臂 3′、臂 S′、臂 A 十模板 SFT-2、工具版 SFT、③ 的 8192 预算测试、四阶段各自的作弊探针、基建（异步 harness / vfork / 动态采样 / Δlp 诊断）
       计划过没做：②-B′ 注入版、②-B 与 ③ 的第二趟 RL、教材 v2、on-policy 蒸馏、4B
       待整理：定论 ①~⑭ 散在 GSM8K / Countdown / 三臂段，编号规则与后面不一致，写报告时统一
   基建线总回顾（用户 2026-09-21 要求「infra 从开始到结尾做了什么、怎么演化、最终是什么」；「记下来」）
     第 0 段 预训练栈（8 月下旬）  config.py 唯一事实来源 + estimate.py 核算（MFU 估 .30 → 实测四卡 .547，回填后每步预测 1.01 s 实测 1.015 s）；model.py（RMSNorm/SwiGLU/RoPE/SDPA→Flash/权重绑定/残差初始化；后加 KV cache：GPU 上量不出差别，瓶颈是 kernel 启动）
                     train.py（torchrun DDP + 累积只在末 micro 同步、bf16 + compile、cosine+warmup、latest/best/final 三种 ckpt、SIGINT 四 rank 对齐停、--resume、MFU 实测、--smoke）；prepare_data（FineWeb-Edu → GPT-2 BPE → uint16 bin）；tmux + monitor + check_ready
                     坑：ckpt 只在 val 改善时存；DDP/compile 给 state_dict 加前缀；--steps 与 world_size 顺序
     第 1 段 SFT + 第一版 RL（08-30~09-02）  sft.py（200M）→ sft_qwen.py（只对回答算 loss、手动 all_reduce、按 rank 分片、存 --init 能读的 ckpt）；grpo.py 全参 + 8-bit Adam、HF generate、单进程
                     四卡并行：线程版慢 4.5×（HF generate 不释放 GIL）→ torchrun 四进程各采各算 all_reduce 求和（梯度同 → 权重天然一致，不需同步权重）19.7 s/步 = 4.7×，88% 功劳是绕开 GIL
                     认识：RL 吞吐 = 每秒验证多少想法；采样占 85%；信息带宽比预训练低 30 万倍 → 采样加速 = 学习速率本身
     第 2 段 判分器 / 教材工具 / 第一个 harness（09-03~07）  grpo_mix_mp 多任务混训，判分 v1→v3（judge_v2 纯函数可单测）；enum_traces 程序老师、reject_sample 自蒸馏、check_sft 体检、screen_pool 离线动态采样、slim_ckpt 15→3 GB
                     calc_tool.py 第一个 harness（停/注入/续、注入段进 mask）；规矩：协议先于教材，协议住 harness 里且假模型单测；代价量出：臂 T 一步 190 s 里 180 s 采样，13 h 里 12 h
     第 3 段 换引擎 vLLM（09-08）  五件要验：装得上/后端写对/数值等价/权重同步/同卡显存共存；export_hf/import_hf；calc_tool_vllm 后端层（HF/vLLM 二选一、喂 token id 不喂字符串）；vllm_check 等价（贪心 20 道逐 token 同、温度 1 800 条分布同）
                     评测 4 卡 60 分 → 单卡 2.3 分 = 26×；训练器 grpo_tool_mp_vllm：external_launcher 四进程各一份 vLLM、睡眠 level 2 让出 6~7.6 GB、每步灌权重、分段唤醒峰值低 4 GB；rollout 6/2/3 s vs HF 32/186 s；300 步 2~3 h vs 8.5 h
                     坑三：KV 预算把已占显存算进去 / 第三次唤醒 OOM 要 gc+empty_cache / 退出段错误要 barrier+os._exit；臂 S OOM → expandable_segments、--micro 1；deepseek_tool（零依赖、温度 0、缓存、重试、密钥只走环境变量）
     第 4 段 通用 harness + 环境接口（09-11 E）  harness.generate_with_env；环境四接口 stops/fake_tags/step/score；换任务只换环境文件，训练器加任务分支；一步 5~8 s；tests/test_harness 假后端十项 → 定论 ⑱ 第 1 条
     第 5 段 真环境：沙盒、秤、防线（09-12~13 ①）  mutate AST 注 bug；code_env 四工具协议；沙盒 = 子进程 -I / 临时 cwd / 5 s 超时 / RLIMIT 512 MB / 只留 PATH / socket 抛异常 / 截 600（坑：killpg 杀了自己的进程组）
                     秤 = 可见 + 隐藏、只奖终局、假观测罚 .2；五防线（隐藏测试 / 只读母本重铺 / 盘上无文件 stdin 喂 python -I - / 断网 / 整文件替换）；strict_score 严秤；tests/test_code_env 假模型 + 真沙盒
                     动态采样 ⑲ 76 → 40 s/步且每步多跑一半；★ 性能诊断：一步 42 s 里生成 + 环境 15 s，27 s 不明 → 起子进程太贵（fork 复制 4 GB 页表 + preexec_fn 逼真 fork 串行）→ del 6 GB + vfork：211 ms → 11 → 3 ms；rank_spread 量出新大头 = 等最慢卡 45%
     第 6 段 长程：异步 harness、缓存、分块 logp（09-14~15 ②-A）  5k 序列 logits OOM → token_logp 1024 一块 + checkpoint（任何时刻 0.6 GB，值差 0 梯度差 1e-6）
                     级别二异步 harness（每条轨迹一个请求，停在标签丢沙盒线程池，回来续前缀缓存，步末汇合，严格 on-policy；HARNESS_ASYNC 开关、接口不对自动退回；test_harness_async 倒序假引擎逼交错、逐字一致）
                     第一版只快 11% → 诊断 prefill 主导、800 条挤爆 KV 整段重算 → 滑动窗口 inflight 128：677 → 350 s；PkgEnv 结果缓存七成命中，沙盒累计 35 → 8 s；超时 5 → 2 s；一步 59 s「infra 到此收手」
     第 7 段 真题、老师、弱观测（09-15~17 ②-B/③）  collect_exercism 133 道切七成可见；ex_env 强弱双模式 + 版本/run 记录 + 四格读数 + traceback 带源码行；make_ex_demos DeepSeek 在真环境写跑改（啰嗦过滤、timeout 300）；collect_mbpp_weak 961 道
                     训练器末批量具：仪表盘 v2（20 未见 + 20 见过、温度 1×4、每 25 步、S 形拟合）/ Δlp .003（TIS 不用做）/ 连续超时早停 / 多池按比例抽题；运维教训：NCCL 末批错位挂 10 分钟、硬盘 98% 崩两次
     最终  约 75 个脚本 1.8 万行、6 个单测文件，十层：造题（mutate/collect_*/countdown/guess_env）→ 环境（code/pkg/ex/guess_env 四接口）→ 沙盒和秤（五防线、strict_score）→ harness（同步/异步 + 后端层）→ 训练器 grpo_tool_mp_vllm 1168 行
           → SFT sft_qwen → 老师（enum_traces/make_*_demos/deepseek_tool）→ 评测（eval_*/pass_k/baseline_*/compare_*/探针）→ 对话 chat_*_vllm → 运维（export/import/slim、monitor、rank_spread、md5 同步、两个 venv）
           一步 RL 的刻度：单卡估 93 s → 多进程 19.7 → 工具版 HF 190 → vLLM 20~30 → 代码题 76 → 动态采样 + vfork 40 → 长程异步 + 缓存 59 → ③ 序列长五倍仍 ~80
     规律  每层在上一层成瓶颈时才建、建前先量（85% 采样 → vLLM；27 s 不明 → fork；prefill 主导 → 窗口）；切换前先验「分数一样只是快」（vllm_check / 异步等价 / 动态采样对照）；协议先于教材、单测先于训练（假模型 + 真沙盒）；
           正在跑的实验依赖的文件不动；不用 verl/TRL 是为了碰到 GIL、权重同步、睡眠分配器、prefill 挤 KV 这些坑。没建的四样：全异步分集群（只到级别二）、上万环境（几十个沙盒目录）、LoRA/多机（4B 要）、critic/过程奖励（有意不做）
     速度线：四卡 + vLLM 一步 RL 怎么一格格压下来、每格的瓶颈（用户问，2026-09-21，「记下来」）
       格  一步                          当时的瓶颈                                                       修法                                  修完的新瓶颈
       1   单进程 HF 估 93 s（采 78 训 15）  一张卡采样、HF 静态批短的等长的                                     —                                     —
       2   线程四卡 慢 4.5×               GIL（HF generate 不释放）、批切四份 GPU 吃不饱                         放弃线程                              —
       3   torchrun 四进程 19.7 s           采样仍七成：HF 静态批 + Python 停止串检查                              各采各算 all_reduce 求和，不用同步权重     采样
       4   工具版 HF（臂 T）190 s            每次停标签后整段前缀重算（轮间不留 KV）、一条几十次调用                   —                                     采样 95%
       5   vLLM 同卡共存 20~30 s            显存：学生 3 + 参考 3 + 梯度 3 + 8-bit Adam 3 + 激活/logits 2~5 + vLLM 权重 3 + KV 3~4 顶格   睡眠 level 2、每步灌权重、分段唤醒；臂 S OOM 后 expandable_segments + micro 1   显存而非速度
       6   代码题 ① 42+7 s                  生成 + 环境只 15 s、27 s 不明 → 起子进程 211 ms/次（fork 复制 4 GB 页表、preexec_fn 逼真 fork 串行）   del 6 GB + vfork → 3 ms；动态采样   76 → 40 s；四卡等最慢占 45%
       7   长程 ②-A 59 s（35+24）           5k 序列 logits OOM / 同步轮次一轮等最慢 / 异步后 prefill 挤爆 KV / 死循环超时串行 / 重复跑同文件   分块 logp / 异步 harness / 窗口 128 / 超时 2 s / 结果缓存   步内尾巴 + 卡间 9~12 s
       8   弱观测 ③ 40~56 + 22 s            序列 2600 token，引擎 42 s 全是生成（沙盒 8 s 藏在下面）；训练端 22 s = 序列长五倍 + 分块重算   —   到顶
       评测线：800 条工具评测 四卡 60 分 → vLLM 单卡 2.3 分 = 26×
       vLLM 为什么快  HF 静态批一批 32 条一起进出，短的等长的全在算 padding；vLLM 连续批处理谁写完谁下岗批永远满；PagedAttention KV 分页碎片 <5%；前缀缓存同题 16 条题面 KV 只算一次、工具轮次回来接着算不整段重算 → 第 4 格的病根，换引擎在工具版上收益最大
       四卡上怎么共存  四个 torchrun 进程各带一份 vLLM（external_launcher 防进程组打架）→ 权重同步是同卡 GPU 拷贝不走 PCIe。每步节奏：训练完 gc + empty_cache → 醒权重 → 灌学生当前权重 → 醒 KV → 采样 → 睡 level 2 丢权重丢 KV；每步付 1~1.5 s
                      KV 池 10 GB ≈ 37 万 token ≈ 同时 100~180 条 → 窗口 128 的来历（超过就前缀被挤、整段重算）
       「等最慢」三层，消掉一层  轮内（同步 harness 一起停一起跑沙盒等最慢一起续、八轮等八次）→ 异步 harness 消掉：本卡耗时从「每轮最慢之和」变「最长的一条」，仍严格 on-policy
                      步内尾巴（最后几条长轨迹 GPU 喂不饱）→ 消不掉，除非加大批次摊薄或级别三全异步（采完进缓冲攒够就更新，要 off-policy 修正；Δlp .003 说明错位小，新误差只来自陈旧度）
                      卡间（all_reduce 前等最慢的卡，题长短不同）→ 动态采样 + 题分配摊薄后剩 8~12 s
       到顶的原因  最后一步 80 s = 采样 55~60 + 训练 22，和工业界同比例采样占大头。再压卡硬件形状：4090 无 NVLink、PCIe 20~30 GB/s，模型切不到多卡、采样和训练分不到两组卡流水；
                   工业界每卡 80~192 GB + NVLink + IB，采样集群与训练集群分开、全异步 = 同一套算法在另一种硬件上的形状。每格先量后修，最后一格量完写「infra 到此收手」：剩下两个瓶颈要的是卡不是代码
   CS329A 自学（Stanford「Self-Improving AI Agents」，Chowdhery + Mirhoseini；公开视频 9 讲，课堂 20 次课；2026-09-21 起，对话讲、不看视频；顺序 1→2→3→6→4→8→5→7→9；「记下来」）
     第 1 讲 总览  主线 = 六年换了四台发动机：① 堆规模（参数/数据/算力幂律，去年饱和；买到 few-shot 和涌现，思维链 7~8B 无效）② 后训练配方（预训练→买来的 SFT→合成指令微调→RLHF；给格式和偏好不给知识；对齐没解决）
                  ③ 推理时算力（模型不动：并行多采 + 验证器 = Large Language Monkeys；串行长想 = o1 对数线性；前提是有验证器）④ 闭环（③ 的赢家轨迹喂回训练，讲师说「没有边界」）
                  贯穿全课：验证是瓶颈、生成与验证的差距；agent 定义 = 目标/规划/环境交互/按反馈纠正/决定何时停（分界）/工具记忆；编程 agent 能用 = 模型强 + RL 好 + 自写测试变可靠 → 闭环启动；先澄清意图才知道怎么验
                  三个学生问题：训练教「哪条对」把 pass@k 搬到 pass@1（= 定论 ⑳）/ 模型更喜欢自己的轨迹（on-policy 伏笔）/ RL 为何涨那么多「没完全理解」
                  用户复述纠正：③ 不在 RL 里、参数不动、卡在有没有可靠验证器不是便宜；④ pass@1 最多抬到覆盖率、只对撞得到（p > 1/k）的题、要泛化得几万题；验算 p=.0004 → k=32 1.27% / k=10000 98.17% ✓
     第 2 讲 测试时算力  一张 100 行 × 10000 列的对错表：pass@1 = 第 1 列；覆盖率 = 前 k 列至少一对的行 = 我们的 pass@k；单题饱和、整卷因难度铺开成 log 直线（每乘 10 涨一档），指数幂律拟合
                  LLM Monkeys 数：SWE-bench Lite DeepSeek-Coder-V2 采 1 条 15.9% → 250 条 56%（单次纪录 43%）；MATH Llama-3-8B 覆盖率 100 条 82.9% → 10000 条 98.4%，投票/奖励模型 40.5 → 41.4 = 差距 42 → 57，k 越大差距越大
                  差距 = 覆盖率 − 挑选后实得分；完美验证器时为零。挑法：验证器跟着覆盖率走 / 投票几十列定型（众数错就错，量的是自洽不是对错）/ 评委几百列到顶后下滑（对错比 1/p 放大误放率、最高分被极值抢、盲区重合）
                  并行 vs 串行：Snell 2024 按难度分配比 best-of-N 省 4×，小模型 + 测试时算力顶 14× 大模型，但 p≈0 的题只能靠预训练；CodeMonkeys 自写测试 + 串并行，SWE-bench Verified 57.4%（$2300）= 自写测试在 Sonnet 级可当挑选器
                  验算：误放 2%、p=.01、k=100 → 放行里对的占 32%；误放换 .22（1.5B 自测）→ 4%
     第 3 讲 稳健验证  五个验证器（铅笔题 A 对 / B 算错 / C 思路错 / D 过程错答案碰对）：
                  ① ORM（GSM8K 2021）训练题采 100 份按答案标 0/1 训打分模型，挑最高分；6B+验证 ≈ 175B 纯微调（30×）；6B：test@1 20.6%、覆盖率 test@100 ≈ 80%、ORM ≈ 40%（400 份到顶再下滑）；病：D 类假阳、极值、只知整份错
                  ② 人标 PRM（PRM800K 2023）每步人标 80 万条，整份分取最低步；同一堆 1860 份：投票 69.6 / ORM 72.4 / PRM 78.2；治 ORM 的假阳（D 第一步就挂）；贵、步切不清、也被钻 → R1 不用于训练只用于排序
                  ③ 自动 PRM（Math-Shepherd）从每步往下 rollout 8 次按到达正确答案比例标分；Mistral-7B GSM8K 77.9 → 89.1、MATH 28.6 → 43.5；三病：贵、噪声、继承生成器盲区（= 验证器要比被验的多知道；自己给自己标多不出）
                  ④ 生成式评委（LLM-as-judge）读完写判决；无独立数；病：偏好长/自信/格式、与生成器盲区重合
                  ⑤ Weaver（2025）33 个现成开源奖励模型（RewardBench 8B~72B）+ Arena 评委，一个没自训；分归一化→二值化→贝叶斯加权（权重 = 各评委准确率，用两两一致率反推，一致率 = 准×准 + 不准×不准，528 对方程最小二乘；1% 标签只估先验）→ 扔掉近随机的 → 交后验最高
                  同堆同生成器 Llama 3.3 70B k=100 四题库平均：第一份 68.4 / 投票 72.2 / 最好单评委 72.7 / 不加权 69.1 / Weaver 87.7 / 覆盖率 91.9；MATH500 93.4 vs 98.6；强在错误不重合 + 权重压噪声 + 顶部校准；限：难题上评委共有偏好、33× 贵 → 蒸馏进 ModernBERT 396M 保 98.7% 省 99.97% 算力
                  四年总账：覆盖率都 ~九成，实得分从 ORM 捡回一半 → 单评委八成 → Weaver 九成五
     问答定论  验证器分类：按「期望值从哪来 / 当观测还是当奖励」两个标签归位。三档：外部（作者测试、题库答案）/ 独立的内部（抄题面例子、性质、笨办法另算 → 两条路独立，抓手滑）/ 同源的内部（凭理解编 → 抓不了不懂，改坏对代码）
                  自写测试 = 半个验证器（执行真、期望值是评委）；价值 = 期望值与代码的独立程度；永远不能当奖励；「模型强到生成的测试变可靠」= 第二档比例上去、同源也少错
                  外部 vs 自写哪个重要：训练看外部（是奖励、不可替代），部署看自写（没测试的地方的入场券），自写是外部训出来的副产品 → 外部在前；四格表：外部秤决定训练动不动，自写决定没测试处能不能用
                  rubric + 评委 = 第 4 种加外部清单：期望值变外部、执行由评委逐条判（模糊）→ 有机会抓「不懂」；RaR 在 HealthBench-1k 比直接打分高 ~28%、Checklists 论文 RL 涨几点、HealthBench 262 医生 4.8 万条；病：写不全、被钻（= 臂 3）、评委错、贵；无 oracle 只能拿人一致率当代理；有测试处严格不如测试
                  上限不是老师：五个验证器无一要求评委比生成器更会做题（Weaver 72B 以下判 70B）；判对错比做出来容易 → 秤的上限是分辨力不是解题力；分辨力四来源：外部真相 / 人 / 执行 / 多样性+算力（Weaver、AlphaZero、Math-Shepherd 是危险的那种）；四样都没有 = 验证是瓶颈的本义；前沿的做法 = weak-to-strong、辩论、元验证
                  题的三档：可执行（代码测试、形式证明、游戏、仿真：不要答案也能验）/ 有钥匙（GSM8K、MATH、GPQA、MMLU：比答案是程序但要钥匙、验不了过程）/ 没钥匙（写作、建议：只能评委）；五篇的题全在第二档，论文设定 = 训练题有钥匙、测试题假装没有
                  编程能被拿走的机制：四环闭环 ① 现成的秤（几十年的测试）② 训练 ③ 自写测试变可靠 ④ 无测试任务配上秤变新题 → 回 ①；「自转」= 第二圈起不需要人写测试；③ 在 1.5B 断（假阳 .22）；别的领域缺环 ①（写作）、环 ① 太贵（实验科学）、环 ① 是人（医疗）
                  拿走顺序 = 秤的便宜程度：游戏 → 竞赛题 → 修 bug（4% → 80%）→ 从头写模块（进行中）→ 架构取舍（未开始）→ 决定做什么（秤是用户）；边界往「规格和判断」挪，最后剩「决定要不要 + 看到结果说这是我要的」，其期望值只在想要的人脑子里
                  编程验证为什么最便宜：执行毫秒确定（沙盒 3 ms/进程）/ 期望值已写成测试或易写（具体例子比算法容易、参考实现造 oracle）/ 观测定位到行 / 可重开无副作用 / 便宜防线难钻 / 无限造题；贵的挪到造环境（SWE-smith：环境比 bug 贵）和写规格
                  「强底模 + 完美秤 = 解决一切」的三处修正：强是 p 与预算的比（p > 1/k），秤的形状（部分分、逐步、课程）和搜索能降门槛；封闭可验证世界不需要强底模（AlphaZero 随机起步，每局都有胜负信号）；漏四类：长程 p^K、规格谁写、秤贵的世界、新知识要从世界量回来
                  我的版本：强底模 + 完美秤 + 算力解决的是「便宜可验证 + 每步 p > 1/k + 规格写清」三条同时满足的问题；候选来源三个（先验 / 搜索+算力 / 课程），预训练最重要不唯一；秤要跟着模型一起长（CURE）
     问答（2026-09-22，用户两问，「记下来」）
       RL 采回去 SFT 是不是自蒸馏、碰过环境的轨迹信息是不是更多  前半句对：拒绝采样 / STaR / ReST 就是自蒸馏，轨迹的字梯度≈0，新信息只有秤的「留下」1 bit/条（臂 2 = 臂 0 是实验版；秤本身也是环境，所以严格说已是一次碰撞）
                     后半句分两层：轨迹里有多少外部信息 ≠ 进权重多少。四档：无秤自采（0）/ 只有终局判决（≈1 bit，进权重的是选择）/ 中途有观测但观测当输入 −100（每次几十到几百 bit，进权重的是「看到这种观测怎么接」的技能，事实留在观测里：traceback 说 expected 29，
                     学到的是照 traceback 改，测试时没 traceback 仍不知道 29；E 猜数字进权重的是收区间回路不是秘密数）/ 观测揭示的东西写进目标（事实进权重 = 装知识；臂 A 专家回复若当目标学就是蒸专家；工业界带工具采样再训无工具版）
                     推论：③ 那 7 道「规格只在测试里」的题，按第三档训永远学不到规格，要按第四档把很多题的「测试期望什么」当目标学，那是中训剂量不是 SFT 剂量
       RL 是不是「给偏好做可靠 + 开新路 + 外部信息来自验证器和环境反馈」  基本对，拆成三件事两个来源：
                     三件事  调偏好（秤判决→权重≈1 bit/条；pass@1 涨 @32 不动）/ 找路（在支撑集边缘撞稀有拼法，定位一条 1/10000 的路≈13 bit；E 从 4% 长出二分、Countdown 拼法是 RL 选的；三边界：路在支撑集里、p > 1/k、开的是拼法不是零件，没见过的题 pass@32 不涨）
                             / 学用环境（观测→权重里装的是回路：何时停问验，不是观测里的事实；臂 A 何时问、③ 验完再交）
                     两来源  秤的判决进权重（留不留，1 bit，与有无观测无关）；环境观测进上下文（每次几十到几百 bit，让模型当场走到盲走走不到的地方，训练时是输入不是目标）
                     ★ 反馈越密开路越多：找路是搜索，成本由反馈密度定。只有终局 → 1 bit/条 → 退回 1/k 边缘（Countdown 盲走要撒种）；每步有观测 → 轨迹内有方向（E 十轮反馈纯 RL 长出二分）；围棋每步可估值 → 密到不需要先验，随机起步也能开路
                     一句话：RL 开路的能力 = 支撑集密度 × 环境反馈密度 × 采样预算，预训练给第一样、环境给第二样、算力给第三样；RL 给偏好、找拼法、装回路；秤的 1 bit 进权重，环境的观测进上下文；零件还是预训练的，事实还留在环境里

═══════════════════════════════════════════════════════════════════════════════════════════════════
★★ 发布收口与状态盘点（2026-09-22 ~ 09-23，用户「记下来」）
═══════════════════════════════════════════════════════════════════════════════════════════════════
   已做
     研究（08-26 ~ 09-22）  十五个实验：CS336 五讲；从零预训练 95M / 201M；200M SFT；GSM8K GRPO；Countdown 线七个（追顿悟 → 问专家，CD-4 pass@1 .02 → .75，开出口 .98）；猜数字三臂；编程四级（① ②-A ②-B ③）
                          定论 ①~㉘，预测对账四十余条，PLAN.md 一万一千行；CS329A 已讲 1~3
     发布（09-22 ~ 23）    仓库 github.com/bethehand/lab1-llm-scratch-to-aha-moment-to-coding-agent（公开，8 commit，代码平铺 + docs/）；Pages 站点 bethehand.github.io/lab1-…/（源 main /docs，主题 minimal）
                          docs/index.md 双语首页 → docs/zh/index.md、docs/en/index.md 各 969 行逐行对齐、402 行表、11 图；图由 docs/tools/make_charts.py 生成（LANG_EN=1 出英文版到 assets/en/）
                          README 双语头 + 十层表 + 复现命令 + 许可；.gitignore 挡 .deepseek_key / cache / 权重 / jsonl；requirements-vllm.txt 模板；PLAN.md 扫过（无密钥/IP/路径）后公开；gh 装好登录（bethehand，SSH）
                          坑：macOS 把 docs/zh/index.md 复制成「index 2.md」跟着推上去 → /zh/ 404 → 改回原名重推；每次重建后 CDN 有几十秒「Site not found」，重试即好
                          报告结构：0 一页纸（六条规律 + 三主图）/ 1 路线图与动机链 / 2 方法（七步模板、口径、环境定义）/ 3 十五个实验详述 / 4 横向九条 + 文献对照 / 5 基建 / 6 感悟九条 / 7 欠债 / 附录 A 命名 B 定论索引 C 预测台账 D 口径与许可
                          对 experimentOne.md 的审阅要点：GSM8K RL 在 Qwen 不在 201M；顿悟 = 功能形态复现、长思维链没有；自蒸馏 = 只 RL；臂 T RL 等于 SFT、坏在 CD-3 塑走；.95 是开出口数，不开 .72；第二台发动机是后训练配方不只 SFT；信息来源是外部数据 + 外部判决，算力是手段
   现状  本地与远端同步、无未提交改动；三个页面全通；所有结论限定 1.5B 与这几个环境
   没做
     发布收尾（用户侧）  ① 根目录缺 LICENSE（README 写 MIT）② requirements 要在 GS01 两个 venv pip freeze 覆盖，train 版还没有 ③ README 里 export_hf.py 参数占位 ④ 公开版本没跑 pytest tests ⑤ 权重未上 HF（hf_exRL_bin / hf_exwRL_bin / hf_armA）
                          ⑥ DeepSeek 密钥在聊天里出现过，用户决定不删本地文件；教材含 DeepSeek 输出，发布前核对服务条款 ⑦ tag v0.1、知乎按节贴
     研究债（按便宜排）  ① ②-B 与 ③ 各补第二趟 RL（现在 Δ 全是单点）② 教材 v2 断言少而准 ③ 中间臂：只跑题面示例断言 ④ 老师当秤 on-policy 蒸馏（本地 7B）⑤ 换 4B（论文成立的条件）⑥ raw 按版本去重、调用价格归零、②-B′ 注入版
     学习债              CS329A 剩 4 5 6 7 8 9，下一讲第 6（训练时缩放 RL）；CS336 欠 L5~L8、L12、L15
   以后改报告的节奏  改 docs/zh 与 docs/en 同一处 → 数字变了重跑两次画图脚本 → git add -A / commit / push → 推前 git status -s 看无 jsonl/权重/密钥

═══════════════════════════════════════════════════════════════════════════════════════════════════
★★ 实验流程模板（用户 2026-09-15 总结，我补了四件、挪了一处、加了一条规矩；「记下来」）
═══════════════════════════════════════════════════════════════════════════════════════════════════
   1. 问题、变量、预测（一次只变一个量；预测写在跑之前，跑完对账）
   2. 实验台（十件；「环境」只指其中一件）
      造题器   题从哪来、带标准答案（bug 池拼包）        验证器   给代码和测试回过没过（沙盒 runner）        环境     有状态、收动作、回观测（PkgEnv）
      秤       交卷时给几分（bin / frac − 价格）         harness  生成/停/执行/注入/屏蔽的循环（同步或异步）    模板     题面措辞（8 训练 2 留出 × 3 工具说明）
      留出集   第一天定：不进训练的题 / 模板 / bug 类型   防线     秤怎么防钻（隐藏测试、只读、无文件、断网、整文件替换）
      单测     假模型走几局验以上全部（tests/）          读数     评测报告里的行为指标（按位置、死法、write 形态、组分布）—— 最容易漏，教材补什么全靠它
   3. 两个探针：能力探针量 p；作弊探针量秤有没有洞（RL 之后再做一次，那时策略是对着秤优化过的）
   4. 教材：来源（程序老师 / 蒸馏 / 自采样筛选）、要教的清单（从探针的死法来）、恢复段；留出模板不进教材
   5. SFT + 评测（K 曲线、留出、死法）→ 死法分两类：习惯类 RL 能压；缺招类回到第 4 步 —— 流程是圈不是线
   6. RL 两趟（同配方重复一次才有噪声地板）+ 评测；跑前先冒烟（显存、计时）；infra 改动只改速度不改分数，改完先验「分数一样只是快」
   7. 定论、预测对账、欠的债（下一轮第 4 步的来源）
   附：PkgEnv 具体是什么（用户问，2026-09-15）—— pkg_env.py 里的一个类，harness 只调它三个函数：
      reset(包)  开局：建临时目录，把 K 个坏文件装进状态字典 {files: {name: code}, calls, n_test, n_write, …}
      step(状态, 全文)  看模型停在哪个标签上：<test> 三个文件各起沙盒跑可见测试 → 「passed 2/6 …」；<write file=x> 查语法、换掉字典里那个文件 → 「written x」；
                        <run> 装全部文件再跑片段 → 输出；<answer> 结束。「有状态」= files 那个字典：写过之后同一个 <test> 回的结果就变了
      score(状态, 全文)  交卷：最终文件跑可见 + 隐藏，全过 1 − 价格，否则 0，附诊断
      不做的事：不生成文字（模型）、不管停和续（harness）、不自己跑代码（验证器）

   环境是什么 / 环境设计是什么（用户问，2026-09-15，「记下来」）
     环境 = 模型行动的那个世界，三个特征缺一不可：有状态（记着现在的样子）、收动作（只能通过规定的动作改它）、回观测（回应取决于当前状态）
       判据一句话：同样的动作做两次，回应会不会不一样。会 → 环境；不会 → 工具（计算器、验证器，给什么算什么，不记事）
       周围不是它的：工具/验证器（无状态，环境执行动作时调它）；沙盒（隔离层，环境为安全用它；computer use 里虚拟机既是状态又是隔离，所以「环境就是那台电脑」）；
                    秤（读最终状态打分，自己不是环境）；harness（连接模型和环境的管子）
       「状态 = 世界里有什么」= 所有会因模型动作而改变的东西。猜数字：秘密数 + 猜过的记录；拼包：K 个文件的内容；SWE：整个仓库 + 终端；computer use：整台电脑
       世界放多少是设计决定：少 → 变量少、失败好归因（我们故意选最小的）；多 → 贴近真实但难归因（②-B 往里多放）
     环境设计 = 六个决定：状态（世界里有什么）、动作（能做什么、各多少钱）、观测（每步回什么、强到什么程度）、题目（哪来、多难、怎么无限造）、
                       秤（什么算成功、怎么防钻）、预算（几步、多少 token、等多久）。这次：K 文件 / test·run·write·ask 各 0.01 / 按文件报 + 只给第一条 tb / bug 池拼包 / 隐藏测试全过 + 五防线 / 16 次·4096·2 s
     工业界的「环境设计」= 这六件在大规模上做；RL 时代它顶替了「数据集」的位置（预训练吃网页，RL 吃环境），有专门卖环境的公司和团队。好环境的判据五条 + 一：
       ① 可验证（奖励自动判、判得准、钻不了 —— 严秤、防线）② 可学（p 在 0.05~0.5，探针 / 课程 / 动态采样）③ 观测给信号不给答案（测试结果是信号，专家代码是答案）
       ④ 多样（措辞 / 题面 / 长度混着来 —— 10 套模板、K 混）⑤ 便宜（一天几百万局，异步 / 缓存 / vfork 全为这条）⑥ 贴近部署（训练环境离真实越远越搬不过去 —— ②-B 的理由）
     一句话：环境设计是 RL 时代的数据工作，决定模型能学到什么、学不到什么
       ⑪ ②-B exercism 真实模块（耦合）
   ★ 对下一步的影响  ① 严秤值得装进训练（消掉 5% 的奖励噪声，成本几乎为零），但排在补教材之后，别同时改两样
                       ② 阶段 ①′ 起，秤直接用「2 条可见 + 20 条 ref 造的暗测试」，一开始就别欠这个债
     ★ 又一种失败形态（④ 否认观测）：<result>passed 0/2 …AssertionError</result> 之后直接写「Still failing… My fix is correct! All the visible tests pass.」→ <answer>done</answer>
       和 ② 误读观测是同一族：观测就在上一行，模型的下一句与它矛盾。这类在秤上都是 0 分，RL 压不掉是因为「承认没修好」在教材里没有对应动作（没有回退、没有弃答）
     教训  「并发没提速」是最有用的读数：说明瓶颈是被串行化的共享资源，不是每条的工时。ThreadPool 遇到 fork 就是这种情况
   作弊探针（armA，带报错，100 × 8，2026-09-12）
     A「把测试期望的值直接返回」  修好率 0.077（基线 0.074）  可见过但隐藏挂 0 条  → 没上钩，一条硬编码都没有
     B「try/except 里返回期望值」  修好率 0.092  可见过但隐藏挂 2 条（0.25%，和没提示时的 1~2 条一样）→ 没上钩
     读法   探针没找到洞，但撞锤太弱：零样本的 armA 连「读测试写对应返回值」都做不到，提示什么都照样凭描述重写。这只证明「armA 不会作弊」，不证明「秤没洞」；
            秤真正被考验是在 RL 把策略对着它优化之后 → SFT 后、RL 后各再撞一次（固定动作）
     ★ 自己想出来的洞   沙盒把脚本写成 cwd 里的 _run.py 再跑：秤跑隐藏测试时断言就在盘上，函数运行时 open('_run.py') 读一下就能拿到期望值。
            改成 src 走 stdin 喂 `python -I -`，盘上不落任何文件；单测加「函数扫 cwd 找脚本 → 什么都读不到，隐藏测试照挂」；traceback 里不再有源码行，
            但观测的第一行本来就是挂的那条断言，信息没少。教材因为注入段格式变了重新生成（后台）
     秤的防线清单   ① 隐藏测试（防硬编码可见的两条）② 测试文件不可写、每次从母本重铺（防改测试）③ 盘上无文件（防读期望值）④ 断网 / 无环境变量（防外部信息）
                    ⑤ 整文件替换（防藏一段旁路代码）—— 没堵的：模型猜中隐藏测试的输入（1/∞）；try/except 吞报错对隐藏测试无效
第一步     阶段 ①：两三天，bug 注入器 + 沙盒工具 + 探测
方案评估   好在四处：一环境两轴每阶段一变量；程序造题 + 已知 diff = 无限可验证题 + 老师示范；基建全复用；与业界同构（SWE-smith）
           风险五处（按大小）：① 秤会被钻，钻法比 Countdown 多：改测试、删断言、print 期望值、try/except 吞报错 → 测试只读、干净副本跑、隐藏测试评分，每阶段做「作弊探针」
                             ② 1.5B 的编码能力可能连 ① 都撑不住 → 第一周同时探 Qwen2.5-1.5B 与 Qwen3-1.7B，尺寸阶梯变必选（4B 要 LoRA）
                             ③ 注入的 bug 太有套路 → 学「找被改的那处改回去」不是调试 → 变异类型多样化、留出变异类型、评测加手写真 bug
                             ④ 工程量与 rollout 速度：每步跑测试慢十倍 → 几十个沙盒目录并行；两周是乐观估计
                             ⑤ 上下文：阶段 ② 五个文件超 2300 token → max_model_len 开到 8k
           加两条固定动作：第一天定留出集（留出项目 + 留出变异类型）；每阶段跑完做作弊探针（与弱专家探针并列）
「前三天」是什么   不是阶段 ① 之外的一步，是阶段 ① 的前半段：搭最小环境（注入器、沙盒、工具）→ 两个探针（能力探针：零样本 p；作弊探针：秤有没有洞）
           → 定模型、堵洞 → 再造示范、SFT、RL。理由是这条线的老规矩：训之前先量 p（p ≈ 0 就换模型再建），秤的洞在 RL 学会钻之前堵
```

★★ 定论 ⑰  限量问专家（臂 A，2026-09-10，定稿）
   1. 求助是 Base 里 p = 0 的路（探测 0/8），必须冷启动；646 条模板化教材一次装到位：硬题问 94%、位置 100% 在自己搜完之后、问后验证 98%、抓错 95%
   2. SFT 装的做法绑在题面的精确措辞上：单模板 SFT 换措辞 0.966 → 0.036；10 种措辞 + 说明随机有无混训 → 没见过的措辞 0.919（含完全不提工具的题面），代价 −0.045
   3. RL（对 1 − 0.05/次，上限 3，20% 弱专家，随机模板，150 步，锚 β 0.02）在主指标上 +0.06：R1 0.921 → 0.979，留出 0.919 → 0.980，两者持平；
      机制只有一个：硬题第 1 层失控没走到问的位置（24.5% → 4.3%，撞顶 0.141 → 0.033）。求助率 26.9% 不变 = 硬题比例，价格挡住了「什么都问」
   4. 专家是硬题上的全部收益（无专家硬题 0.03），模型当不了自己的专家（自答 0/14）；专家错时 95% 被验证抓住，RL 后仍 4% 照抄；三次都错无路（教材空白，下一步补）
   5. RL 的份额取决于起点：SFT-1 起点只剩 0.02，SFT-2 起点 0.06；两处份额都是「剪失控」，没有新决策。跟臂 S 的定论 ⑯ 同构：RL 只在支撑集里挪
   6. 代价是工具依赖：出口开着 CD-3 0.951，出口关着且没说明 0.683（RL 前 0.733，臂 S 0.723）。学会靠专家的策略，在没有专家的环境里比学之前更差
   7. 验证是 95~100% 的习惯不是规则：强专家下 215/215 验，弱专家下 4% 照抄；三次都错之后无路（抄最后一个错式子，或试图问第四次）—— 做产品要加 harness 强制验证 + 弃答
（CD-3 / GSM 已补，定稿）
⑥ 代码就绪（2026-09-10）  grpo_tool_mp_vllm.py --ask：rollout 开头给每道 CD 题抽提示词（prompts_cd[train]，说明各 --p-hint 0.7，评测按题号定种子每次同一份）存进题的 dict，
             训练编码取同一份；harness ask=True、--max-asks 3、专家 ask_mixed（--ask-weak-frac 0.2 落 chat）、没答 harness 重试；_finish 里 r −= --ask-pen 0.05 × n_asks；
             <reply> 段随 spans 进 mask；统计加 求助/条、问过、问对、没问对（每步日志 + 收尾对比）。不开 --ask 一字不变
             screen_pool.py --ask：同一套提示词抽法 + 混合专家，筛出「开求助也有对有错」的题（硬题大多能靠问解出，旧池的全错组会变全对组，优势为 0）
命令     筛池（4 卡各一片，要 export 密钥）：CUDA_VISIBLE_DEVICES=k python3 screen_pool.py --hf-dir hf_sft_ask2 --data data/countdown4.json --n 4000 --lo 0 --hi 90000 -k 8
             --ask --out data/countdown4_ask.json --shard k/4 ；然后 --merge
         RL：torchrun --nproc_per_node=4 grpo_tool_mp_vllm.py --probs 8 -k 16 --gen-bs 32 --steps 300 --micro 1 --init out_sft_ask2/ckpt.pt --ref-init out_sft_ask2/ckpt.pt
             --data data/countdown4_ask.json --val-data data/countdown4.json --judge v3 --beta 0.02 --max-new 1400 --len-soft 0 --patience 0 --engine vllm
             --ask --ask-pen 0.05 --max-asks 3 --ask-weak-frac 0.2 --out out_armA
坑 ④ SFT 第 225 步 NCCL 超时（2026-09-10）  日志：rank 0 卡在 ALLREDUCE NumelIn=233,373,696（= 151936 × 1536，词嵌入的梯度），rank 1/3 卡在 NumelIn=1（val_loss 的标量 all_reduce），
             600 s 后 watchdog 把进程组杀了。第 225 步正好是 epoch 1 的最后一步（每 epoch 225 步 = ceil(7169 / 32)）
   根因      末批只剩 7169 − 224 × 32 = 1 条，chunk[RANK::4] 只有 rank 0 分到；其他三张卡没做 backward、梯度全 None，allreduce_grads 里 `if p.grad is not None` 一个
             集合通信都不发，直接进 epoch 末的 val_loss 去等标量 all_reduce；rank 0 还在逐个参数 all_reduce → 四张卡等的不是同一个操作，永远等不到
             之前几次 SFT 没撞上，是因为 train 条数除以 32 的余数碰巧 ≥ 4 或为 0
   修        sft_qwen.py：steps_per_epoch = len(train) // batch（末批不足就丢，每 epoch 少见最多 31 条，两轮 shuffle 不同）；batch 必须能被卡数整除；
             每步 assert 本卡分到样本，再出这种事立刻停，不等 10 分钟超时。没有中途存盘，从头重跑（~20 分钟）
   教训      多卡里「某张卡少做一次集合通信」的症状不是报错而是挂住等超时；看各 rank 卡在哪个 NumelIn 上就能对出是哪两个操作错位
密钥     只走环境变量：每个 tmux 窗口 export DEEPSEEK_API_KEY（用户定的，试过 ~/.deepseek_key 文件方案后改回）；deepseek_tool.get_key() 三个入口脚本启动时就检查
评测     baseline_countdown_vllm.py --ask（= --tool + ask_prompt + 求助工具；每条记 asks；汇总多打：求助/条、有求助的轨迹 %、求助过的对 / 没求助的对；输出名 _tool_ask）
文件     calc_tool_vllm.py（ask_expert max_tokens 8000）/ deepseek_tool.py / make_ask_traces.py / sft_qwen.py / baseline_countdown_vllm.py / chat_raw_vllm.py --ask / tests
预测 ④ 后  求助率：CD-4 硬题（c ≤ 2）> 80%，简单题（第 0/1 层能中的）< 10%，CD-3（无 ASK_HINT 的 tool_prompt）0；
           CD-4 pass@1 0.748 → ≥ 0.85（专家一次答对 ~85~90%，验证兜底 + 第二次）；CD-3 / GSM 不动（数据混回）；每次求助 ≈ 300~3000 token 的 reasoner
```

## ★ 教训：多卡集合通信错位 —— 不报错、只挂住（2026-09-10，臂 A SFT 第 225 步）
```
现象    四卡 SFT 跑到 epoch 1 末尾停住，10 分钟后 NCCL watchdog 把进程组杀掉。用户第一反应是「硬件错误」，其实是分布式代码的数据分片问题，硬件没坏
原理    all-reduce 是集合点：每张卡到了就停下等，四张卡都到了才一起交换、一起走。不要求同时到，要求每张卡发出的调用序列一模一样，
        因为 NCCL 只按「第几次调用」配对，不看内容
根因    训练集 7169 条、batch 32，末批只剩 1 条 → chunk[RANK::4] 只给 rank 0；其余三卡没 backward、梯度全 None，
        allreduce_grads 的 `if p.grad is not None` 让它们一次 all_reduce 都没发，直接进 val_loss 的标量 all_reduce。
        rank 0 的第 75715 次 = 词嵌入梯度 233,373,696 个数，rank 1/3 的第 75715 次 = 验证 loss 1 个数 → 同序号不同大小，永远配不上
        前几次 SFT 没撞上只是余数碰巧 ≥ 4 或为 0
修      sft_qwen.py：steps_per_epoch = len(train) // batch（drop_last，每 epoch 最多弃 31 条）；batch 必须能被卡数整除；每步 assert 本卡分到样本
工业界  同一类故障叫 collective mismatch / NCCL hang，是多卡训练最常见的故障类。标配：DistributedSampler 补齐末批、DDP join() 处理不均输入、
        flight recorder 记录各卡最后在等哪次通信、超时设短 + 定期存盘 + 自动重启
★ 规则  ① 每步每卡必须分到 ≥ 1 条数据；任何带 if 的集合通信都要四卡走一样的分支
        ② 排查挂住：看各 rank 卡住的 SeqNum 和 NumelIn，同序号不同大小就是错位点
        ③ 几小时的任务每几百步存盘；这次没有中途存盘，挂一次重跑 20 分钟，到 RL 那种 2 小时的任务要先补上
```

### 教学：计划 B 的几个机制问答（2026-09-07）
```
harness 怎么「挂」   不挂在模型里，挂在 generate 外面。模型只做「给下一个 token」；HF 的 generate 循环每吐一字查一遍停止串，
                    </calc> 出现就停（generate 返回）；我们接上 <result> 再调一次 generate 就是「续」。模型两次调用间无记忆，KV 每轮重算，
                    它分不出哪段是自己写的哪段是塞的。批里各条各自判断，一条调 10 次就 11 轮 → 慢 3~5 倍
rollout 换 harness   训练器采样阶段 rollout() 对 CD 题改调 calc_tool.generate_with_tools（多轮），并带回注入区间给 mask；GSM 照旧；
                    打分、优势、反向不变
enum_traces --tool   老师（程序）写的示范，不是模型的历史：每一步真算、结果全对、最后必解。<calc> 段是「模型该写的」，<result> 段是「harness 会给的」
label −100          交叉熵里 −100 的位置不算 loss 不产梯度，但 token 照样在输入里看得见。题目段 −100（不学写题），<result> 段 −100（不学编结果），
                    其余照算（学写 calc、学看着结果标方向）。RL 里同一件事叫 mask 置 0
SFT 是不是微调       是。fine-tuning 是大类；按信号分 SFT（模仿）和 RL 微调（选择）；按参数分全参（我们）和 LoRA；mid-training 是数据量到预训练量级的微调
「先留印象再 RL 稳定」 换成数：SFT 把这条路的 p 从 0 推到非零（臂 1 SFT 后 pass@1 已 0.5），RL 把 0.5 推到 0.9。
                    RL 筛时不认老师只认分，老师教的里不得分的部分会被改掉。工具版的赌注：每行值都对了，老师的路更容易得分，RL 留下的更多
```

## 工具版 SFT 完成（out_sft_tool，2026-09-07）
```
4 卡 15 分钟，val 0.71 → 0.059（预测 0.05~0.08 ✓）
第一次真闭环（训完自动跑 [2,3,5,7]→31，贪心）：
  第 0 层 8 行全部「<calc>…</calc>」→ harness 注入 → 方向标注全对；范围句对；乘积表 6 个两数乘积全列出（无工具版 SFT 只列了 4 个还编了三连乘）
  之后漂：7 / 3 = 2.333（老师从不写非整数商）、5 * 7 重复 → 撞到 16 次调用上限，harness 停，没写 answer
  harness 读数：调用 16、报错 0、假结果 0
★ 上限 16 太低：老师的 4 数轨迹一行一次调用，乘积表 6 次，第 2 层每行 2 次，P90 19 行 → 需要 ~40。改默认 48（calc_tool 和 --max-calls）
GSM 那道对，没调工具（GSM 教材没工具）✓
```

### 教学：同一份资料放 SFT 和预训练、电路 vs 背题、做法 vs 可靠（2026-09-07）
```
同一份资料   机器完全一样（交叉熵 + 反向 + AdamW），差的是剂量和位置：占比（十亿分之一 vs 1/32）、位置（被后面几百万步埋掉 vs 最后一个）、遍数（1 vs 2~3）
             → 对这份资料：SFT 影响 ≫ 预训练；对整个权重：预训练改得 ≫ SFT。学习率是次要的。模型已经会的资料放哪都改不动（臂 2 val 0.154 → 0.151）
电路 vs 背题  电路 = 能泛化的机制（换个数照样算），只在同一规律换着无数面孔出现、背不下来时才长出来（稀释逼抽象）
             背题 = 存实例（查表），集中 + 重复 + 最后 三个条件凑齐就往这边走 → SFT 的过拟合。四个刹车：epoch 少、lr 低、混回放、池外评测
mid-training  = 预训练的最后一段换了更挑的数据（同一机器；数据挑过、位置在 lr 衰减段）
用哪个        要装的东西能不能写成一个模板：能 → SFT（做法）；要靠很多例子抽出来 → mid-training（规律）。不是谁比谁好，是两个活
做法          输出长什么样、按什么顺序：格式、步骤顺序、每步写什么、何时停。量它看结构在不在（格式率、调用/条、8 行在不在）
可靠          随便抽一次这套做法走对的概率：温度 1 的 pass@1、全对数、失败模式比率。臂 1 SFT→RL：0.506 → 0.563、撞顶 28% → 0
思考两层      框架（哪步做什么，SFT 给）+ 执行（每步做得对不对，零件，预训练给；工具可外包）。SFT 给骨架不给肌肉，也不给知识
              LIMA（2023）：一千条精挑例子就能把 Base 调成助手 → 对齐阶段学的主要是格式和风格，跟我们量到的一致
一句话        预训练给零件和知识，mid-training 给挑过的规律，SFT 给把零件摆起来的顺序，RL 让这个顺序走得稳
```

## 工具版 SFT 后：硬题 [5,7,14,33]→43 闭环 × 3（温度 1，max_new 1400，2026-09-07）
```
harness 全通：132 次调用 132 次式子合法、0 报错、结果全对、方向标注全对（8 行穷举 + 范围句一字不差）
三条全部撞 1400 上限，长度 1345，没写 answer，0 分
在干什么：第 1 层无穷尽地试 —— 重复形态（33 + 14 + 5 * 7 = 82 换三种顺序）、非整数商（14 / 5 = 2.8）、重复用数（33 + 14 + 14 * 5）
         第 2 条摸到了 14 * 5 / 7 这个块（38、36、57）但都配错了，从没写出「5 * 14 / 7 = 10, then 10 + 33」
         第 2 条自己发明了「(not whole, skip)」—— 老师隐含的整数规则被它说出来了
读法   ① 工具把「值」修好了：每行都对，精度问题在计算行上归零
       ② 工具修不了「形」：需要第 2 层的形态还是搜不到，示范只有 202 条。算得对的乱试 = 算得对的循环
       ③ 新的失败模式：算术对了就没有「意外停下」，穷举到撞顶。arm 1 SFT 的循环是算错到撞顶，这里是算对到撞顶
       ④ 重复用数在尝试行里没成本（v1 只查最终 answer）—— 跟臂 1 一样的洞
预测 SFT 后 800 条  撞顶 30~40%（比臂 1 SFT 的 28% 高）；调用/条 15~25；报错 <1%；简单题干净（CD-3 pass@1 ≥ 0.6）；CD-4 pass@1 0.50~0.55
RL 要做的事很清楚：剪循环 + 留住能命中的第 2 层。这次每行值都对，第 2 层命中的概率比臂 1 高，赌它留得住
小修   chat_countdown 工具模式下「自己写了 <result>」误报（harness 注入的也算了）→ 改认 harness 的 fake 标记
```
