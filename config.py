"""
从零训练英文 LLM —— 模型与训练配置
单一事实来源：所有脚本都从这里读配置，不要在别处硬编码数字。

═══ 当前：第二轮（12 小时预算）═══
═══ 第一轮的配置备份在 config_run1.py.bak ═══

           第一轮(已完成)      第二轮(本次)
参数        95,233,536        201,035,520     2.11×
C                 768                896
n_layer             8                 16      2×
d_ff             2048               2432
词表占比        40.5%              22.4%      模型变大摊薄的，词表没换
D                1.5B                10B      6.7×
D/N              15.8               49.7      超标 2.5 倍
时长           48 分钟            约 11 小时
实测 loss      3.4068          预测 ~2.77
困惑度          30.17            预测 ~16

★ 词表保持 50257 不变 —— 换词表会让 loss 跟第一轮不可比
"""
from dataclasses import dataclass, asdict


@dataclass
class Config:
    # ══════════ 模型 ══════════
    # 第 1 步定：GPT-2 byte-level BPE
    vocab_size: int = 50257
    # 第二轮：N≈200M 预算 → 减词表 45.0M → 剩 155M → 除以 12C² → 16 层
    #   注意 C 必须能被 head_dim 整除！896 = 64×14 = 128×7  ✅
    #   （900 不行：900/64=14.06, 900/128=7.03）
    C:          int = 896      # hidden_size / d_model / n_embd   (第一轮 768)
    n_layer:    int = 16       #                                  (第一轮 8)
    n_head:     int = 14       # = C / head_dim = 896/64          (第一轮 12)
    head_dim:   int = 64       # L3：固定 64 或 128
    d_ff:       int = 2432     # = 8×896/3 = 2389，圆整到 128 的倍数 (第一轮 2048)
    block_size: int = 1024     # T：上下文长度

    # ── 架构：L3 的五个收敛点，照抄 ──
    norm_type:      str  = "rmsnorm"   # 不是 LayerNorm
    norm_position:  str  = "pre"       # 残差通路上不放任何操作
    activation:     str  = "swiglu"    # 不是 GELU
    pos_encoding:   str  = "rope"      # 不是位置查表
    rope_theta:     float = 10000.0
    bias:           bool = False       # 全模型无 bias
    tie_embeddings: bool = True        # 出口复用词典表，省 38.6M 参数

    # ══════════ 批次 ══════════
    batch_size_per_gpu: int  = 16      # 受 logits 张量显存限制
    grad_accum_steps:   int  = 8
    world_size:         int  = 4       # 4 × RTX 4090
    grad_checkpointing: bool = False   # 显存够，不开，省 33% 算力

    # ══════════ 优化 ══════════
    lr:           float = 5e-4         # L3 表：100M 用 6e-4，1.5B 用 3e-4，200M 插值
    lr_min:       float = 5e-5         # 峰值的 10%
    beta1:        float = 0.9
    beta2:        float = 0.95         # 注意：不是 PyTorch 默认的 0.999
    weight_decay: float = 0.1
    grad_clip:    float = 1.0
    warmup_steps: int   = 500          # 总步数 19,073 的 2.6%
    schedule:     str   = "cosine"

    # ══════════ 数据 ══════════
    # 第二轮：12 小时预算 → 10B token，D/N = 49.7（超标 2.5 倍，为部署优化）
    #   独立 token 只有 5.3B（7 个 parquet），所以是 1.9 个 epoch —— 在 4 遍法则内
    train_tokens: int = 10_000_000_000
    val_tokens:   int =     5_000_000
    data_dir:     str = "data/fineweb_edu"

    # ══════════ 系统 ══════════
    dtype:   str  = "bfloat16"
    compile: bool = True
    seed:    int  = 1337

    # ══════════ 硬件（第 0 步实测）══════════
    peak_tflops_per_gpu: float = 165.0  # RTX 4090 BF16 稠密
    # ★ 实测值（每轮开跑前用 5b/5c 校准）
    #   第一轮 95M：  单卡 0.672  四卡 0.547（fp32 梯度，通信占 18.6%）
    #   第二轮 201M： 单卡 0.658  四卡 0.594（bf16 梯度压缩，通信降到 9.4%）
    #                 每步 1.92 秒，显存 17.3 GB(PyTorch) / 约 19 GB(nvidia-smi)
    assumed_mfu:         float = 0.594  # 第二轮四卡实测

    # ── 派生量 ──
    @property
    def tokens_per_step(self) -> int:
        return (self.batch_size_per_gpu * self.block_size
                * self.grad_accum_steps * self.world_size)

    @property
    def max_steps(self) -> int:
        return self.train_tokens // self.tokens_per_step

    def check(self):
        assert self.C % self.head_dim == 0, "C 必须能被 head_dim 整除"
        assert self.n_head == self.C // self.head_dim, \
            f"n_head 应为 {self.C // self.head_dim}"
        assert self.vocab_size < 65536, "vocab 超出 uint16，.bin 存储要改 int32"
        assert 30 <= self.C / self.n_layer <= 200, \
            f"深宽比 {self.C/self.n_layer:.0f} 超出合理区间"
        return True


cfg = Config()

if __name__ == "__main__":
    cfg.check()
    for k, v in asdict(cfg).items():
        print(f"{k:22s} = {v}")
    print(f"{'tokens_per_step':22s} = {cfg.tokens_per_step:,}")
    print(f"{'max_steps':22s} = {cfg.max_steps:,}")
