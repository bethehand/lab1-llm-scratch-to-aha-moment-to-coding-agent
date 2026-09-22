"""
第 2 步的验收工具：从 config 算出参数量、显存、算力、时长。
改了 config.py 就重跑这个，不用手算。
"""
import math
from config import cfg

GB = 1024 ** 3


def params(c):
    """按矩阵逐个数格子。"""
    attn_per_layer = 4 * c.C * c.C                # wq wk wv wo
    mlp_per_layer  = 3 * c.C * c.d_ff             # SwiGLU: gate up down
    norm_per_layer = 2 * c.C                      # 两个 RMSNorm 的可学习增益
    per_layer = attn_per_layer + mlp_per_layer + norm_per_layer

    embed = c.vocab_size * c.C                    # 绑权重，lm_head 复用
    final_norm = c.C
    total = per_layer * c.n_layer + embed + final_norm
    return dict(attn=attn_per_layer, mlp=mlp_per_layer, norm=norm_per_layer,
                per_layer=per_layer, layers=per_layer * c.n_layer,
                embed=embed, final_norm=final_norm, total=total)


def memory(c, n_params):
    """单卡显存，四块。"""
    # ① 参数 + 梯度 + Adam(m,v)：混合精度和纯 fp32 都是 16 字节/参数
    states = n_params * 16

    # ② 激活值：每层每 token 约 k 个 C 宽的中间量
    k = 1 if c.grad_checkpointing else 16
    acts = k * c.batch_size_per_gpu * c.block_size * c.C * c.n_layer * 2

    # ③ logits + 其梯度：B × T × vocab，这是大词表的代价
    logits = c.batch_size_per_gpu * c.block_size * c.vocab_size * 2 * 2

    # ④ CUDA context / NCCL buffer / 碎片
    overhead = 1.5 * GB
    return dict(states=states, acts=acts, logits=logits,
                overhead=overhead, total=states + acts + logits + overhead)


def compute(c, n_params):
    attn_frac = c.block_size / (6 * c.C)          # L2：6ND 不含的那一项
    flops_6nd = 6 * n_params * c.train_tokens
    flops_all = flops_6nd * (1 + attn_frac)
    eff = c.peak_tflops_per_gpu * 1e12 * c.assumed_mfu * c.world_size
    return dict(attn_frac=attn_frac, flops_6nd=flops_6nd,
                flops_all=flops_all, eff_flops=eff, seconds=flops_all / eff)


def main():
    c = cfg
    c.check()
    p, m, f = params(c), None, None
    m = memory(c, p["total"])
    f = compute(c, p["total"])

    print("=" * 62)
    print(f"  配置  C={c.C}  n_layer={c.n_layer}  n_head={c.n_head}  "
          f"d_ff={c.d_ff}  T={c.block_size}  vocab={c.vocab_size}")
    print("=" * 62)

    print("\n【参数量】")
    print(f"  每层 attention (4C²)      {p['attn']:>14,}")
    print(f"  每层 MLP (3·C·d_ff)       {p['mlp']:>14,}")
    print(f"  每层 RMSNorm              {p['norm']:>14,}")
    print(f"  每层小计                  {p['per_layer']:>14,}")
    print(f"  × {c.n_layer} 层                    {p['layers']:>14,}   {100*p['layers']/p['total']:5.1f}%")
    print(f"  词表 (vocab × C, 绑权重)  {p['embed']:>14,}   {100*p['embed']/p['total']:5.1f}%  ⚠️")
    print(f"  {'-'*54}")
    print(f"  总计                      {p['total']:>14,}   = {p['total']/1e6:.1f}M")

    print("\n【单卡显存】")
    print(f"  ① 参数+梯度+Adam (16 B/参数)   {m['states']/GB:7.2f} GB")
    print(f"  ② 激活值 (梯度检查点={c.grad_checkpointing})    {m['acts']/GB:7.2f} GB")
    print(f"  ③ logits + 梯度 (B·T·vocab)    {m['logits']/GB:7.2f} GB  ← 大词表的代价")
    print(f"  ④ CUDA context / 碎片          {m['overhead']/GB:7.2f} GB")
    print(f"  {'-'*46}")
    print(f"  合计                           {m['total']/GB:7.2f} GB / 24.00 GB "
          f"({100*m['total']/(24*GB):.0f}%)  {'✅' if m['total'] < 20*GB else '⚠️'}")

    print("\n【算力与时长】")
    print(f"  6ND                       {f['flops_6nd']:.3e} FLOPs")
    print(f"  attention 额外 T/(6C)     +{100*f['attn_frac']:.1f}%")
    print(f"  真实总量                  {f['flops_all']:.3e} FLOPs")
    print(f"  有效算力 ({c.world_size}卡 × MFU {c.assumed_mfu})  {f['eff_flops']/1e12:.0f} TFLOPS")
    print(f"  预计时长                  {f['seconds']/3600:.2f} 小时")

    print("\n【训练规模】")
    print(f"  每步 token   {c.tokens_per_step:,}  "
          f"({c.world_size}卡 × {c.batch_size_per_gpu} × {c.block_size} × {c.grad_accum_steps})")
    print(f"  总步数       {c.max_steps:,}")
    print(f"  warmup       {c.warmup_steps} ({100*c.warmup_steps/c.max_steps:.1f}%)")
    print(f"  每步耗时     {f['seconds']/c.max_steps:.2f} 秒")
    print(f"  D / N        {c.train_tokens/p['total']:.1f}   (Chinchilla 建议 20)")

    print("\n【跑起来后对照这几个数】")
    print(f"  ★ 初始 loss   ln({c.vocab_size}) = {math.log(c.vocab_size):.3f}")
    print(f"  ★ 显存占用    约 {m['total']/GB:.1f} GB")
    print(f"  ★ 每步耗时    约 {f['seconds']/c.max_steps:.2f} 秒")
    print(f"  ★ MFU         实测后回填 config.assumed_mfu，重跑本脚本")
    print()


if __name__ == "__main__":
    main()
