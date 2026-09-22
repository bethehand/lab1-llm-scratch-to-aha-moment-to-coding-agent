"""
模型定义 —— 把 L3「架构与超参」学的东西写成代码。

跟 GPT-2 的区别（全是 L3 讲的那五个现代收敛点）：
    LayerNorm    →  RMSNorm      不减均值，快 10~20%
    GELU         →  SwiGLU       三个矩阵，d_ff = 8C/3
    学习式位置表  →  RoPE         没有长度硬上限，以后能扩窗口
    有 bias      →  全部去掉      效果无差异，更稳
    post-norm    →  pre-norm      残差通路上不放任何操作
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════
#  RMSNorm  —— L3 收敛点 ②
# ══════════════════════════════════════════════════════════════
class RMSNorm(nn.Module):
    """
    LayerNorm:  (x − mean) / std × gain + bias     要算均值，有 bias
    RMSNorm:    x / rms(x) × gain                  不减均值，无 bias

    少一次跨维度的 reduction。norm 是访存受限的算子，所以这省的是真时间。
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))     # 只有缩放，没有偏移

    def forward(self, x):
        # 在 fp32 里算平方和：bf16 只有 7 位尾数，累加大量平方容易掉精度
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.to(dtype) * self.weight


# ══════════════════════════════════════════════════════════════
#  RoPE  —— L3 收敛点 ④
# ══════════════════════════════════════════════════════════════
def precompute_rope(head_dim: int, max_seq: int, theta: float = 10000.0):
    """
    预先算好每个位置的旋转角度的 cos / sin。
    这里没有任何可学习参数 —— 位置是「算」出来的，不是「查表」查出来的。
    这就是为什么以后调大 theta 就能把窗口从 1024 扩到 128k。
    """
    # 每两维一组，第 i 组的角频率 = theta^(-2i/head_dim)
    # 低维转得快（管局部），高维转得慢（管长程）
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(max_seq).float()
    freqs = torch.outer(pos, inv_freq)               # (T, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)          # (T, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    """
    x: (B, n_head, T, head_dim)   cos/sin: (T, head_dim)

    只作用在 q 和 k 上，不碰 v，更不碰残差流。
    旋转后 q_m · k_n 只依赖 (m − n) —— 这就是「只看相对距离」的由来。
    """
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


# ══════════════════════════════════════════════════════════════
#  Attention  —— 沿 T 轴通信
# ══════════════════════════════════════════════════════════════
class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head   = cfg.n_head
        self.head_dim = cfg.head_dim
        C = cfg.C
        # 四个 C×C 矩阵，这就是 estimate.py 里那个 4C²
        self.wq = nn.Linear(C, cfg.n_head * cfg.head_dim, bias=cfg.bias)
        self.wk = nn.Linear(C, cfg.n_head * cfg.head_dim, bias=cfg.bias)
        self.wv = nn.Linear(C, cfg.n_head * cfg.head_dim, bias=cfg.bias)
        self.wo = nn.Linear(cfg.n_head * cfg.head_dim, C, bias=cfg.bias)

    def forward(self, x, cos, sin, cache=None, pos=0):
        """
        cache: (k_past, v_past) 或 None。给了就走 KV cache 路径（推理用）。
        pos:   本次这批 token 的起始绝对位置 —— RoPE 要用它才能转对角度。
        训练时 cache=None, pos=0，行为跟以前完全一样。
        """
        B, T, C = x.shape
        # (B,T,C) → (B, n_head, T, head_dim)：把 C 切成 n_head 段
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # RoPE 用「绝对位置」，所以带 cache 解码时要从 pos 开始切
        q = apply_rope(q, cos[pos:pos+T], sin[pos:pos+T])     # 只旋转 q 和 k
        k = apply_rope(k, cos[pos:pos+T], sin[pos:pos+T])

        new_cache = None
        if cache is not None:
            if cache[0] is not None:
                k = torch.cat([cache[0], k], dim=2)           # 拼上历史的 k
                v = torch.cat([cache[1], v], dim=2)           # 和 v
            new_cache = (k, v)

        # 掩码规则：
        #   q 和 k 一样长  = prefill / 训练，一次算 T 个位置 → 必须遮未来
        #   q 比 k 短      = 带 cache 解码，新 token 本来就该看到全部历史 → 不遮
        # SDPA 会自动走 FlashAttention：不物化 T×T 矩阵，显存 O(T) 而非 O(T²)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(q.size(2) == k.size(2)))

        y = y.transpose(1, 2).contiguous().view(B, T, -1)     # 拼回 (B,T,C)
        y = self.wo(y)                                        # 让各个头互相混合
        return (y, new_cache) if cache is not None else y


# ══════════════════════════════════════════════════════════════
#  SwiGLU MLP  —— L3 收敛点 ③，沿 C 轴加工
# ══════════════════════════════════════════════════════════════
class MLP(nn.Module):
    """
    标准 FFN:  W2 · GELU(W1·x)                      2 个矩阵, d_ff = 4C  → 8C² 参数
    SwiGLU:    W_down · (SiLU(W_gate·x) ⊙ W_up·x)   3 个矩阵, d_ff = 8C/3 → 8C² 参数
                                            ↑
                                      逐元素相乘 = 门控

    3 × C × (8C/3) = 8C²，跟标准 FFN 参数量完全相等 —— 这就是 8/3 的来历。
    整段没有出现 t：每个 token 关起门来自己算。
    """
    def __init__(self, cfg):
        super().__init__()
        self.w_gate = nn.Linear(cfg.C, cfg.d_ff, bias=cfg.bias)
        self.w_up   = nn.Linear(cfg.C, cfg.d_ff, bias=cfg.bias)
        self.w_down = nn.Linear(cfg.d_ff, cfg.C, bias=cfg.bias)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ══════════════════════════════════════════════════════════════
#  Block  —— 通信 + 加工
# ══════════════════════════════════════════════════════════════
class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.C)
        self.attn  = Attention(cfg)
        self.norm2 = RMSNorm(cfg.C)
        self.mlp   = MLP(cfg)

    def forward(self, x, cos, sin, cache=None, pos=0):
        # pre-norm：norm 在分支里，残差通路 (x + ...) 上一个操作都没有
        # → 梯度能从最后一层直通第一层
        if cache is None:
            x = x + self.attn(self.norm1(x), cos, sin)          # 通信：跨 T 轴
            x = x + self.mlp(self.norm2(x))                     # 加工：沿 C 轴
            return x
        a, new_cache = self.attn(self.norm1(x), cos, sin, cache, pos)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x, new_cache


# ══════════════════════════════════════════════════════════════
#  GPT
# ══════════════════════════════════════════════════════════════
class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte    = nn.Embedding(cfg.vocab_size, cfg.C)      # 词典表
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.C)
        self.lm_head = nn.Linear(cfg.C, cfg.vocab_size, bias=False)

        # 权重绑定：出口复用词典表。入口「id→向量」，出口「向量→跟每行比相似度」
        if cfg.tie_embeddings:
            self.lm_head.weight = self.wte.weight               # 省 38.6M 参数

        # RoPE 的 cos/sin 是常量，注册成 buffer（不是参数，不参与训练）
        cos, sin = precompute_rope(cfg.head_dim, cfg.block_size, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # 残差分支的「出口」矩阵按 1/sqrt(2·n_layer) 缩小初始化
        # 让每个 block 一开始接近恒等映射，深层网络训练更稳（GPT-2 的做法）
        for name, p in self.named_parameters():
            if name.endswith(("wo.weight", "w_down.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    # ── 参数量：跟 estimate.py 对账用 ──
    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wte.weight.numel()
        return n

    def forward(self, idx, targets=None, kv_cache=None, pos=0):
        B, T = idx.shape
        assert pos + T <= self.cfg.block_size, \
            f"位置 {pos}+{T} 超过 block_size {self.cfg.block_size}"

        x = self.wte(idx)                                   # (B,T) → (B,T,C)
        cos, sin = self.rope_cos, self.rope_sin

        if kv_cache is None:                                # 训练路径，跟以前完全一样
            for blk in self.blocks:                         # (B,T,C) → (B,T,C)，形状不变
                x = blk(x, cos, sin)
        else:                                               # 推理路径，逐层带缓存
            for i, blk in enumerate(self.blocks):
                x, kv_cache[i] = blk(x, cos, sin, kv_cache[i], pos)
        x = self.norm_f(x)

        if targets is not None:
            # 训练：T 个位置全都要，每个位置一份 loss
            logits = self.lm_head(x)                        # (B,T,C) → (B,T,vocab)  ← 显存大户
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
            return logits, loss
        else:
            # 推理：只算最后一个位置，省掉 (T−1)/T 的 lm_head 计算和显存
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

    # ── 优化器：2 维以上的参数才加 weight decay ──
    def configure_optimizers(self, weight_decay, lr, betas, device_type="cuda"):
        params = [p for p in self.parameters() if p.requires_grad]
        decay     = [p for p in params if p.dim() >= 2]     # 矩阵：加 wd
        no_decay  = [p for p in params if p.dim() <  2]     # RMSNorm 的 gain：不加
        groups = [
            {"params": decay,    "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused = device_type == "cuda" and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        opt = torch.optim.AdamW(groups, lr=lr, betas=betas,
                                **({"fused": True} if fused else {}))
        return opt, len(decay), len(no_decay), sum(p.numel() for p in decay), \
               sum(p.numel() for p in no_decay)

    # ── MFU：实测算力利用率，用来修正第 0 步的预算 ──
    def estimate_mfu(self, tokens_processed, dt, peak_flops):
        N = self.num_params()
        flops = 6 * N * tokens_processed
        flops *= (1 + self.cfg.block_size / (6 * self.cfg.C))   # 加上 attention 的 T² 项
        return flops / dt / peak_flops

    # ── 采样 ──
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8,
                 top_k=None, top_p=0.92, repetition_penalty=1.15, use_cache=True):
        """
        三种解码控制，缺一不可：

        temperature         整体的「保守 ↔ 发散」旋钮
        top_k / top_p       截断长尾。top_k 固定取前 k 个；
                            top_p（nucleus）按累积概率截断，能自适应分布形状 —— 通常更好
        repetition_penalty  对已出现过的 token 降权。
                            专门治「重复循环」—— 模型越强越容易陷进去，
                            因为它对「继续重复已建立的模式」置信度极高
        """
        # ── KV cache：把算过的 k/v 存下来，每步只算 1 个新 token ──
        #    不用 cache：每步重算整个序列  →  O(T²)
        #    用 cache：  prefill 一次 + 每步 1 个 token  →  O(T)
        cache, pos = None, 0
        if use_cache:
            cache = [(None, None) for _ in range(self.cfg.n_layer)]
            logits, _ = self(idx, kv_cache=cache, pos=0)    # prefill：一次吃下整个 prompt
            pos = idx.size(1)

        for _ in range(max_new_tokens):
            if not use_cache:                               # 老路径：每步重算全部
                logits, _ = self(idx[:, -self.cfg.block_size:])
            logits = logits[:, -1, :].float()

            # ① 重复惩罚（CTRL 论文的做法：正 logit 除、负 logit 乘）
            if repetition_penalty and repetition_penalty != 1.0:
                for b in range(idx.size(0)):
                    seen = torch.unique(idx[b])
                    l = logits[b, seen]
                    logits[b, seen] = torch.where(l > 0, l / repetition_penalty,
                                                  l * repetition_penalty)

            logits = logits / max(temperature, 1e-5)

            # ② top-k：固定保留分数最高的 k 个
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            # ③ top-p / nucleus：保留累积概率达到 p 的最小集合
            if top_p and top_p < 1.0:
                sl, si = torch.sort(logits, descending=True, dim=-1)
                cum = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
                drop = cum > top_p
                drop[..., 1:] = drop[..., :-1].clone()      # 右移一位，保证至少留一个
                drop[..., 0] = False
                sl = sl.masked_fill(drop, -float("inf"))
                logits = torch.full_like(logits, -float("inf")).scatter_(-1, si, sl)

            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)   # 采样，不是 argmax
            idx = torch.cat((idx, nxt), dim=1)

            if use_cache:                                   # decode：只喂新的那 1 个 token
                if pos >= self.cfg.block_size:
                    break                                   # 缓存满了（超过 block_size）
                logits, _ = self(nxt, kv_cache=cache, pos=pos)
                pos += 1
        return idx


if __name__ == "__main__":
    from config import cfg
    cfg.check()

    torch.manual_seed(cfg.seed)
    model = GPT(cfg)

    print("=" * 60)
    print("参数量核对（应与 estimate.py 完全一致）")
    print("=" * 60)
    total = model.num_params()
    print(f"  实际  {total:>13,}")
    print(f"  预期      95,233,536")
    print(f"  {'✅ 完全一致' if total == 95_233_536 else '❌ 对不上！'}")
    print(f"  非词表部分 {model.num_params(non_embedding=True):>12,}")
    print(f"  词表占比   {100*model.wte.weight.numel()/total:>11.1f}%")

    print("\n各模块参数量：")
    for name, mod in [("wte 词典表", model.wte), ("单个 block", model.blocks[0]),
                      ("norm_f", model.norm_f)]:
        print(f"  {name:<12} {sum(p.numel() for p in mod.parameters()):>12,}")
    b = model.blocks[0]
    print(f"    ├ attention {sum(p.numel() for p in b.attn.parameters()):>10,}  (4C²)")
    print(f"    ├ mlp       {sum(p.numel() for p in b.mlp.parameters()):>10,}  (3·C·d_ff = 8C²)")
    print(f"    └ 2×RMSNorm {sum(p.numel() for p in b.norm1.parameters())*2:>10,}")

    print("\n" + "=" * 60)
    print("前向 + 反向冒烟测试")
    print("=" * 60)
    B, T = 2, 64
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    tgt = torch.randint(0, cfg.vocab_size, (B, T))
    logits, loss = model(idx, tgt)
    print(f"  输入   {tuple(idx.shape)}")
    print(f"  logits {tuple(logits.shape)}   ← 出口从 C={cfg.C} 撑到 vocab={cfg.vocab_size}")
    print(f"  ★ 初始 loss = {loss.item():.4f}")
    print(f"    ln(vocab) = {math.log(cfg.vocab_size):.4f}")
    d = abs(loss.item() - math.log(cfg.vocab_size))
    print(f"    偏差 {d:.4f}  {'✅ 初始化正确' if d < 0.3 else '❌ 初始化可能有问题'}")

    loss.backward()
    ng = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"  反向：{ng} 个参数张量拿到了梯度  ✅")

    print("\n" + "=" * 60)
    print("RoPE 自检：旋转后的内积只依赖相对距离")
    print("=" * 60)
    cos, sin = precompute_rope(cfg.head_dim, 64, cfg.rope_theta)
    q = torch.randn(1, 1, 64, cfg.head_dim)
    qr = apply_rope(q, cos, sin)
    # 位置 (10,15) 和 (30,35) 距离都是 5，若 RoPE 正确，同一向量在两处的内积应相等
    v = torch.randn(cfg.head_dim)
    a = torch.stack([v, v]).view(1, 1, 2, -1)
    a1 = apply_rope(a, cos[10:12], sin[10:12])
    a2 = apply_rope(a, cos[30:32], sin[30:32])
    d1 = (a1[0, 0, 0] * a1[0, 0, 1]).sum()
    d2 = (a2[0, 0, 0] * a2[0, 0, 1]).sum()
    print(f"  位置 (10,11) 的内积  {d1.item():.6f}")
    print(f"  位置 (30,31) 的内积  {d2.item():.6f}")
    print(f"  {'✅ 相等 → 只看相对距离' if abs(d1-d2) < 1e-4 else '❌ 不等，RoPE 实现有误'}")

    print("\n" + "=" * 60)
    print("采样冒烟测试（未训练，输出必然是乱码）")
    print("=" * 60)
    out = model.generate(torch.zeros(1, 1, dtype=torch.long), max_new_tokens=12)
    print(f"  生成 {out.shape[1]} 个 token: {out[0].tolist()}  ✅ 采样通路正常")
    print()
