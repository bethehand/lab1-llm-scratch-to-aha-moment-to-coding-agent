"""
训练脚本 —— 第 4 步。

用法:
    python3 train.py --smoke                        # 5a 冒烟：玩具配置+随机数据，不用 GPU 也能跑
    python3 train.py --steps 100                    # 5b 单卡：目标配置，看初始 loss 和 MFU
    torchrun --nproc_per_node=4 train.py --steps 200 --smoke-data   # 5c 四卡：验 DDP
    numactl --cpunodebind=1 --membind=1 \
        torchrun --nproc_per_node=4 train.py        # 6  正式训练
"""
import os, sys, math, time, argparse, signal
from contextlib import nullcontext
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group

from config import Config
from model import GPT

# ══════════════════════════════════════════════════════════════
#  命令行
# ══════════════════════════════════════════════════════════════
ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true", help="5a：玩具配置+随机数据，几分钟跑完")
ap.add_argument("--smoke-data", action="store_true", help="用随机数据但保持目标配置")
ap.add_argument("--steps", type=int, default=None, help="覆盖总步数")
ap.add_argument("--out", type=str, default="out", help="checkpoint 目录")
ap.add_argument("--resume", action="store_true", help="从 out/ckpt.pt 续训")
ap.add_argument("--eval-every", type=int, default=250)
ap.add_argument("--sample-every", type=int, default=500)
ap.add_argument("--log-every", type=int, default=10)
ap.add_argument("--save-every", type=int, default=100,
                help="每 N 步无条件存一份 ckpt_latest.pt（崩了最多丢这么多步）")
ap.add_argument("--wandb", action="store_true")
ap.add_argument("--no-bf16-grad", action="store_true",
                help="关掉 bf16 梯度压缩（默认开：通信量减半，11 小时的训练能省约 1 小时）")
args = ap.parse_args()

cfg = Config()
# world_size 必须最先确定：train_tokens 和 max_steps 都依赖它
cfg.world_size = int(os.environ.get("WORLD_SIZE", 1))
if args.smoke:                      # 玩具配置：秒级迭代，只验代码通不通
    cfg.C, cfg.n_layer, cfg.n_head, cfg.head_dim = 128, 2, 4, 32
    cfg.d_ff, cfg.block_size = 384, 128
    cfg.batch_size_per_gpu, cfg.grad_accum_steps = 4, 2
    cfg.warmup_steps, cfg.compile = 10, False
if args.steps:                      # 直接指定步数（world_size 已确定，换算才准）
    cfg.train_tokens = args.steps * cfg.tokens_per_step
cfg.check()
USE_RANDOM_DATA = args.smoke or args.smoke_data

# ══════════════════════════════════════════════════════════════
#  DDP
# ══════════════════════════════════════════════════════════════
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    init_process_group(backend="nccl")
    rank       = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)
    master = rank == 0
    assert world_size == cfg.world_size, "WORLD_SIZE 与 config 不一致"
else:
    rank, local_rank, world_size, master = 0, 0, 1, True
    device = "cuda" if torch.cuda.is_available() else "cpu"

# ── 防呆：没有 GPU 却在跑真实配置，一定是环境坏了，不能静默回退到 CPU ──
#    （典型原因：venv 里的 PyTorch 是为更高版本 CUDA 编译的，驱动不支持 → 静默降级）
if not torch.cuda.is_available() and not args.smoke:
    print("\n" + "!" * 66)
    print("  ❌ CUDA 不可用 —— 拒绝在 CPU 上跑真实配置")
    print("!" * 66)
    print(f"  torch 版本      {torch.__version__}")
    print(f"  编译用的 CUDA   {torch.version.cuda}")
    print(f"  device_count    {torch.cuda.device_count()}")
    print("\n  常见原因：venv 里的 PyTorch 为更高版本 CUDA 编译，本机驱动不支持。")
    print("  检查：nvidia-smi 看驱动支持的 CUDA 版本，跟上面的『编译用的 CUDA』对比。")
    print("  修复：pip install torch --index-url https://download.pytorch.org/whl/cuXXX")
    print("        （XXX 换成驱动支持的版本，如 cu124）")
    print("\n  只想在 CPU 上验证代码，加 --smoke。\n")
    sys.exit(1)

device_type = "cuda" if "cuda" in device else "cpu"
torch.manual_seed(cfg.seed + rank)              # 每张卡不同种子 → 数据不重复
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# bf16 只在 CUDA 上用；CPU 冒烟测试走 fp32
use_bf16 = device_type == "cuda" and torch.cuda.is_bf16_supported()
ptdtype  = torch.bfloat16 if use_bf16 else torch.float32
ctx = (torch.amp.autocast(device_type="cuda", dtype=ptdtype)
       if use_bf16 else nullcontext())


def log(*a):
    if master:
        print(*a, flush=True)


# ── Ctrl-C 优雅退出：先标记，本步跑完存盘再走；再按一次才强制退出 ──
_interrupted = False


def _sigint(signum, frame):
    global _interrupted
    if _interrupted:                      # 第二次 Ctrl-C：真的退
        raise KeyboardInterrupt
    _interrupted = True
    if master:
        print("\n⚠️  收到中断，本步结束后存盘退出（再按一次 Ctrl-C 强制退出）", flush=True)


signal.signal(signal.SIGINT, _sigint)


# ══════════════════════════════════════════════════════════════
#  数据
# ══════════════════════════════════════════════════════════════
_rand_cache = {}


def get_batch(split):
    B, T = cfg.batch_size_per_gpu, cfg.block_size
    if USE_RANDOM_DATA:
        # 随机 token：只验证形状和梯度能不能流通，loss 不会真的下降
        if split not in _rand_cache:
            _rand_cache[split] = np.random.randint(
                0, cfg.vocab_size, size=2_000_000, dtype=np.uint16)
        data = _rand_cache[split]
    else:
        path = os.path.join(cfg.data_dir, f"{split}.bin")
        # 每次重新 memmap：避免长时间训练下的内存泄漏（nanoGPT 的已知坑）
        data = np.memmap(path, dtype=np.uint16, mode="r")

    ix = np.random.randint(0, len(data) - T - 1, size=B)
    x = torch.from_numpy(np.stack([data[i:i+T].astype(np.int64) for i in ix]))
    y = torch.from_numpy(np.stack([data[i+1:i+1+T].astype(np.int64) for i in ix]))
    if device_type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# ══════════════════════════════════════════════════════════════
#  模型
# ══════════════════════════════════════════════════════════════
os.makedirs(args.out, exist_ok=True)

# 开跑前检查磁盘：三个 checkpoint + 一个 .tmp 中间文件
if master:
    import shutil
    n_par = (cfg.n_layer * (4*cfg.C**2 + 3*cfg.C*cfg.d_ff + 2*cfg.C)
             + cfg.vocab_size * cfg.C + cfg.C)
    need_gb = n_par * 12 * 4 / 1024**3        # 参数+m+v，四份（三个 ckpt + 一个 tmp）
    free_gb = shutil.disk_usage(args.out).free / 1024**3
    print(f"  磁盘 需要 ~{need_gb:.1f} GB（3 个 checkpoint + 1 个临时文件），"
          f"可用 {free_gb:.1f} GB "
          f"{'✅' if free_gb > need_gb * 1.5 else '⚠️ 偏紧！'}", flush=True)
    if free_gb < need_gb:
        raise SystemExit(f"磁盘不够：需要至少 {need_gb:.1f} GB")

model = GPT(cfg)
start_step, best_val = 0, 1e9

if args.resume:
    # 优先用最新的（无条件存的），其次用 best（val 最好的）
    cand = [os.path.join(args.out, n) for n in ("ckpt_latest.pt", "ckpt.pt")]
    cand = [c for c in cand if os.path.exists(c)]
    if not cand:
        raise FileNotFoundError(
            f"{args.out}/ 下没有 ckpt_latest.pt 或 ckpt.pt —— 之前那次可能还没存过盘")
    log(f"↻ 载入 {cand[0]}")
    ck = torch.load(cand[0], map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    start_step, best_val = ck["step"], ck["best_val"]
    log(f"↻ 从 step {start_step} 续训，best_val={best_val:.4f}")

model.to(device)
opt, nd, nn_, pd, pn = model.configure_optimizers(
    cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2), device_type)
if args.resume:
    opt.load_state_dict(ck["optimizer"]); del ck

raw_model = model                                  # 采样/存盘用未包装的
if cfg.compile and device_type == "cuda":
    log("torch.compile 编译中（首次约 1~2 分钟）…")
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[local_rank])
    if not args.no_bf16_grad:
        # 梯度默认是 fp32（201M × 4B = 804 MB），压成 bf16 后减半。
        # 你的 all-reduce 只有 3.7 GB/s，这一项直接把通信开销从 18.5% 砍到 9.3%。
        # 业界标准做法，对收敛几乎无影响。
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
        model.register_comm_hook(None, default_hooks.bf16_compress_hook)

max_steps = cfg.train_tokens // cfg.tokens_per_step

log("=" * 64)
log(f"  设备 {device} × {world_size}   dtype {ptdtype}   compile {cfg.compile}")
log(f"  模型 {raw_model.num_params():,} 参数  "
    f"(C={cfg.C} L={cfg.n_layer} H={cfg.n_head} d_ff={cfg.d_ff} T={cfg.block_size})")
log(f"  优化 加wd的张量 {nd} 个 / {pd:,} 参数；不加wd {nn_} 个 / {pn:,} 参数")
log(f"  批次 {cfg.tokens_per_step:,} token/步 × {max_steps:,} 步 "
    f"= {max_steps*cfg.tokens_per_step/1e9:.2f}B token")
log(f"  数据 {'随机（冒烟）' if USE_RANDOM_DATA else cfg.data_dir}")
if ddp:
    log(f"  通信 bf16 梯度压缩 {'关' if args.no_bf16_grad else '开'}"
        f"   eval 跨卡平均 开（{world_size}× 样本量）")
log(f"  ★ 初始 loss 应该 ≈ ln({cfg.vocab_size}) = {math.log(cfg.vocab_size):.4f}")
log("=" * 64)

if args.wandb and master:
    import wandb
    wandb.init(project="scratch-llm", config=vars(cfg))


# ══════════════════════════════════════════════════════════════
#  学习率：warmup + cosine 衰减到 10%
# ══════════════════════════════════════════════════════════════
def get_lr(it):
    if it < cfg.warmup_steps:
        return cfg.lr * (it + 1) / (cfg.warmup_steps + 1)
    if it >= max_steps:
        return cfg.lr_min
    r = (it - cfg.warmup_steps) / max(1, max_steps - cfg.warmup_steps)
    return cfg.lr_min + 0.5 * (1 + math.cos(math.pi * r)) * (cfg.lr - cfg.lr_min)


def save_ckpt(name, step_now):
    """
    原子写盘：先写 .tmp，再 os.replace 覆盖。
    直接 torch.save 到目标路径的话，2.4 GB 要写好几秒 ——
    如果正好在这个窗口崩溃，新文件写了一半、旧文件已被覆盖，两个都没了。
    os.replace 在 POSIX 上是原子的：要么是旧文件，要么是新文件，不存在中间态。
    """
    if not master:
        return
    path = os.path.join(args.out, name)
    tmp = path + ".tmp"
    torch.save({"model": raw_model.state_dict(), "optimizer": opt.state_dict(),
                "step": step_now, "best_val": best_val, "cfg": vars(cfg)}, tmp)
    os.replace(tmp, path)          # 原子替换


@torch.no_grad()
def evaluate(iters=20):
    model.eval()
    out = {}
    for split in (["train"] if USE_RANDOM_DATA else ["train", "val"]):
        losses = torch.zeros(iters, device=device)
        for k in range(iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss
        m = losses.mean()
        if ddp:
            # 四张卡各自随机采样，平均起来等于 4 倍样本量 —— 免费降噪。
            # 第一轮只用了 rank 0 的 20 个 batch，另外三张卡白算了。
            dist.all_reduce(m, op=dist.ReduceOp.AVG)
        out[split] = m.item()
    model.train()
    return out


@torch.no_grad()
def sample(n_tokens=80):
    if USE_RANDOM_DATA:
        return None
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    model.eval()
    ids = torch.tensor([enc.encode_ordinary("The meaning of life is")],
                       dtype=torch.long, device=device)
    with ctx:
        out = raw_model.generate(ids, n_tokens, temperature=0.8, top_k=50)
    model.train()
    return enc.decode(out[0].tolist())


# ══════════════════════════════════════════════════════════════
#  训练循环
# ══════════════════════════════════════════════════════════════
peak_flops = cfg.peak_tflops_per_gpu * 1e12 * world_size
X, Y = get_batch("train")
t0 = time.time()
model.train()

for step in range(start_step, max_steps):
    # 所有 rank 对「是否停止」达成一致，否则有的 rank 退了、有的还在等 all-reduce → 死锁
    if ddp:
        flag = torch.tensor([1.0 if _interrupted else 0.0], device=device)
        dist.all_reduce(flag)            # 一个 float，开销可忽略
        should_stop = flag.item() > 0
    else:
        should_stop = _interrupted
    if should_stop:
        save_ckpt("ckpt_latest.pt", step)
        log(f"  💾 已存盘于 step {step} → {args.out}/ckpt_latest.pt")
        log(f"  续训: 同样的命令加 --resume")
        break

    lr = get_lr(step)
    for g in opt.param_groups:
        g["lr"] = lr

    # ── 梯度累积：只在最后一个 micro-step 同步梯度 ──
    #    你的 all-reduce 只有 3.7 GB/s，这一步把通信开销从 40% 压到 5%
    loss_accum = 0.0
    for micro in range(cfg.grad_accum_steps):
        if ddp:
            model.require_backward_grad_sync = (micro == cfg.grad_accum_steps - 1)
        with ctx:
            _, loss = model(X, Y)
            loss = loss / cfg.grad_accum_steps
        X, Y = get_batch("train")            # 预取下一批，跟反向重叠
        loss.backward()
        loss_accum += loss.item()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    opt.step()
    opt.zero_grad(set_to_none=True)

    if device_type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    t0 = time.time()

    if step % args.log_every == 0 or step == max_steps - 1:
        mfu = raw_model.estimate_mfu(cfg.tokens_per_step, dt, peak_flops)
        mem = (torch.cuda.max_memory_allocated() / 1024**3) if device_type == "cuda" else 0
        log(f"step {step:>6,}/{max_steps:,} | loss {loss_accum:.4f} | lr {lr:.2e} | "
            f"norm {norm:.2f} | {dt*1000:6.0f}ms | MFU {mfu*100:5.1f}% | {mem:.1f}GB")
        if args.wandb and master:
            import wandb; wandb.log({"loss": loss_accum, "lr": lr, "mfu": mfu}, step=step)

    if step > 0 and step % args.save_every == 0:
        save_ckpt("ckpt_latest.pt", step)      # 无条件存，不看 val

    if step > 0 and step % args.eval_every == 0:
        losses = evaluate()
        log(f"  ── eval @ {step}: " +
            "  ".join(f"{k} {v:.4f}" for k, v in losses.items()))
        if args.wandb and master:
            import wandb; wandb.log({f"eval/{k}": v for k, v in losses.items()}, step=step)
        v = losses.get("val", losses["train"])
        if v < best_val:
            best_val = v
            save_ckpt("ckpt.pt", step)
            log(f"  ✅ 存盘 best_val={best_val:.4f}")

    if step > 0 and step % args.sample_every == 0 and master:
        s = sample()
        if s:
            log(f"  ── 生成 @ {step} ──\n    {s[:400]}\n")

# ── 收尾 ──
if not should_stop:
    losses = evaluate()
    log("=" * 64)
    log("  训练结束  " + "  ".join(f"{k} {v:.4f}" for k, v in losses.items()))
    save_ckpt("ckpt_final.pt", max_steps)
    log(f"  最终权重 → {args.out}/ckpt_final.pt")
    txt = sample()
    if txt:
        log(f"\n  最终样本:\n    {txt}")

if ddp:
    destroy_process_group()               # 干净退出，不再警告资源泄漏
