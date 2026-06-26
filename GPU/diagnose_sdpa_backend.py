"""
诊断：PyTorch SDPA 在当前环境走的哪个 backend
"""
import torch
import torch.nn.functional as F
import math

print(f"PyTorch 版本  : {torch.__version__}")
print(f"CUDA 版本    : {torch.version.cuda}")
print(f"显卡         : {torch.cuda.get_device_name()}")
print(f"Compute Cap  : {torch.cuda.get_device_capability()}")

def cuda_timer(fn, n_warmup=3, n_repeat=10):
    for _ in range(n_warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_repeat): fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n_repeat

# ── 测试配置
# Flash Attention 要求 4D：[batch, num_heads, seq_len, head_dim]
# 3D [batch, seq, dim] 会导致所有 fused kernel 报错退化到 Math backend
BATCH, NUM_HEADS, SL, D_HEAD = 1, 8, 8192, 64
Q = torch.randn(BATCH, NUM_HEADS, SL, D_HEAD, dtype=torch.float16, device='cuda')
K = torch.randn(BATCH, NUM_HEADS, SL, D_HEAD, dtype=torch.float16, device='cuda')
V = torch.randn(BATCH, NUM_HEADS, SL, D_HEAD, dtype=torch.float16, device='cuda')
scale = 1.0 / math.sqrt(D_HEAD)
D = D_HEAD  # 保持后续兼容

# ── Step 1：检查各 backend 是否可用
print("\n" + "=" * 55)
print("Step 1：各 SDPA backend 可用性")
print("=" * 55)

backends = {
    "Flash Attention"   : dict(enable_flash=True,  enable_math=False, enable_mem_efficient=False),
    "Memory Efficient"  : dict(enable_flash=False, enable_math=False, enable_mem_efficient=True),
    "Math (naive-like)" : dict(enable_flash=False, enable_math=True,  enable_mem_efficient=False),
}

available = {}
for name, kwargs in backends.items():
    try:
        with torch.backends.cuda.sdp_kernel(**kwargs):
            _ = F.scaled_dot_product_attention(Q, K, V, scale=scale)
        available[name] = kwargs
        print(f"  {name:<22}: ✓ 可用")
    except Exception as ex:
        print(f"  {name:<22}: ✗ 不可用  ({ex})")

# ── Step 2：对每个可用 backend 计时
print("\n" + "=" * 55)
print("Step 2：各 backend 实测耗时（batch=1, seq=8192）")
print("=" * 55)

timings = {}
for name, kwargs in available.items():
    def fn(kw=kwargs):
        with torch.backends.cuda.sdp_kernel(**kw):
            return F.scaled_dot_product_attention(Q, K, V, scale=scale)
    ms = cuda_timer(fn)
    timings[name] = ms
    print(f"  {name:<22}: {ms:.3f} ms")

# naive baseline（4D 用 matmul，支持多头）
def naive():
    # [B, H, N, D] @ [B, H, D, N] -> [B, H, N, N]
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    P = torch.softmax(S, dim=-1)
    return torch.matmul(P, V)

t_naive = cuda_timer(naive)
print(f"  {'Naive (explicit S/P)':<22}: {t_naive:.3f} ms  ← 对照基准")

# ── Step 3：结论
print("\n" + "=" * 55)
print("Step 3：结论")
print("=" * 55)

if "Flash Attention" in available:
    t_fa = timings["Flash Attention"]
    if t_fa < t_naive:
        print(f"  Flash Attention 比 naive 快 {t_naive/t_fa:.2f}x ✓")
    else:
        print(f"  Flash Attention 比 naive 慢 {t_fa/t_naive:.2f}x")
        print("  → 在当前 seq_len 下 flash 的 tiling overhead > IO 节省")
        print("  → 需要更大 seq_len 才能翻正（参考 Step 4）")
else:
    print("  ✗ Flash Attention kernel 不可用！")
    print("  SDPA 实际走的是其他 backend，和 Flash Attention 无关。")

# ── Step 4：找到 Flash Attention 开始占优的 seq_len 拐点
if "Flash Attention" in available:
    print("\n" + "=" * 55)
    print("Step 4：找到 FA 占优的 seq_len 拐点")
    print("=" * 55)
    print(f"{'seq_len':>10} {'naive':>10} {'FA':>10} {'FA/naive':>10} {'谁更快':>10}")
    print("-" * 55)
    for sl in [1024, 2048, 4096, 8192, 16384, 32768]:
        sp_gb = 2 * BATCH * NUM_HEADS * sl * sl * 2 / 1e9
        if sp_gb > 10:
            print(f"{sl:>10}  跳过（S/P 约 {sp_gb:.1f}GB，节省显存）")
            continue
        q = torch.randn(BATCH, NUM_HEADS, sl, D_HEAD, dtype=torch.float16, device='cuda')
        k = torch.randn(BATCH, NUM_HEADS, sl, D_HEAD, dtype=torch.float16, device='cuda')
        v = torch.randn(BATCH, NUM_HEADS, sl, D_HEAD, dtype=torch.float16, device='cuda')

        def n_fn(q=q, k=k, v=v):
            S = torch.matmul(q, k.transpose(-2, -1)) * scale
            P = torch.softmax(S, dim=-1)
            return torch.matmul(P, v)

        def fa_fn(q=q, k=k, v=v, kw=available["Flash Attention"]):
            with torch.backends.cuda.sdp_kernel(**kw):
                return F.scaled_dot_product_attention(q, k, v, scale=scale)

        t_n = cuda_timer(n_fn)
        t_f = cuda_timer(fa_fn)
        winner = "← FA 胜" if t_f < t_n else "← naive 胜"
        print(f"{sl:>10} {t_n:>8.3f}ms {t_f:>8.3f}ms {t_f/t_n:>10.2f}x  {winner}")
        del q, k, v
        torch.cuda.empty_cache()
