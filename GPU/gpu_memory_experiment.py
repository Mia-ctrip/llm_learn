"""
GPU 内存层次实验：在真正学 Flash Attention 之前的感性认知
============================================================
目标：
  1. 实测 HBM 带宽，和理论峰值对比
  2. 感受 Memory-Bound vs Compute-Bound 的本质差异
  3. 量化 Attention 中间矩阵的 HBM IO 代价随 seq_len 如何增长
  4. 用 naive attention vs PyTorch SDPA 验证 Flash Attention 的真实加速

硬件前提：L20 48G（或 H20）
  - L20 HBM 理论带宽：  864 GB/s
  - L20 FP16 Tensor Core 峰值算力：约 119 TFLOPS
  - L20 每个 SM 的 SRAM（共享内存）：128 KB
  - SM 数量：72 个

GPU 内存层次速查（从快到慢）：
  ┌─────────────────┬───────────────────┬─────────────────────┐
  │   层次           │  容量              │  带宽/延迟           │
  ├─────────────────┼───────────────────┼─────────────────────┤
  │ 寄存器 Register  │ ~64KB/SM (per线程) │ 最快，无显式带宽概念  │
  │ SRAM(共享内存)   │ 128KB per SM       │ ~19 TB/s            │
  │ L2 Cache        │ 64MB 全卡共享       │ ~4 TB/s             │
  │ HBM (显存)      │ 48 GB              │ 864 GB/s (理论)     │
  └─────────────────┴───────────────────┴─────────────────────┘

  Flash Attention 核心：把 N×N 中间矩阵（S, P）留在 SRAM，不让它落到 HBM。
  本实验就是量化"落到 HBM 这个绕路"到底有多贵。
"""

import torch
import math
import sys

# ─── 基础工具 ────────────────────────────────────────────────
def cuda_timer(fn, n_warmup=5, n_repeat=30):
    """
    用 CUDA Event 精确计时，返回平均耗时（ms）。
    注意：必须用 CUDA Event，不能用 time.perf_counter()，
    因为 CPU 看不到 GPU 真正什么时候完成。
    """
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    t_start = torch.cuda.Event(enable_timing=True)
    t_end   = torch.cuda.Event(enable_timing=True)

    t_start.record()
    for _ in range(n_repeat):
        fn()
    t_end.record()
    torch.cuda.synchronize()

    return t_start.elapsed_time(t_end) / n_repeat  # ms


def sep(title=""):
    width = 65
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─' * pad} {title} {'─' * (width - pad - len(title) - 2)}")
    else:
        print("─" * width)


# ─── 前置检查 ────────────────────────────────────────────────
if not torch.cuda.is_available():
    print("❌ 没有检测到 CUDA GPU，请确认环境配置。")
    sys.exit(1)

props = torch.cuda.get_device_properties(0)
device = "cuda"

print("=" * 65)
print("GPU 内存层次实验  —  Flash Attention 前置感性认知")
print("=" * 65)
print(f"显卡型号 : {props.name}")
print(f"总显存   : {props.total_memory / 1e9:.1f} GB")
print(f"SM 数量  : {props.multi_processor_count}")
print(f"L2 Cache : {props.l2_cache_size / 1e6:.0f} MB")
print(f"Compute  : {props.major}.{props.minor}")


# ════════════════════════════════════════════════════════════
# 实验 1：实测 HBM 带宽
# ════════════════════════════════════════════════════════════
sep("实验 1：实测 HBM 带宽")

print("""
原理：分配一大块显存，做 clone()（一次完整的读 + 写），
      根据 数据量 / 耗时 算出实际带宽。
""")

SIZE_GB = 2
n_elem  = SIZE_GB * (1024**3) // 2   # float16 = 2 bytes
x = torch.empty(n_elem, dtype=torch.float16, device=device)

ms_bw   = cuda_timer(lambda: x.clone(), n_warmup=5, n_repeat=20)
bytes_rw = n_elem * 2 * 2  # 读 + 写，各 SIZE_GB
bw_actual = bytes_rw / (ms_bw / 1000) / 1e9  # GB/s

print(f"数据量    : {SIZE_GB} GB (float16)")
print(f"耗时      : {ms_bw:.2f} ms")
print(f"实测带宽  : {bw_actual:.1f} GB/s")
print(f"理论峰值  : 864 GB/s (L20)")
print(f"利用率    : {bw_actual / 864 * 100:.1f}%")
print("""
→ 小结：实测带宽通常能达到理论峰值的 80-95%。
  下面所有实验里，数据搬运速度就受限于这个数字。
""")

del x
torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════
# 实验 2：Memory-Bound vs Compute-Bound 的量化对比
# ════════════════════════════════════════════════════════════
sep("实验 2：Memory-Bound vs Compute-Bound 直觉建立")

print("""
核心概念：Arithmetic Intensity（算术强度） = FLOP / Byte
  - 低 AI（< ~100）→ Memory-Bound：GPU 在等数据，算力闲着
  - 高 AI（> ~100）→ Compute-Bound：GPU 算力打满，带宽还好

L20 的 "屋顶线" 拐点 AI ≈ (119 TFLOPS) / (0.864 TB/s) ≈ 138 FLOP/Byte
""")

N = 8192
a = torch.randn(N, N, dtype=torch.float16, device=device)
b = torch.randn(N, N, dtype=torch.float16, device=device)

# ── 场景 A：逐元素加法（Memory-Bound）
ms_add  = cuda_timer(lambda: a + b, n_warmup=5, n_repeat=50)
flop_add = N * N                 # 每个元素 1 次加法
byte_add = N * N * 2 * 3        # 读a + 读b + 写c，float16=2B
ai_add   = flop_add / byte_add  # FLOP/Byte
bw_add   = byte_add / (ms_add / 1000) / 1e9

# ── 场景 B：矩阵乘法（Compute-Bound）
ms_mm   = cuda_timer(lambda: torch.mm(a, b), n_warmup=5, n_repeat=20)
flop_mm  = 2 * N**3             # N×N×N 乘加
byte_mm  = N * N * 2 * 3        # 同样 3 个矩阵
ai_mm    = flop_mm / byte_mm
tflops_mm = flop_mm / (ms_mm / 1000) / 1e12

print(f"矩阵规模：{N} × {N}，float16\n")
print(f"{'操作':<20} {'耗时':>8} {'AI (FLOP/B)':>14} {'带宽利用':>12} {'备注'}")
print("-" * 70)
print(f"{'逐元素加法':<20} {ms_add:>7.3f}ms {ai_add:>13.3f} {bw_add:>9.1f} GB/s  ← Memory-Bound")
print(f"{'矩阵乘法':<20} {ms_mm:>7.3f}ms {ai_mm:>13.1f} {'~':>9}       ← Compute-Bound ({tflops_mm:.1f} TFLOPS)")

ratio_time = ms_mm / ms_add
ratio_flop = flop_mm / flop_add
print(f"""
直觉总结：
  矩阵乘法做了 {flop_mm/flop_add:,.0f}x 更多计算，却只慢了 {ratio_time:.1f}x
  → 说明 GPU 的计算能力极其充裕，"搬数据"才是稀缺资源
  → 逐元素加法的 AI={ai_add:.3f} 远低于拐点138，它的瓶颈就在 HBM 带宽
""")

del a, b
torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════
# 实验 3：N×N 中间矩阵的 HBM IO 代价随 seq_len 平方增长
# ════════════════════════════════════════════════════════════
sep("实验 3：Attention 中间矩阵 S/P 的 IO 代价")

print("""
标准 Attention：S = QK^T 写到 HBM → P = softmax(S) 读 HBM 写 HBM → O = PV 读 HBM
每轮需要对 N×N 矩阵做 2~4 次 HBM 读写。
下面只测"把一个 N×N float16 矩阵写到 HBM 再读回来"的纯 IO 代价。
""")

print(f"{'seq_len':>10} {'矩阵大小':>10} {'写+读耗时':>12} {'折算带宽':>12}  说明")
print("-" * 60)

for sl in [256, 512, 1024, 2048, 4096, 8192]:
    mat = torch.zeros(sl, sl, dtype=torch.float16, device=device)
    # 模拟：写 HBM（clone）→ 读 HBM（sum/add）
    def io_round_trip():
        tmp = mat.clone()      # 写到 HBM（新分配）
        return tmp + 0.0       # 从 HBM 读回（简单触发读）
    
    ms_io     = cuda_timer(io_round_trip, n_warmup=5, n_repeat=100)
    mat_bytes = sl * sl * 2      # float16
    rw_bytes  = mat_bytes * 2    # 写一次 + 读一次
    bw_used   = rw_bytes / (ms_io / 1000) / 1e9

    flag = "← 注意增速" if sl >= 2048 else ""
    print(f"{sl:>10} {mat_bytes/1e6:>7.1f} MB {ms_io:>10.3f} ms {bw_used:>9.1f} GB/s  {flag}")
    del mat
    torch.cuda.empty_cache()

print("""
→ 注意：耗时随 seq_len² 增长。seq_len 翻倍，IO 代价 ×4。
  这正是标准 Attention 不能用于长序列的根本原因。
""")


# ════════════════════════════════════════════════════════════
# 实验 4：naive Attention vs Flash Attention（SDPA）真实加速
# ════════════════════════════════════════════════════════════
sep("实验 4：naive Attention vs Flash Attention 实测加速")

print("""
native_attention：显式把 S、P 写到 HBM（PyTorch 默认行为）
sdpa_attention  ：torch.nn.functional.scaled_dot_product_attention
                  → 需要 4D 张量 [batch, heads, seq, head_dim]
                  → PyTorch 2.0+ 在满足条件时自动调用 Flash Attention

注意：Flash Attention 强制要求 4D 输入。
      3D [batch, seq, dim] 会导致所有 fused kernel 报错退化到 Math backend。
""")

NUM_HEADS = 8
D_HEAD    = 64
BATCH     = 4
scale     = 1.0 / math.sqrt(D_HEAD)

print(f"配置：batch={BATCH}, num_heads={NUM_HEADS}, d_head={D_HEAD}")
print(f"{'seq_len':>10} {'naive (ms)':>12} {'SDPA/FA (ms)':>14} {'加速比':>8} {'S/P per head':>14}")
print("-" * 68)

for sl in [512, 1024, 2048, 4096, 8192]:
    # 正确 shape：[batch, num_heads, seq_len, head_dim]
    Q = torch.randn(BATCH, NUM_HEADS, sl, D_HEAD, dtype=torch.float16, device=device)
    K = torch.randn(BATCH, NUM_HEADS, sl, D_HEAD, dtype=torch.float16, device=device)
    V = torch.randn(BATCH, NUM_HEADS, sl, D_HEAD, dtype=torch.float16, device=device)

    def naive_attn():
        # [B, H, N, D] @ [B, H, D, N] -> [B, H, N, N]  每步完整落 HBM
        S = torch.matmul(Q, K.transpose(-2, -1)) * scale
        P = torch.softmax(S, dim=-1)
        return torch.matmul(P, V)

    def sdpa_attn():
        return torch.nn.functional.scaled_dot_product_attention(Q, K, V, scale=scale)

    t_naive = cuda_timer(naive_attn, n_warmup=5, n_repeat=20)
    t_sdpa  = cuda_timer(sdpa_attn,  n_warmup=5, n_repeat=20)
    # S/P 单头大小（每个 head 的 N×N 矩阵）
    sp_per_head_mb = sl * sl * 2 / 1e6

    print(f"{sl:>10} {t_naive:>10.3f} ms {t_sdpa:>12.3f} ms {t_naive/t_sdpa:>7.2f}x {sp_per_head_mb:>10.1f} MB/head")
    del Q, K, V
    torch.cuda.empty_cache()

print("""
→ 上面 seq_len 较小时 S/P 矩阵可能仍命中 L2 Cache（101MB），抹平了部分差距。
  见下方实验 4b：参数调大让矩阵彻底超出 L2。
""")


# ════════════════════════════════════════════════════════════
# 实验 4b：超出 L2 Cache 后的真实加速对比
# ════════════════════════════════════════════════════════════
sep("实验 4b：超出 L2 Cache 后的真实加速")

print(f"""
目标：让 S/P 矩阵超出 L2 Cache（{props.l2_cache_size/1e6:.0f} MB），使 naive attention 真正打到 HBM。

显存占用估算（naive 同时持有 S 和 P，含所有 head）：
  显存 = 2 × batch × num_heads × seq_len² × 2B
  batch=1, heads=8, seq=4096  → 约 512 MB   ← 超出L2，安全
  batch=1, heads=8, seq=8192  → 约   2 GB   ← 安全
  batch=4, heads=8, seq=8192  → 约   8 GB   ← 安全（L20有48GB）
""")

print(f"{'seq_len':>10} {'batch':>6} {'S/P总显存':>11} {'naive (ms)':>12} {'SDPA/FA (ms)':>14} {'加速比':>8}")
print("-" * 68)

NUM_HEADS_B = 8
D_HEAD_B    = 64
scale_b     = 1.0 / math.sqrt(D_HEAD_B)

for batch_l2, sl in [(1, 4096), (1, 8192), (4, 8192), (4, 16384)]:
    sp_total_mb = 2 * batch_l2 * NUM_HEADS_B * sl * sl * 2 / 1e6
    if sp_total_mb > 20_000:
        print(f"{sl:>10} {batch_l2:>6} {sp_total_mb:>8.0f} MB  跳过（预计OOM）")
        continue

    Q = torch.randn(batch_l2, NUM_HEADS_B, sl, D_HEAD_B, dtype=torch.float16, device=device)
    K = torch.randn(batch_l2, NUM_HEADS_B, sl, D_HEAD_B, dtype=torch.float16, device=device)
    V = torch.randn(batch_l2, NUM_HEADS_B, sl, D_HEAD_B, dtype=torch.float16, device=device)

    def naive_attn_l2():
        S = torch.matmul(Q, K.transpose(-2, -1)) * scale_b
        P = torch.softmax(S, dim=-1)
        return torch.matmul(P, V)

    def sdpa_attn_l2():
        return torch.nn.functional.scaled_dot_product_attention(Q, K, V, scale=scale_b)

    t_naive = cuda_timer(naive_attn_l2, n_warmup=3, n_repeat=10)
    t_sdpa  = cuda_timer(sdpa_attn_l2,  n_warmup=3, n_repeat=10)

    print(f"{sl:>10} {batch_l2:>6} {sp_total_mb:>8.0f} MB {t_naive:>10.3f} ms {t_sdpa:>12.3f} ms {t_naive/t_sdpa:>8.2f}x")
    del Q, K, V
    torch.cuda.empty_cache()

print("""
→ 当 S/P 总量超出 L2（>101MB），naive 每步都必须真正打到 HBM，
  Flash Attention 省掉 HBM 绕路的收益才真正显现。
""")


# ════════════════════════════════════════════════════════════
# 实验 5（加餐）：SRAM 大小的物理约束——为什么必须 Tiling
# ════════════════════════════════════════════════════════════
sep("实验 5（加餐）：SRAM 约束 —— 为什么必须 Tiling")

sram_per_sm_kb = 128
sm_count       = props.multi_processor_count
total_sram_mb  = sram_per_sm_kb * sm_count / 1024

print(f"""
L20 SRAM（共享内存）：
  每个 SM：{sram_per_sm_kb} KB
  SM 数量 ：{sm_count} 个
  总计    ：{total_sram_mb:.1f} MB（所有 SM 加起来，也不超过这个数）

N×N 矩阵（float16）占多少空间？
""")

print(f"{'seq_len':>10} {'N×N 矩阵':>12}  vs SRAM 总量 {total_sram_mb:.1f} MB")
print("-" * 45)
for sl in [512, 1024, 2048, 4096]:
    mat_kb = sl * sl * 2 / 1024   # KB
    fits   = "✓ 勉强能放" if mat_kb < total_sram_mb * 1024 * 0.3 else "✗ 远超 SRAM"
    print(f"{sl:>10} {mat_kb/1024:>9.1f} MB   {fits}")

print(f"""
→ 核心限制：
  - seq_len ≥ 1024 时，N×N 矩阵（2MB+）已远超单个 SM 的 SRAM
  - 就算把所有 SM 的 SRAM 合起来也装不下 seq_len=4096 的情况
  - 所以 Flash Attention 必须 Tiling（分块）：每次只把一小块送进 SRAM
  - Tiling 不是优化技巧，是让"不绕路 HBM"这件事在物理上可行的唯一手段

至此：
  实验1 → HBM 带宽是真实瓶颈（≈ 700-850 GB/s，而算力有 119 TFLOPS）
  实验2 → Memory-Bound 时 GPU 算力大量浪费，AI 越低浪费越严重
  实验3 → N×N IO 代价以平方速度膨胀，seq_len=8192 单次绕路就要搬 256MB
  实验4 → Flash Attention 省掉这个绕路，seq_len 越长加速越明显
  实验5 → Tiling 是物理必须，不是可选的工程优化
""")

print("=" * 65)
print("实验完成！现在再回头看 Flash Attention 的论文，数字应该有感觉了。")
print("=" * 65)
