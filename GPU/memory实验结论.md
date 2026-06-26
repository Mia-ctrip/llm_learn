=================================================================
GPU 内存层次实验  —  Flash Attention 前置感性认知
=================================================================
显卡型号 : NVIDIA L20
总显存   : 51.0 GB
SM 数量  : 92
L2 Cache : 101 MB
Compute  : 8.9

──────────────────────── 实验 1：实测 HBM 带宽 ─────────────────────────

原理：分配一大块显存，做 clone()（一次完整的读 + 写），
      根据 数据量 / 耗时 算出实际带宽。

数据量    : 2 GB (float16)
耗时      : 5.66 ms
实测带宽  : 758.5 GB/s
理论峰值  : 864 GB/s (L20)
利用率    : 87.8%

→ 小结：实测带宽通常能达到理论峰值的 80-95%。
  下面所有实验里，数据搬运速度就受限于这个数字。


──────────── 实验 2：Memory-Bound vs Compute-Bound 直觉建立 ────────────

核心概念：Arithmetic Intensity（算术强度） = FLOP / Byte
  - 低 AI（< ~100）→ Memory-Bound：GPU 在等数据，算力闲着
  - 高 AI（> ~100）→ Compute-Bound：GPU 算力打满，带宽还好

L20 的 "屋顶线" 拐点 AI ≈ (119 TFLOPS) / (0.864 TB/s) ≈ 138 FLOP/Byte

矩阵规模：8192 × 8192，float16

操作                         耗时    AI (FLOP/B)         带宽利用 备注
----------------------------------------------------------------------
逐元素加法                  0.570ms         0.167     706.6 GB/s  ← Memory-Bound
矩阵乘法                   9.816ms        2730.7         ~       ← Compute-Bound (112.0 TFLOPS)

直觉总结：
  矩阵乘法做了 16,384x 更多计算，却只慢了 17.2x
  → 说明 GPU 的计算能力极其充裕，"搬数据"才是稀缺资源
  → 逐元素加法的 AI=0.167 远低于拐点138，它的瓶颈就在 HBM 带宽


──────────────── 实验 3：Attention 中间矩阵 S/P 的 IO 代价 ────────────────

标准 Attention：S = QK^T 写到 HBM → P = softmax(S) 读 HBM 写 HBM → O = PV 读 HBM
每轮需要对 N×N 矩阵做 2~4 次 HBM 读写。
下面只测"把一个 N×N float16 矩阵写到 HBM 再读回来"的纯 IO 代价。

   seq_len       矩阵大小        写+读耗时         折算带宽  说明
------------------------------------------------------------
       256     0.1 MB      0.012 ms      22.8 GB/s  
       512     0.5 MB      0.011 ms      92.8 GB/s  
      1024     2.1 MB      0.011 ms     370.6 GB/s  
      2048     8.4 MB      0.011 ms    1483.6 GB/s  ← 注意增速
      4096    33.6 MB      0.039 ms    1741.7 GB/s  ← 注意增速
      8192   134.2 MB      0.708 ms     379.0 GB/s  ← 注意增速

→ 注意：耗时随 seq_len² 增长。seq_len 翻倍，IO 代价 ×4。
  这正是标准 Attention 不能用于长序列的根本原因。


───────── 实验 4：naive Attention vs Flash Attention 实测加速 ──────────

native_attention：显式把 S、P 写到 HBM（PyTorch 默认行为）
sdpa_attention  ：torch.nn.functional.scaled_dot_product_attention
                  → 需要 4D 张量 [batch, heads, seq, head_dim]
                  → PyTorch 2.0+ 在满足条件时自动调用 Flash Attention

注意：Flash Attention 强制要求 4D 输入。
      3D [batch, seq, dim] 会导致所有 fused kernel 报错退化到 Math backend。

配置：batch=4, num_heads=8, d_head=64
   seq_len   naive (ms)   SDPA/FA (ms)      加速比   S/P per head
--------------------------------------------------------------------
       512      0.056 ms        0.034 ms    1.67x        0.5 MB/head
      1024      0.390 ms        0.093 ms    4.18x        2.1 MB/head
      2048      2.128 ms        0.350 ms    6.07x        8.4 MB/head
      4096      8.646 ms        1.355 ms    6.38x       33.6 MB/head
      8192     32.732 ms        5.160 ms    6.34x      134.2 MB/head

→ 上面 seq_len 较小时 S/P 矩阵可能仍命中 L2 Cache（101MB），抹平了部分差距。
  见下方实验 4b：参数调大让矩阵彻底超出 L2。


─────────────────── 实验 4b：超出 L2 Cache 后的真实加速 ────────────────────

目标：让 S/P 矩阵超出 L2 Cache（101 MB），使 naive attention 真正打到 HBM。

显存占用估算（naive 同时持有 S 和 P，含所有 head）：
  显存 = 2 × batch × num_heads × seq_len² × 2B
  batch=1, heads=8, seq=4096  → 约 512 MB   ← 超出L2，安全
  batch=1, heads=8, seq=8192  → 约   2 GB   ← 安全
  batch=4, heads=8, seq=8192  → 约   8 GB   ← 安全（L20有48GB）

   seq_len  batch      S/P总显存   naive (ms)   SDPA/FA (ms)      加速比
--------------------------------------------------------------------
      4096      1      537 MB      2.303 ms        0.354 ms     6.51x
      8192      1     2147 MB      9.025 ms        1.368 ms     6.60x
      8192      4     8590 MB     32.729 ms        5.163 ms     6.34x
     16384      4    34360 MB  跳过（预计OOM）

→ 当 S/P 总量超出 L2（>101MB），naive 每步都必须真正打到 HBM，
  Flash Attention 省掉 HBM 绕路的收益才真正显现。


─────────────── 实验 5（加餐）：SRAM 约束 —— 为什么必须 Tiling ────────────────

L20 SRAM（共享内存）：
  每个 SM：128 KB
  SM 数量 ：92 个
  总计    ：11.5 MB（所有 SM 加起来，也不超过这个数）

N×N 矩阵（float16）占多少空间？

   seq_len       N×N 矩阵  vs SRAM 总量 11.5 MB
---------------------------------------------
       512       0.5 MB   ✓ 勉强能放
      1024       2.0 MB   ✓ 勉强能放
      2048       8.0 MB   ✗ 远超 SRAM
      4096      32.0 MB   ✗ 远超 SRAM

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

=================================================================
实验完成！现在再回头看 Flash Attention 的论文，数字应该有感觉了。
=================================================================