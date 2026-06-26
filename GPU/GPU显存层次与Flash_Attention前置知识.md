# GPU 显存层次与 Flash Attention 前置知识

> 学习日期：2026-06-26  
> 学习方式：在 NVIDIA L20 48G 上通过 5 组实验量化验证  
> 目标：在读 Flash Attention 论文之前，建立对 HBM / SRAM / IO 代价的感性认知

---

## 一、GPU 内存层次结构

GPU 内部有多个层次的存储，速度和容量呈反比，是所有性能优化的物理基础。

```
速度快 ◄──────────────────────────────────► 速度慢
容量小 ◄──────────────────────────────────► 容量大

  寄存器          SRAM           L2 Cache        HBM（显存）
(Register)    (共享内存)                        (High Bandwidth Memory)
  ~6 MB/SM    128 KB/SM         101 MB            48 GB
  最快         ~19 TB/s         ~4 TB/s           758 GB/s（实测）
  线程私有      SM 内共享         全卡共享           全卡共享
```

**L20 实测各层参数（本机数据）：**

| 层次 | 容量 | 带宽 | 访问范围 |
|------|------|------|---------|
| 寄存器 | ~6 MB/SM（per 线程私有） | 极快，无显式带宽 | 单线程私有 |
| SRAM（共享内存） | 128 KB/SM，92个SM共 11.5 MB | ~19 TB/s | 同一 SM 内线程共享 |
| L2 Cache | 101 MB | ~4 TB/s（推算） | 全卡共享，自动缓存 |
| HBM（显存） | 48 GB | **758 GB/s**（实测） | 全卡共享 |

### 关键规则

> **所有计算只能在 SRAM（寄存器 + 共享内存）里发生，数据必须先从 HBM 搬进来。**

```
HBM ──(读)──► SRAM ──(计算)──► SRAM ──(写)──► HBM
                  ↑
             真正干活的地方
```

HBM 的每一次读写都是 IO 开销，**IO 次数 = 速度瓶颈的根源**。

---

## 二、HBM 带宽是真实上限（实验 1）

**实测数据（L20，float16，2GB 数据 clone = 一次读 + 一次写）：**

| 指标 | 值 |
|------|-----|
| 实测带宽 | **758.5 GB/s** |
| 理论峰值 | 864 GB/s |
| 利用率 | 87.8% |

这个数字是后续所有分析的物理上限。不管 kernel 怎么写，每秒最多只能搬 758 GB 的数据。

---

## 三、Memory-Bound vs Compute-Bound（实验 2）

### Arithmetic Intensity（算术强度）

```
AI = FLOP / Byte（每搬一个字节数据，能做多少次浮点运算）
```

- **AI 低** → Memory-Bound：GPU 在等数据，算力大量闲置
- **AI 高** → Compute-Bound：算力打满，带宽反而够用

**L20 的 Roofline 拐点：**

```
AI_拐点 = 峰值算力 / HBM带宽 = 119 TFLOPS / 0.864 TB/s ≈ 138 FLOP/Byte

AI < 138  → Memory-Bound（被带宽卡住）
AI > 138  → Compute-Bound（被算力卡住）
```

### 实测对比（8192×8192，float16）

| 操作 | 耗时 | Arithmetic Intensity | 带宽利用 | 瓶颈 |
|------|------|----------------------|---------|------|
| 逐元素加法（a+b） | 0.570 ms | **0.167 FLOP/Byte** | 706.6 GB/s | Memory-Bound |
| 矩阵乘法（A@B） | 9.816 ms | **2730 FLOP/Byte** | — | Compute-Bound（112 TFLOPS） |

**关键直觉：矩阵乘法做了 16,384 倍更多的计算，却只慢了 17 倍。**

这说明：
- GPU 的计算资源（Tensor Core）极其充裕，"算"本身不是瓶颈
- 逐元素加法 AI=0.167，远低于拐点 138，HBM 带宽已经打满，算力全部闲着
- **"搬数据"才是稀缺资源**

---

## 四、L2 Cache 的存在：实验 3 的意外收获

实验 3 测"把一个 N×N 矩阵写到 HBM 再读回来"的 IO 代价时，出现了异常：

```
seq_len=2048  →  折算带宽 1483 GB/s  ← 超过了HBM峰值 758 GB/s ！
seq_len=4096  →  折算带宽 1741 GB/s  ← 更离谱
seq_len=8192  →  折算带宽  379 GB/s  ← 恢复正常
```

**带宽超过 HBM 峰值，意味着数据根本没走 HBM，而是命中了 L2 Cache。**

L20 有 101MB 的 L2 Cache：
- seq_len≤4096 的矩阵（≤33MB）→ 命中 L2 → 速度是 HBM 的 2~4 倍
- seq_len=8192 的矩阵（134MB）→ 超出 L2 → 真正打到 HBM → 带宽恢复正常

**完整的三层访问速度（从实验数据推算）：**

```
SRAM（共享内存） > L2 Cache（~4 TB/s） >> HBM（758 GB/s）

seq_len     矩阵大小    实际来源      折算带宽
256~4096    0.1~33 MB   L2 Cache     > 1000 GB/s
8192        134 MB      HBM          ~379 GB/s
```

> 这就是为什么 Flash Attention 对"中小 seq_len"的加速效果不明显：naive attention 的 S/P 矩阵还在 L2 里跑，没有真正暴露 HBM IO 的代价。

---

## 五、Attention 的 HBM IO 代价随 seq_len 平方膨胀（实验 3）

标准 Attention 的每一步都要把 N×N 中间矩阵落到 HBM：

```
S = QK^T  →  写 HBM（S 矩阵）
P = softmax(S)  →  读 HBM → 写 HBM（P 矩阵）
O = PV  →  读 HBM（P 矩阵）→ 写 HBM（O 矩阵）
```

**单次"写到 HBM 再读回来"的 IO 代价（单头，float16）：**

| seq_len | 矩阵大小 | IO 耗时 | 实际来源 |
|---------|---------|---------|---------|
| 256 | 0.1 MB | 0.012 ms | L2 Cache |
| 512 | 0.5 MB | 0.011 ms | L2 Cache |
| 1024 | 2.1 MB | 0.011 ms | L2 Cache |
| 2048 | 8.4 MB | 0.011 ms | L2 Cache |
| 4096 | 33.6 MB | 0.039 ms | L2 Cache |
| **8192** | **134.2 MB** | **0.708 ms** | **HBM（真正打到）** |

**seq_len 翻倍，矩阵大小 ×4，IO 代价 ×4（平方关系）。**

这就是标准 Attention 无法扩展到长序列的根本原因——每步的 HBM IO 代价以平方速度增长。

---

## 六、Flash Attention 的实测加速（实验 4/4b）

### PyTorch SDPA 的重要前提

`torch.nn.functional.scaled_dot_product_attention` 只有在输入为 **4D 张量** `[batch, num_heads, seq_len, head_dim]` 时才能触发 Flash Attention kernel。

```python
# ✓ 正确：4D，触发 Flash Attention
Q = torch.randn(batch, num_heads, seq_len, head_dim)

# ✗ 错误：3D，所有 fused kernel 报错，退化到 Math backend（和 naive 一样慢）
Q = torch.randn(batch, seq_len, dim)
```

### 实测加速（batch=4, num_heads=8, d_head=64）

| seq_len | naive (ms) | Flash Attn (ms) | 加速比 | S/P per head |
|---------|-----------|-----------------|--------|-------------|
| 512 | 0.056 | 0.034 | 1.67x | 0.5 MB（L2内） |
| 1024 | 0.390 | 0.093 | 4.18x | 2.1 MB |
| 2048 | 2.128 | 0.350 | **6.07x** | 8.4 MB |
| 4096 | 8.646 | 1.355 | **6.38x** | 33.6 MB |
| 8192 | 32.732 | 5.160 | **6.34x** | 134.2 MB |

### 超出 L2 后的干净对比（实验 4b）

| seq_len | batch | S/P 总显存 | naive (ms) | Flash Attn (ms) | 加速比 |
|---------|-------|----------|-----------|-----------------|--------|
| 4096 | 1 | 537 MB | 2.303 | 0.354 | **6.51x** |
| 8192 | 1 | 2147 MB | 9.025 | 1.368 | **6.60x** |
| 8192 | 4 | 8590 MB | 32.729 | 5.163 | **6.34x** |

### 为什么加速比在 seq≥2048 后稳定在 6x 而不继续增大？

Flash Attention 本身也要做 GEMM（QK^T 和 PV），这部分是 Compute-Bound 的固定代价，无法消除。加速比的上限由 `省掉的HBM IO / 总耗时` 决定，两者到达一定规模后等比增长，比值趋于稳定。

```
总耗时 = GEMM耗时（固定）+ HBM IO耗时（Flash Attention省掉的部分）
加速比 = 总耗时_naive / 总耗时_FA ≈ 稳定在 6x（L20 上）
```

---

## 七、Tiling 是物理必须，不是工程选项（实验 5）

如果想让 S/P 矩阵"不落 HBM"，直觉上似乎只需要在 SRAM 里处理完就行。但有一个硬约束：

**L20 每个 SM 只有 128 KB 的 SRAM，全卡 92 个 SM 加起来也只有 11.5 MB。**

| seq_len | N×N 矩阵（单头） | 能否放入 SRAM（11.5 MB 全卡） |
|---------|----------------|---------------------------|
| 512 | 0.5 MB | ✓ 勉强 |
| 1024 | 2.0 MB | ✓ 勉强 |
| 2048 | **8.0 MB** | ✗ 超出单头上限 |
| 4096 | **32.0 MB** | ✗ 远超全卡 SRAM |

**结论：seq_len ≥ 2048（生产环境常见范围）时，N×N 矩阵根本塞不进 SRAM。**

因此 Flash Attention 必须 **Tiling（分块）**：

```
每次只把一小块 Q/K/V 读入 SRAM
 → 在 SRAM 里计算这块的 S_block、P_block
 → 用完即丢（不写回 HBM）
 → 只把 O_block 写回 HBM

N×N 矩阵从未在 HBM 中出现
```

**Tiling 不是可选的优化技巧，是让"不绕路 HBM"这件事在物理上可行的唯一手段。**

---

## 八、知识链条总结

```
实验 1：HBM 实测带宽 758 GB/s，是搬运数据的物理上限
    ↓
实验 2：算力充裕（112 TFLOPS），AI < 138 时 GPU 在等数据，搬数据才是稀缺资源
    ↓
实验 3：L2 Cache 的存在（101MB）会掩盖 HBM IO 代价
        N×N IO 代价随 seq_len² 平方膨胀，seq=8192 单次 IO 已需 0.7ms
    ↓
实验 4/4b：Flash Attention 省掉 S/P 矩阵的 HBM IO，稳定实现 6x 加速
            seq_len 越大，S/P 越大，节省越显著
    ↓
实验 5：SRAM 只有 11.5 MB，装不下完整 N×N 矩阵
        Tiling 是唯一手段，让 S/P 分块留在 SRAM 里处理完即丢
```

---

## 九、对 Flash Attention 论文的启示

带着这些数字再看论文，以下几点会有具体感受：

| 论文说的 | 实验里的对应数字 |
|----------|---------------|
| "标准 Attention 是 Memory-Bound" | 实验2：逐元素加法 AI=0.167，带宽打满而算力闲置 |
| "HBM IO 是瓶颈" | 实验3：seq=8192 单次 IO 耗时 0.708ms，是整个 attention 耗时的主要部分 |
| "Flash Attention 显著加速" | 实验4：L20 上稳定 6x 加速 |
| "必须 Tiling 因为 SRAM 不够" | 实验5：全卡 SRAM 仅 11.5MB，seq≥2048 已装不下 |
| "Online Softmax 是 Tiling 的数学保障" | 待学习：分块时无法看到全局最大值，需要维护 running max `m` 和 running sum `l` |

---

## 十、待深入

- [ ] **Online Softmax**：Tiling 时如何正确计算 softmax（核心：维护 running max `m` 和 running sum `l`，每处理一个新块修正之前结果）
- [ ] **Flash Attention v2 的改进点**：更好的并行策略，减少 SM 间通信
- [ ] **反向传播的 Recompute**：为什么重算比存储 N×N 矩阵更划算（计算换显存）

---

> 实验环境：NVIDIA L20 48G，PyTorch 2.9.0+cu130，CUDA 13.0  
> 实验脚本：`GPU/gpu_memory_experiment.py`，`GPU/diagnose_sdpa_backend.py`  
> 原始数据：`GPU/memory实验结论.md`
