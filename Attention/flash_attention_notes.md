# Flash Attention 学习笔记

> 学习日期：2026-06-25  
> 前置知识：Transformer 架构、Attention 计算、KV Cache、Prefill/Decode 阶段

---

## 一、核心问题：标准 Attention 的瓶颈在哪里？

标准 Attention 的计算流程：

```
Q, K, V  →  S = QK^T  →  P = softmax(S)  →  O = PV
  [小]        [N×N 大]     [N×N 大]           [小]
              ↑ 写 HBM      ↑ 写 HBM
              ↓ 读 HBM      ↓ 读 HBM
```

**瓶颈不是 Q/K/V，而是 S 和 P 这两个 N×N 矩阵：**
- seq_len = 2048 时：N×N = 16MB
- seq_len = 4096 时：N×N = 64MB
- seq_len = 8192 时：N×N = 256MB

这类操作属于 **Memory-Bound**：GPU 算力充足，但大量时间耗在等数据从 HBM 搬运。

---

## 二、GPU 内存层次背景知识

| | SRAM（片上缓存） | HBM（显存）|
|--|----------------|-----------|
| 容量 | 几十 KB per SM | 几十 GB |
| 带宽 | ~19 TB/s | ~2 TB/s |
| 速度 | **快约 10 倍** | 慢 |
| 类比 | CPU 的 L1 Cache | CPU 的内存条 |

**关键规则：所有计算只能在 SRAM 里发生，数据来自 HBM。**

```
HBM → (读) → SRAM → (计算) → SRAM → (写) → HBM
                ↑
           真正干活的地方
```

HBM 读写 = IO，IO 次数 = 速度瓶颈。

---

## 三、Flash Attention 的核心思路

### 目标
让 N×N 的中间矩阵（S 和 P）永远不出现在 HBM 里。

### 为什么需要切片（Tiling）？

直觉上"不回写 HBM"不需要切片，但有一个物理约束：
> **SRAM 只有几十 KB，根本装不下整个 N×N 矩阵。**

所以必须切片：每次只把一小块 Q/K/V 放进 SRAM，在 SRAM 里算完这块结果，直接丢弃中间矩阵，只把输出 O 写回 HBM。

```
Flash Attention 数据流：
  HBM                        SRAM
  Q_block ──读──►
  K_block ──读──►   在 SRAM 里计算 S_block、P_block（用完即丢）
  V_block ──读──►   
                  ──写──► O_block 回 HBM
  
  N×N 矩阵从未落到 HBM
```

**切片是"不回写 HBM"这个目标的唯一实现手段，因为 SRAM 装不下完整矩阵。**

### 切片带来的两个收益（同一机制的两面）

| 收益 | 原因 |
|------|------|
| 速度提升 | HBM IO 次数大幅减少，Memory-Bound 缓解 |
| 显存减少 | N×N 矩阵不需要在 HBM 中分配空间 |

---

## 四、训练 vs 推理的收益对比

Flash Attention 分两个部分：

| 部分 | 作用于 | 核心手段 | 主要收益 |
|------|--------|---------|---------|
| 前向传播 Tiling | **训练 + 推理** | 分块计算，S/P 不落 HBM | 速度提升 |
| 反向传播 Recompute | 只有训练 | backward 时重算 attention，不存 N×N | 显存大幅降低 |

**反向传播的逻辑：**
- 没有缓存 N×N 矩阵 → backward 时需要重新计算
- 但：GPU 的重新计算耗时 < HBM 读写耗时（计算换 IO，净收益为正）
- 主要收益是**省显存**（O(N²) → O(N)），而非省时间

**综合对比：**
- **推理**：主要收益是速度（前向 Tiling IO 减少）
- **训练**：速度 + 显存双收益，显存收益更革命性（使超长序列训练成为可能）

---

## 五、待学习

- [ ] Online Softmax：分块计算时如何正确算出 softmax（不需要全局最大值）
  - 核心：维护 running max `m` 和 running sum `l` 两个跑动变量
  - 每处理一个新块，修正之前的结果，最终等效于看过全部数据

---

## 六、关键问答记录

**Q：Flash Attention 减少的是训练反向传播的耗时吗？**  
A：不完全是。提速主要来自前向传播的 Tiling（训练和推理都有）。反向传播的 Recompute 主要是省显存，时间上有轻微代价（多算一遍），但 IO 节省更多，综合仍有收益。

**Q：切片是不是只为节省显存，和提速无关？**  
A：不对。切片本身就是减少 HBM IO 的手段——通过切片让数据在 SRAM 里处理完，不绕路去 HBM，IO 少了自然提速。显存节省和速度提升是同一个机制的两个结果。

**Q：不回写 HBM 不需要切片也能实现吧？**  
A：想法对，但忘了 SRAM 的物理限制——SRAM 只有几十 KB，根本装不下完整的 N×N 矩阵。切片是唯一能让"不回写 HBM"可行的方式。

---

> 参考论文：  
> - Flash Attention v1: https://arxiv.org/abs/2205.14135  
> - Flash Attention v2: https://arxiv.org/abs/2307.08691
