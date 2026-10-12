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

## 一-b、GPU 计算单元：CUDA Core vs Tensor Core

理解 GPU 的两种计算单元，是分析性能瓶颈和选择优化策略的前提。

### CUDA Core —— 通用计算单元

CUDA Core 是 GPU 的**基础计算单元**，负责所有通用浮点/整数运算。

**擅长的计算类型：**
- 逐元素运算（element-wise）：`x + y`、`relu(x)`、`x * 0.5`
- 归约运算（reduction）：`sum()`、`mean()`、`softmax()` 中的 exp 和 sum
- 小尺寸矩阵/向量运算
- 任何**不规则、非矩阵乘法**的计算

**本质**：每个 CUDA Core 每个时钟周期执行一次 FP32 乘加（FMA）运算，是标量级别的计算。GPU 的绝大多数晶体管都是 CUDA Core。

### Tensor Core —— 矩阵乘法专用加速单元

Tensor Core 是专门为**矩阵乘加（MMA, Matrix Multiply-Accumulate）**设计的硬件单元。

**擅长的计算类型：**
- **矩阵乘法**（GEMM）：`C = A × B + C`
- 一个 Tensor Core 每个时钟周期完成一个小矩阵块的乘加（如 4×4 × 4×4），等效于一次完成 64 次乘加运算

**不适用的计算：**
- softmax、LayerNorm、激活函数等**非矩阵乘法**操作
- 逐元素运算、条件分支、归约运算

### 核心区别对比

| 维度 | CUDA Core | Tensor Core |
|------|-----------|-------------|
| 计算粒度 | 标量（一次一个乘加） | 矩阵块（一次几十个乘加） |
| 擅长 | 通用计算、逐元素、归约 | 矩阵乘法（GEMM） |
| 不擅长 | 大规模矩阵乘法（慢） | 非矩阵乘法运算 |
| 精度支持 | FP32 / FP64 / INT32 | FP16 / BF16 / TF32 / INT8 / FP8 |
| 典型占比 | GPU 的绝大多数晶体管 | 少量专用单元 |

### LLM 训练/推理中的计算分配

以 GPT 模型的一次前向传播为例：

```
Embedding Lookup          → CUDA Core（查表 + 缩放）
Q/K/V Linear (x @ W)      → ⚡ Tensor Core（矩阵乘法）
Q @ K^T (attention scores) → ⚡ Tensor Core（矩阵乘法）
Softmax                    → CUDA Core（exp + 归一化）
attn @ V                   → ⚡ Tensor Core（矩阵乘法）
FFN Linear layers          → ⚡ Tensor Core（矩阵乘法）
GELU 激活                  → CUDA Core（逐元素）
LayerNorm                  → CUDA Core（均值/方差 + 归一化）
Cross Entropy Loss         → CUDA Core（log + 归约）
```

**矩阵乘法部分 → Tensor Core（条件满足时）**，**其他所有操作 → CUDA Core**。

### Tensor Core 被触发的条件

需要**同时满足**以下条件：

1. **运算是矩阵乘法**（GEMM 类型）
2. **矩阵维度是特定值的倍数**：
   - FP16/BF16：维度需为 8 或 16 的倍数
   - TF32（FP32 输入）：维度需为 4 的倍数
3. **矩阵规模足够大**：太小的矩阵，Tensor Core 的启动开销（数据重排、tile 划分）反而比直接算更慢，cuBLAS 会自动选择不用 Tensor Core
4. **cuBLAS / cuDNN 选择了 Tensor Core kernel**：必须走标准算子路径（如 `nn.Linear`、`torch.matmul`、`F.scaled_dot_product_attention`），手写算子可能绕过优化路由

### Tensor Core 不被使用的常见场景

| 场景 | 原因 |
|------|------|
| 模型太小（如 embed=256, batch=32） | 矩阵规模不够，开销 > 收益，cuBLAS 选择普通 kernel |
| 非矩阵乘法运算（softmax、norm、激活） | 硬件上不支持 |
| 维度不是对齐倍数 | kernel 无法正确分 tile |
| 手写算子绕过了 cuBLAS 路由 | 优化器没机会选择 Tensor Core kernel |
| CPU 模式 | Tensor Core 只存在于 GPU |

### 实操案例：mini_gpt 训练时 Tensor Core 活跃度为 0

在 mini_gpt 手写训练脚本（embed=256, heads=4, batch=32, seq=128）中，DCGM 监控显示 Tensor Core 活跃度为 0，原因分析：

1. **模型太小**：最大矩阵才 32×128×256，cuBLAS 判定 Tensor Core 启动开销不划算，选择了 CUDA Core 路径
2. **手写 attention**：`torch.matmul(q, k.T)` + `torch.softmax()` 的手动拼接，没有走 `F.scaled_dot_product_attention`（SDPA）优化路径，SDPA 是触发 Flash Attention / Tensor Core kernel 的主要入口
3. **纯 FP32**：Tensor Core 在 FP16/BF16 下效率最高，FP32 下收益有限

**解决方式**：
- 改用 `F.scaled_dot_product_attention` 替代手写 attention
- 启用混合精度训练（AMP，`torch.amp.autocast`）
- 增大模型规模（embed ≥ 512，batch ≥ 64）

> **与实验 2 的关联**：实验 2 中 8192×8192 矩阵乘法达到 112 TFLOPS，正是 Tensor Core 全力运转的结果。而 mini_gpt 的微小矩阵远未达到这个规模。

---

## 一-c、训练精度：FP32、FP16 与 AMP 混合精度

理解训练时的数值精度，是分析训练稳定性、显存占用和速度的基础。

### PyTorch 默认精度是 FP32

不指定精度时，模型参数、梯度、优化器状态全部为 `torch.float32`（FP32）。这是最稳定的默认选择。

### FP32 vs FP16 数值范围对比

```
             FP32                          FP16
最大值：  3.4 × 10^38                   65,504
最小正规数：1.2 × 10^-38                6.0 × 10^-5
精度位数：  23 bit 尾数                   10 bit 尾数
```

**FP16 的最大值只有 65504，这是一个非常小的数。**

### 从 FP32 切到 FP16 的两种风险

**上溢（Overflow）**：FP32 下完全正常的值，在 FP16 下超出最大值 65504 → 变成 `inf`

```python
# Attention 点积中某个中间值
score = 80000.0   # FP32：完全正常，远小于 3.4×10^38
                  # FP16：❌ 超过 65504 → 变成 inf

# softmax 前的 logit
logit = 12.0
exp(12.0) = 162754  # > 65504 → FP16 下溢出为 inf → loss = NaN
```

容易触发上溢的地方：
- Attention 的点积 `Q @ K^T`（head_dim 大时累加值可达数万）
- `softmax` / `cross_entropy` 内部的 `exp()` 操作

**下溢（Underflow）**：很小的梯度在 FP16 下直接变成 0

```python
grad = 1e-6  # FP32：完全正常（FP32 最小 1.2e-38）
             # FP16：❌ 小于 6.0e-5 → 变成 0（梯度消失）
```

### 纯 FP16 训练为什么不可行

```python
model.half()  # 简单地把所有参数转为 FP16
# 问题：
# 1. 中间激活值可能上溢（inf）
# 2. 微小梯度下溢（变成 0）
# 3. 优化器状态也用 FP16，参数更新累积误差
# → loss 容易 NaN，训练崩溃
```

### AMP 混合精度：工业界标准解法

AMP（Automatic Mixed Precision）的核心思路：**计算时用 FP16 提速，但关键数值用 FP32 保护。**

**两个核心组件：**

```
1. torch.amp.autocast('cuda', dtype=torch.float16)
   - Linear / matmul → 自动用 FP16（速度快，Tensor Core 发力）
   - LayerNorm / softmax / cross_entropy → 自动保持 FP32（防止溢出）

2. torch.amp.GradScaler('cuda')
   - scaler.scale(loss)：将 loss 放大（默认 ×65536）
   - 反向传播：梯度也跟着放大，微小梯度回到 FP16 可表达范围
   - scaler.step(optimizer)：更新前自动缩放梯度回来
   - scaler.update()：动态调整缩放因子
```

**AMP 的工作流程：**

```
前向传播：FP16 计算 → 速度快，显存省一半
反向传播：FP16 梯度 → 速度快
参数更新：GradScaler 放大梯度防止下溢
         optimizer.step() 用 FP32 主权重更新 → 数值稳定
```

### AMP 改造代码（只需 3 处改动）

```python
# ═══════════════════ FP32 原版 ═══════════════════
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for x, y in data_loader:
    x, y = x.to(device), y.to(device)
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ═══════════════════ AMP 混合精度版 ═══════════════════
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler('cuda')                          # ← 新增

for x, y in data_loader:
    x, y = x.to(device), y.to(device)
    with torch.amp.autocast('cuda', dtype=torch.float16):       # ← 包裹前向传播
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    optimizer.zero_grad()
    scaler.scale(loss).backward()                               # ← 替换 loss.backward()
    scaler.step(optimizer)                                      # ← 替换 optimizer.step()
    scaler.update()                                             # ← 新增
```

### AMP 注意事项

| 注意点 | 说明 |
|--------|------|
| **不需要 `model.half()`** | AMP 自动在需要时用 FP16，模型参数始终保持 FP32 |
| **不需要改模型代码** | 模型文件完全不需要动 |
| **学习率不用改** | `scaler` 会自动处理梯度缩放 |
| **loss.item() 不受影响** | 返回的是原始 loss 值，缩放不影响读取 |
| **loss 出现 NaN 时** | `scaler` 自动跳过该次更新，不会崩溃 |

### AMP 的效果

| 维度 | FP32 | AMP（FP16 混合） | 说明 |
|------|------|-------------|------|
| 训练速度 | 基准 | **快 1.5~2 倍** | 矩阵乘法走 FP16 Tensor Core |
| 显存占用 | 基准 | **省约 40~50%** | 激活值、梯度用 FP16 |
| 模型保存大小 | 基准 | **不变** | 参数存盘仍是 FP32 |
| 训练效果 | 基准 | **几乎一致** | 主权重始终 FP32，差异 < 0.1% |
| 泛化能力 | 基准 | **有时更好** | FP16 的轻微噪声有正则化效果 |

**显存节省明细：**

```
模型参数：  不变（仍 FP32）
梯度：    减半（FP16）
激活值：  减半（FP16）
优化器状态：不变（AdamW 的动量/方差仍 FP32）
────────────────────────────────
总体：约节省 40~50% 显存
```

### AMP 实操验证结果（DCGM 监控）

在 mini_gpt 训练上实测，从 FP32 切换到 AMP 后，DCGM 监控指标变化：

| 指标 | FP32 训练 | AMP 训练 | 原因 |
|------|-----------|----------|------|
| `PROF_PIPE_FP16_ACTIVE` | 0 | ✅ 有数据 | `autocast` 让 Linear/matmul 走 FP16 |
| `PROF_PIPE_FP32_ACTIVE` | 较高 | ✅ 大幅下降 | 大部分计算转移到 FP16，FP32 只剩 LayerNorm/softmax 等 |
| **Tensor Core 利用率** | 0 | ✅ 有利用率 | FP16 矩阵乘法触发了 Tensor Core |
| **SM 占用率** | 较低 | ✅ 35~50% | 显存省出后可以增大 batch_size，更多线程块填满 SM |
| **显存拷贝引擎** | 较低 | ✅ 利用率上升 | batch_size 增大后每个 batch 搬运更多数据 |
| **训练速度** | 基准 | ✅ 明显提升 | FP16 计算快 + batch_size 更大 + Tensor Core 介入 |

**为什么 SM 只有 35~50% 而没有接近 100%？**

```
SM 执行计算（忙）──► 等待 HBM 数据（闲）──► 执行计算（忙）──► ...
                     ↑
              这段时间 SM 是空的
```

小模型中 softmax、LayerNorm、GELU、cross_entropy 等 Memory-Bound 操作占比较高，SM 在等数据搬运，无法打满。生产环境大模型（如 LLaMA）因为矩阵规模巨大，Compute-Bound 比例高，SM 才能接近 80~95%。

**常见报错及修复：**

```python
# ❌ 报错：FP16 下 -1e9 超出最大值 65504，触发上溢
attn = attn.masked_fill(mask == 0, -1e9)
# RuntimeError: value cannot be converted to type at::Half without overflow

# ✅ 修复：用 float('-inf') 替代，所有精度都支持 inf 特殊值
attn = attn.masked_fill(mask == 0, float('-inf'))
# 经过 softmax 后 exp(-inf) = 0，效果与 -1e9 完全一致
```

### 一句话总结

> **AMP 混合精度 = 训练时用 FP16 加速计算、省显存，但用 FP32 保存主权重保证精度。模型效果不变，速度快一倍，显存省一半。**

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
计算单元：CUDA Core 处理通用计算，Tensor Core 处理矩阵乘法
        小模型/手写算子 → Tensor Core 不触发，全走 CUDA Core
        大模型 + SDPA + FP16 → Tensor Core 全力运转（112 TFLOPS）
    ↓
训练精度：默认 FP32，FP16 有上溢/下溢风险
        AMP 混合精度：FP16 计算提速，FP32 保护主权重
        autocast 控制哪些算子用 FP16，GradScaler 防止梯度下溢
    ↓
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
| “标准 Attention 是 Memory-Bound” | 实验2：逐元素加法 AI=0.167，带宽打满而算力闲置 |
| “HBM IO 是瓶颈” | 实验3：seq=8192 单次 IO 耗时 0.708ms，是整个 attention 耗时的主要部分 |
| “Flash Attention 显著加速” | 实验4：L20 上稳定 6x 加速 |
| “必须 Tiling 因为 SRAM 不够” | 实验5：全卡 SRAM 仅 11.5MB，seq≥2048 已装不下 |
| “Online Softmax 是 Tiling 的数学保障” | 待学习：分块时无法看到全局最大值，需要维护 running max `m` 和 running sum `l` |
| “SDPA 是 Tensor Core 的触发入口” | 手写 matmul 链不走 SDPA，Tensor Core 活跃度为 0；改用 SDPA + FP16 后可触发 |

---

## 十、待深入

- [ ] **Online Softmax**：Tiling 时如何正确计算 softmax（核心：维护 running max `m` 和 running sum `l`，每处理一个新块修正之前结果）
- [ ] **Flash Attention v2 的改进点**：更好的并行策略，减少 SM 间通信
- [ ] **反向传播的 Recompute**：为什么重算比存储 N×N 矩阵更划算（计算换显存）
- [x] **AMP 实操验证**：在 mini_gpt 上对比 FP32 vs AMP 的训练速度、显存占用、最终 Loss 差异 —— 已验证，DCGM 监控指标变化见「AMP 实操验证结果」
- [ ] **BF16 vs FP16**：BF16 数值范围与 FP32 相同（无溢出风险），但精度更低，什么场景下选 BF16

---

> 实验环境：NVIDIA L20 48G，PyTorch 2.9.0+cu130，CUDA 13.0  
> 实验脚本：`GPU/gpu_memory_experiment.py`，`GPU/diagnose_sdpa_backend.py`  
> 原始数据：`GPU/memory实验结论.md`
