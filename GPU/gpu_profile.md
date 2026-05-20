# RTX PRO 5000 Blackwell — GPU 档案

容器探查日期：2026-05-18
Driver：580.126.09（open kernel module） · CUDA Runtime：13.0 · nvcc 13.0.88

---

## 1. 身份与架构

| 项 | 值 |
|---|---|
| 产品名 | NVIDIA RTX PRO 5000 Blackwell（工作站 / Pro Vis 线） |
| 架构 | **Blackwell**（第 5 代 Tensor Core，第 4 代 RT Core，GDDR7） |
| Compute Capability | **12.0 (sm_120)** |
| GPU UUID | GPU-0729f2a8-e8d9-8e2b-1382-e03325d53ab1 |
| Board P/N | 900-5G153-0050-000 · GPU P/N 2BB3-850-A1 |
| VBIOS | 98.02.A5.00.02 · GSP firmware 580.126.09 |
| 虚拟化 | Pass-Through（直通模式） |

**命名说明**：Blackwell 首发是 2024 年的数据中心 B100 / B200（sm_100 / sm_103）。我们这块"RTX PRO Blackwell"是第二波的 **工作站 / Pro Vis 线**，使用 GB20x 系列芯片，**sm_120**。两条线虽然都叫 Blackwell，但 FP4、第二代 Transformer Engine 的硬件细节不完全一致——网上看 "Blackwell FP8 性能" 时一定要分清是 B200 还是 sm_120。

**GSP firmware**：Blackwell 必须使用 **GSP-RM 驱动模式**——GPU 内有 RISC-V 协处理器跑 GSP firmware，主机 driver 只是个薄壳，所以**必须用 open kernel module**，老的闭源 driver path 不再支持。这是用户提到的"必须 open kernel"的根本原因。

## 2. 计算资源（来自 cudaGetDeviceProperties + 属性查询）

| 项 | 值 |
|---|---|
| **SM 数量** | **110** |
| CUDA core / SM | 128（FP32 ALU） |
| **CUDA core 总数** | **14 080** |
| Tensor Core / SM | 4（第 5 代） |
| **Tensor Core 总数** | **440** |
| Warp size | 32 |
| 每 block 最大线程数 | 1024 |
| 每 SM 最大线程数 | 1536 |
| 每 SM 最大 block 数 | 24 |
| 每 SM 寄存器数 | 65 536（32-bit，即 256 KB） |
| Shared memory / block | 48 KB 默认 · 99 KB opt-in |
| Shared memory / SM | 100 KB |
| **L2 cache** | **96 MB** ← Blackwell sm_120 的关键卖点 |
| Boost SM 时钟 | 2 377 MHz（supported_clocks 表里最高 3 090 MHz） |

**SM 是什么**：把 GPU 想成一个工厂，**每个 SM 就是一条独立的生产线**，里面有自己的 CUDA core、Tensor Core、寄存器堆、shared memory、warp scheduler。我们这卡 110 条线同时干活。RTX PRO 6000 Blackwell 是 188 SM 满血版，5000 Pro 砍了约 41% 是这一档定位。

**CUDA core**：每个 SM 里 128 个 FP32 ALU——做标量加减乘除的"普通"计算单元。注意 A100/H100 是 64/SM（数据中心更看重 Tensor Core），Ada / Blackwell 工作站系是 128/SM。

**Tensor Core**：不是"一次一个数"的标量单元，而是**一次做一整块小矩阵乘加（MMA）**。一个 Tensor Core 一拍可以做类似 16×8×16 的矩阵乘。这就是为什么 BF16 比 FP32 快 5 倍——不是带宽差异，而是吞吐密度根本不同。第 5 代 Tensor Core 新增原生 **FP8 (E4M3/E5M2)、FP6、FP4**，配合 block scaling（MXFP8 / NVFP4）。

**Warp = 32 线程**：GPU 的最小调度单位，32 个线程同进同出。所有 GPU 性能优化的起点都是"对齐到 32 的倍数"。

**寄存器 65536/SM**：决定一个 kernel 能开多少线程——例如每线程用 64 个寄存器时，一个 SM 能塞 1024 线程（≈32 个 warp）。是写 CUDA kernel 时的核心 trade-off。

**L2 = 96 MB**：上一代 Ada L20 是 60 MB，数据中心 Blackwell B200 也是 60 MB/die。**96 MB 是 sm_120 这一代相对的优势**。意义：LLM 推理时一层激活、一段 KV-cache 可以完全留在 L2，下层不用再读 DRAM——后面带宽测试的"16MB 块跑出 2042 GB/s"就是 L2 命中速度。

## 3. 显存子系统

| 项 | 值 |
|---|---|
| 总显存 | 47.27 GiB 可用（48 935 MiB 报告，534 MiB ECC 预留） |
| 总线宽度 | 384-bit |
| 显存类型 | **GDDR7** |
| 显存时钟 | 14 001 MHz（内部命令时钟，GDDR7 实际数据速率是其倍数） |
| 理论带宽（GDDR5/6 公式估算） | 1 344 GB/s ← 此公式对 GDDR7 PAM3 是低估值，真实理论上限更高 |
| **实测峰值 D2D 带宽（大块拷贝）** | **~555 GB/s 持续** = **~1110 GB/s** DRAM 流量（read+write）。约 83% 理论效率 |
| 实测 L2 命中（16 MB 块） | 2042 GB/s（数据完全装进 96 MB L2） |
| ECC | **已开启**（约消耗 1% 容量与一部分带宽） |
| BAR1 | 64 GB（整张卡的显存全部可被 CPU 地址空间映射） |

**ECC 决策**：训练 / 长跑场景强烈建议保留 ECC，避免单 bit flip 误差悄悄污染权重。需要榨性能再考虑 `nvidia-smi -e 0` 关闭。

**BAR1 = 64 GB**：普通消费卡只有 256 MB BAR1，64 GB 是企业级标配。意义：GPUDirect Storage、cuStreamWriteValue、vLLM 中 KV-cache mmap 等都依赖大 BAR1。

## 4. PCIe 与互联

| 项 | 值 |
|---|---|
| PCIe 槽位 | **Gen 5 × 16**（最大），P8 idle 时降为 Gen1，负载下自动升档 |
| NVLink | **无**（工作站 / Pro Vis 单卡线产品） |
| CPU NUMA 亲和 | node 0，CPU 0–15 |
| H2D pinned（大块） | **56.9 GB/s** ← 验证 PCIe Gen5 实际工作 |
| D2H pinned | 56.6 GB/s |
| H2D pageable | 47–53 GB/s（比 pinned 慢 10–15%） |
| D2H pageable | ~36 GB/s（driver 内部 staged copy） |

**对照 L20**：L20 是 PCIe **Gen4 × 16**，理论带宽 ~32 GB/s，实测约 26-28 GB/s。**5000 Pro 在 host↔device 数据装入这一层快接近一倍**，对训练数据加载、推理首 token 延迟有直接影响。

**实践提醒**：任何吞吐敏感的代码路径**必须**使用 pinned host buffer（`cudaMallocHost` / DataLoader `pin_memory=True`）。

## 5. 计算吞吐（cuBLAS GEMM 实测，方阵 M=N=K，FP32 累加器）

| 尺寸 | FP32 | TF32 | FP16 | BF16 |
|------|------|------|------|------|
| 1024 |  29.9 |  65.9 | 127.1 | 128.3 |
| 2048 |  45.5 |  99.6 | 209.4 | 208.5 |
| 4096 |  50.0 | 129.0 | 245.6 | 244.8 |
| 8192 | **51.9** | **138.2** | **255.1** | **253.2** |

（单位 TFLOPS = 2·M·N·K / 单次耗时）

**怎么读这张表**：
- **FP32 → TF32 = 2.7×**：从 ALU 切到 Tensor Core，且尾数从 23-bit 降到 10-bit（19-bit 总宽），吞吐提升 2.7 倍
- **TF32 → BF16 = 1.85×**：同样 Tensor Core，从 19-bit 降到 16-bit，吞吐再翻一倍
- **FP16 ≈ BF16**：第 5 代 Tensor Core 上两者吞吐一致——意味着**不必为吞吐选 FP16，BF16 数值范围更稳就用 BF16**

**FP32 ~52 TFLOPS** = 2 × 14 080 × 2.377 GHz × 0.77 效率，与理论 ~67 TFLOPS（满频满 issue）相比 77% 效率，是健康水平。

NVIDIA 官网标称 5000 Pro 的 **dense BF16 Tensor TFLOPS 约 380**（with sparsity 760）。我们实测 253 是 **dense 实际可达值**——cuBLAS 默认不用 sparsity，对得上。

**FP8 (E4M3) 在 cuBLASLt 上 NOT_SUPPORTED** ← 唯一异常信号。CUDA 13 在 sm_120 上的 FP8 GEMM 似乎只接受 **block-scaled (MXFP8)** 配置而非简单 per-tensor scale，或需要走 TransformerEngine。**这条留到 Stage 2 用 PyTorch `torch._scaled_mm` 重新验证。**

## 6. 功耗与散热

| 项 | 值 |
|---|---|
| 默认功耗上限 | **300 W**（最低 250 W，最高 300 W，调节范围窄） |
| Idle | P8 状态，~5 W，风扇 30%，26 °C |
| 热降频温度阈值 | T.Limit 66 °C 以上开始（当前余量大） |
| Persistence Mode | 已开启（避免每次 nvidia-smi 调用都重启 GPU 上下文） |

**对照 L20**：L20 TDP 275 W，5000 Pro 多 25 W。功耗-性能比要在 Stage 3 实测下评估。

## 7. Open kernel driver 状态总览

- Driver 580.126.09 与 GSP firmware 580.126.09 版本一致（无 mismatch）
- ECC 启用，volatile / aggregate 错误计数全 0
- 无 remapped rows、无 retired pages、无 pending channel/TPC repair → 显存无坏块迹象
- nvidia-smi、cudaGetDeviceProperties、cuBLAS GEMM (FP32/TF32/FP16/BF16) **全部正常**

**结论**：静态层面 driver 表现完好。**唯一异常是 cuBLASLt FP8 路径不支持简单 scale 配置**——但大概率不是 driver 问题（GPU 已被正确识别为 sm_120），而是 cuBLAS API 在 sm_120 上要求新的 block-scaled 配置。Stage 2 会换路径复测确认。

## 8. 留给 Stage 2 的待验证问题

1. `torch._scaled_mm` 在 sm_120 上能否跑通 FP8 GEMM？
2. cuDNN 9.x 是否包含 sm_120 的 conv 预编译 kernel，还是会 PTX JIT fallback？
3. PyTorch stable cu128 是否带 sm_120 二进制？没带的话首个 matmul 会因 PTX JIT 显著变慢——这本身就是验证项。
4. SDPA / FlashAttention 路径在 cap=(12,0) 上选哪个 backend？
5. 长跑训练是否触发 ECC correctable 错误（早期坏块预警信号）？

## 9. 关键比较参数（为 Stage 3 与 L20 对比预备）

| 参数 | 5000 Pro Blackwell (sm_120) | L20 Ada (sm_89) |
|---|---|---|
| SM 数 | 110 | 92 |
| CUDA core | 14 080 | 11 776 |
| Tensor Core | 440（5th gen，原生 FP8/FP6/FP4） | 368（4th gen，原生 FP8） |
| L2 | **96 MB** | 60 MB |
| 显存 | 48 GB **GDDR7** 384-bit | 48 GB **GDDR6** 384-bit |
| 理论显存带宽 | ~1.34 TB/s（公式低估） | 0.864 TB/s |
| PCIe | **Gen5 × 16**（~64 GB/s） | Gen4 × 16（~32 GB/s） |
| TDP | 300 W | 275 W |
| BF16 dense TFLOPS（实测/标称） | 253 / 380 | 待测 / 119 |
| FP8 dense TFLOPS（标称） | 待 stage2 验证 / 760 | 待测 / 239 |

预期结论：**5000 Pro 在显存带宽（GDDR7）、L2 容量、PCIe 带宽、低精度（FP8/FP4）四点上优势明显**；BF16/FP16 计算约 2× 优势；FP32 优势相对小。Stage 3 实测验证。
