# GPU 监控

> 后续学习。本文档先存放为参考资料，目前**不深入展开**。
> 上下文：阶段 1 学习 nvidia-smi 时延伸出来的话题——`nvidia-smi -q` 是瞬时快照，要分析训练/推理效果需要时序埋点。

---

## 1. 核心问题：nvidia-smi -q 是瞬时快照

每次跑 `nvidia-smi -q` 是一次**当下的查询**——driver 把那一刻的寄存器值读出来给你。下一秒训练 step 切到下一个 kernel，数字全变了。

直接用它分析训练效果有三个致命问题：

| 问题 | 为什么坑 |
|---|---|
| **采样率太低** | 5 秒跑一次，但训练 step 可能 50 ms。99% 的事件你看不见 |
| **没有时间维度** | 单点数字告诉不了你"功耗一直在 280W 还是偶尔抖到 320W" |
| **手动 grep 没法做曲线** | `-q` 是给人读的，不是给监控吃的 |

所以**生产监控不会用 `-q`**——会用下面这些有时序结构的工具/接口。

---

## 2. 监控工具的"采集层级"

按从轻到重排：

### 2.1 `nvidia-smi --query-gpu=... --format=csv` —— 轻量埋点

`-q` 的 CSV 兄弟接口，专为脚本设计：

```bash
nvidia-smi \
  --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu,clocks_throttle_reasons.active \
  --format=csv \
  -l 1
```

`-l 1` = 每秒采样一次，stdout 输出 CSV。重定向到文件就是最便宜的埋点方案：

```bash
nvidia-smi --query-gpu=... --format=csv -l 1 > train_$(date +%s).csv &
```

训完 `kill %1`，CSV 用 pandas 画图。**适合单机一次性 benchmark**。

### 2.2 `nvidia-smi dmon` —— 多卡多列

```bash
nvidia-smi dmon -s pucvmet -d 1
```

`-s` 是 select：

| 字母 | 含义 |
|---|---|
| `p` | power |
| `u` | utilization |
| `c` | clocks |
| `v` | power violation（throttle 计数器） |
| `m` | memory |
| `e` | ECC errors |
| `t` | pcie throughput |

这个工具**就是给采样设计的**，输出已经是固定列宽好解析的格式，**多卡时一行一卡**。

### 2.3 DCGM —— 数据中心标配 ⭐

`nvidia-smi` 看不到的关键指标，DCGM 都有：

| 指标 | 含义 |
|---|---|
| **SMACT** (SM Active) | 真正的"SM 利用率"，比 GPU-Util 准 100 倍 |
| **SMOCC** (SM Occupancy) | SM 内 warp 占用率 |
| **TENSO** (Tensor Active) | Tensor Core 真实用了多少 |
| **DRAMA** (DRAM Active) | 显存带宽实际占用 |
| **PCIE TX/RX** | PCIe 实际吞吐 |
| 各种 NVLINK 指标 | NVLink 链路占用 |

DCGM 用法：

```bash
# 实时看
dcgmi dmon -e 203,1002,1003,1004,1005 -d 1000

# 长跑导出 Prometheus
docker run -d --gpus all -p 9400:9400 nvcr.io/nvidia/k8s/dcgm-exporter:latest
# curl http://localhost:9400/metrics 就是 Prometheus 格式
```

事件 ID 表：

| ID | 名字 | 含义 |
|---|---|---|
| 1002 | SMACT | SM Active —— 真利用率 |
| 1003 | SMOCC | SM Occupancy |
| 1004 | TENSO | Tensor Active |
| 1005 | DRAMA | DRAM Active |
| 1009 | PCITX | PCIe TX bytes |
| 1010 | PCIRX | PCIe RX bytes |
| 203 | GPUUTIL | 同 nvidia-smi 的 GPU-Util |

完整列表见 [dcgm 官方文档](https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html)。

**做严肃训练性能优化时 DCGM 是必装**。

### 2.4 NVML / pynvml —— 程序内埋点

`nvidia-smi` 的 C 库叫 NVML，Python 绑定是 `pynvml`：

```python
import pynvml, time
pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)

for step in range(steps):
    train_step(...)
    util = pynvml.nvmlDeviceGetUtilizationRates(h)
    mem  = pynvml.nvmlDeviceGetMemoryInfo(h)
    pwr  = pynvml.nvmlDeviceGetPowerUsage(h) / 1000  # mW → W
    log({"step": step, "gpu_util": util.gpu, "mem_used": mem.used, "power_w": pwr})
```

- **好处**：和训练 step 严格同步，不丢点。
- **坏处**：嵌进训练代码里，开销略高（每次调 NVML 几十 μs）。

### 2.5 PyTorch profiler / Nsight Systems —— kernel 级追踪

前面四种是**采样监控**（最高 10 Hz）；这一类是**事件追踪**（纳秒级精度），能告诉你**每个 kernel 跑了多久**。

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./trace'),
) as p:
    for step in range(5):
        train_step(...)
        p.step()
```

或者命令行：

```bash
nsys profile -o my_run python train.py
```

**用途**：定位 "为什么这个 step 慢了"——看到底是 kernel 慢、cudaMemcpy 卡、还是 H2D 等数据。

---

## 3. 不同需求选不同工具

| 你想知道什么 | 用什么 |
|---|---|
| "训练时 GPU 大概在干嘛？" | `nvidia-smi --query-gpu=... -l 1 > csv` |
| "训练有没有触发降频？" | `nvidia-smi --query-gpu=clocks_throttle_reasons.* -l 1` |
| "Tensor Core 利用率多少？" | DCGM 的 TENSO |
| "我的 batch size 还能加多少？" | 看显存（NVML 或 nvidia-smi） |
| "为什么这个 step 卡了 200ms？" | Nsight / PyTorch profiler |
| "集群所有卡的实时仪表盘" | dcgm-exporter + Prometheus + Grafana |

---

## 4. 阶段 3 实战方案（5000 Pro vs L20 对比）

最便宜可行的埋点方案：

```bash
# benchmark 期间后台跑这个
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv -l 1 > 5000pro_run.csv &

python benchmark.py  # benchmark 脚本

kill %1
```

跑出来不仅是"5000 Pro BF16 = X TFLOPS"这一个数字，还是**整个 benchmark 期间的功耗、频率、降频曲线**——能确认两边都跑在峰值频率下，对比才公平。

L20 那台同样跑一份，最后画两条功耗-时钟曲线对比就能看出差异。

---

## 5. 一句话原则

`nvidia-smi -q` 看的是**瞬时快照**，生产里几乎不直接用。要分析训练/推理效果**必须埋点采样为时序数据**。从“读 GPU”到“运维 GPU”的关键一步就是这个意识转变。

---

## 6. DCGM 指标诊断训练性能瓶颈

> 学习日期：2026-07-07  
> 来源：在 2×H20 上用 DataParallel 训练 mini_gpt，观察指标并分析瓶颈

### 6.1 核心认知：活动率 ≠ 性能

DCGM 的活动率指标（SM_ACTIVE、FP16_ACTIVE 等）衡量的是 **“这条流水线有活干的时间占比”**，不是吞吐量。

```
活动率 = "该计算单元处于忙碌状态的时间" / "总时间"
```

**为什么活动率低反而可能是好事？**

比如 SDPA（融合 kernel） vs 手写 attention（3 个独立 kernel）：

```
手写 attention：
  kernel1(QK^T) → 间隙 → kernel2(softmax) → 间隙 → kernel3(×V) → 间隙
       ↑忙            ↑闲         ↑忙              ↑闲       ↑忙      ↑闲
  → 活动率高（3 段“忙”），但每段效率低

SDPA：
  kernel(全部融合) → 间隙
       ↑忙（高效）     ↑闲
  → 活动率低（只有 1 段“忙”），但这段效率更高，完成更快
```

**结论：活动率低可能是因为“做得快所以占用的时间少”。真正判断性能看 batch/s 或 epoch 耗时。**

### 6.2 第一层：看速度（最重要）

| 指标 | 怎么看 | 说明 |
|------|--------|------|
| **samples/sec / batch/s** | tqdm 进度条 | 最直接的训练速度 |
| **每 epoch 耗时** | tqdm 的 s/epoch | 对比不同配置的唯一标准 |

其他所有 DCGM 指标都是为了解释“为什么快/慢”，不是直接衡量快慢。

### 6.3 第二层：定位瓶颈类型（SM vs DRAM 组合看）

核心方法：**对比 SM 活动率和显存控制器活动率**。

```
SM_ACTIVE  (dcgm_prof_sm_active,  DCGM ID 1002)  → GPU 计算单元忙不忙
DRAM_ACTIVE (dcgm_prof_dram_active, DCGM ID 1005)  → 显存控制器忙不忙
```

| SM_ACTIVE | DRAM_ACTIVE | 瓶颈类型 | 含义 | 优化方向 |
|-----------|-------------|---------|------|----------|
| **高 (>70%)** | 低 (<40%) | **Compute-Bound** | 算力不够，计算打满 | 更多卡 / 更大算力 GPU |
| **低 (<40%)** | **高 (>70%)** | **Memory-Bound** | 带宽不够，等数据 | 优化数据搬运、增大矩阵 |
| **低 (<40%)** | **低 (<40%)** | **Latency-Bound** | GPU 在等 CPU | DataLoader、减少 Python 开销 |
| 高 | 高 | 两者都打满 | 接近最优 | 收益递减 |

### 6.4 第三层：细分原因（针对性指标）

| 你想知道的 | DCGM 指标 | ID | 怎么用 |
|-----------|-----------|------|--------|
| Tensor Core 有没有用上 | `PROF_PIPE_TENSOR_ACTIVE` | 1004 | >30% 说明大矩阵在跑 TC |
| FP16 计算有没有触发 | `PROF_PIPE_FP16_ACTIVE` | 1001 | AMP 下应 >0，为 0 说明没走 FP16 |
| FP32 占比是否过高 | `PROF_PIPE_FP32_ACTIVE` | 1002 | 远高于 FP16 说明很多操作没被 autocast 覆盖 |
| 显存带宽打满了吗 | `PROF_DRAM_BYTES_READ/WRITE` | 1005/1006 | 接近理论峰值 = 带宽瓶颈 |
| PCIe 传输是瓶颈吗 | `PROF_PCIE_BYTES` | 1009/1010 | DP gather/scatter 时观察 |
| NVLink 占用 | `PROF_NVLINK_BYTES` | 1011+ | 多卡训练时观察 |
| 降频了吗 | `clocks_throttle_reasons.*` | nvidia-smi | 任何 Active=1 说明在降频 |

**DCGM 实时监控命令（训练性能诊断常用组合）：**

```bash
# SM 活动率 + Tensor Core + DRAM 活动率 + FP16 + FP32（5 个指标组合看瓶颈）
dcgmi dmon -e 1002,1004,1005,1001,1003 -d 1000

# 加上 PCIe 和 NVLink（多卡场景）
dcgmi dmon -e 1002,1004,1005,1009,1010 -d 1000
```

### 6.5 实战案例：mini_gpt 在 2×H20 DataParallel 上的诊断

**观测数据：**

| 指标 | 观测值 | 判断 |
|------|--------|------|
| SM_ACTIVE | 30-40% | 低 |
| DRAM_ACTIVE | 未测 | - |
| FP16_ACTIVE | < 1% | 极低 |
| FP32_ACTIVE | < 3% | 极低 |
| TENSOR_CORE | 12.7% | 低 |
| NVLink | 14.4 GB/s / 900 GB/s = 1.6% | 极低 |
| batch/s | 1.27 → 1.68 (SDPA 后) | SDPA 提速 33% |

**诊断结论：Latency-Bound**

SM 低 + FP16/FP32 都低 → GPU 大量时间在空闲等待，而不是在计算。原因拆解：

1. **Python 调度开销**：每个 batch 要经过 Python 循环 → DataLoader → .to(device) → model() → loss → backward → step，中间 GPU 是空的
2. **DataLoader 瓶颈**：`num_workers=0` 数据加载在 CPU 主线程，GPU 等数据
3. **小矩阵**：attention 矩阵太小，kernel 执行几微秒就完成了，调度开销比计算还长
4. **DataParallel 固有问题**：多线程 GIL + 每 step replicate 模型 + gather 回 GPU0

**优化路径：**

```
1. DataLoader num_workers=4~8（让数据加载和 GPU 计算并行）
2. torch.compile(model)（融合算子，减少 kernel launch 次数）
3. 增大模型维度（embed=1024, head_dim=128）让每次 kernel 执行时间变长
4. DDP 替换 DataParallel（去掉 GIL 和不对称问题）
```

**SDPA 效果验证：**

切换到 SDPA 后，虽然 DCGM 活动率全部下降，但 batch/s 从 1.27 提升到 1.68（+33%），每 epoch 从 154 分钟降到 116 分钟。**活动率下降是因为同样的活做得更快了**，不是性能变差。
