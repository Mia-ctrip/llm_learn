# DCGM 监控专题

> DCGM（Data Center GPU Manager）面向数据中心 GPU 的长期遥测、健康检查、诊断、策略管理和工作负载观测。它适合回答“哪张卡、哪个节点、哪个时间段出现异常”，不负责解释单个 CUDA Kernel 的指令级性能。

---

## 1. DCGM 在监控体系中的位置

DCGM 处在 Driver/NVML 之上，为数据中心 GPU 提供统一的设备和实体模型，并向命令行、API 与 Prometheus Exporter 暴露指标。

```text
NVIDIA Driver / NVML / GPU Hardware Counters
                    ↓
               DCGM Host Engine
              ↙       ↓        ↘
          dcgmi      DCGM API   dcgm-exporter
                                  ↓
                           Prometheus / Grafana
```

它主要用于：

- GPU 资产与拓扑发现；
- 温度、功耗、时钟、显存和利用率遥测；
- ECC、Xid、Retired Pages、Row Remap 等健康状态；
- SM、Tensor、DRAM、PCIe、NVLink 等 Profiling Metrics；
- 集群告警、容量评估和多卡负载对比；
- 主动诊断与压力测试。

它不擅长：

- 显示每个 CUDA Kernel 的起止时间；
- 将硬件指标直接归属到某一行代码；
- 给出 Warp Stall 的具体原因；
- 单凭一个指标证明性能优化有效。

---

## 2. 组件与部署模型

### 2.1 Host Engine

Host Engine 负责管理 GPU 实体、维护字段 Watch、缓存采样值并协调 Profiling Counter。它可以：

- 作为独立进程运行；
- 由调用程序以 Embedded Mode 嵌入；
- 被 `dcgmi`、Exporter 或业务平台通过 API 连接。

### 2.2 `dcgmi`

`dcgmi` 是管理与诊断命令行工具，适合：

- 查询 GPU、Group、Field；
- 查看支持的 Profiling Metric Group；
- 临时流式观察指标；
- 启停诊断；
- 暂停和恢复 Profiling Counter。

### 2.3 DCGM Exporter

DCGM Exporter 将字段转换为 Prometheus Metrics，适合长期监控和告警。

在 Kubernetes 中，典型部署不是每个业务 Pod 一个 sidecar，而是：

```text
每个 GPU Node
  └─ nvidia-dcgm-exporter DaemonSet Pod
       └─ 采集本节点 GPU / MIG / NVSwitch / NVLink
            └─ Prometheus 拉取
```

GPU Operator 可以统一管理 Driver、Container Toolkit、Device Plugin、DCGM 和 DCGM Exporter。

当启用 Kubernetes 映射时，Exporter 可以给指标增加 Namespace、Pod、Container 等标签。这种标签映射不会改变指标本身的设备级语义。

---

## 3. 两类指标

### 3.1 基础遥测字段

这类指标通常来自 Driver/NVML，适合长期运行：

| 类别 | 示例 | 用途 |
|---|---|---|
| 利用率 | GPU Util、Memory Util | 判断设备是否有工作 |
| 容量 | FB Used、FB Free | 观察显存容量 |
| 功耗 | Power Usage、Power Limit | 能耗与功耗封顶 |
| 温度 | GPU Temperature、Memory Temperature | 散热与热风险 |
| 时钟 | SM Clock、Memory Clock | 降频和性能状态 |
| 健康 | ECC、Xid、Row Remap | 硬件与 Driver 异常 |
| 互联 | PCIe、NVLink 错误与吞吐 | 数据传输和链路状态 |

这些指标适合 Prometheus 长期保留、告警和容量统计。

### 3.2 Profiling Metrics

Profiling Metrics 来自 GPU 硬件性能计数器，用于解释 GPU 内部不同单元的活动情况。

常用字段如下：

| Field ID | Field Name | 含义 |
|---:|---|---|
| 1001 | `DCGM_FI_PROF_GR_ENGINE_ACTIVE` | 图形/计算引擎活动率 |
| 1002 | `DCGM_FI_PROF_SM_ACTIVE` | SM 至少驻留一个 Warp 的周期比例 |
| 1003 | `DCGM_FI_PROF_SM_OCCUPANCY` | 驻留 Warp 相对于理论最大 Warp 数的比例 |
| 1004 | `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | Tensor Pipe 活动率 |
| 1005 | `DCGM_FI_PROF_DRAM_ACTIVE` | 显存接口活动率 |
| 1006 | `DCGM_FI_PROF_PIPE_FP64_ACTIVE` | FP64 Pipe 活动率 |
| 1007 | `DCGM_FI_PROF_PIPE_FP32_ACTIVE` | FP32 Pipe 活动率 |
| 1008 | `DCGM_FI_PROF_PIPE_FP16_ACTIVE` | FP16 Pipe 活动率 |
| 1009 | `DCGM_FI_PROF_PCIE_TX_BYTES` | PCIe 发送字节速率 |
| 1010 | `DCGM_FI_PROF_PCIE_RX_BYTES` | PCIe 接收字节速率 |

不同 GPU 架构支持的字段和可同时采集的组合不同。使用前应以目标 GPU 的运行时查询结果为准，不要假设所有型号都支持相同字段。

---

## 4. 正确理解核心指标

### 4.1 GPU Util 与 GR Engine Active

这类指标主要回答“观测窗口内是否有图形或计算工作”。

高值不等于 GPU 所有 SM 或计算流水线都饱和。一个并行度很低、但持续运行的 Kernel 也可能让该指标很高。

### 4.2 SM Active

`SM Active` 表示 SM 至少分配了一个 Warp 的周期比例，并在所有 SM 上做平均。

例如 20% 可能来自：

- 所有 SM 只工作了观测窗口的 20%；
- 20% 的 SM 一直工作；
- 两者的任意组合。

此外，等待显存或数据依赖的 Warp 仍然可能被视为 Active。因此它不是“有效 FLOPS 占峰值的比例”。

### 4.3 SM Occupancy

Occupancy 描述驻留 Warp 数相对于硬件上限的比例，受以下因素影响：

- Threads per Block；
- Registers per Thread；
- Shared Memory per Block；
- 每个 SM 的最大 Block 和 Warp 数；
- Kernel 的 Grid 大小。

Occupancy 太低可能无法隐藏延迟，但 Occupancy 越高不一定越快。计算密集 Kernel 达到足够并发后，继续提高 Occupancy 可能没有收益。

### 4.4 Tensor Active

`Tensor Active` 表示 Tensor Pipe 发出 Tensor 指令的周期比例。

它能回答“Tensor Core 是否真实参与”，但不能直接回答：

- 实际 TFLOPS；
- Tensor Core 指令效率；
- 矩阵尺寸是否合理；
- FP16/BF16/TF32/FP8 的具体吞吐；
- 业务是否因此更快。

即使是高效 GEMM，也不应机械追求 Tensor Active 100%，因为还存在数据加载、地址计算、同步和其他指令。

### 4.5 DRAM Active

`DRAM Active` 表示显存接口存在流量的活动程度，适合与 SM、Tensor 和业务吞吐组合判断。

它不是显存容量，也不一定等于“实际 GB/s / 产品标称 GB/s”。如果需要绝对吞吐，应使用字节速率字段或专门的带宽指标。

---

## 5. 基本使用

### 5.1 查看目标 GPU 支持的 Profiling Fields

当前版本推荐先查询目标 GPU 的运行时 Metric Catalogue：

```bash
dcgmi profile --list --entity-id gpu:0
```

旧版本常见写法：

```bash
dcgmi profile -l -i 0
```

输出会显示字段以及可兼容采集的 Metric Group。

### 5.2 临时流式观察

```bash
dcgmi dmon \
  --entity-id gpu:0 \
  --field-id 1002,1004,1005 \
  --delay 1000
```

观察：

- `1002`：SM Active；
- `1004`：Tensor Active；
- `1005`：DRAM Active。

多卡分析时，应分别保留 GPU ID，不要先把所有 GPU 求平均，否则会掩盖负载不均衡。

### 5.3 暂停和恢复 Profiling Metrics

当 Nsight Systems 或 Nsight Compute 需要使用相同硬件计数器时：

```bash
dcgmi profile --pause
```

分析结束后：

```bash
dcgmi profile --resume
```

暂停和恢复作用于整个 Host Engine，而不是单个 GPU 或单个客户端。暂停期间，相关 Profiling Watch 可能返回空值。

---

## 6. Kubernetes 与 Prometheus

### 6.1 典型数据流

```text
GPU Node
  ├─ Business Pod A
  ├─ Business Pod B
  └─ DCGM Exporter DaemonSet
          ↓ /metrics
      Prometheus
          ↓
       Grafana / Alertmanager
```

DCGM 负责节点级长期观测；业务系统应同时暴露请求量、Token、延迟、Batch、队列长度等应用指标。

### 6.2 指标归属

Kubernetes Label 能帮助建立：

```text
GPU UUID / Device ID
    ↔ Node
      ↔ Namespace / Pod / Container
```

但 Profiling Counter 仍是设备级采样。在 GPU 共享、MPS 或 vGPU 环境中，不能只凭 Pod Label 认定全部活动来自某个进程。

### 6.3 告警分层

建议分为：

- **硬件健康**：Xid、不可纠正 ECC、温度异常、Row Remap。
- **资源容量**：显存持续接近上限、GPU 分配不足。
- **性能状态**：长期低利用率、时钟异常、功耗封顶。
- **业务异常**：吞吐下降、延迟升高、队列积压。

硬件活动率更适合用于趋势和诊断，不适合脱离业务背景直接设置统一阈值。

---

## 7. 诊断方法

### 7.1 先看业务速度

训练优先观察：

- tokens/s 或 samples/s；
- step time；
- epoch time；
- DataLoader time；
- 多卡每 Rank 的 step time。

推理优先观察：

- 请求吞吐；
- tokens/s；
- TTFT；
- TPOT；
- P95/P99；
- Batch、输入长度和输出长度。

DCGM 指标用于解释业务速度变化，不替代业务速度。

### 7.2 SM 与 DRAM 组合

| 现象 | 值得怀疑的方向 | 还需要验证 |
|---|---|---|
| SM 高、DRAM 相对低 | 计算密集 | Tensor/FP Pipe、实际 FLOPS、Kernel 时间 |
| SM 低、DRAM 高 | 显存访问主导 | 实际带宽、Cache、算术强度 |
| SM 与 DRAM 都低 | GPU 供给不足或工作过小 | CPU、DataLoader、Kernel 空洞、Batch |
| SM 与 DRAM 都高 | 两类资源都较忙 | 业务吞吐是否已接近目标 |

这里的“高低”必须相对于同一模型的基线、相同输入和相同 GPU 判断，不建议固定使用一组全局阈值。

### 7.3 Tensor Active 的判断

常见模式：

- Tensor 接近 0：可能没有 Tensor Core Kernel，或采集字段/架构不支持。
- Tensor 有活动但业务不快：可能矩阵太小、Kernel 间隙多或数据供给不足。
- Tensor 上升且业务吞吐上升：说明混合精度或矩阵 Kernel 优化可能有效。
- Tensor 下降但业务吞吐上升：可能融合减少了总工作时间，不能据此判定退化。

### 7.4 多卡与通信

需要同时比较：

- 每张 GPU 的 SM/Tensor/DRAM；
- PCIe/NVLink TX/RX；
- 每 Rank 的业务吞吐和 Step Time；
- NCCL Collective 的持续时间。

如果只有部分 GPU 长期空闲，应优先调查：

- 数据或 Batch 分配不均；
- Rank 同步等待；
- 参数服务器或主卡瓶颈；
- DataParallel 的 Gather/Scatter；
- 通信拓扑和 NUMA 亲和。

---

## 8. 实战案例：mini_gpt

实验背景：

- 2×H20；
- PyTorch DataParallel；
- mini_gpt 小模型；
- 手写 Attention 与 SDPA 对比。

观测：

| 指标 | 现象 |
|---|---|
| SM Active | 约 30%～40% |
| FP16/FP32 Pipe | 活动较低 |
| Tensor Active | 约 12.7% |
| NVLink | 相对于链路能力较低 |
| Batch/s | 使用 SDPA 后从 1.27 提升到 1.68 |

仅看硬件活动率容易得出“SDPA 后利用率下降”的错误结论。业务 Batch/s 提升约 33%，说明相同工作完成得更快。

进一步值得用 Nsight Systems 验证：

- DataLoader 是否让 GPU 等待；
- Python 与 DataParallel 是否产生 Kernel 间隙；
- 小矩阵是否导致 Grid 规模不足；
- SDPA 是否通过融合减少 Kernel Launch 与中间显存访问。

这个案例说明：

> DCGM 负责发现“整体活动模式发生了变化”，业务吞吐判断优化是否有效，Nsight Systems 解释变化发生在时间线的哪个位置。

---

## 9. 常见误区

### 误区一：SM Active 高就是算力打满

Warp 等待内存时仍可能是 Active。需要结合 SM Issue、Tensor/FP Pipe、DRAM 和业务吞吐。

### 误区二：Occupancy 越高越好

Occupancy 的目标是提供足够并发隐藏延迟，而不是永远追求理论上限。

### 误区三：DCGM 可以定位具体 Kernel

DCGM 是区间平均和设备级遥测。具体 Kernel 归属需要 Nsight Systems，Kernel 内部原因需要 Nsight Compute。

### 误区四：所有指标都能同时高频采集

硬件计数器存在兼容组和 Multiplexing。字段组合不当或采样过快可能产生空值、零值或统计误差。

### 误区五：DCGM 和 Nsight GPU Metrics 可以无条件同时运行

它们可能争用同一 Profiling Counter。正式采集前必须确认并协调暂停/恢复。

---

## 10. 官方资料

- [DCGM Documentation](https://docs.nvidia.com/datacenter/dcgm/latest/)
- [DCGM Profiling](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/profiling.html)
- [DCGM Field Identifiers](https://docs.nvidia.com/datacenter/dcgm/latest/reference/dcgm-api/dcgm-api-field-ids.html)
- [DCGM Exporter](https://docs.nvidia.com/datacenter/dcgm/latest/installation/install-dcgm-exporter.html)
- [NVIDIA GPU Telemetry](https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/latest/)
