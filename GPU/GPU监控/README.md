# GPU 监控

> 面向 AI Infra 工程师的 GPU 可观测性入口。目标不是收集越多指标越好，而是回答三个问题：GPU 是否健康、资源是否够用、业务为什么快或慢。

---

## 1. 先建立监控分层

GPU 监控工具不是互相替代的关系，而是观察层级不同：

| 层级 | 常用工具 | 时间粒度 | 主要回答 |
|---|---|---:|---|
| 状态快照 | `nvidia-smi` | 单次查询 | GPU 型号、显存、温度、功耗、进程 |
| 连续遥测 | `nvidia-smi dmon`、NVML、DCGM | 毫秒到秒级 | 哪张卡长期异常、是否存在资源闲置或不均衡 |
| 框架级分析 | PyTorch Profiler | Operator/Kernel | 哪个 PyTorch Operator 消耗时间或显存 |
| 系统级时间线 | Nsight Systems | 微秒级事件与高频采样 | CPU、CUDA API、Kernel、通信之间为什么出现等待 |
| Kernel 深度分析 | Nsight Compute | Kernel/指令级 | 单个 Kernel 为什么慢、受哪种硬件资源限制 |

完整性能诊断通常采用：

```text
业务指标
  → DCGM 发现异常 GPU 或异常阶段
    → Nsight Systems 定位 CPU/GPU/通信时间线
      → Nsight Compute 深挖可疑 Kernel
```

---

## 2. 四个最容易混淆的概念

### 2.1 GPU-Util 不等于算力利用率

`nvidia-smi` 的 `GPU-Util` 主要表示采样周期内有 Kernel 执行的时间比例。只要持续有 Kernel 运行，即使并行度很低，也可能接近 100%。

它适合回答“GPU 有没有工作”，不能单独回答“GPU 是否跑满”。

### 2.2 显存占用不等于显存带宽

- `memory.used`：容量问题，回答还能否装下更大的模型、Batch 或 KV Cache。
- DRAM Read/Write、DRAM Active：吞吐问题，回答显存接口是否繁忙。

显存占了 90% 不代表带宽繁忙；显存只占 20% 也可能已经受到带宽限制。

### 2.3 活动率不等于业务性能

SM、Tensor Core、DRAM 等活动率描述硬件在观测窗口内“有活干”的比例，不直接代表完成了多少请求或 Token。

融合 Kernel 可能用更短时间完成相同工作，使平均活动率下降，但业务吞吐反而提高。因此任何硬件指标都必须和以下业务指标一起看：

- 训练：samples/s、tokens/s、step time、epoch time。
- 推理：TTFT、TPOT、tokens/s、请求吞吐、P95/P99 延迟。

### 2.4 设备级指标不一定属于当前进程

DCGM Profiling Metrics 和 Nsight Systems GPU Metrics 都以 GPU 设备为主要观测对象。在共享 GPU、MPS、vGPU 等环境中，指标可能包含其他进程的工作。

需要进程归属时，应结合：

- Kubernetes GPU 独占关系；
- CUDA Trace；
- GPU Context Switch Trace；
- NVTX 业务阶段标记。

---

## 3. 专题导航

### [DCGM 监控专题](./DCGM监控专题.md)

适合生产环境和 Kubernetes 集群的长期监控，重点包括：

- DCGM、Host Engine、DCGM Exporter 的关系；
- 基础字段与 Profiling Fields；
- SM、Tensor、DRAM、PCIe、NVLink 指标；
- Prometheus/Grafana 与 Kubernetes 部署；
- 如何结合业务吞吐诊断训练和推理；
- 与 Nsight Systems、Nsight Compute 的计数器冲突。

### [Nsight Systems GPU Metrics 专题](./Nsight_Systems_GPU_Metrics专题.md)

适合短时间、高频性能分析，重点包括：

- CUDA Trace 与 GPU Metrics 的区别；
- `GR Active`、`SMs Active`、`SM Issue`、`Tensor Active` 等指标；
- 如何与 CUDA Kernel、CPU 线程和 NCCL 时间线关联；
- 命令行、GUI 和 SQLite 导出；
- 容器与 Kubernetes 中的使用方式；
- DCGM 计数器冲突和常见错误。

---

## 4. 轻量监控工具

### 4.1 `nvidia-smi` 快照

```bash
nvidia-smi
nvidia-smi -q -d MEMORY,POWER,TEMPERATURE,CLOCK
```

适合快速确认：

- GPU、Driver、CUDA Driver API 兼容上限；
- 显存占用和进程；
- 温度、功耗、时钟、P-State；
- ECC、Retired Pages、Row Remap 等健康状态。

### 4.2 结构化时序数据

```bash
nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu \
  --format=csv \
  -l 1
```

适合一次性 Benchmark 或没有 DCGM 的环境。输出重定向为 CSV 后，可与业务吞吐曲线对齐。

### 4.3 `nvidia-smi dmon`

```bash
nvidia-smi dmon -s pucvmet -d 1
```

适合终端观察多卡功耗、利用率、时钟、显存、ECC 和 PCIe 流量，但不适合作为完整生产监控平台。

### 4.4 NVML

NVML 是 `nvidia-smi` 和大量监控工具使用的底层管理接口，适合在平台服务中读取显存、功耗、温度、时钟和错误状态。

NVML 查询与训练 Step 同步，不代表底层数据就具有相同刷新率。部分指标的驱动采样周期明显长于函数调用耗时，增加查询频率不一定产生更多有效数据。

---

## 5. 推荐诊断流程

### 第一步：先确认业务是否真的慢

建立可比较的基线：

- 模型、输入长度、Batch、精度一致；
- GPU 型号、时钟、功耗上限一致；
- 排除模型加载、CUDA JIT 和缓存预热；
- 至少保留平均值和 P95/P99，而不是只看一次结果。

### 第二步：用 DCGM 判断异常类型

观察一段代表性窗口：

- GPU 是否长期空闲；
- 多卡是否负载不均衡；
- SM、Tensor、DRAM、PCIe、NVLink 哪类资源更活跃；
- 是否出现温度、功耗、ECC、Xid 等运维问题。

DCGM 的结论应描述为“值得怀疑的方向”，而不是仅凭一个百分比认定瓶颈。

### 第三步：用 Nsight Systems 找到时间位置

重点检查：

- Kernel 之间是否存在空洞；
- CPU 是否晚于 GPU 提交工作；
- H2D/D2H 是否阻塞计算；
- NCCL 是否与计算重叠；
- Prefill、Decode、Forward、Backward 等阶段分别发生了什么。

### 第四步：必要时进入 Nsight Compute

只有在已经定位到具体可疑 Kernel 后，再分析：

- Warp Stall Reason；
- 指令吞吐；
- Cache Hit Rate；
- 寄存器、Shared Memory 和 Occupancy；
- Roofline 与算术强度。

---

## 6. 数据采集原则

1. **采集代表性窗口**：避开模型加载、首次编译和缓存预热，除非它们就是调查目标。
2. **控制采集时长**：高频追踪用于秒级到分钟级诊断，不用于全天候监控。
3. **先记录业务指标**：没有 tokens/s、step time 或延迟，硬件指标失去判断基准。
4. **记录环境信息**：GPU、Driver、CUDA、框架版本、模型配置、Batch 和输入长度必须可追溯。
5. **避免共享负载污染**：设备级指标不能自动区分同一 GPU 上的多个业务。
6. **逐层增加开销**：先使用轻量遥测，再打开 CUDA Trace、CPU Sampling 和更多硬件计数器。
7. **不要追求所有指标 100%**：不同工作负载天然受不同资源限制，优化目标是业务性能和成本。

---

## 7. 官方资料

- [NVIDIA NVML API](https://docs.nvidia.com/deploy/nvml-api/)
- [NVIDIA DCGM Documentation](https://docs.nvidia.com/datacenter/dcgm/latest/)
- [NVIDIA Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
