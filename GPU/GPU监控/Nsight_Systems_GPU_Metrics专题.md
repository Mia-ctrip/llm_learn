# Nsight Systems GPU Metrics 专题

> Nsight Systems GPU Metrics 用高频周期采样观察 GPU 硬件单元的活动，并把这些指标放到 CPU、CUDA API、CUDA Kernel、内存拷贝和通信的统一时间线上。它适合解释“GPU 为什么没有高效工作”，不是生产环境的全天候监控系统。

---

## 1. 先区分三个概念

### 1.1 Nsight Systems

Nsight Systems 是系统级性能分析工具，主要展示：

- CPU 线程运行与调度；
- CUDA Runtime/Driver API；
- CUDA Kernel；
- H2D、D2H、D2D 内存操作；
- CUDA Stream、Context 和同步；
- NVTX 业务阶段；
- cuBLAS、cuDNN、NCCL、MPI 等库调用；
- GPU、CPU、NIC 等硬件指标。

它的核心价值不是给出一个平均利用率，而是回答：

```text
哪个时间点
  → 哪个 CPU 线程
    → 调用了哪个 CUDA API
      → 启动了哪个 Kernel 或传输
        → 当时 GPU 硬件处于什么状态
```

### 1.2 CUDA Trace

CUDA Trace 是事件追踪，记录离散事件：

- CUDA API 何时开始、何时结束；
- Kernel 在哪个 Stream 上执行；
- Kernel 持续多长时间；
- 内存拷贝何时发生；
- CPU 提交和 GPU 执行之间有多少延迟。

它回答“发生了什么”。

### 1.3 GPU Metrics

GPU Metrics 是周期采样，定期读取 GPU 硬件计数器，例如：

- SM 是否活跃；
- Warp Scheduler 是否发出指令；
- Tensor Pipe 是否工作；
- DRAM、PCIe、NVLink 是否繁忙；
- GPU 时钟如何变化。

它回答“当时 GPU 内部资源处于什么状态”。

CUDA Trace 和 GPU Metrics 结合后，才能把硬件活动和具体时间段关联起来。

---

## 2. 与其他工具的边界

| 工具 | 主要数据 | 适合场景 |
|---|---|---|
| `nvidia-smi` / NVML | 状态与低频利用率 | 快速检查、平台状态采集 |
| DCGM | 长期遥测、健康、Profiling 区间平均 | 集群监控、异常发现、告警 |
| Nsight Systems CUDA Trace | CPU/CUDA/GPU 事件时间线 | 定位等待、同步、传输、通信 |
| Nsight Systems GPU Metrics | 高频设备级硬件活动 | 判断 SM、Tensor、DRAM、互联的活动模式 |
| Nsight Compute | 单个 Kernel 的详细硬件计数器 | Stall、Cache、Occupancy、Roofline |

推荐工作流：

```text
DCGM：发现某张 GPU 或某段业务异常
  → Nsight Systems：定位异常发生在哪个阶段
    → GPU Metrics：判断当时 GPU 哪类硬件资源活跃
      → Nsight Compute：解释具体 Kernel 为什么慢
```

---

## 3. 工作原理与数据语义

GPU Metrics 使用专用硬件机制周期性采样，在单次采集中获得多个 GPU 单元的统计信息。它不会像部分 Nsight Compute 分析那样重放 Kernel，因此适合观察真实系统时间线。

需要牢记三个语义：

### 3.1 设备级，而非进程级

GPU Metrics 给出的是目标 GPU 的设备级信息，本身不知道采样值属于哪个进程或 CUDA Context。

如果一张 GPU 同时运行多个进程，指标是这些负载的合成结果。CUDA Trace 只能帮助识别被追踪进程的 Kernel，不能自动从设备级指标中扣除其他进程。

### 3.2 时间与空间被聚合

例如 `SMs Active = 50%` 可能表示：

- 所有 SM 在一半时间内有 Warp；
- 一半 SM 在整个窗口内有 Warp；
- 两者的任意组合。

因此百分比不等于“50% 峰值 FLOPS”。

### 3.3 活跃不等于有效计算

一个 Warp 已经驻留在 SM 上，即使正在等待显存或数据依赖，也可能使 SM 被计为 Active。

判断效率需要组合：

- `SMs Active`；
- `SM Issue`；
- `Tensor Active`；
- DRAM Read/Write；
- Kernel 时间；
- 业务吞吐。

---

## 4. 支持条件

当前 Nsight Systems GPU Metrics 的主要条件：

- Linux x86-64、Linux aarch64 或 Windows Target；
- NVIDIA Turing 或更新架构；
- GPU Driver 与 Nsight Systems 版本满足目标架构要求；
- 具有 GPU Performance Counter 权限；
- 没有其他工具占用同一组硬件计数器。

不同 GPU 架构支持不同 Metric Set。不要照搬其他 GPU 的 Metric Set 名称，应以运行时查询结果为准。

### 4.1 检查版本

```bash
nsys --version
```

### 4.2 检查系统环境

```bash
nsys status --environment
```

该命令主要检查 CPU Sampling 所需的 Kernel、`perf_event_open` 和 `perf_event_paranoid` 环境。即使 CPU Sampling 不可用，部分 CUDA Trace 仍可能工作；GPU Metrics 还需要单独的 GPU Counter 权限。

### 4.3 查询支持的 GPU

当前版本使用复数参数：

```bash
nsys profile --gpu-metrics-devices=help
```

可能看到：

```text
0: NVIDIA ... PCI[...]
all: Select all supported GPUs
cuda-visible: Select GPUs that match CUDA_VISIBLE_DEVICES
none: Disable GPU Metrics
```

较旧版本可能使用：

```bash
--gpu-metrics-device
```

如果命令报 `unrecognized option`，以本机 `nsys profile --help` 为准。

### 4.4 查询 Metric Set

```bash
nsys profile \
  --gpu-metrics-devices=cuda-visible \
  --gpu-metrics-set=help
```

默认会选择支持目标 GPU 的第一个 Metric Set。第一次分析通常不需要手动指定。

---

## 5. 三个核心参数

### 5.1 `--gpu-metrics-devices`

```bash
--gpu-metrics-devices=0
--gpu-metrics-devices=0,1
--gpu-metrics-devices=all
--gpu-metrics-devices=cuda-visible
```

在容器和调度系统中推荐 `cuda-visible`，避免物理 GPU 编号与容器内 CUDA 编号映射造成误解。

### 5.2 `--gpu-metrics-set`

```bash
--gpu-metrics-set=<alias>
```

Metric Set 决定采集哪些硬件计数器。可用 Alias 与 GPU 架构相关，也可以使用自定义 Metric 配置文件。

### 5.3 `--gpu-metrics-frequency`

```bash
--gpu-metrics-frequency=10000
```

当前支持范围为 10 Hz～200 kHz，默认 10 kHz，约每 100 微秒采样一次。

并非越高越好：

- GPU 越大、SM 越多，单次样本数据越多；
- 负载和系统压力越高，越容易出现 Buffer Overflow；
- 采样过快可能在时间线上产生 `Missing Data`；
- Metric Set 可能有自己的推荐频率。

第一次使用保持默认 10 kHz。只有在确实需要观察更短事件、并确认没有数据缺口时才提高频率。

---

## 6. 推荐采集命令

### 6.1 训练或推理程序

```bash
sudo nsys profile \
  --trace=cuda,nvtx,osrt \
  --gpu-metrics-devices=cuda-visible \
  --gpu-metrics-frequency=10000 \
  --delay=30 \
  --duration=15 \
  --kill=none \
  -o /reports/llm_gpu_metrics \
  python inference.py
```

参数含义：

| 参数 | 作用 |
|---|---|
| `--trace=cuda,nvtx,osrt` | 采集 CUDA、NVTX 和常见 OS Runtime 调用 |
| `--delay=30` | 程序启动 30 秒后开始，跳过加载和预热 |
| `--duration=15` | 采集 15 秒代表性窗口 |
| `--kill=none` | 采集结束后不终止长时间运行的业务 |
| `-o` | 指定报告路径，不需要手写 `.nsys-rep` 后缀 |

对于普通离线 Job，如果分析结束时允许任务退出，可以不使用 `--kill=none`。

### 6.2 只保留 CUDA 与 GPU 数据

如果容器没有 CPU Sampling 权限，或者希望先降低开销：

```bash
nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --gpu-metrics-devices=cuda-visible \
  --duration=15 \
  -o /reports/cuda_gpu_only \
  python train.py
```

关闭 CPU Sampling 后仍然可以分析 CUDA API、Kernel、内存操作和 GPU Metrics，但会失去部分 CPU 调度证据。

### 6.3 已运行进程：只看设备级 GPU Metrics

如果业务已经运行，而 `nsys` 没有包裹其启动过程，可以采集同一时间段的设备级指标：

```bash
nsys profile \
  --trace=none \
  --sample=none \
  --gpu-metrics-devices=cuda-visible \
  --duration=15 \
  -o /reports/device_metrics \
  sleep 15
```

这能看到设备级 SM、Tensor、DRAM 等变化，但报告中没有既有业务进程的 CUDA Kernel Trace，因此无法把指标精确对应到 Kernel。

### 6.4 用 NVTX 控制采集窗口

对迭代式训练和推理，推荐给业务阶段增加 NVTX：

```python
import nvtx

with nvtx.annotate("prefill", color="blue"):
    run_prefill()

with nvtx.annotate("decode", color="green"):
    run_decode()
```

NVTX 可以标记：

- data loading；
- forward；
- backward；
- optimizer step；
- prefill；
- decode；
- NCCL communication；
- 某个 Batch 或请求。

相比固定 `delay`，业务阶段标记更稳定，也更容易跨多次实验比较。

---

## 7. 主要指标及意义

不同架构和 Metric Set 显示的指标不同。下面是最常见的计算类指标。

### 7.1 Clock

| 指标 | 含义 |
|---|---|
| `GPC Clock Frequency` | Graphics/Compute 主时钟，常对应应用或 Graphics Clock |
| `SYS Clock Frequency` | GPU 前端、Copy Engine 和性能监控相关时钟 |

用途：

- 识别负载阶段；
- 判断是否发生明显降频；
- 对比相同业务在不同时钟下的表现。

仅凭时钟降低不能判断原因，应结合功耗、温度和 Clock Event Reason。

### 7.2 `GR Active`

表示 Graphics/Compute Engine 处于活动状态的周期比例。

它适合回答：

- GPU 是否存在工作；
- Kernel 之间是否存在空闲阶段。

它不能回答：

- 所有 SM 是否都在工作；
- Warp Scheduler 是否高效发射；
- Tensor Core 或 FP Pipe 是否饱和。

可以把它理解为比 `GPU-Util` 更适合时间线观察的“引擎是否忙”指标，而不是实际算力利用率。

### 7.3 `SMs Active`

表示 SM 至少有一个 Warp in Flight 的周期比例。

用途：

- 判断工作是否覆盖足够多的 SM；
- 观察 Kernel 尾部效应；
- 发现 Grid 太小、Batch 太小或 GPU 供给不足。

`SMs Active` 高只代表 Warp 驻留，不代表持续发出有效指令。

### 7.4 `SM Issue`

表示 SM 子分区的 Warp Scheduler 发出指令的周期比例。

组合解释：

| SMs Active | SM Issue | 可能现象 |
|---|---|---|
| 高 | 高 | Warp 驻留并持续发出指令 |
| 高 | 低 | Warp 存在但经常等待内存、依赖或同步 |
| 低 | 低 | GPU 供给不足、工作规模太小或时间线有空洞 |

`SM Issue` 低只能说明值得进一步分析，具体 Stall Reason 需要 Nsight Compute。

### 7.5 `Tensor Active`

表示 Tensor Pipe 发出 Tensor 指令的周期比例。

用途：

- 验证 GEMM 是否使用 Tensor Core；
- 比较 FP32、TF32、FP16、BF16、FP8 等路径；
- 观察 Prefill 与 Decode 的计算模式差异。

注意：

- Tensor Active 不是实际 Tensor TFLOPS；
- 不应期望真实 Kernel 达到 100%；
- 某些架构可能使用不同名称或共享 Pipe 指标；
- Tensor Active 低也可能是矩阵小、Kernel 短或非 GEMM 操作占比较高。

### 7.6 Compute Warps in Flight

表示 Compute Shader Warp 的驻留程度，可辅助判断：

- Kernel 是否具有足够并发；
- 是否存在并行度不足；
- Occupancy 是否可能受资源限制。

它不能替代 Kernel 级 Occupancy 分析，因为时间线指标经过设备和时间聚合。

### 7.7 Active SM Unused Warp Slots

表示已经活跃的 SM 中仍未使用的 Warp Slot。

可能原因：

- Registers per Thread 较高；
- Shared Memory per Block 较大；
- Block/SM 上限；
- Threads per Block 不合适；
- Kernel 本身并不需要更多 Warp。

该指标提示“活跃 SM 内可能还有并发空间”，但不能单独证明提高 Occupancy 会提升性能。

### 7.8 Idle SM Unused Warp Slots

表示由于部分 SM 完全空闲产生的 Warp Slot。

常见原因：

- CPU 没及时提交 GPU 工作；
- Grid 太小，Block 数少于可用 SM；
- Kernel 尾部只剩少量 Block；
- 当前工作结束但后续工作被同步阻塞；
- Batch 或并发请求不足。

它对识别小模型、小矩阵和单 Token Decode 的并行度不足非常有价值。

### 7.9 DRAM Read/Write Bandwidth

表示 DRAM 接口进行读写操作的活跃程度或相对于持续峰值的比例。

用途：

- 判断 Kernel 是否可能受显存带宽影响；
- 对比 Prefill 和 Decode；
- 观察 KV Cache、Attention、Embedding、Elementwise Kernel；
- 验证 Kernel Fusion 是否减少中间结果读写。

高 DRAM 指标不自动等于 Memory-Bound，仍需结合 SM Issue、Kernel 时间和 Nsight Compute Roofline。

### 7.10 PCIe Read/Write Throughput

表示 PCIe 接口收发流量相对于理论能力的比例。

常见来源：

- H2D/D2H；
- Unified Memory Migration；
- GPUDirect RDMA；
- GPUDirect Storage；
- 多卡 P2P；
- BAR1 访问。

需要与 CUDA Memory Copy、NIC 和存储时间线结合，避免把正常数据传输误判为瓶颈。

### 7.11 NVLink RX/TX

表示 NVLink 接收和发送流量。

在多 GPU 训练或推理中，用于分析：

- AllReduce；
- AllGather；
- ReduceScatter；
- AllToAll；
- Tensor Parallel P2P；
- Pipeline Parallel 激活传输。

NVLink 高只说明链路繁忙，不证明通信就是瓶颈。关键是通信是否延长 Critical Path，以及是否与计算重叠。

---

## 8. LLM 场景诊断模式

### 8.1 GPU 时间线有大量空洞

表现：

- `GR Active`、`SMs Active` 周期性降到很低；
- Kernel 之间有明显间隙；
- CPU 线程、DataLoader 或同步调用占据间隙。

可能方向：

- DataLoader/Tokenizer 慢；
- Python 调度开销；
- `cudaDeviceSynchronize` 或隐式同步；
- 请求到达不连续；
- Dynamic Batching 没有聚合到足够请求；
- CPU 到 GPU 数据准备不及时。

### 8.2 GPU-Util 很高，但 SMs Active/SM Issue 不高

说明 GPU 持续有 Kernel，但 GPU 内部并行度或指令发射不充分。

常见于：

- 小模型；
- 小 Batch；
- 小矩阵；
- Grid 太小；
- Kernel 尾部；
- 单 Token Decode。

这也是 GPU Metrics 相对于 `nvidia-smi GPU-Util` 的核心价值。

### 8.3 DRAM 高、Tensor 低

可能偏向显存访问：

- KV Cache 读取；
- LLM Decode；
- Elementwise；
- LayerNorm；
- Embedding；
- 非融合 Attention 中间结果。

可能优化方向：

- Continuous Batching；
- Kernel Fusion；
- FlashAttention；
- Paged KV Cache；
- 合理量化；
- 减少中间 Tensor；
- 改善数据布局和 Cache Locality。

是否真正 Memory-Bound 仍需对可疑 Kernel 使用 Nsight Compute 验证。

### 8.4 Tensor 高、DRAM 相对低

通常说明较大的矩阵计算占主导，例如：

- 大 Batch Prefill；
- 大模型训练 GEMM；
- 大尺寸 Linear；
- Tensor Core 友好的矩阵维度。

进一步应检查：

- 实际业务吞吐；
- Kernel 间隙；
- 精度路径；
- cuBLAS/cuBLASLt Kernel 选择；
- 矩阵尺寸与对齐。

### 8.5 SMs Active 高、SM Issue 低

Warp 已经驻留，但没有持续发射指令，值得怀疑：

- 显存延迟；
- Cache Miss；
- 数据依赖；
- Barrier；
- Shared Memory 冲突；
- 指令 Pipe 依赖。

这些只是候选原因，必须进入 Nsight Compute 才能确定。

### 8.6 NCCL 时间长、NVLink/PCIe 活跃

需要判断：

- 通信是否处于 Critical Path；
- 通信是否与计算重叠；
- 不同 Rank 是否同时进入 Collective；
- 是否有某个 Rank 因 DataLoader 或计算慢而拖延其他 Rank；
- 拓扑是否走了预期链路。

建议同时启用 NCCL Trace 或 NVTX 标记，而不是只看链路百分比。

### 8.7 Prefill 与 Decode

典型趋势：

| 阶段 | 常见特征 |
|---|---|
| Prefill | 大矩阵、并行度高、Tensor Active 更明显 |
| Decode | 每次生成一个 Token、矩阵较窄、KV Cache 访问占比上升 |

这是趋势而不是定律。Batch、序列长度、模型架构、量化和推理框架都会改变结果。

---

## 9. 查看与导出报告

### 9.1 GUI

```bash
nsys-ui /path/to/llm_gpu_metrics.nsys-rep
```

远程服务器通常只安装 CLI，将报告复制到本地工作站，用相同或更新版本的 GUI 打开。

推荐展开顺序：

```text
Process / Threads / NVTX
  → CUDA API
    → CUDA Context / Streams
      → Kernels / Memory Operations
        → GPU Metrics
```

分析时先选择某个业务阶段，再横向查看同一时间段的 CUDA 与 GPU Metrics。

### 9.2 汇总统计

```bash
nsys stats llm_gpu_metrics.nsys-rep
```

可查看 CUDA API、Kernel、Memory Operation 等汇总。汇总适合找耗时大户，但会丢失事件顺序；等待和重叠问题仍要看时间线。

### 9.3 导出 SQLite

```bash
nsys export -t sqlite llm_gpu_metrics.nsys-rep
```

查询 `SMs Active` 示例：

```sql
SELECT timestamp, value
FROM GPU_METRICS
JOIN TARGET_INFO_GPU_METRICS USING (metricId)
WHERE value != 0
  AND metricName LIKE 'SMs Active%'
ORDER BY timestamp;
```

原始值常以 0～100 的整数百分比保存。具体表结构应以当前报告导出的 SQLite Schema 为准。

---

## 10. 容器中执行

`nsys` 可以在容器内执行，官方推荐容器内使用 CLI，而不是安装 GUI。

需要满足四个条件：

### 10.1 CLI 在容器内可见

三种方式：

1. 在诊断镜像中安装 `nsight-systems-cli`；
2. 将宿主机 Nsight Systems 安装目录只读挂载到容器；
3. 使用 Nsight Operator 注入。

宿主机安装了 `nsys` 不代表业务容器自动可见。

### 10.2 GPU 和 Driver Library 可见

容器需要通过 NVIDIA Container Toolkit、GPU Device Plugin 或相应 Runtime 获得：

- `/dev/nvidia*`；
- Driver User-space Library；
- 被分配的 GPU/MIG 设备。

### 10.3 报告目录可写

建议将 `/reports` 映射到 `emptyDir`、PVC 或宿主机目录：

```yaml
volumeMounts:
  - name: reports
    mountPath: /reports

volumes:
  - name: reports
    emptyDir: {}
```

生产环境需要确保 Pod 退出前将 `.nsys-rep` 复制到持久存储。

### 10.4 权限满足

CPU Sampling 可能需要允许 `perf_event_open`，常见做法：

```yaml
securityContext:
  capabilities:
    add:
      - SYS_ADMIN
```

或者使用受控的 Seccomp Profile 只放行所需 Syscall。`privileged: true` 权限更大，只有在隔离诊断环境并经过安全审批时使用。

GPU Metrics 还需要 GPU Performance Counter 权限。Driver 默认限制时，即使 `nsys` 存在，也可能报：

```text
ERR_NVGPUCTRPERM
```

权限策略由节点管理员管理，不能只在普通业务容器内修改。

---

## 11. Kubernetes 使用方式

### 11.1 直接放进业务容器

最简单，但会增加镜像体积，并要求修改镜像或启动命令。

适合：

- 学习环境；
- 专用 Benchmark Pod；
- 临时诊断镜像。

### 11.2 手工 Sidecar

普通 Sidecar 并不能自动分析兄弟容器：

- 它可能看不到分配给业务容器的 GPU Device；
- 默认不共享 PID Namespace；
- 无法自动包裹已经运行的业务进程；
- CUDA Injection 和报告生命周期需要额外处理。

因此不要简单理解为“增加一个装有 `nsys` 的 Sidecar 就可以分析业务”。

### 11.3 诊断副本

对于 Deployment/StatefulSet，可以创建一个配置相同的诊断副本：

- 使用相同模型、环境变量和资源规格；
- 将启动命令改为 `nsys profile ... <original command>`；
- 只引入少量真实或可重放流量；
- 报告写入 PVC；
- 分析结束后回收。

这是对生产业务侵入较小、结果又容易归属的方式。

### 11.4 Nsight Operator

Nsight Operator 面向 Kubernetes 容器化应用，可根据 Label/Rule 注入 Profiling Agent，并按需开始、停止、存储和下载报告。

适合：

- 需要反复分析不同业务 Pod；
- 多租户集群；
- 不希望把 CLI 永久加入业务镜像；
- 希望统一控制采集窗口和报告存储。

即使使用 Operator，GPU Metrics 仍需要额外权限，并需要处理 DCGM Exporter 的计数器冲突。

---

## 12. 与 DCGM 的计数器冲突

GPU 硬件 Profiling Counter 通常只能由一个订阅者占用。Nsight Systems GPU Metrics 不能与以下工具无条件同时使用：

- DCGM Profiling Metrics；
- Nsight Compute；
- Nsight Graphics；
- 直接使用 CUPTI Sampling 的应用；
- 其他占用相同硬件计数器的工具。

采集前：

```bash
dcgmi profile --pause
```

采集结束后：

```bash
dcgmi profile --resume
```

注意：

- 这是 Host Engine 级操作，会影响同一 Host Engine 的所有 Profiling Watch；
- 基础温度、功耗、显存等 NVML 类指标不等同于 Profiling Fields；
- Kubernetes 中应由平台管理员协调，不要从普通业务 Pod 擅自停止全节点监控；
- 如果无法暂停冲突工具，可以先只做 CUDA Trace，不启用 `--gpu-metrics-devices`。

---

## 13. 常见错误

### 13.1 `nsys: command not found`

原因：

- 容器/节点没有安装 CLI；
- CLI 已挂载但不在 `PATH`；
- 宿主机安装了，容器不可见；
- 使用 `sudo` 后 `secure_path` 丢失安装目录。

检查：

```bash
command -v nsys
find /opt/nvidia /usr/local/cuda* -type f -name nsys 2>/dev/null
```

找到后优先使用绝对路径验证：

```bash
/opt/nvidia/nsight-systems/<version>/bin/nsys --version
```

### 13.2 包管理器提示 `No match for argument`

说明当前启用的软件源没有该包，常见原因：

- NVIDIA DevTools Repository 未配置；
- Repository 与发行版版本不匹配；
- x86_64 与 aarch64 Repository 选错；
- 容器是精简发行版；
- Repository Metadata 未刷新。

不要反复尝试不同包名，应先确认：

```bash
cat /etc/os-release
uname -m
dnf repolist --enabled
dnf list --available '*nsight-systems*'
```

### 13.3 `ERR_NVGPUCTRPERM`

说明进程无权访问 GPU Performance Counter。

处理方向：

- 节点管理员允许相应 GPU Profiling Capability；
- 容器增加经过审批的 `SYS_ADMIN`/Profiler 权限；
- 在专用诊断节点运行；
- 如果暂时无法取得权限，只采集 CUDA Trace。

### 13.4 `Missing Data` / `Inconsistent Data`

常见原因：

- 采样频率过高；
- GPU SM 数较多；
- 系统负载高；
- Metric Set 的推荐频率不匹配；
- 内部 Buffer Overflow。

优先恢复默认 10 kHz，检查 Diagnostics Summary，再逐步调整。

### 13.5 GPU 不出现在 `--gpu-metrics-devices=help`

检查：

- GPU 架构是否早于 Turing；
- 容器是否真正获得 GPU Device；
- Driver 与 Nsight Systems 版本；
- GPU/MIG/vGPU 模式是否受当前版本支持；
- 是否选错物理设备编号。

### 13.6 有 GPU Metrics，但没有业务 CUDA Kernel

常见原因：

- `nsys` 没有启动或注入业务进程；
- 只对 `sleep` 做了设备级采样；
- 业务运行在兄弟容器；
- CUDA Trace 被关闭；
- 进程在采集窗口之前已经完成目标阶段。

---

## 14. 建议练习

### 练习一：小矩阵与大矩阵

分别运行小尺寸和大尺寸 GEMM，对比：

- `GR Active`；
- `SMs Active`；
- `SM Issue`；
- `Tensor Active`；
- Idle SM Unused Warp Slots。

目标：理解“GPU 持续有 Kernel”与“GPU 具有足够并行度”的区别。

### 练习二：Prefill 与 Decode

给推理程序增加 NVTX：

```text
request
  ├─ prefill
  └─ decode
```

对比两个阶段的 Tensor、DRAM、Kernel 尺寸和间隙。

### 练习三：改变 Batch

固定模型和输入长度，只改变 Batch：

- 记录 tokens/s、TTFT/TPOT；
- 对比 SM、Tensor、DRAM；
- 判断 Batch 增加是提高并行度，还是导致显存/延迟代价。

### 练习四：DataLoader

训练任务分别使用不同的：

- `num_workers`；
- `pin_memory`；
- Prefetch；
- H2D Async Copy。

观察 Kernel 空洞、Memcpy 和 CPU Thread。

### 练习五：DCGM → Nsight

1. 先用 DCGM 发现低 SM Active 或多卡不均衡；
2. 暂停 DCGM Profiling；
3. 用 Nsight Systems 采集 10～20 秒；
4. 找到具体空洞、通信或 Kernel；
5. 恢复 DCGM Profiling。

---

## 15. 最终判断原则

1. `GPU-Util` 只能说明有没有 Kernel，不代表算力打满。
2. `GR Active` 说明引擎是否忙，不代表所有 SM 饱和。
3. `SMs Active` 说明 Warp 是否驻留，不代表持续有效计算。
4. `SM Issue` 帮助判断 Warp 是否持续发出指令，但不给出具体 Stall 原因。
5. `Tensor Active` 验证 Tensor Pipe 参与，不等于实际 Tensor FLOPS。
6. DRAM、PCIe、NVLink 高只表示接口繁忙，不自动等于业务瓶颈。
7. 设备级 Metrics 必须结合 CUDA Trace、NVTX 和独占关系判断归属。
8. 所有硬件指标最终都要回到 tokens/s、step time、TTFT、TPOT 和成本。

---

## 16. 官方资料

- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [Nsight Systems User Guide：GPU Metrics](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#gpu-metrics)
- [Nsight Systems Installation Guide](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html)
- [Nsight Systems Post-Collection Analysis Guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)
- [GPU Performance Counter Permissions](https://developer.nvidia.com/ERR_NVGPUCTRPERM)
- [Nsight Operator Documentation](https://docs.nvidia.com/nsight-operator/)
- [DCGM Profiling](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/profiling.html)
