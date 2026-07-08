
# 驱动层 
## 1. nvidia-smi
nvidia-smi 是 NVIDIA 提供的命令行工具，背后是个驱动 IOCTL 接口。它不需要 CUDA，只要驱动装好就能用。
nvidia-smi 是顺势快照，是一次当下实时情况的查询，driver会把当前的寄存器值返回，无法用于长时间监控。

 * 表头
 NVIDIA-SMI 580.126.09 ： nvidia-smi 这个命令行工具自己的版本
 Driver Version: 580.126.09 ： 内核态 NVIDIA 驱动的版本（cat /proc/driver/nvidia/version可查看）
 CUDA Version: 13.0 :  这是"该 driver 最高支持的 CUDA runtime"，不是你装的 CUDA 版本,  **实际用的 CUDA 看 nvcc --version** 
 * 身份列
  * 0 —— GPU 编号（本机第几块卡，从 0 开始）。多卡机器才有意义。
  * NVIDIA RTX PRO 5000 Blackwell 卡型
  * Persistence-M: On —— ⭐ 重要。Persistence Mode 开着，意思是 GPU 驱动模块常驻内存，不会因为没进程引用就卸载。
    * 关掉时：每次 python -c "import torch" 第一次访问 GPU，driver 要重新初始化，多 1~3 秒延迟
    * 开启的代价：idle 时多耗几瓦
    * 训练/推理服务器必须开。命令：sudo nvidia-smi -pm 1
 * 总线 / 显示列
  * Bus-Id: 定位显卡物理位置的"坐标"
    00000000:00:03.0 —— PCIe 地址，格式 domain:bus:device.function。多卡时按 bus-id 区分卡的物理位置，NUMA 亲和也靠它。
  * Disp.A: Off —— Display Active，没有显示器接在这块卡上（纯计算用途）。台式机插显示器时为 On，会有几百 MB 显存被显示占用，且 GPU 偶尔被显示中断打扰。
 * ECC 列
  * Volatile Uncorr. ECC: 0 —— 自上次重启以来的不可纠正 ECC 错误数。0 = 健康；非 0 警惕显存坏 cell。
    * Volatile 是相对 Aggregate（出厂以来累计）。完整 ECC 信息：nvidia-smi -q -d ECC
 * 风扇 / 温度列
  * 30% —— 风扇转速百分比，idle 时 30% 是这卡下限默认。
  * 26C —— 当前 GPU 核心温度。Blackwell 工作站卡 thermal limit 一般 88-90°C，26°C 是冷启动。
  * 主表不显示 memory 温度、hotspot 温度。要看用 nvidia-smi -q -d TEMPERATURE（多数 GPU 不暴露）。
 * 性能态列
  * P8 —— ⭐ NVIDIA GPU 的 P-state（性能状态），P0=最高频满负载，P8=最深度省电。
    * idle 时降到 P8 是正常行为：SM 时钟从 2377 MHz 降到 ~180 MHz，PCIe 从 Gen5 降到 Gen1，省电
    * 一旦有 CUDA kernel 启动，自动跳到 P0（< 100ms 内）
    * watch -n 0.5 nvidia-smi 跑 GEMM 时能观察到 P8 → P0 的切换
    * 当前 PCIe Gen 与 P-state 强关联，需要 nvidia-smi -q -d PCI 才看得到
 * 功耗列
  * 4W / 300W —— 当前功耗 / 功耗上限
    * 4W idle 是 P8 的特征
    * 300W 是这卡 TDP，可调范围 250-300W，工作站卡调节窄
    * 调节命令（root）：sudo nvidia-smi -pl 280
 * 显存列
  * 2MiB / 48935MiB —— 已用 / 总共
    * 总共 48935 MiB ≈ 47.79 GiB ≈ 48 GB（注意 GiB vs GB：1 GiB = 1024³，1 GB = 10⁹）
    * 用 2MiB 是 driver 自己的小开销
    * cudaGetDeviceProperties 里 totalGlobalMem 报 47.27 GiB，差的 ~1.4% 是 ECC reserved 走的
 * 利用率列
  * GPU-Util: 0% —— ⚠️ 最容易被误解的字段
    * 真实含义：过去 1 秒内有多少时间至少有一个 kernel 在 SM 上跑
    * 不告诉你 SM 内部利用率！1 个 SM 跑满 1 秒，这字段也显示 100%
    * 训练时可能显示 95% 但实际算力只用了 30% —— kernel 在等内存
    * 真正"算力打满了多少"要看 DCGM 的 SMACT（SM Active）和 TPCACT
    * 入门记着：这个数字只是"GPU 有没有在干活"的二值化
  * Compute M.: Default —— Compute Mode
    * Default = 多进程都能用同一块 GPU
    * Exclusive_Process = 一次只允许一个进程使用
    * Prohibited = 谁都不让用
    * 训练用 Default 即可
  * MIG M.: N/A —— Multi-Instance GPU 开关。
    MIG 是 A100/H100/H20 数据中心卡才有（一块卡切成多个隔离子卡），消费/工作站 Blackwell 不支持，所以 N/A。支持时会显示为 Disabled/Enable。

 * 进程表
  * No running processes found —— 字面意思，当前没进程在用这块 GPU。有的话会列 PID、进程名、占的显存。
  * 更细粒度的进程监控：nvidia-smi pmon -i 0（每秒一行，看具体进程的 SM/Mem 利用）

 * 一句话总结当前状态
  * 空闲、健康、待命：driver 580 + CUDA 13、ECC 没出错、P8 省电中、显存全空、显示器没接、persistence 已开。

## 2. nvidia-smi的结构化查询
### 2.1 -q / -q -d X —— 人读用，分组、分章
-q（query） 加上之后会把所有字段以可读格式打全部出来
-d (domain)  display section  把 GPU 的所有可查询属性按主题分类后的分组 
eg：
nvidia-smi -q -d ECC                       # 单看 ECC
nvidia-smi -q -d MEMORY,POWER              # 看两个
nvidia-smi -q                              # 不加 -d 就是全部 section


**GPU主要的14个分组**
section名|涵盖什么|什么时候用
|---|---|---|
MEMORY|显存容量、已用、空闲、reserved；BAR1 | 看显存占用
UTILIZATION | GPU/Memory/Encoder/Decoder util | 监控负载
PCI|当前 Gen、最大 Gen、link width、replay counter、Tx/Rx 吞吐|PCIe 链路
ECC|全部 ECC 错误计数（volatile + aggregate）|排查显存可疑错误
TEMPERATURE|核心温度、上下限、降频阈值|散热排查
POWER|当前功耗、上限、可调范围|调功耗策略
CLOCK|Graphics / SM / Memory / Video 时钟 + Application/Default/Max|看降频/锁频
CLOCKS_EVENT_REASONS|⭐ 降频原因位（power cap / thermal / sw slowdown 等）|降频原因排查
COMPUTE|Compute Mode、容许的 compute apps|设独占模式
PIDS|占用该卡的进程 PID 列表|找谁在用卡
PERFORMANCE|P-state（性能状态）|排查性能下降
SUPPORTED_CLOCKS|所有合法的 (memory, graphics) 频率组合|锁频前先查能锁到哪些点
PAGE_RETIREMENT|退役页（旧卡）/ row remap（新卡）|显存坏块预警
ACCOUNTING|进程级资源会计|集群计费
ENCODER_STATS|NVENC 编码器使用率|视频编码场景
FBC_STATS|Frame Buffer Capture|远程桌面/串流


#### 频率与降频（throttle）

**"频"是什么**：GPU 内部各部件每秒"跳节拍"的次数，单位 MHz。每跳一次节拍完成一组操作。一块 GPU 上有**多个独立时钟**，不是一个：

| 时钟名 | 控制谁 | 5000 Pro 的值 |
|---|---|---|
| **SM Clock** | ⭐ **CUDA core / Tensor core 运算速度**——算力的根本 | idle 180 MHz / boost 2377 MHz / 上限 3090 MHz |
| Graphics Clock | 图形管线（光栅/ROP）；NVIDIA 卡上和 SM Clock **几乎总是相等** | 同 SM |
| **Memory Clock** | ⭐ **显存控制器**——决定显存带宽 | idle 405 MHz / 满载 14001 MHz |
| Video Clock | NVENC / NVDEC 编解码器 | 600~3090 MHz |

**对深度学习最关键是 SM Clock 和 Memory Clock**：
- 算力峰值 = 算力峰值 = SM数 × (每个SM里的核数) × (SM频率) × 2
- 显存带宽峰值 = 总线宽度 × 显存数据率 （每根线每秒能传多少 bit）

**降频（throttle）**：GPU 主动降低自己频率的自我保护机制。常见触发原因：

| 降频原因（throttle reason） | 触发条件 |
|---|---|
| Idle | 空闲时降到 P8 省电（正常行为） |
| HW Thermal Slowdown | 核心温度逼近上限（5000 Pro ~88°C） |
| SW Power Cap | 当前功耗逼近 power limit（300W） |
| HW Power Brake | 主板/电源给 GPU 发 PWR_BRAKE 信号（电源不够） |
| Applications Clocks Setting | 用户/驱动手动锁低了频率 |
| Sync Boost | 多卡间同步降频，让最慢的拖累其他 |

**怎么查降频**：
```bash
nvidia-smi -q -d CLOCK,PERFORMANCE
```

输出里两个关键段：
- **Clocks Event Reasons**：当前是否在降频（每项 Active / Not Active,任何一个变成 Active 都意味着降频中）
- **Clocks Event Reasons Counters**：自上次重启以来累计降频微秒数 ⭐ **事后排查训练慢的金矿**——比如 `SW Power Capping = 30 minutes` 说明功耗封顶在拖慢算力

**为什么训练时降频是真问题**：满频 SM 2377 MHz → BF16 253 TFLOPS。如果热降频到 1800 MHz，算力下降 ~24%，8 小时任务变 10.5 小时，曲线上看不出来，只在 throttle counter 里能查到。

**主动锁频（做 benchmark 时用）**：
```bash
nvidia-smi -q -d SUPPORTED_CLOCKS  # 列所有合法的频率组合
sudo nvidia-smi -lgc 2400          # 锁 SM clock 在 2400 MHz
sudo nvidia-smi -lmc 14001         # 锁 memory clock
sudo nvidia-smi -rgc               # 解锁 SM clock
sudo nvidia-smi -rmc               # 解锁 memory clock
```
阶段 3 做 5000 Pro vs L20 对比时会用——避免 boost 行为差异让两次跑分失真。



### 2.2 --query-gpu=... —— 脚本/监控用，CSV 输出，字段名是 dot-notation
--query-gpu= 是"查询接口"——查一次出一次（瞬时快照，和 -q 一样）。
-l N 是"循环触发"，让它每 N 秒重新查一次。两者组合起来可实现"持续监控"。

#### 用法基本结构

```bash
nvidia-smi --query-gpu=<字段1>,<字段2>,... --format=<格式> [-l N | -lms N]
```

- `--query-gpu=` 后面接**逗号分隔的字段名**，字段名是 NVIDIA 定义的 dot-notation
- `--format=` 决定输出格式
- `-l N` / `-lms N` 决定是否循环（不加就是查一次退出）

#### 字段名怎么查

```bash
nvidia-smi --help-query-gpu             # 看完整字段字典（200+ 条）
nvidia-smi --query-gpu=<字段> --format=csv  # 测试这块卡支不支持某字段（不支持返回 [N/A]）
```

#### --format= 格式参数

| 写法 | 含义 |
|---|---|
| `--format=csv` | 标准 CSV，第一行是表头 |
| `--format=csv,noheader` | 不要表头（适合追加日志） |
| `--format=csv,nounits` | 不要单位（值变成纯数字，方便 awk/pandas 处理） |
| `--format=csv,noheader,nounits` | ⭐ 最适合监控用 |

带单位 vs 不带单位：
```bash
nvidia-smi --query-gpu=power.draw,clocks.sm --format=csv,noheader
# 输出: 4.56 W, 180 MHz
nvidia-smi --query-gpu=power.draw,clocks.sm --format=csv,noheader,nounits
# 输出: 4.56, 180
```

#### -l vs -lms（循环采样）

| 命令 | 含义 |
|---|---|
| 不加 | 查一次就退出（瞬时快照） |
| `-l 1` | 每 1 秒查一次，持续输出 |
| `-lms 100` | 每 100 ms 查一次，**毫秒级**采样 |

`-lms` 最高大概到 50-100 ms，再快下面 driver 那层会扛不住，数据开始不准。

⚠️ **某些字段的 driver 内部刷新率慢于查询频率**——查得再快也是重复读同一个值：

| 字段 | driver 内部刷新率 |
|---|---|
| `power.draw` | ~50 ms |
| `clocks.*` | ~100 ms |
| `temperature.gpu` | ~1 s |
| `utilization.gpu` | ~1 s（过去 1 秒平均） |
| `memory.used` | 实时 |


#### 三个最常用的"配方"

**配方 1：训练监控（最常用）**

```bash
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw,clocks.sm,clocks.mem,temperature.gpu,clocks_throttle_reasons.active --format=csv -l 1
```

涵盖了算力、显存、功耗、频率、降频五件事。一行一秒，直接重定向成 CSV 训练完用 pandas 画图。

**配方 2：身份卡片（一次性查清这块卡是什么）**

```bash
nvidia-smi --query-gpu=name,uuid,driver_version,vbios_version,compute_cap,memory.total,pcie.link.gen.max,pcie.link.width.max --format=csv
```

输出：
```
name, uuid, driver_version, vbios_version, compute_cap, memory.total [MiB], pcie.link.gen.max, pcie.link.width.max
NVIDIA RTX PRO 5000 Blackwell, GPU-0729..., 580.126.09, 98.02.A5..., 12.0, 48935 MiB, 5, 16
```

把这一行存到 README 里就是这块卡的"档案"。

**配方 3：降频体检**

```bash
nvidia-smi --query-gpu=clocks_throttle_reasons.active,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.hw_power_brake_slowdown,clocks_throttle_reasons.applications_clocks_setting --format=csv
```

每个值是 0/1，1 = 当前正在因为这个原因降频。配合 `-l 1` 可以做实时降频告警。

#### 一个直观练习：观察 P-state 跳变

开两个终端，左边跑：
```bash
nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,power.draw --format=csv -l 1
```

右边随便跑个 GPU 任务（最简单的：装 torch 后随便 matmul 一下）。
左边的 pstate 会从 P8 → P0、SM 时钟从 180 → 2377、功耗从 4W → 一两百瓦的实时跳变。
这是**最直观感受 GPU "醒过来"** 的场景。



## 3 nvidia子命令
nvidia-smi -q（query 模式）虽然能查询非常多的静态/动态信息，但确实有一些功能只能通过专门的子命令实现，无法通过 -q -d 获取。
1. 拓扑信息（Topology）
nvidia-smi topo -m       # GPU 互联矩阵
nvidia-smi topo -p       # 路径
nvidia-smi topo -n       # 邻居
2. NVLink 状态和统计
nvidia-smi nvlink -s     # NVLink 状态
nvidia-smi nvlink -c     # NVLink 能力
nvidia-smi nvlink -g 0   # NVLink 错误计数器
nvidia-smi nvlink -e     # 错误统计
注意：-q 只能看到一些 NVLink 的基础信息，但详细的链路状态、带宽、错误计数必须用 nvlink 子命令。
3. MIG（Multi-Instance GPU）管理
nvidia-smi mig -lgip     # 列出 GPU instance profiles
nvidia-smi mig -lcip     # 列出 compute instance profiles
nvidia-smi mig -cgi ...  # 创建 GPU instance
nvidia-smi mig -dci      # 删除 compute instance
-q 只能看到 MIG 是否启用，无法管理。
4. dmon / pmon 实时监控
nvidia-smi dmon          # 设备级实时监控（滚动输出）
nvidia-smi pmon          # 进程级实时监控
-q 是快照式查询，无法做持续滚动监控（虽然 -l 可以循环，但格式不同）。
5. 统计数据流（stats）
nvidia-smi stats         # 持续输出时间序列统计
6. 持久化模式 / 配置类操作
这些是设置而非查询，但与 -q 的查询能力互补：
nvidia-smi -pm 1         # 持久化模式
nvidia-smi -pl 250       # 功耗上限
nvidia-smi -ac 5001,1590 # 应用时钟
nvidia-smi -lgc 1000,1500 # 锁定 GPU 时钟
nvidia-smi -rgc          # 重置 GPU 时钟
nvidia-smi -e 0/1        # ECC 开关
nvidia-smi -c 0/1/2/3    # 计算模式
nvidia-smi --gpu-reset   # GPU 重置
7. vGPU 相关
nvidia-smi vgpu -q       # vGPU 查询（独立子命令）
nvidia-smi vgpu -c       # vGPU 能力
虽然也叫 -q，但是是 vgpu 子命令下的，主 -q 看不到 vGPU 详情。

## 4. GPU 架构与算力规格

理解 GPU 的算力指标（TFLOPS、SM 数量等）是怎么来的，以及不同架构/型号之间的本质差异。

### 4.1 SM —— GPU 的核心计算单元

**SM（Streaming Multiprocessor，流式多处理器）** 是 GPU 的基本调度与计算单元，类似 CPU 的“核”。

每个 SM 内部包含：
- **CUDA Core**（通用计算单元，处理 FP32/FP64/INT32）
- **Tensor Core**（矩阵乘法专用加速单元）
- **共享内存（SRAM）**（SM 内线程共享的高速缓存）
- **寄存器文件**（线程私有的最快存储）

**SM 数量由芯片设计决定，是物理固定的**，同一架构下通过“闲割” SM 数量来区分产品档次。

### 4.2 TFLOPS 算力怎么算

TFLOPS（每秒万亿次浮点运算）是峰值算力的衡量单位，由架构参数算出：

```
峰值 TFLOPS = SM数量 × 每SM的TensorCore数 × 每TC每周期FMA操作数 × 2 × SM频率
```

其中“×2”是因为一次 FMA（乘加）算 2 次浮点操作。

以 Tensor Core FP16 为例：

| 参数 | H100 SXM | H20 |
|------|----------|------|
| 架构 | Hopper (GH100) | Hopper (GH100，闲割版) |
| SM 数量 | 132 | ~60 |
| 每 SM Tensor Core | 4（第4代） | 4（第4代） |
| 每 TC 每周期 FMA 操作数 | 256 | 256 |
| Boost 频率 | ~1.83 GHz | ~1.5-1.8 GHz |
| **FP16 Tensor Core 算力** | **~990 TFLOPS（含稀疏）** | **~448 TFLOPS（含稀疏）** |

### 4.3 架构决定什么，不决定什么

| 特性 | 由什么决定 | 举例 |
|------|-----------|------|
| SM 内部结构 | **架构代际** | Hopper SM 比 Ada 多了异步执行引擎 |
| Tensor Core 代数 | **架构代际** | Hopper/Ampere = 第3/4代 TC |
| SM 数量 | **芯片型号** | 同架构不同型号 SM 数不同 |
| 显存类型/带宽 | **芯片封装** | H20 用 HBM3 (4TB/s)，L20 用 GDDR6 (864GB/s) |
| NVLink 带宽 | **互联设计** | H20 NVLink 900GB/s，L20 不支持 NVLink |
| 时钟频率 | **功耗/散热设计** | 数据中心卡 vs 消费卡频率不同 |

### 4.4 常见 GPU 算力对比表

| GPU | 架构 | SM数 | 显存 | 显存带宽 | FP16 Tensor Core | NVLink | 定位 |
|-----|------|------|------|---------|-----------------|--------|------|
| L20 | Ada (AD102) | ~142 | 48GB GDDR6 | 864 GB/s | ~300 TFLOPS | ❌ 不支持 | 中国特供工作站 |
| L40S | Ada (AD102) | 142 | 48GB GDDR6 | 864 GB/s | ~366 TFLOPS | ❌ | 工作站/推理 |
| A100 | Ampere (GA100) | 108 | 80GB HBM2e | 2039 GB/s | ~312 TFLOPS | 600 GB/s | 数据中心 |
| H100 SXM | Hopper (GH100) | 132 | 80GB HBM3 | 3350 GB/s | ~990 TFLOPS | 900 GB/s | 数据中心 |
| **H20** | Hopper (GH100) | **~60** | 96GB HBM3 | **4000 GB/s** | **~448 TFLOPS** | **900 GB/s** | **中国特供** |
| B200 | Blackwell (GB202) | 192 | 192GB HBM3e | 8000 GB/s | ~4500 TFLOPS | 1800 GB/s | 下一代旗舰 |

### 4.5 中国特供卡的设计逻辑

H20、L20 等中国特供卡受美国出口管制，但“闲割”策略不同：

| 型号 | 原版 | 闲割策略 | 保留了什么 |
|------|------|---------|------------|
| **H20** | H100 | 闲割 SM（132→~60），算力大减 | HBM3 带宽(4TB/s)、NVLink(900GB/s)、大显存(96GB) |
| **L20** | L40S/L40 | 闲割 SM、降频率 | GDDR6 显存容量(48GB) |

**H20 的设计定位是“大模型推理卡”**：算力被砍了，但显存带宽和互联保留了 H100 的顶级规格。这意味着：
- 适合大模型推理/长序列场景（吃显存带宽，不吃算力）
- 不适合小模型训练（小模型用不到大带宽，反而因 SM 少而慢）

### 4.6 为什么小模型用大卡是浪费

以 mini_gpt（embed=256, 10层, ~25M参数）在 H20 上为例：

```
每个 step 的 FLOPs ≈ 6 × 参数量 × batch_tokens
                 = 6 × 25M × (32 × 128)
                 ≈ 600 GFLOPs

H20 理论算力 448 TFLOPS，利用率 ≈ 600G / 448T ≈ 0.13%
```

就像用卡车拉一箱快递——严重大材小用。小模型在 L20 上可能更快，因为 L20 的 SM 数量更多（~142 vs ~60），小矩阵调度更充分。

**真正能打满 GPU 算力的条件**：
- 大模型（参数量 > 1B）
- 大 batch size（> 256）
- 长序列（seq_len > 1024）
- AMP 混合精度（触发 Tensor Core）

### 4.7 驱动层查询算力信息

```bash
# 查看 GPU 计算能力（compute capability）
nvidia-smi --query-gpu=name,compute_cap --format=csv

# 查看 SM 数量和每个 SM 的核心数
nvidia-smi -q -d TOPOLOGY | grep -i "sm"

# 通过 CUDA 查询详细算力参数
python -c "
import torch
dev = torch.cuda.get_device_properties(0)
print(f'GPU: {dev.name}')
print(f'SM 数量: {dev.multi_processor_count}')
print(f'每 SM CUDA Core: 128 (Hopper) / 128 (Ada)')
print(f'每 SM Tensor Core: 4')
print(f'显存: {dev.total_mem / 1024**3:.1f} GB')
print(f'计算能力: {dev.major}.{dev.minor}')
"
```



