# GPU 通讯架构完全指南

## 前言：澄清核心概念

GPU 通讯涉及两个容易混淆的层面：

| 层面 | 概念 | 类比 |
|------|------|------|
| **物理层** | 插槽形态、引脚定义、供电规格 | 就像 USB-A vs USB-C 接口 |
| **协议层** | 数据如何编码、传输、路由 | 就像 USB 2.0 vs USB 3.0 协议 |

**关键澄清**：
- **PCIe** 既是物理标准（插槽设计）又是通讯协议（数据传输规则）
- **SXM** 只是物理接口（一种更高密度的插槽形态）
- **NVLink** 只是通讯协议（GPU 间高速互联的数据传输方式）

**核心结论**：
> SXM 插槽可以同时跑 PCIe 协议（CPU ↔ GPU）和 NVLink 协议（GPU ↔ GPU）。
>
> 即使使用了 NVLink，CPU 和 GPU 之间的通讯仍然走 PCIe 协议。

---

## 一、物理接口形态

### 1.1 PCIe 插槽（可插拔式）

**形态特点**：
```
┌─────────────────────────┐
│    GPU PCB（印刷电路板） │  ← 独立显卡
│    散热器 + 风扇         │
│    供电接口（6/8 pin）   │
└──────┬──────────────────┘
       │ 金手指（x16）
   ════╧════════════════════  ← PCIe x16 插槽（主板）
```

**技术参数**：
- **物理尺寸**：标准 PCIe x16 插槽（89mm）
- **引脚数**：164 个触点（正反各 82 个）
- **供电能力**：插槽 75W + 外接供电（8-pin×3 可达 525W）
- **典型功耗**：150W - 450W

**典型产品**：
- 消费级：RTX 4090、RTX 4080、RTX 3090
- 工作站：RTX 6000 Ada、RTX A6000
- 数据中心：T4、L4、L40S、A100 PCIe 版

**优势**：
- ✅ 灵活插拔，易于维护和升级
- ✅ 兼容标准服务器和工作站
- ✅ 单卡购买成本较低

**限制**：
- ⚠️ NVLink 链路数少（最多 4 条）
- ⚠️ 供电能力有上限（<600W）
- ⚠️ 散热需要独立风扇（占用空间）

---

### 1.2 SXM 插槽（焊接式）

**形态特点**：
```
┌─────────────────────────┐
│    GPU 计算模块          │  ← 无风扇，直接暴露芯片
│    HBM 内存堆叠          │
│    （无独立 PCB）        │
└─────────────────────────┘
    ║║║║║║║║║║║║║║║║
════╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩═══════  ← SXM 插槽（焊接在底板）
```

**技术参数**：
- **物理尺寸**：SXM4（92×114mm）、SXM5（更大）
- **引脚数**：>4000 个（远超 PCIe 的 164 个）
- **供电能力**：集中供电，单卡可达 700W（H100 SXM5）
- **典型功耗**：400W - 700W

**典型产品**：
- A100 SXM4（400W）
- H100 SXM5（700W）
- H20 SXM5（700W，中国特供）
- B200 SXM6（仅有 SXM 版本）

**优势**：
- ✅ 更多引脚 → 支持 12-18 条 NVLink 链路
- ✅ 集中供电和散热（液冷/风冷由机箱统一管理）
- ✅ 更高功耗上限（适合顶级计算芯片）
- ✅ 更紧凑的布局（8 卡系统体积更小）

**限制**：
- ❌ 焊接在底板，无法单独更换（坏了需要换整个模块）
- ❌ 必须购买整机系统（如 DGX、HGX）
- ❌ 初期投资成本高

---

### 1.3 接口选择的决定因素

**关键问题**：PCIe 和 SXM 是由什么决定的？

**答案**：不是随便选，而是**产品定位 + 物理需求**决定：

| GPU 型号 | 架构 | PCIe 版本 | SXM 版本 | 决策因素 |
|---------|------|----------|---------|---------|
| **B200** | Blackwell | ❌ 无 | ✅ 有 | 功耗 1000W+，PCIe 无法承载 |
| **H100** | Hopper | ✅ 有（350W）| ✅ 有（700W）| 两种场景都覆盖 |
| **A100** | Ampere | ✅ 有（250W）| ✅ 有（400W）| 两种场景都覆盖 |
| **L4** | Ada Lovelace | ✅ 只有 PCIe | ❌ 无 | 推理卡，72W 低功耗 |
| **L40S** | Ada Lovelace | ✅ 只有 PCIe | ❌ 无 | 推理/图形，350W |
| **RTX 4090** | Ada Lovelace | ✅ 只有 PCIe | ❌ 无 | 消费级，450W |

**决策树**：
```
是否需要 8 卡 NVSwitch 全连接？
├─ 是 → 必须 SXM（需要 12+ 条 NVLink）
└─ 否 → 看功耗和场景
    ├─ 功耗 >600W → SXM（PCIe 供电不够）
    ├─ 需要灵活部署 → PCIe（可插拔）
    └─ 推理/消费级 → PCIe（足够用且便宜）
```

**重要结论**：
> 同一架构（如 Ampere）可能既有 PCIe 版本又有 SXM 版本，差异在于功耗和 NVLink 链路数。
>
> **应用场景决定了你只能选哪个**。

---

## 二、通讯协议层级

### 2.1 PCIe 协议：CPU ↔ GPU 的唯一通道

**作用**：负责 CPU 和 GPU 之间的数据传输。

**代际演进**：

| PCIe 代际 | 单通道带宽 | x16 总带宽 | 双向带宽 | 典型 GPU |
|----------|-----------|-----------|---------|---------|
| PCIe 3.0 | 985 MB/s | 15.75 GB/s | 31.5 GB/s | GTX 10 系、RTX 20 系 |
| PCIe 4.0 | 1.97 GB/s | 31.5 GB/s | 63 GB/s | RTX 30 系、A100 PCIe |
| PCIe 5.0 | 3.94 GB/s | 63 GB/s | 126 GB/s | RTX 40 系、H100 PCIe |
| PCIe 6.0 | 7.88 GB/s | 126 GB/s | 252 GB/s | 2025+ 新品 |

**典型使用场景**：
```
训练阶段：
CPU 内存 ──PCIe──> GPU 显存    # 加载数据批次（如 64 张图片）
GPU 显存 ──PCIe──> CPU 内存    # 返回日志、指标（很小）

推理阶段：
CPU 内存 ──PCIe──> GPU 显存    # 输入数据（如用户查询）
GPU 显存 ──PCIe──> CPU 内存    # 推理结果
```

**性能瓶颈分析**：
- **小模型推理**：PCIe 3.0 就足够（每次传输几 MB）
- **大 batch 训练**：需要 PCIe 4.0+（每次传输几 GB）
- **模型加载**：一次性传输（如 70B 模型 = 140 GB FP16），受 PCIe 带宽限制

**关键点**：
> **即使使用了 NVLink，CPU ↔ GPU 的通讯仍然走 PCIe**。
>
> NVLink 只负责 GPU ↔ GPU，无法绕过 PCIe。

---

### 2.2 NVLink 协议：GPU ↔ GPU 高速互联

**作用**：GPU 之间的专用高速通道，绕过 PCIe 的带宽限制。

**代际演进**：

| NVLink 代际 | 单条带宽 | 典型 GPU | PCIe 版本链路数 | SXM 版本链路数 | 总带宽对比 |
|------------|---------|---------|---------------|--------------|----------|
| NVLink 1.0 | 20 GB/s | P100 | 4 条 | 4 条 | 80 GB/s |
| NVLink 2.0 | 25 GB/s | V100 | 4 条 | 6 条 | 100 GB/s / 150 GB/s |
| NVLink 3.0 | 25 GB/s | A100 | 4 条 | 12 条 | 100 GB/s / 300 GB/s |
| NVLink 4.0 | 25 GB/s | H100 | 4 条 | 18 条 | 100 GB/s / 450 GB/s |
| NVLink 5.0 | 50 GB/s | B200 | — | 18 条 | — / 900 GB/s |

**关键澄清**：
> **PCIe 插槽的 GPU 也能有 NVLink**，只是链路数少（2-4 条）。
>
> 不是"SXM 独占 NVLink"，而是"SXM 能提供更多链路"（12-18 条 vs 2-4 条）。

**架构支持情况**：

| 架构 | 首次引入 | 消费级支持 | 数据中心支持 | 说明 |
|------|---------|-----------|-------------|------|
| Maxwell | — | ❌ | ❌ | 无 NVLink |
| **Pascal** | 2016 | ❌（消费级无）| ✅ P100 | **首次引入** |
| Volta | 2017 | ⚠️ Titan V | ✅ V100 | NVLink 2.0 |
| Turing | 2018 | ❌ RTX 20 系无 | ⚠️ Quadro RTX 部分有 | 选择性启用 |
| Ampere | 2020 | ✅ RTX 3090 有 | ✅ A100 全系 | RTX 3060/3080 无 |
| Ada Lovelace | 2022 | ✅ RTX 4090 有 | ⚠️ L4 无，L40S 无 | 按产品定位区分 |
| Hopper | 2022 | — | ✅ 全系支持 | NVLink 4.0 |
| Blackwell | 2024 | ✅ RTX 50 系 | ✅ 全系支持 | NVLink 5.0 |

**为什么同架构不同型号支持情况不同？**
1. **成本控制**：NVLink 收发器增加芯片面积和成本（~5-10%）
2. **功耗考虑**：每条 NVLink 约 5W 功耗
3. **市场定位**：游戏/推理场景不需要 GPU 间频繁通讯

**典型拓扑结构**：

**双卡 PCIe 系统（RTX 4090 ×2）**：
```
┌────────┐     NVLink 桥接      ┌────────┐
│ GPU 0  │◄──────────────────►│ GPU 1  │
│        │   (100 GB/s)        │        │
└───┬────┘                     └────┬───┘
    │ PCIe 4.0 (64 GB/s)           │ PCIe 4.0
    └──────────┬───────────────────┘
           主板 PCIe 交换芯片
               │
             CPU
```

**8 卡 SXM + NVSwitch 系统（H100 ×8）**：
```
       ┌─────── NVSwitch 交换矩阵 ────────┐
       │      （4 个 NVSwitch 芯片）      │
GPU0 ══╪══ 每个 GPU 18 条 NVLink ════╪══ GPU4
GPU1 ══╪══ 连接到 4 个 NVSwitch  ════╪══ GPU5
GPU2 ══╪══ 实现任意两卡全速直连   ════╪══ GPU6
GPU3 ══╪══ (450 GB/s)           ════╪══ GPU7
       └─────────────────────────────────┘

任意 GPU 到任意 GPU：全速 450 GB/s，无需中转
```

**NVLink 的核心优势**：
1. **带宽高**：H100 的 450 GB/s vs PCIe 5.0 的 126 GB/s（3.5 倍）
2. **延迟低**：~1-2 μs vs PCIe 的 5-10 μs
3. **缓存一致性**：支持 GPU 间直接访问对方的缓存和显存
4. **直接内存访问**：GPU0 可以直接读写 GPU1 的显存，无需 CPU 中转

---

### 2.3 NVSwitch：GPU 交换机

**问题**：如果只有点对点 NVLink，8 卡系统会遇到什么问题？

**场景对比**：

**方案 1：无 NVSwitch（PCIe 版 A100 ×4，菊花链拓扑）**：
```
GPU0 ←NVLink→ GPU1 ←NVLink→ GPU2 ←NVLink→ GPU3

GPU0 到 GPU3 的通讯：
- 路径：GPU0 → GPU1 → GPU2 → GPU3
- 需要中转两次，带宽减半，延迟累加
- GPU1 和 GPU2 成为瓶颈
```

**方案 2：NVSwitch（SXM 版 A100 ×8，全连接拓扑）**：
```
       NVSwitch 交换矩阵
   ┌───────────────────────┐
   │   4 × NVSwitch 芯片   │
GPU0 ──┤   任意两个 GPU     ├── GPU4
GPU1 ──┤   之间都是全速     ├── GPU5
GPU2 ──┤   直连通道         ├── GPU6
GPU3 ──┤   (无需中转)       ├── GPU7
   └───────────────────────┘
```

**NVSwitch 技术规格**：

| 指标 | NVSwitch 2.0（A100）| NVSwitch 3.0（H100）| NVSwitch 4.0（B200）|
|------|-------------------|-------------------|-------------------|
| 单芯片端口数 | 18 个 | 32 个 | 72 个 |
| 单芯片总带宽 | 900 GB/s | 3.6 TB/s | 7.2 TB/s |
| DGX 系统配置 | 6 个芯片（8 卡）| 4 个芯片（8 卡）| 2 个芯片（8 卡）|

**为什么只有 SXM 卡能用 NVSwitch？**

计算需求：
```
8 卡全连接拓扑：
每个 GPU 需要连接到 4 个 NVSwitch 芯片
每个连接需要多条链路（冗余 + 带宽）
→ 每个 GPU 至少需要 12-18 条 NVLink 通道

PCIe 插槽的物理限制：
164 个引脚需要分配给：
- PCIe 数据通道（x16 = 32 对差分信号）
- 供电（12V、3.3V、地线）
- 控制信号
→ 最多只能留出 4 条 NVLink 的空间

SXM 插槽：
>4000 个引脚 → 可以轻松容纳 18 条 NVLink + PCIe + 供电
```

**结论**：
> NVSwitch 不是"功能更高级"，而是"需要物理上更多的链路"，这是 PCIe 插槽的物理限制做不到的。

---

### 2.4 没有 NVLink 怎么办？PCIe P2P

**场景**：L4、T4、RTX 3080 等不支持 NVLink 的 GPU。

**解决方案**：PCIe Peer-to-Peer（P2P）

```
不支持 NVLink 的多卡系统（如 4×L4）：

GPU0 ─┐
GPU1 ─┤
GPU2 ─┤─ PCIe 交换芯片（PLX/Broadcom）─ CPU
GPU3 ─┘

GPU0 → GPU1 的数据传输路径：
GPU0 显存 → PCIe 总线 → 交换芯片 → GPU1 显存
```

**性能对比**：

| 指标 | PCIe P2P | NVLink（PCIe 卡）| NVLink + NVSwitch（SXM）|
|------|---------|----------------|----------------------|
| 带宽 | 12-16 GB/s | 100 GB/s | 450 GB/s |
| 延迟 | 5-10 μs | 1-2 μs | 1-2 μs |
| 缓存一致性 | ❌ 不支持 | ✅ 支持 | ✅ 支持 |
| 直接内存访问 | ⚠️ 通过 BAR | ✅ 支持 | ✅ 支持 |

**PCIe P2P 的应用场景**：
- ✅ 推理服务（GPU 间通讯很少）
- ✅ 数据并行训练（只在梯度同步时需要通讯）
- ❌ 模型并行（频繁传输激活值，PCIe P2P 太慢）
- ❌ 流水线并行（需要低延迟传递中间结果）

---

## 三、NUMA 与 GPU 亲和性

### 3.1 什么是 NUMA？

**NUMA (Non-Uniform Memory Access)** = 非一致性内存访问

**单 CPU 系统（UMA）**：
```
    ┌─────┐
    │ CPU │
    └──┬──┘
       │ 内存总线
    ┌──┴──┐
    │ RAM │  ← 所有内存访问延迟一致
    └─────┘
```

**多 CPU 系统（NUMA）**：
```
┌─────────────────────┬─────────────────────┐
│   NUMA 节点 0       │   NUMA 节点 1       │
│   ┌─────┐           │   ┌─────┐           │
│   │CPU0 │           │   │CPU1 │           │
│   └──┬──┘           │   └──┬──┘           │
│      │              │      │              │
│   ┌──┴──┐           │   ┌──┴──┐           │
│   │RAM0 │           │   │RAM1 │           │
│   └─────┘           │   └─────┘           │
└─────────────────────┴─────────────────────┘
         │                     │
         └──── QPI/UPI ────────┘
              (跨节点互联)
```

**关键特性：**
- CPU0 访问 RAM0：**本地访问**，快（~100ns）
- CPU0 访问 RAM1：**远程访问**，慢（~140ns，延迟增加 40%）
- CPU 和内存成组，每组是一个 NUMA 节点

---

### 3.2 GPU 和 NUMA 的关系

**典型双路服务器 + 4 卡 GPU**：
```
┌────────── NUMA 节点 0 ──────────┐  ┌────────── NUMA 节点 1 ──────────┐
│  CPU0          RAM0            │  │  CPU1          RAM1            │
│   │             │              │  │   │             │              │
│   └── PCIe Root Complex 0      │  │   └── PCIe Root Complex 1      │
│        │          │            │  │        │          │            │
│      GPU0       GPU1           │  │      GPU2       GPU3           │
└────────────────────────────────┘  └────────────────────────────────┘
```

**性能影响：**

| 访问路径 | 延迟 | 带宽 | 影响 |
|---------|------|------|------|
| CPU0 → GPU0 | 低 | 满带宽 | ✅ 本地访问（推荐）|
| CPU0 → GPU2 | 高 | 降低 30% | ⚠️ 跨 NUMA（性能下降）|
| GPU0 → RAM0 | 低 | 满带宽 | ✅ 本地内存 |
| GPU0 → RAM1 | 高 | 降低 30% | ⚠️ 跨 NUMA |

**实测影响（数据加载）**：
```bash
# 正确绑定（本地 NUMA）
numactl --cpunodebind=0 --membind=0 python train.py --gpu=0
→ 数据加载：2.5 GB/s

# 错误绑定（跨 NUMA）
numactl --cpunodebind=1 --membind=1 python train.py --gpu=0
→ 数据加载：1.7 GB/s（慢 32%）
```

---

### 3.3 查看 NUMA 拓扑

**查看 NUMA 节点数：**
```bash
numactl --hardware

# 输出示例：
available: 2 nodes (0-1)
node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11
node 0 size: 128 GB
node 1 cpus: 12 13 14 15 16 17 18 19 20 21 22 23
node 1 size: 128 GB
node distances:
node   0   1
  0:  10  21    ← 本地 10，远程 21（延迟比）
  1:  21  10
```

**查看 GPU 的 NUMA 亲和性：**
```bash
nvidia-smi topo -m

# 输出示例：
        GPU0  GPU1  CPU Affinity  NUMA Affinity
GPU0     X    NV4    0-11          0           ← GPU0 属于 NUMA 0
GPU1    NV4    X     0-11          0
GPU2    SYS   SYS   12-23          1           ← GPU2 属于 NUMA 1
GPU3    SYS   SYS   12-23          1
```

**关键字段解释：**
- **CPU Affinity**：推荐绑定的 CPU 核心范围
- **NUMA Affinity**：GPU 所属的 NUMA 节点
- **SYS**：跨 NUMA 访问（慢）

---

### 3.4 如何正确绑定 NUMA

**方法 1：使用 numactl（命令行）**
```bash
# 绑定进程到 NUMA 0，使用 GPU0
numactl --cpunodebind=0 --membind=0 python train.py --gpu=0

# 绑定进程到 NUMA 1，使用 GPU2
numactl --cpunodebind=1 --membind=1 python train.py --gpu=2
```

**方法 2：容器环境（Docker/K8s）**
```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: training
    resources:
      limits:
        nvidia.com/gpu: 1
    env:
    - name: CUDA_VISIBLE_DEVICES
      value: "0"
  # Kubernetes Topology Manager 自动处理 NUMA 亲和性
  topologySpreadConstraints:
  - topologyKey: kubernetes.io/hostname
```

**方法 3：PyTorch DataLoader（自动绑定）**
```python
import torch

# PyTorch 会自动将 DataLoader 的 worker 绑定到 GPU 的 NUMA 节点
train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    num_workers=8,  # 自动绑定到 GPU 的 NUMA 节点
    pin_memory=True
)
```

---

### 3.5 NUMA 优化建议

**生产环境最佳实践：**

1. **查看拓扑后再分配**
   ```bash
   nvidia-smi topo -m  # 先看 NUMA 分布
   # 根据输出分配任务到对应 NUMA 节点
   ```

2. **避免跨 NUMA 访问**
   - ❌ 错误：CPU0 控制 GPU2（跨 NUMA）
   - ✅ 正确：CPU0 控制 GPU0/GPU1（本地 NUMA）

3. **多进程训练时分组**
   ```bash
   # 4 卡训练，按 NUMA 分组
   # 进程 0,1 在 NUMA 0
   CUDA_VISIBLE_DEVICES=0,1 numactl --cpunodebind=0 --membind=0 python -m torch.distributed.launch --nproc_per_node=2 train.py &
   
   # 进程 2,3 在 NUMA 1
   CUDA_VISIBLE_DEVICES=2,3 numactl --cpunodebind=1 --membind=1 python -m torch.distributed.launch --nproc_per_node=2 train.py &
   ```

4. **K8s 环境启用 Topology Manager**
   ```yaml
   # kubelet 配置
   topologyManagerPolicy: single-numa-node  # 强制单 NUMA 分配
   ```

---

## 四、跨节点通讯：多服务器互联

### 4.1 传统网络通讯的瓶颈

**问题**：GPU 和远程服务器上的 GPU 通讯，数据需要经过多次拷贝：

```
传统网络通讯路径：
节点 A:
GPU 显存 ──PCIe──> CPU 内存 ──PCIe──> 网卡 ──→ 网络

节点 B:
网络 ──→ 网卡 ──PCIe──> CPU 内存 ──PCIe──> GPU 显存

瓶颈：
1. 多次内存拷贝（GPU→CPU→网卡，4 次拷贝）
2. CPU 参与每次传输（占用 CPU 资源）
3. 带宽损失（实际可用带宽只有网卡的 60-70%）
```

---

### 3.2 GPUDirect RDMA

**技术原理**：让网卡直接访问 GPU 显存，绕过 CPU。

**实现方式**：
```
GPUDirect RDMA 路径：
节点 A:
GPU 显存 ──直接──> 网卡 ──→ 网络

节点 B:
网络 ──→ 网卡 ──直接──> GPU 显存

优势：
1. 零拷贝（GPU 显存直接到网卡）
2. CPU 不参与（释放 CPU 资源）
3. 延迟降低 50%+
```

**技术要求**：
1. ✅ 支持 RDMA 的网卡（InfiniBand、RoCE）
2. ✅ NVIDIA GPUDirect 驱动模块
3. ✅ 网卡厂商提供的 GPU 内存注册接口

**性能提升**：

| 场景 | 传统 TCP/IP | GPUDirect RDMA | 提升 |
|------|------------|---------------|------|
| 带宽利用率 | 60-70% | 90-95% | +30-50% |
| 延迟 | 10-20 μs | 4-6 μs | 减少 50%+ |
| CPU 占用 | 40-60% | <5% | 释放 CPU |

---

### 4.3 NIC（网卡）与网络技术

#### 4.3.1 NIC 的作用

**NIC (Network Interface Card)** = 网卡，负责节点间的网络通讯。

**传统 NIC（以太网）**：
```
应用 → CPU 内存 → 网卡驱动 → NIC 硬件 → 网络
         ↑ CPU 参与每次传输
```

**RDMA NIC（InfiniBand/RoCE）**：
```
应用 → GPU 显存 → NIC 硬件 → 网络
       ↑ 绕过 CPU，零拷贝
```

**GPU 训练集群的 NIC 配置：**
```
典型 8 卡训练节点：
┌─────────────────────────┐
│  8×GPU (H100/A100)      │
│  ↓ PCIe/NVLink          │
│  主板                    │
│  ↓ PCIe                 │
│  8×NIC (每 GPU 一个)     │  ← 每 GPU 配一个专用网卡
└────────┬────────────────┘
         │
      交换机
```

**为什么每 GPU 配一个 NIC？**
- 单 NIC 带宽不够（200 Gbps vs 8×GPU 需要 1600+ Gbps）
- GPUDirect RDMA 性能最佳（GPU 直连网卡，避免竞争）
- 降低单点故障风险

---

#### 4.3.2 InfiniBand 详解

**架构特点：**
```
InfiniBand 完整栈：
┌────────────────────────┐
│  应用 (PyTorch/NCCL)   │
├────────────────────────┤
│  Verbs API (IB 接口)   │
├────────────────────────┤
│  IB 驱动 (OFED)        │  ← Mellanox OFED 驱动
├────────────────────────┤
│  IB HCA (Host Channel  │  ← InfiniBand 网卡
│         Adapter)       │
├────────────────────────┤
│  IB 交换机              │
└────────────────────────┘
```

**代际演进：**

| 代际 | 速率 | 延迟 | 发布年份 | 典型产品 |
|------|------|------|---------|---------|
| FDR | 56 Gb/s | 0.7 μs | 2011 | ConnectX-3 |
| EDR | 100 Gb/s | 0.5 μs | 2014 | ConnectX-4 |
| HDR | 200 Gb/s | 0.6 μs | 2017 | ConnectX-6 |
| NDR | 400 Gb/s | 0.65 μs | 2020 | ConnectX-7 |
| XDR | 800 Gb/s | — | 2024+ | ConnectX-8 |

**拓扑设计：**

**Leaf-Spine 架构（标准设计）**：
```
        ┌──── Spine 交换机 ────┐
        │   (核心层，高带宽)   │
    ┌───┴───┬───────┬───────┬───┴───┐
    │       │       │       │       │
Leaf 1  Leaf 2  Leaf 3  Leaf 4  Leaf 5
  │       │       │       │       │
 节点   节点    节点    节点    节点
(8卡)  (8卡)  (8卡)  (8卡)  (8卡)

- 每个节点连接到 1 个 Leaf 交换机
- 每个 Leaf 上联所有 Spine
- 任意两节点间：2 跳（Leaf → Spine → Leaf）
```

**DGX SuperPOD 拓扑**：
```
Fat-Tree 拓扑（3 层）：
        Core (核心)
          │
        Aggregation (汇聚)
          │
        Access (接入)
          │
        计算节点

- 单 SuperPOD：32 个 DGX 节点（256 × H100）
- 总带宽：102.4 Tbps
- 任意两 GPU 间：<2 μs 延迟
```

**InfiniBand 的优势：**
- ✅ **RDMA 原生支持**（协议层内置，无需 CPU）
- ✅ **无损网络**（硬件级流控，丢包率 <10⁻¹²）
- ✅ **低延迟**（硬件卸载，<1 μs）
- ✅ **生态成熟**（NCCL、MPI 深度优化）
- ✅ **QoS 保证**（服务质量，优先级控制）

**劣势：**
- ❌ **成本高**（网卡 + 交换机贵 2-3 倍）
- ❌ **专用设备**（不兼容以太网）
- ❌ **厂商锁定**（主要是 NVIDIA/Mellanox）

---

#### 4.3.3 RoCE（RDMA over Converged Ethernet）

**什么是 RoCE？**
- 在**以太网**物理层上跑 **RDMA** 协议
- 复用现有以太网基础设施
- 成本比 InfiniBand 低 30-50%

**RoCE 版本：**

| 版本 | 协议栈 | 路由能力 | 适用场景 |
|------|--------|---------|---------|
| **RoCE v1** | Ethernet + IB | ❌ 二层（不可路由）| 小规模（同子网）|
| **RoCE v2** | Ethernet + UDP + IB | ✅ 三层（可路由）| **生产推荐** |

**RoCE v2 协议栈：**
```
┌────────────────────────┐
│  应用 (PyTorch)        │
├────────────────────────┤
│  Verbs API             │
├────────────────────────┤
│  InfiniBand 传输层     │  ← RDMA 核心
├────────────────────────┤
│  UDP/IP                │  ← 封装在以太网内
├────────────────────────┤
│  Ethernet (物理层)     │
└────────────────────────┘
```

**关键要求：无损以太网**

RoCE 需要配置以太网交换机支持：

1. **PFC (Priority Flow Control)**：
   ```
   作用：防止丢包
   原理：接收端缓冲区满时发送 PAUSE 帧，暂停发送
   配置：在交换机启用 PFC（通常优先级 3）
   ```

2. **ECN (Explicit Congestion Notification)**：
   ```
   作用：拥塞控制
   原理：网络拥塞时标记数据包，发送端降速
   配置：交换机和网卡同时启用 ECN
   ```

3. **DCQCN (Data Center Quantized Congestion Notification)**：
   ```
   作用：动态调整发送速率
   原理：根据 ECN 反馈调整拥塞窗口
   ```

**交换机配置示例（Mellanox）**：
```bash
# 启用 PFC
interface ethernet 1/1
  dcb priority-flow-control mode on force
  dcb priority-flow-control priority 3 enable

# 启用 ECN
  traffic-class 3
    congestion-control ecn minimum-absolute 150 maximum-absolute 1500
```

**性能对比：**

| 指标 | InfiniBand | RoCE v2 | TCP/IP 以太网 |
|------|-----------|---------|-------------|
| 带宽 | 200-400 Gbps | 100-200 Gbps | 10-100 Gbps |
| 延迟 | <1 μs | 2-5 μs | 10-100 μs |
| CPU 占用 | <1% | <5% | 30-60% |
| 丢包率 | ~0 | <10⁻⁹ (配置正确) | 10⁻⁶ |
| 成本 | 高 | 中 | 低 |
| 配置复杂度 | 低 | 高（需要调优）| 低 |

**RoCE 的适用场景：**
- ✅ **预算有限的训练集群**（成本比 IB 低）
- ✅ **已有以太网基础设施**（复用现有投资）
- ✅ **中小规模训练**（16-64 卡）
- ⚠️ **需要专业网络调优**（否则性能不稳定）

**RoCE 的坑：**
- ⚠️ 交换机必须支持无损以太网（不是所有以太网交换机都行）
- ⚠️ 配置错误导致丢包 → 性能暴跌
- ⚠️ 多租户环境隔离困难（不同任务争抢带宽）

---

#### 4.3.4 网络技术选择决策

**决策矩阵：**

| 场景 | 规模 | 推荐方案 | 原因 |
|------|------|---------|------|
| **推理服务** | 任意 | **TCP/IP 以太网** | 通讯量小，成本优先 |
| **小规模训练** | <16 卡 | **RoCE v2** 或以太网 | 成本敏感，单节点为主 |
| **中规模训练** | 16-128 卡 | **RoCE v2** | 性价比最优 |
| **大规模训练** | 128+ 卡 | **InfiniBand** | 性能和稳定性必需 |
| **超大规模训练** | 1000+ 卡 | **InfiniBand** | 唯一选择 |

**成本对比（200 Gbps 网络）**：

| 方案 | 网卡成本 | 交换机成本 | 总成本（8 节点）|
|------|---------|-----------|---------------|
| InfiniBand HDR | $1500/张 | $50000 | ~$110k |
| RoCE 200G | $800/张 | $30000 | ~$70k |
| 以太网 100G | $300/张 | $10000 | ~$30k |

**性能/成本比：**
- InfiniBand：1.0（基准）
- RoCE：0.7 性能 / 0.6 成本 = **1.17**（性价比高）
- 以太网：0.3 性能 / 0.3 成本 = 1.0

---

### 3.4 GPUDirect Storage

**问题**：训练时从存储加载数据，传统路径：

```
NVMe SSD → CPU 内存 → GPU 显存

瓶颈：
1. 数据需要先到 CPU 内存
2. 占用 CPU 和 PCIe 带宽
```

**GPUDirect Storage**：NVMe SSD 直接写入 GPU 显存

```
NVMe SSD ──直接──> GPU 显存

带宽：
- 单个 NVMe：7 GB/s
- 8 × NVMe RAID：50+ GB/s 直达 GPU
```

**适用场景**：
- ✅ 大规模数据集训练（ImageNet、视频数据）
- ✅ 推荐系统（海量特征数据）
- ⚠️ 需要支持的文件系统（GPUDirect Storage 驱动）

---

## 四、GPU 计算中的通讯模式

### 4.1 为什么需要多 GPU 通讯？

**单 GPU 的限制**：
- ❌ 显存不够（GPT-3 175B 需要 350 GB+，单卡最大 80 GB）
- ❌ 算力不够（训练时间太长）
- ❌ 批次大小受限（影响训练效果）

**多 GPU 的挑战**：
- ⚠️ 如何切分模型？
- ⚠️ 如何切分数据？
- ⚠️ 如何同步梯度？
- ⚠️ 如何最小化通讯开销？

---

### 4.2 数据并行（Data Parallelism）

**原理**：每个 GPU 拥有完整模型，处理不同的数据。

```
              完整模型副本
GPU0: [模型] ← batch 0-7
GPU1: [模型] ← batch 8-15
GPU2: [模型] ← batch 16-23
GPU3: [模型] ← batch 24-31

前向传播（各自独立）
    ↓
反向传播（各自计算梯度）
    ↓
梯度同步（AllReduce）← 需要 GPU 间通讯
    ↓
更新参数（各自更新）
```

**通讯模式**：**AllReduce**（全规约）

```
步骤：
1. 每个 GPU 计算自己的梯度
2. 所有 GPU 交换梯度并求平均
3. 每个 GPU 得到相同的平均梯度
4. 各自更新参数

通讯量：
- 模型大小：7B 参数 × 4 字节 = 28 GB
- 每次迭代需要传输：28 GB（梯度）
- 频率：每个 batch 都要同步
```

**通讯开销分析**：

| 场景 | 通讯方式 | 单次通讯量 | 耗时 | 瓶颈 |
|------|---------|-----------|------|------|
| 8×A100 NVLink | AllReduce | 28 GB | ~100 ms | ✅ 不是瓶颈 |
| 8×L4 PCIe P2P | AllReduce | 28 GB | ~2 秒 | ⚠️ 通讯成瓶颈 |
| 跨节点（InfiniBand） | AllReduce | 28 GB | ~300 ms | ⚠️ 需要优化 |

**优化技术**：
- **梯度累积**：多个 batch 累积后再同步（减少通讯次数）
- **混合精度训练**：FP16 梯度（减半通讯量）
- **梯度压缩**：量化或稀疏化梯度

**适用场景**：
- ✅ 模型小于单卡显存（如 7B 模型 × A100 80GB）
- ✅ 需要大 batch size（提升训练效果）
- ⚠️ 模型太大（如 70B）就放不下了

---

### 4.3 模型并行（Model Parallelism）

**原理**：把模型切分到多个 GPU，每个 GPU 只保存部分层。

**张量并行（Tensor Parallelism）**：切分单个层内部的矩阵

```
Transformer 层的线性层：
Y = X @ W    # W 是权重矩阵（4096 × 4096）

切分方式（列切分）：
GPU0: Y0 = X @ W0  # W0 是前半部分（4096 × 2048）
GPU1: Y1 = X @ W1  # W1 是后半部分（4096 × 2048）

最后拼接：Y = [Y0, Y1]
```

**通讯模式**：

```
前向传播：
GPU0: X → 计算 Y0 ──┐
GPU1: X → 计算 Y1 ──┤→ AllGather → 拼接 Y

反向传播：
GPU0: ← 梯度 G0 ──┐
GPU1: ← 梯度 G1 ──┤→ ReduceScatter → 分发梯度

通讯量：
- 每层需要 2 次通讯（前向 + 反向）
- 单次传输：激活值大小（batch_size × hidden_size）
```

**流水线并行（Pipeline Parallelism）**：切分不同的层

```
4 卡流水线：
GPU0: [Layer 1-8]
GPU1: [Layer 9-16]
GPU2: [Layer 17-24]
GPU3: [Layer 25-32]

执行流程（Micro-batch 流水线）：
时间 t0: GPU0 处理 batch0
时间 t1: GPU0 处理 batch1, GPU1 处理 batch0
时间 t2: GPU0 处理 batch2, GPU1 处理 batch1, GPU2 处理 batch0
...

通讯量：
- 只需要传递中间激活值（layer 输出）
- 通讯量 << 模型大小
```

**通讯开销对比**：

| 并行方式 | 通讯频率 | 单次通讯量 | 对带宽要求 |
|---------|---------|-----------|----------|
| **数据并行** | 每个 batch | 28 GB（梯度）| 中等 |
| **张量并行** | 每层前向+反向 | 几百 MB（激活值）| **极高**（需要 NVLink）|
| **流水线并行** | 每层输出 | 几百 MB | 低（可用 PCIe P2P）|

**关键结论**：
> **张量并行必须用 NVLink**，否则通讯延迟会严重拖慢训练速度。
>
> 流水线并行可以跨节点（InfiniBand 够用）。

---

### 4.4 3D 并行（现代大模型训练）

**结合三种并行**：

```
GPT-3 175B 训练（示例）：
- 数据并行：64 个节点（每节点 8 卡）
- 张量并行：8 卡（节点内 NVLink）
- 流水线并行：64 个节点（跨节点 InfiniBand）

总卡数：64 × 8 = 512 × A100

通讯层次：
1. 节点内（NVLink）：张量并行的 AllGather/ReduceScatter
2. 跨节点（InfiniBand）：流水线并行的激活值传递
3. 全局（InfiniBand）：数据并行的梯度同步
```

**为什么需要这样设计？**

| 并行维度 | 解决的问题 | 通讯特点 |
|---------|-----------|---------|
| **数据并行** | 扩大 batch size | 低频高带宽（每 batch 同步一次）|
| **张量并行** | 单层放不下单卡 | 高频高带宽（每层都要通讯）|
| **流水线并行** | 模型太深 | 中频中带宽（层间传递）|

**通讯优化原则**：
1. **高频通讯用 NVLink**（节点内张量并行）
2. **中频通讯用 InfiniBand**（跨节点流水线）
3. **低频通讯可以慢一点**（数据并行梯度同步可以和计算重叠）

---

### 4.5 通讯库：NCCL

**NVIDIA Collective Communications Library**：GPU 集合通讯的标准库。

**支持的通讯原语**：

| 操作 | 说明 | 数据并行 | 张量并行 |
|------|------|---------|---------|
| **AllReduce** | 所有卡求和并广播结果 | ✅ 梯度同步 | ✅ |
| **AllGather** | 收集所有卡的数据并拼接 | ⚠️ | ✅ 拼接输出 |
| **ReduceScatter** | 求和后分发不同部分 | ⚠️ | ✅ 分发梯度 |
| **Broadcast** | 一个卡广播到所有卡 | ✅ 分发参数 | ⚠️ |

**NCCL 的拓扑感知**：
```
自动检测硬件拓扑：
- 节点内：优先使用 NVLink
- 跨节点：自动使用 InfiniBand/RoCE
- 多级拓扑：自动选择最优路径

性能：
- NVLink（节点内）：接近硬件峰值（300+ GB/s）
- InfiniBand（跨节点）：接近网络峰值（180+ Gb/s）
```

---

## 五、常见误区澄清

### 误区 1：SXM 是更高级的协议

**错误理解**：SXM 比 PCIe 更先进，性能更好。

**正确理解**：
- SXM 只是物理接口（插槽形态），不是协议
- SXM 和 PCIe 插槽都运行 PCIe 协议（CPU ↔ GPU）
- SXM 的优势是**更多引脚 → 更多 NVLink 链路 + 更高供电**

---

### 误区 2：PCIe 卡不支持 NVLink

**错误理解**：只有 SXM 卡才有 NVLink。

**正确理解**：
- PCIe 卡也能有 NVLink（如 RTX 3090、A100 PCIe）
- 差异在于链路数：PCIe 卡 2-4 条，SXM 卡 12-18 条
- 物理限制：PCIe 插槽引脚不够，装不下更多链路

---

### 误区 3：有了 NVLink 就不需要 PCIe

**错误理解**：NVLink 可以替代 PCIe。

**正确理解**：
- NVLink 只负责 GPU ↔ GPU
- CPU ↔ GPU 永远走 PCIe 协议
- 即使是 SXM 卡，也同时有 PCIe 和 NVLink

---

### 误区 4：同架构的 GPU 接口一样

**错误理解**：Ampere 架构的 GPU 都有相同的接口。

**正确理解**：
- A100 有 PCIe 和 SXM 两个版本
- RTX 3090（Ampere）只有 PCIe 版本
- 接口由产品定位决定，不只是架构

---

### 误区 5：推理服务需要 NVLink

**错误理解**：多卡推理必须用 NVLink。

**正确理解**：
- 推理通常是数据并行（每卡处理独立请求）
- GPU 间通讯很少，PCIe P2P 足够
- L4、T4 这些推理卡故意不配 NVLink（降成本）

---

## 六、实际部署案例

### 案例 1：消费级 AI 工作站

**配置**：
- 2 × RTX 4090（PCIe 插槽）
- NVLink 桥接
- PCIe 4.0 主板

**通讯架构**：
```
GPU0 ←NVLink→ GPU1  (100 GB/s)
  ↓ PCIe 4.0    ↓ PCIe 4.0
      主板 (64 GB/s 每卡)
          ↓
        CPU
```

**适用场景**：
- ✅ 小模型训练（7B - 13B）
- ✅ 推理服务
- ✅ 数据并行训练
- ❌ 张量并行（NVLink 带宽不够）

---

### 案例 2：单节点训练服务器

**配置**：
- 4 × A100 PCIe 80GB
- 无 NVSwitch（菊花链拓扑）
- PCIe 4.0 服务器

**通讯架构**：
```
GPU0 ←NVLink→ GPU1 ←NVLink→ GPU2 ←NVLink→ GPU3
  ↓           ↓           ↓           ↓
        PCIe 4.0 交换芯片
                ↓
              CPU
```

**瓶颈**：
- ⚠️ GPU0 到 GPU3 需要中转（带宽减半）
- ⚠️ 张量并行性能受限

**适用场景**：
- ✅ 数据并行（每卡独立）
- ✅ 流水线并行（顺序传递）
- ⚠️ 张量并行（非对称带宽影响性能）

---

### 案例 3：DGX H100（8 卡 SXM）

**配置**：
- 8 × H100 SXM5 80GB
- 4 × NVSwitch 3.0
- 8 × InfiniBand NDR 400 Gb/s

**通讯架构**：
```
节点内：
   ┌─── 4×NVSwitch ───┐
GPU0──┤  全连接拓扑    ├──GPU4
GPU1──┤  450 GB/s      ├──GPU5
GPU2──┤  任意两卡      ├──GPU6
GPU3──┤                ├──GPU7
   └───────────────────┘

跨节点：
8×InfiniBand NDR (每卡一个，400 Gb/s)
→ GPUDirect RDMA
```

**能力**：
- ✅ 完美支持张量并行（节点内全连接）
- ✅ 流水线并行（InfiniBand 跨节点）
- ✅ 数据并行（多节点扩展）
- ✅ 3D 并行（三种并行混合）

**价格**：~$300,000+

---

### 案例 4：云推理集群

**配置**：
- 4 × L4 24GB（PCIe 插槽）
- 无 NVLink
- 10 GbE 以太网

**通讯架构**：
```
GPU0 ─┐
GPU1 ─┤
GPU2 ─┤─ PCIe 交换芯片 ─ CPU ─ 10 GbE
GPU3 ─┘
```

**场景**：
- ✅ 推理服务（每卡处理独立请求）
- ✅ 小 batch 推理（延迟优先）
- ⚠️ GPU 间几乎无通讯
- ❌ 不适合训练

**成本**：~$40,000（比 DGX 便宜 7 倍）

---

## 七、选择决策指南

### 7.1 决策树：我该选什么配置？

```
问题 1：用途是什么？
├─ 推理服务
│   └→ PCIe 插槽 + 无 NVLink（L4、T4）
│
├─ 小规模训练（<70B 模型）
│   ├─ 预算 <$50k → PCIe 卡 + NVLink 桥接（RTX 4090 ×2）
│   └─ 预算 >$100k → SXM 卡 + NVSwitch（DGX）
│
└─ 大规模训练（>70B 模型）
    ├─ 单节点不够 → 多节点 + InfiniBand
    └─ 必须 SXM + NVSwitch（张量并行需求）
```

### 7.2 关键问题检查清单

**硬件选择**：
1. ✅ 单卡显存够吗？（模型大小 × 1.2 < 显存）
2. ✅ 需要几张卡？（算力需求 + 显存需求）
3. ✅ 需要张量并行吗？（模型 >70B 或单层太大）
4. ✅ 预算是多少？（PCIe 灵活便宜，SXM 性能好但贵）

**通讯需求**：
1. ✅ GPU 间通讯频率？（每层 vs 每 batch）
2. ✅ 单次通讯量？（几 MB vs 几 GB）
3. ✅ 是否跨节点？（需要 InfiniBand 吗？）
4. ✅ 延迟敏感度？（推理低延迟 vs 训练高吞吐）

**决策矩阵**：

| 场景 | GPU 型号 | 接口 | NVLink | 网络 | 原因 |
|------|---------|------|--------|------|------|
| 推理服务 | L4 / T4 | PCIe | ❌ 无 | 10 GbE | 通讯少，成本优先 |
| 小模型训练 | RTX 4090 ×2 | PCIe | ✅ 桥接 | — | 数据并行够用 |
| 中等模型训练 | A100 PCIe ×4 | PCIe | ✅ 4条 | — | 流水线 + 数据并行 |
| 大模型训练 | H100 SXM ×8 | SXM | ✅ 18条 | InfiniBand | 3D 并行 |
| 超大规模训练 | 多节点 H100 | SXM | ✅ 18条 | InfiniBand | 千卡集群 |

---

## 八、实用命令和检测

### 8.1 检测 GPU 接口类型

```bash
# 查看 GPU 型号
nvidia-smi --query-gpu=name --format=csv

# 查看 PCIe 版本和带宽
nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv

# 输出示例：
# pcie.link.gen.current, pcie.link.width.current
# 4, 16    ← PCIe 4.0 x16
```

### 8.2 检测 NVLink 状态

```bash
# 查看 NVLink 拓扑
nvidia-smi nvlink --status

# 输出示例：
# GPU 0: A100-SXM4-80GB
# 	 Link 0: <connected to GPU 1>
# 	 Link 1: <connected to GPU 2>
#  ...

# 查看 NVLink 带宽利用率
nvidia-smi nvlink -g 0 -l
```

### 8.3 测试 GPU 间通讯带宽

```bash
# 使用 NCCL 官方测试工具
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 2

# 输出示例：
#       size         time   algbw   busbw
#     4194304       0.01   42.0GB/s   42.0GB/s    ← 实际带宽
```

### 8.4 nvidia-smi topo -m 详细解读

**基础命令：**
```bash
nvidia-smi topo -m
```

**完整输出示例（8 卡 DGX H100）：**
```
        GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7  mlx5_0 mlx5_1  CPU Affinity  NUMA Affinity
GPU0     X    NV18  NV18  NV18  NV18  NV18  NV18  NV18   SYS    SYS      0-23         0
GPU1    NV18   X    NV18  NV18  NV18  NV18  NV18  NV18   SYS    SYS      0-23         0
GPU2    NV18  NV18   X    NV18  NV18  NV18  NV18  NV18   SYS    SYS      0-23         0
GPU3    NV18  NV18  NV18   X    NV18  NV18  NV18  NV18   SYS    SYS      0-23         0
GPU4    NV18  NV18  NV18  NV18   X    NV18  NV18  NV18   PHB    SYS     24-47         1
GPU5    NV18  NV18  NV18  NV18  NV18   X    NV18  NV18   SYS    PHB     24-47         1
GPU6    NV18  NV18  NV18  NV18  NV18  NV18   X    NV18   SYS    SYS     24-47         1
GPU7    NV18  NV18  NV18  NV18  NV18  NV18  NV18   X     SYS    SYS     24-47         1
mlx5_0  SYS   SYS   SYS   SYS   PHB   SYS   SYS   SYS     X      PIX
mlx5_1  SYS   SYS   SYS   SYS   SYS   PHB   SYS   SYS    PIX     X

Legend:
  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```

---

#### 8.4.1 符号详细解释

**核心符号性能排序（从快到慢）：**

| 符号 | 全称 | 含义 | 典型带宽 | 延迟 | 适用场景 |
|------|------|------|---------|------|---------|
| **X** | Self | 自己 | N/A | N/A | 自身 |
| **NV#** | NVLink × # 条 | 通过 # 条 NVLink 直连 | 100-900 GB/s | 1-2 μs | **最快**，GPU 间高频通讯 |
| **PIX** | PCIe Internal Exchange | 通过单个 PCIe 桥接芯片 | 32-64 GB/s | 3-5 μs | 同 PCIe 交换芯片下的设备 |
| **PXB** | PCIe Cross Bridge | 通过多个 PCIe 桥接芯片 | 16-32 GB/s | 5-8 μs | 跨多个 PCIe 交换芯片 |
| **PHB** | PCIe Host Bridge | 通过 PCIe 主桥（CPU） | 12-32 GB/s | 8-12 μs | 跨 PCIe root complex |
| **NODE** | NUMA Node | 同 NUMA 节点，跨 PCIe 主桥 | 8-16 GB/s | 12-20 μs | 同 NUMA，不同 PCIe root |
| **SYS** | System | 跨 NUMA 节点 + PCIe | 4-12 GB/s | 20-40 μs | **最慢**，跨 CPU socket |

---

#### 8.4.2 实际案例解析

**案例 1：双路服务器 + 4×A100 PCIe**
```
        GPU0  GPU1  GPU2  GPU3  CPU Affinity  NUMA Affinity
GPU0     X    NV4   PHB   SYS    0-23          0
GPU1    NV4    X    SYS   PHB    0-23          0
GPU2    PHB   SYS    X    NV4   24-47          1
GPU3    SYS   PHB   NV4    X    24-47          1
```

**解读：**
- **GPU0 ↔ GPU1**：`NV4` = 4 条 NVLink（100 GB/s）
  - 属于同一 NUMA 节点 0
  - 最快的通讯路径
  
- **GPU0 ↔ GPU2**：`PHB` = 通过 PCIe 主桥（CPU）
  - 跨 PCIe root complex
  - 带宽 ~16 GB/s
  
- **GPU0 ↔ GPU3**：`SYS` = 跨 NUMA 节点
  - 需要通过 QPI/UPI 跨 CPU socket
  - 带宽 ~8 GB/s，延迟最高

**优化建议：**
- ✅ 训练时配对：(GPU0+GPU1) 和 (GPU2+GPU3)
- ✅ 绑定 NUMA：进程 0-1 用 GPU0-1（NUMA 0），进程 2-3 用 GPU2-3（NUMA 1）
- ❌ 避免：GPU0 和 GPU3 频繁通讯（跨 NUMA，最慢）

---

**案例 2：8 卡 DGX H100（NVSwitch 全连接）**
```
        GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
GPU0     X    NV18  NV18  NV18  NV18  NV18  NV18  NV18
GPU1    NV18   X    NV18  NV18  NV18  NV18  NV18  NV18
...（所有都是 NV18）
```

**解读：**
- **任意两个 GPU**：`NV18` = 18 条 NVLink（450 GB/s）
- **NVSwitch 的作用**：实现全连接拓扑
- **性能一致**：任意 GPU 对之间带宽相同

**训练优势：**
- ✅ 张量并行无瓶颈（任意切分都是全速）
- ✅ 流水线并行灵活（不需要考虑拓扑）
- ✅ 数据并行高效（AllReduce 均匀分布）

---

**案例 3：推理集群 4×L4（无 NVLink）**
```
        GPU0  GPU1  GPU2  GPU3  CPU Affinity  NUMA Affinity
GPU0     X    PIX   PXB   PHB    0-63          0
GPU1    PIX    X    PIX   PXB    0-63          0
GPU2    PXB   PIX    X    PIX    0-63          0
GPU3    PHB   PXB   PIX    X     0-63          0
```

**解读：**
- **无 NVLink**：所有通讯走 PCIe
- **GPU0 ↔ GPU1**：`PIX` = 同一个 PCIe 交换芯片（较快）
- **GPU0 ↔ GPU3**：`PHB` = 跨 PCIe root（较慢）

**推理场景：**
- ✅ 每 GPU 处理独立请求（无通讯）
- ✅ PCIe 带宽够用（只传输少量输入/输出）
- ⚠️ 不适合需要频繁 GPU 间通讯的训练

---

#### 8.4.3 网卡（NIC）的拓扑

**案例：8 卡 + 8 网卡（每 GPU 一个 IB 网卡）**
```
        GPU0  mlx5_0  mlx5_1  mlx5_2  ...
GPU0     X     PHB     SYS     SYS    ...
GPU1    NV18   SYS     PHB     SYS    ...
GPU2    NV18   SYS     SYS     PHB    ...
...

mlx5_0: InfiniBand 网卡（Mellanox ConnectX-7）
```

**解读：**
- **GPU0 ↔ mlx5_0**：`PHB` = 通过 PCIe 主桥
  - GPU0 和 mlx5_0 在同一 PCIe root
  - GPUDirect RDMA 性能最佳
  
- **GPU0 ↔ mlx5_1**：`SYS` = 跨 NUMA
  - 网卡不在同一 NUMA，性能下降

**最佳配置：**
- ✅ 每 GPU 配对一个**本地 NUMA 的网卡**
- ✅ 避免跨 NUMA 访问网卡
- ✅ 8 卡 → 8 网卡（1:1 配置）

---

#### 8.4.4 查看更多拓扑信息

**查看 P2P 读写能力：**
```bash
nvidia-smi topo -p2p r

# 输出示例（读能力）：
        GPU0  GPU1  GPU2  GPU3
GPU0     -     OK    OK    NOK    ← GPU0 无法从 GPU3 读取
GPU1    OK     -     NOK   OK
GPU2    OK    NOK    -     OK
GPU3    NOK   OK     OK    -

NOK = P2P 不可用（通常因为跨 NUMA 或 IOMMU 限制）
```

**查看 P2P 写能力：**
```bash
nvidia-smi topo -p2p w

# 写能力和读能力可能不同（取决于硬件支持）
```

**查看完整路径信息：**
```bash
nvidia-smi topo --matrix

# 显示详细的 PCIe 路径和带宽
```

---

#### 8.4.5 拓扑优化建议

**训练任务分配原则：**

1. **优先级 1：NVLink 连接**
   ```
   张量并行 → 必须用 NVLink 连接的 GPU
   例如：GPU0-GPU3 有 NV18 → 4 卡张量并行
   ```

2. **优先级 2：同 NUMA 节点**
   ```
   数据并行 → 同 NUMA 节点的 GPU 组
   例如：NUMA 0 有 GPU0-3 → 一组数据并行
         NUMA 1 有 GPU4-7 → 另一组数据并行
   ```

3. **避免：跨 NUMA 频繁通讯**
   ```
   ❌ 错误：GPU0（NUMA 0）和 GPU7（NUMA 1）做张量并行
   ✅ 正确：GPU0-3（NUMA 0）内部做张量并行
   ```

**K8s/SLURM 调度器配置：**
```yaml
# K8s Device Plugin 配置
# 优先调度同 NUMA 节点的 GPU
apiVersion: v1
kind: Pod
spec:
  nodeSelector:
    nvidia.com/gpu.topology: nvlink-enabled
  resources:
    limits:
      nvidia.com/gpu: 4
      nvidia.com/gpu-group: "0"  # 指定同一 NUMA/NVLink 组
```

---

### 8.5 通讯层级和带宽系统对比

**完整通讯层级表（从快到慢）：**

| 层级 | 技术 | 典型带宽 | 延迟 | 使用场景 | 成本 |
|------|------|---------|------|---------|------|
| **GPU 内部** | HBM3e | 4800 GB/s | <100 ns | 显存访问 | 已含 |
| **GPU 间（同节点）** | NVLink 5.0 | 900 GB/s | 1-2 μs | 张量并行 | $$ |
| **GPU 间（同节点）** | NVLink 4.0 | 450 GB/s | 1-2 μs | 张量并行 | $$ |
| **GPU 间（同节点）** | NVLink 3.0 | 300 GB/s | 1-2 μs | 张量并行 | $$ |
| **GPU-CPU** | PCIe 5.0 x16 | 128 GB/s（双向）| 5-10 μs | 数据加载 | 已含 |
| **GPU-CPU** | PCIe 4.0 x16 | 64 GB/s（双向）| 5-10 μs | 数据加载 | 已含 |
| **GPU 间（无 NVLink）** | PCIe P2P (PIX) | 32-64 GB/s | 10-20 μs | 数据并行 | 已含 |
| **GPU 间（跨 NUMA）** | PCIe P2P (SYS) | 8-16 GB/s | 30-50 μs | ❌ 避免 | 已含 |
| **GPU-存储** | GPUDirect Storage | 50 GB/s | 10-20 μs | 数据加载 | $ |
| **跨节点（最快）** | InfiniBand NDR | 50 GB/s（400 Gbps）| <2 μs | 大规模训练 | $$$$ |
| **跨节点（中等）** | RoCE v2 200G | 25 GB/s（200 Gbps）| 5-10 μs | 中规模训练 | $$ |
| **跨节点（基础）** | Ethernet 100G | 12.5 GB/s（100 Gbps）| 50-100 μs | 推理/小训练 | $ |

**带宽可视化对比：**
```
GPU HBM3e        ████████████████████████████████████████████████  4800 GB/s
NVLink 5.0       █████████                                          900 GB/s
NVLink 4.0       █████                                              450 GB/s
PCIe 5.0         ███                                                128 GB/s
IB NDR           █                                                   50 GB/s
PCIe 4.0         █                                                   64 GB/s
PCIe P2P (good)  █                                                   32 GB/s
RoCE 200G        █                                                   25 GB/s
Ethernet 100G    █                                                   12 GB/s
PCIe P2P (SYS)   ▌                                                    8 GB/s
```

**延迟可视化对比：**
```
GPU 内存访问    ▌                   <0.1 μs
NVLink          ██                  1-2 μs
InfiniBand      ██                  1-2 μs
PCIe (local)    █████               5-10 μs
RoCE            ██████              5-10 μs
PCIe P2P (PIX)  ██████████          10-20 μs
PCIe P2P (SYS)  ████████████████████ 30-50 μs
Ethernet 100G   ██████████████████████████████  50-100 μs
```

---

### 8.6 性能测试工具

**带宽测试（GPU 间）：**
```bash
# CUDA 示例测试 P2P 带宽
/usr/local/cuda/samples/bin/p2pBandwidthLatencyTest

# 输出示例：
Unidirectional P2P=Enabled Bandwidth Matrix (GB/s)
   D\D     0      1      2      3
     0  350.2   42.1   42.1   12.3   ← GPU0 到其他 GPU 的带宽
     1   42.1  350.2   12.3   42.1
     2   42.1   12.3  350.2   42.1
     3   12.3   42.1   42.1  350.2
```

**延迟测试：**
```bash
# NCCL 延迟测试
./nccl-tests/build/all_reduce_perf -b 8 -e 8 -f 1 -g 2

# 输出包含延迟信息
```

**网络带宽测试（跨节点）：**
```bash
# iperf3 测试网络带宽
# 节点 A（服务端）
iperf3 -s

# 节点 B（客户端）
iperf3 -c <节点A IP> -t 30 -P 8

# 输出：
[SUM]  0.0-30.0 sec  88.2 GBytes  25.3 Gbits/sec  ← 实际带宽
```

**GPUDirect RDMA 测试：**
```bash
# perftest 套件（IB/RoCE）
ib_write_bw -a -d mlx5_0 -x 0 <远程节点IP>

# -x 0 启用 GPUDirect RDMA
# 输出显示 GPU 到 GPU 的跨节点带宽
```

---

## 九、总结

### 核心要点

| 概念 | 层面 | 作用 | 关键点 |
|------|------|------|--------|
| **PCIe** | 物理 + 协议 | CPU ↔ GPU 通讯 | 永远需要，无法替代 |
| **SXM** | 物理（插槽） | 高密度封装 | 更多引脚 = 更多 NVLink |
| **NVLink** | 协议 | GPU ↔ GPU 高速互联 | PCIe 卡也能有（2-4 条）|
| **NVSwitch** | 硬件（交换机） | 多 GPU 全连接 | 需要 12+ 条链路（只有 SXM）|
| **InfiniBand** | 网络 | 跨节点通讯 | GPUDirect RDMA 必备 |

### 记忆口诀

```
物理接口看需求：
- 推理/消费级 → PCIe（灵活便宜）
- 顶级训练卡 → SXM（高密度高功耗）

通讯协议分场景：
- CPU ↔ GPU → 永远是 PCIe
- GPU ↔ GPU（节点内）→ NVLink（高频通讯必需）
- GPU ↔ GPU（跨节点）→ InfiniBand（大规模训练）

并行策略看模型：
- 小模型 → 数据并行（PCIe P2P 够用）
- 大模型 → 张量并行（必须 NVLink）
- 超大模型 → 3D 并行（NVLink + InfiniBand）
```

### 选择原则

**推理优先成本**：
- L4 / T4（无 NVLink）
- PCIe 插槽
- 以太网

**训练优先性能**：
- A100 / H100（高端 NVLink）
- SXM 插槽 + NVSwitch
- InfiniBand

**工作站平衡**：
- RTX 4090（有 NVLink 桥接）
- PCIe 插槽
- 无需网络

---

## 附录：术语表

| 术语 | 全称 | 说明 |
|------|------|------|
| PCIe | Peripheral Component Interconnect Express | 计算机扩展总线标准 |
| SXM | Server PCI Express Module | NVIDIA 服务器专用 GPU 插槽 |
| NVLink | NVIDIA Link | NVIDIA GPU 间高速互联协议 |
| NVSwitch | NVIDIA Switch | GPU 交换机芯片 |
| RDMA | Remote Direct Memory Access | 远程直接内存访问 |
| GPUDirect | — | NVIDIA 的 GPU 直接访问技术集合 |
| NCCL | NVIDIA Collective Communications Library | GPU 集合通讯库 |
| InfiniBand | — | 高性能计算网络标准 |
| RoCE | RDMA over Converged Ethernet | 在以太网上实现 RDMA |
| DGX | NVIDIA DGX | NVIDIA 的 AI 训练服务器产品线 |
| HGX | NVIDIA HGX | NVIDIA 的 GPU 服务器主板平台 |
| AllReduce | — | 所有节点求和并广播结果 |
| AllGather | — | 收集所有节点数据并拼接 |
| ReduceScatter | — | 求和后分发不同部分给各节点 |
