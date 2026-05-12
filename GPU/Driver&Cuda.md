# Driver 与 CUDA 

## Driver 与 CUDA Toolkit 官方定义对照表

| 项目 | 官方描述 |
|------|---------|
| **NVIDIA Driver** | 运行于操作系统底层的硬件驱动程序，包含内核模块 (Kernel Mode Driver) 与用户态驱动库 (User Mode Driver)，负责 GPU 的资源管理、指令调度与硬件抽象。必须与 GPU 架构匹配。 |
| **驱动工作层级** | 同时存在于内核态 (`nvidia.ko` / `nvidia-open.ko`) 与用户态 (`libcuda.so`、`libnvidia-ml.so` 等)，是上层软件访问 GPU 的唯一通道。 |
| **与操作系统关系** | 作为内核模块加载进 Linux Kernel，通过 PCIe 与 GPU 通信，并向上层提供统一的系统调用接口。 |
| **CUDA Toolkit** | NVIDIA 面向开发者提供的并行计算开发套件 (SDK)，包含 CUDA Runtime (`libcudart`)、编译器 (`nvcc`)、数学库 (cuBLAS/cuFFT 等)、调试与性能分析工具；运行于用户态，通过 CUDA Driver API 间接访问 GPU。 |
| **版本依赖关系** | CUDA Toolkit 的运行依赖底层 NVIDIA Driver 提供的 CUDA Driver API，每个 CUDA Toolkit 版本都有对应的 **Minimum Required Driver Version**。 |
| **兼容性规则** | NVIDIA Driver 对 CUDA Toolkit **向后兼容 (Backward Compatible)**：在不启用 Forward Compatibility 的情况下，系统可运行的 CUDA Toolkit 版本不得高于当前 Driver 所支持的最高 CUDA 版本。 |

---

## 二、官方分层架构（NVIDIA Software Stack）

```
┌──────────────────────────────────────────────┐
│  用户应用 (PyTorch / TensorFlow / 自己的代码)  │   ← 用户空间
├──────────────────────────────────────────────┤
│  CUDA Runtime API (libcudart)                │   ← CUDA Toolkit 提供
│  CUDA Libraries (cuBLAS, cuDNN, NCCL...)     │
├──────────────────────────────────────────────┤
│  CUDA Driver API (libcuda.so)                │   ← Driver 提供（用户态）
├──────────────────────────────────────────────┤
│  NVIDIA Kernel Module (nvidia.ko)            │   ← Driver 提供（内核态）
├──────────────────────────────────────────────┤
│  GPU 硬件                                     │
└──────────────────────────────────────────────┘
```

---

## 三、精确定义

### 1. NVIDIA Driver（驱动）

包含**两部分**：

- **内核模块 (Kernel Mode Driver)**：`nvidia.ko` / `nvidia-open.ko`
  - 加载进 Linux 内核
  - 直接管理 GPU 硬件（显存、调度、中断）
  - 必须与 GPU 架构匹配（Blackwell 需要 ≥570）
  
- **用户态驱动库 (User Mode Driver)**：`libcuda.so`、`libnvidia-ml.so` 等
  - 提供 **CUDA Driver API**
  - 应用程序通过它和内核模块通信
  - `nvidia-smi` 也是基于这个

> 📌 **关键点**：`libcuda.so` 属于 **Driver**，不是 CUDA Toolkit！这是最容易混淆的地方。

---

### 2. CUDA Toolkit（工具包）

是**给开发者的 SDK**，包含：

- **CUDA Runtime** (`libcudart.so`) — 比 Driver API 更高层、更易用
- **编译器** (`nvcc`)
- **数学库** (cuBLAS, cuFFT, cuRAND, cuSPARSE…)
- **深度学习相关** (cuDNN 单独发布但属于这层)
- **调试/性能工具** (nsight, cuda-gdb, nvprof)
- **头文件和示例**

> 📌 CUDA Toolkit **不直接操作硬件**，它最终调用的是 Driver 提供的 `libcuda.so`。

---

## 四、版本兼容性（你提到的"受限关系"）

NVIDIA 官方规则：**Driver 向后兼容 CUDA Toolkit**，反之不成立。

| Driver 版本 | 最高支持的 CUDA Toolkit |
|-------------|------------------------|
| 525.xx      | CUDA 12.0              |
| 535.xx      | CUDA 12.2              |
| 550.xx      | CUDA 12.4              |
| 570.xx      | CUDA 12.8              |
| 580.xx      | CUDA 13.0              |

官方叫这个机制为 **CUDA Forward Compatibility / Minimum Driver Version**。

📖 官方文档：
https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#id4

### 例外：CUDA Forward Compatibility Package
NVIDIA 有个"兼容包"可以让**老 Driver 跑新 CUDA**，但仅限数据中心卡（Tesla/A100/H100 等），消费卡和 Workstation 卡（如 RTX PRO）不支持。

---

## 五、容器场景（一个常见疑惑）

在 Docker / NGC 容器里：

- **宿主机**只需要装 **Driver**
- **容器内部**带自己的 **CUDA Toolkit** 版本

所以你可以宿主机 Driver 580（CUDA 13 能力），容器里跑 CUDA 11.8 的 PyTorch，完全没问题。这正是你说的 "**Driver 版本一定时，CUDA 只能运行 ≤ 对应版本**"。

---

## 六、总结对照表

|              | Driver                              | CUDA Toolkit                       |
|--------------|-------------------------------------|------------------------------------|
| 运行层级     | 内核态 + 用户态底层                 | 用户态                             |
| 是否必需     | ✅ 必需（没它 GPU 不工作）           | ❌ 可选（只跑已编译程序时不需要）  |
| 与硬件关系   | 强绑定（按 GPU 架构）               | 弱绑定（通过 Driver 间接访问）     |
| 谁安装       | 系统管理员（一次装好）              | 开发者（可多版本并存）             |
| 典型文件     | `nvidia.ko`, `libcuda.so`, `nvidia-smi` | `nvcc`, `libcudart.so`, `cuBLAS` |
| 升级频率     | 低（按需）                          | 高（跟随项目）                     |

---

## 七、一句话记忆

> **Driver 让 GPU 能用；CUDA Toolkit 让你能"写程序"用 GPU。**
> **Driver 决定 CUDA 能用到哪一代；CUDA 决定你能用到哪些开发功能。**
