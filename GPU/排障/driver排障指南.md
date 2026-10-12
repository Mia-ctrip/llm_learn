# CUDA 前向兼容（Forward Compatibility）排障指南

> 案例背景：H20 × 8，内核驱动 **570.133.20（CUDA 12.8 上限）**，软件栈是 **CUDA 13.0 Toolkit + torch/vllm cu13**，启动 vllm 报
> `RuntimeError: The NVIDIA driver on your system is too old (found version 12080)`。
> 现象：手动 `export LD_LIBRARY_PATH=/usr/local/cuda/compat` 无效；单跑 python 测试 OK，但 vllm worker 仍报 12080。

---

## 0. 三十秒速查表

```bash
# ① 版本三连（先搞清矛盾在哪）
nvidia-smi --query-gpu=name,driver_version --format=csv   # 内核驱动 → 运行时天花板
nvcc -V                                                   # Toolkit（跟驱动无关！）
python -c "import torch;print(torch.version.cuda)"        # 框架编译时 CUDA

# ② compat 包是否齐全（注意看版本号是否 ≥ 需求）
ls -l /usr/local/cuda-13.0/compat/
cat /proc/driver/nvidia/version

# ③ 真·可用性验证（cuInit + matmul，缺一不可）
LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat python -c "
import ctypes;l=ctypes.CDLL('libcuda.so.1');v=ctypes.c_int()
l.cuDriverGetVersion(ctypes.byref(v));print('api',v.value,'cuInit',l.cuInit(0))"

# ④ 进程实况（唯一可信的“生效”证据）
grep libcuda /proc/<pid>/maps
tr '\0' '\n' < /proc/<pid>/environ | grep -i ld_library
```

---

## 1. 先建立正确的心智模型（90% 的误判源于此）

### 1.1 三个"CUDA 版本"完全是不同的东西

| 概念 | 由谁提供 | 怎么看 | 能否随意更换 |
|---|---|---|---|
| **内核态驱动**<br>`nvidia.ko` + `libcuda.so` | 驱动安装包（`.run`/dkms） | `nvidia-smi` 右上角 `CUDA Version`<br>`cat /proc/driver/nvidia/version` | 需要 root + 卸载模块/重启 |
| **CUDA Toolkit** | `cuda-toolkit-xx-x` 包 | `nvcc -V`，装在 `/usr/local/cuda-13.0` | 随便装，可多版本共存 |
| **框架编译时 CUDA** | pip wheel（cu121/cu128/cu130） | `torch.version.cuda` | pip 换轮子即可 |

> **`yum install cuda-13.0` ≠ 驱动升级到 13.0。**
> `nvcc -V = 13.0` 只说明你有 13.0 的**编译器**；真正跑 kernel 的是驱动里的 `libcuda.so`。
> PyTorch 报的 `found version 12080` 来自 `cuDriverGetVersion()`，读的是**驱动**，不是 toolkit。

### 1.2 驱动版本 → 支持的 CUDA 上限（常用对照）

| 驱动分支 | CUDA 上限 |
|---|---|
| 525.60.13+ | 12.0 |
| 535.54.03+ | 12.2 |
| 550.54.14+ | 12.4 |
| 560.28.03+ | 12.6 |
| **570.26+** | **12.8** ← 本案例 |
| 575.51+ | 12.9 |
| **580.65.06+** | **13.0** ← 需求 |

### 1.3 Forward Compatibility 到底做了什么

```
        用户态                          内核态
  ┌──────────────────┐          ┌───────────────────┐
  │ libcuda.so.580   │ ←compat→ │  nvidia.ko 570    │   ✅ 允许
  │ (来自 compat 包) │          │  (不动它)          │
  └──────────────────┘          └───────────────────┘
```

compat 只替换**用户态** `libcuda.so` / `nvvm` / `ptxjitcompiler`，内核模块仍是宿主的旧驱动。
**硬性前提（缺一条就走不通）：**

1. GPU 必须是 **数据中心卡**（A100/H100/H20/L40S/V100/T4…）。GeForce、Jetson **不支持**，会报 `forward compatibility was attempted on non-supported HW`；
2. 内核驱动 **≥ R418**，且是受支持分支；
3. compat 库版本 **必须高于**已装驱动（本案例 580.167.08 > 570.133.20 ✅）；
4. 用户态 / 内核态 **open 与 proprietary 分支需一致**（`/proc/driver/nvidia/version` 里有 `Open Kernel Module` 字样的要注意）。

---

## 2. 决策树

```
报错 "driver too old / insufficient"
│
├─ Step 1: nvidia-smi 的 CUDA Version < torch.version.cuda ?
│    └─ 否 → 不是版本问题，查 stub 库 / 设备节点（见 §6.4、§6.5）
│
├─ Step 2: GPU 是数据中心卡吗？
│    └─ 否 → compat 路线作废，只能升驱动 or 降 cu 版本（§5）
│
├─ Step 3: compat 目录存在且版本足够吗？（§3.1）
│    └─ 否 → 装 cuda-compat-XX-X；注意 /usr/local/cuda 软链可能没指对
│
├─ Step 4: cuInit == 0 且 matmul 成功吗？（§3.2）
│    ├─ 否（803/804/802）→ 分支不匹配 / 设备节点缺失 → §6.5，或走 §5
│    └─ 是 → compat 本身没问题！问题在**环境传递**
│
└─ Step 5: 目标进程（worker）的 /proc/<pid>/maps 用的是 compat 吗？（§4）
     └─ 否 → 按启动方式注入环境 or 用 ldconfig（§4.2 / §4.3）
```

---

## 3. 验证 compat 本身是否可用

### 3.1 检查 compat 目录

```bash
ls -ld /usr/local/cuda            # ← 先确认软链指向哪个版本！
ls -l  /usr/local/cuda-13.0/compat/
```

**合格的样子**（本案例实测）：

```
libcuda.so.1          -> libcuda.so.580.167.08        ← 必须有 .so.1 软链
libcuda.so.580.167.08
libnvidia-nvvm.so.4   -> libnvidia-nvvm.so.580.167.08 ← CUDA 12+ 必需
libnvidia-ptxjitcompiler.so.1 -> ...580.167.08        ← PTX JIT 必需
libnvidia-gpucomp.so.580.167.08                       ← 580 新增
libcudadebugger.so.1 -> ...
```

**常见坑：**

| 坑 | 表现 / 修法 |
|---|---|
| `/usr/local/cuda` 没指向 13.0 或不存在 | `export .../cuda/compat` 等于写了个空路径 → **一律用绝对版本号路径** |
| 目录为空 / 没装包 | `yum install -y cuda-compat-13-0`（RHEL 系）/ `apt install -y cuda-compat-13-0` |
| 只有 `libcuda.so.580.x`，没有 `libcuda.so.1` | dlopen 找的是 `.so.1` → `ln -sf libcuda.so.580.* libcuda.so.1` |
| 装的是 12.x 的 compat（<580） | 对 cu13 依然不够，要装对应大版本 |
| 容器里目录被改名成 `lib.real` | NVIDIA Container Toolkit 干的，进容器 `ls` 确认 |

### 3.2 功能验证——**两个陷阱一定要避开**

> ⚠️ **陷阱 1**：`cuDriverGetVersion()` 是纯查询接口，**不需要 cuInit 成功**也能返回 13000。
> 只看到 `driver API = 13000` **不能**证明能用，只证明库被 dlopen 了。
>
> ⚠️ **陷阱 2**：`torch._C._cuda_getDriverVersion` 在新版 torch（cu13 / 2.9+）已移除，
> 报 `AttributeError` 跟 compat 无关，别被带跑偏。

**完整验证脚本**（本案例最终全绿）：

```bash
LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat python - <<'EOF'
import ctypes, os, torch

l = ctypes.CDLL('libcuda.so.1')
v = ctypes.c_int(); l.cuDriverGetVersion(ctypes.byref(v))
print('driver API  =', v.value)            # 期望 13000
print('cuInit      =', l.cuInit(0))        # 期望 0  ← 真正的关卡