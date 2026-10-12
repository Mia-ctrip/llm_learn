# NVIDIA GPU 驱动相关组件作用详解

| 组件 | 作用 | 必需性 |
|------|------|--------|
| **kernel-devel** | 提供当前运行内核对应的**头文件和构建脚本**（`/usr/src/kernels/<kver>/`），是编译任何内核模块的基础依赖；版本必须与 `uname -r` 严格一致，否则 DKMS 或 `.run` 安装时会因找不到匹配头文件而编译失败 | ✅ 走DKMS/源码编译路径时必需 |
| **gcc / make / dkms** | **编译工具链三件套**：`gcc` 负责把 NVIDIA 驱动 C 源码编译成 `.ko` 文件；`make` 驱动内核 Kbuild 构建流程；`dkms` 作为动态内核模块管理框架，在每次内核升级后自动重新编译驱动，避免"升级内核后驱动失效" | ✅ 走DKMS/源码编译路径时必需 |
| **nvidia-driver** | NVIDIA 驱动的**本体元包**，包含：① 内核模块（`nvidia.ko`、`nvidia-uvm.ko`、`nvidia-modeset.ko`、`nvidia-drm.ko`、`nvidia-peermem.ko`）② 用户态库（`libcuda.so`、`libnvidia-ml.so`、NVENC/NVDEC 等）③ 命令行工具（`nvidia-smi`、`nvidia-debugdump` 等）④ 配套的 GSP 固件文件。没有它 GPU 无法被系统识别和使用 | ✅ 无条件必需 |
| **GSP 固件** | 安装在 `/lib/firmware/nvidia/<ver>/gsp_*.bin` 的**二进制 blob**，会被内核模块加载并推送到 GPU 内部的 **GSP 协处理器（RISC-V 架构）**上运行，负责电源管理、时钟调节、显存初始化、温度监控等底层硬件管理逻辑。**开源内核模块的工作前提**（把原本在闭源 `nvidia.ko` 里的管理逻辑下放到了 GPU 侧）；随驱动包自动释放，无需单独下载 | ✅ 随驱动自带 |
| **nvidia-persistenced** | **后台守护进程**，通过持续持有 GPU 句柄让 GPU 保持在"已初始化"状态，避免进程退出后 GPU 反初始化带来的冷启动延迟（每次几秒）；同时保留 ECC 错误计数、时钟设置等运行时状态，是 AI 训练/推理、K8s GPU 节点等生产场景的标配 | ⭐ 推荐，不装不会报错 |
| **nvidia-fabric-manager** | 管理 **NVSwitch（NVLink 交换芯片）**的守护进程，负责多 GPU 全互联拓扑的初始化、路由配置、链路状态监控；**仅 DGX / HGX 等带 NVSwitch 的 8 卡互联系统需要**，单卡工作站（如 RTX PRO 5000）或普通 2-4 卡服务器装了反而会因找不到 NVSwitch 硬件而启动失败报错 | ⚠️ 按机型条件必需 |
| **nvidia-container-toolkit** | 让**容器运行时（Docker / containerd / CRI-O）能够使用宿主机 GPU** 的桥接组件，核心作用是在容器启动时自动注入 GPU 设备节点（`/dev/nvidia*`）、驱动库（`libcuda.so` 等）和必要的环境变量；包含 `libnvidia-container`、`nvidia-ctk`、`nvidia-container-runtime` 等子组件，是 K8s 集群通过 Device Plugin 调度 GPU 的底层基础 | ⭐ K8s/容器场景必需 |

必需性标注的四种不同性质，不能用同一个判断逻辑：
- **✅无条件必需**（nvidia-driver）：任何场景都要装，没有例外。
- **✅走DKMS/源码编译路径时必需**（kernel-devel、gcc/make/dkms）：如果走的是Linux发行版仓库里**预编译**的驱动包（如`apt install nvidia-driver-535`，模块已经编译好、匹配好内核版本），这几个工具用不上；只有走NVIDIA官方`.run`安装包或DKMS管理这条路径（现场编译内核模块）才必需。生产服务器通常推荐DKMS路径（能在内核升级后自动重新编译驱动），所以实践上建议都装，但严格说不是任何安装方式下都必需。
- **⭐推荐，不装不会报错**（nvidia-persistenced）：不装不会导致CUDA初始化失败，只是每次进程退出后GPU反初始化、下次使用有几秒冷启动延迟，长期看还可能有时钟/ECC状态不连续的问题——是"质量下降"级别的影响，装了没有副作用，可以无条件装上。
- **⚠️按机型条件必需**（nvidia-fabric-manager）：必须先判断机型是不是**NVSwitch**拓扑（看是不是DGX/HGX系列SXM型号，或`lspci | grep -i nvswitch`查NVSwitch芯片是否存在），再决定装不装——判断标准是"有没有NVSwitch芯片"，不是"支不支持NVLink"：工作站场景两张卡通过NVLink桥接卡**点对点**直连，也算"支持NVLink"，但这种拓扑没有交换芯片介入，不需要也不能装fabric-manager（装了会因为找不到NVSwitch硬件启动失败）。该装的机型不装会导致CUDA初始化彻底失败（Error 802，见`排障/显卡识别案例记录.md`）；不该装的机型装了会直接报错——两个方向都会出问题，不是"建议安装的都装上"这种无条件逻辑能覆盖的。
- **⭐K8s/容器场景必需**（nvidia-container-toolkit）：只有GPU workload会跑在容器里才需要；裸机直接跑训练/推理用不上，但装了闲置也没有副作用。


# 驱动
export LD_LIBRARY_PATH=/usr/local/cuda/compat:${LD_LIBRARY_PATH}
