# 卸载
阶段1：清理550驱动 
具体卸载步骤需要参考下载文件                                                                            
  
## 1. 停止相关服务
sudo systemctl stop nvidia-fabricmanager.service 2>/dev/null || true
sudo systemctl stop nvidia-persistenced.service 2>/dev/null || true
sudo systemctl disable nvidia-fabricmanager.service 2>/dev/null || true
sudo systemctl disable nvidia-persistenced.service 2>/dev/null || true

## 2. 卸载驱动包
sudo dnf remove -y nvidia-driver nvidia-driver-* kmod-nvidia-* \
  nvidia-fabric-manager nvidia-persistenced nvidia-modprobe \
  kmod-nvidia-latest-dkms kmod-nvidia-open-dkms

## 3. 卸载容器组件（你的安装脚本安装了这些）
sudo rpm -e libnvidia-container1 libnvidia-container-tools \
  nvidia-container-toolkit nvidia-container-runtime --nodeps 2>/dev/null || true

## 4. 移除 local repo（注意版本号）
sudo rpm -e nvidia-driver-local-repo-rhel8-550.90.07 --nodeps 2>/dev/null || true

## 5. 清理 DKMS
sudo dkms remove nvidia/550.90.07 --all 2>/dev/null || true
sudo rm -rf /usr/src/nvidia-550.90.07 2>/dev/null || true

## 6. 清理配置文件
sudo rm -f /etc/modprobe.d/nvidia-gsp.conf

## 7. 清理缓存
sudo dnf clean all

## 8. 重启系统
sudo reboot

# 判断卸载情况
重启后执行以下检查：

## 1.检查 NVIDIA 内核模块是否还存在
lsmod | grep nvidia

## 2.检查 RPM 包是否残留
rpm -qa | grep -i nvidia

## 3.检查 nvidia-smi 是否还能运行
nvidia-smi 2>&1

## 4.检查相关文件是否清理
ls -la /usr/src/ | grep nvidia
ls -la /etc/modprobe.d/ | grep nvidia
ls /usr/lib/modules/$(uname -r)/extra/ | grep nvidia

## 5. 检查服务状态
systemctl list-units | grep nvidia

判断标准：
- ✅ lsmod | grep nvidia 无输出
- ✅ rpm -qa | grep nvidia 无输出
- ✅ nvidia-smi 提示找不到命令
- ✅ 上述文件/目录检查无 nvidia 相关内容


# 显卡诊断
## 1. 检查驱动版本                                                                               cat /proc/driver/nvidia/version
                                                                                                
## 2. 检查加载的模块
lsmod | grep nvidia

## 3. 检查模块详情（是open还是专有）
modinfo nvidia | head -20

## 4. 检查DKMS构建状态
dkms status

## 5. 查看安装的内核模块文件
ls -la /lib/modules/$(uname -r)/extra/ | grep nvidia

## 6. 检查dmesg日志
sudo dmesg | grep -i nvidia | tail -30

## 7. 检查安装了哪些nvidia包
rpm -qa | grep nvidia | sort

# 安装
 重新 SSH 连接后：

## 1. 进入临时目录
  cd /tmp

## 2. 下载 driver 580.126.09 并安装本地仓库
sudo wget http://download2.ctripcorp.com/cdos/nvidia/nvidia-driver-local-repo-rhel8-580.126.09-1.0-1.x86_64.rpm
 (测试时通过hadoop上传下载的)
sudo rpm -ivh nvidia-driver-local-repo-rhel8-580.126.09-1.0-1.x86_64.rpm

## 3. 将仓库默认的driver内核从专有改成open kernel
sudo dnf module reset nvidia-driver -y 2>/dev/null || true
sudo dnf module enable nvidia-driver:580-open -y 2>/dev/null || true
sudo dnf clean all

## 3. 确保 kernel-devel 已安装
# Unlock kernel-devel if locked
if grep -q kernel-devel /etc/yum/pluginconf.d/versionlock.list; then
    sudo sed -i 's/^kernel/#kernel/g' /etc/yum/pluginconf.d/versionlock.list
fi

sudo dnf install -y kernel-devel-$(uname -r) dkms

## 4. 安装DKMS工具
sudo dnf install -y kmod-nvidia-open-dkms-580.126.09

## 5. 安装驱动（关键：--dkms --open 参数）
sudo dnf install -y nvidia-open-580.126.09

## 6.  安装其他组件
sudo dnf install -y nvidia-persistenced
# Install fabric manager (required for NVLink-enabled GPUs like H20, B100, B200)
# On GPUs without NVLink (L20, RTX PRO 5000), the service will fail but GPU works normally
sudo dnf install -y nvidia-fabric-manager


# 安装完成测试

✅ 基础验证（必须全部通过）

## 1. 驱动加载检查
nvidia-smi
# 预期：显示 GPU 信息，无错误

## 2. 内核模块检查
lsmod | grep nvidia
## 预期：看到 nvidia, nvidia_modeset, nvidia_uvm 等模块

## 3. CUDA 设备节点检查
ls -la /dev/nvidia*
# 预期：看到 /dev/nvidia0, /dev/nvidiactl, /dev/nvidia-uvm 等设备文件

## 4. nvidia-container-cli 检查
nvidia-container-cli info
# 预期：显示 GPU 信息，包括型号、UUID、架构等

## 5. 服务状态检查
systemctl status nvidia-persistenced
systemctl status nvidia-fabricmanager
## 预期：服务正常运行（active）

## 6. 简单的容器测试（方案 A）
sudo docker run --rm \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --cap-add=SYS_ADMIN \
  --security-opt seccomp=unconfined \
  hub.cloud.ctripcorp.com/ml_environments/peta/train/peta-training:0.2.0-torch270-workbench-py310-cuda128-cudnn9-alma92 \
  nvidia-smi
# 预期：在容器内成功显示 GPU 信息

✅ 深度验证（CUDA 可用性）

## 7. PyTorch CUDA 测试（方案 A）
sudo docker run --rm \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --cap-add=SYS_ADMIN \
  --security-opt seccomp=unconfined \
  hub.cloud.ctripcorp.com/ml_environments/peta/train/peta-training:0.2.0-torch270-workbench-py310-cuda128-cudnn9-alma92 \
  python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
# 预期：CUDA available: True, GPU count: 1

✅ 配置文件验证

## 8. 检查 GSP 固件禁用配置
cat /etc/modprobe.d/nvidia-gsp.conf
# 预期：options nvidia NVreg_EnableGpuFirmware=0

## 9. 检查安装的包
rpm -qa | grep nvidia
# 预期：看到 nvidia-driver, nvidia-fabric-manager, nvidia-persistenced 等包

判断标准

✅ 你的镜像制作流程是正确的，如果：

1. 宿主机层面：
  - nvidia-smi 正常显示 GPU
  - nvidia-container-cli info 正常显示 GPU
  - 服务正常运行
2. 容器层面（使用方案 A 参数）：
  - 容器内 nvidia-smi 正常
  - 容器内 torch.cuda.is_available() 返回 True


  # NVIDIA Open Kernel Module 安装与运维防踩坑说明

本文档根据 **RTX PRO 5000（Blackwell）** 在 **Alibaba Cloud Linux 3（RHEL 8 / al8）** 上通过 **RPM + local-repo** 装机、以及 **Docker + PyTorch** 联调过程中的实际问题整理，供后续换机、换版本、排障时对照。

---

## 1. 适用范围与前提

- **GPU**：NVIDIA Blackwell 架构工作站卡（如 RTX PRO 5000）在 Linux 上需使用 **Open GPU Kernel Module**（下文简称 open 模块）。
- **系统**：与 RHEL 8 兼容的发行版（如 Alibaba Cloud Linux 3），内核需与 `kernel-devel` 版本一致以便 DKMS 编译。
- **网络**：若主机无法访问 `developer.download.nvidia.com` 等外网，应使用 **内部镜像 + `nvidia-driver-local-repo-rhel8-*.rpm`**，不要假设 `dnf config-manager --add-repo` 一定可用。

---

## 2. 为什么不能用「专有内核模块」装 Blackwell

- **专有栈**通常对应 `kmod-nvidia-latest-dkms` 与元包 `nvidia-driver` 所拉起的依赖链。
- **Blackwell** 在官方支持矩阵上要求使用 **open 内核模块**；仅用专有模块时可能出现初始化失败、`nvidia-smi` 异常或无法稳定工作。
- **结论**：装机目标应是 **`kmod-nvidia-open-dkms-<版本>` + `nvidia-open-<版本>`** 这一套 open 栈，而不是裸装 `nvidia-driver`。

---

## 3. `nvidia-driver` 与 `nvidia-open-<版本>` 不要混用

| 命令 / 包 | 含义 | Blackwell 场景 |
|-----------|------|----------------|
| `yum install nvidia-driver` | **专有栈元包**，易拉 `kmod-nvidia-latest-dkms` | 易与 open 冲突，不推荐作为主线 |
| `yum install nvidia-open-580.126.09`（示例） | **open 栈元包**，依赖与 open kmod 一致 | **推荐**（版本号与 local-repo 一致） |
| `yum install nvidia-driver-580*` 等通配符 | 可能匹配到 **专有元包** 或错误组合 | **高风险**，避免 |

**经验**：用户态安装以 **`nvidia-open-<与仓库一致的版本>`** 为准；不要图省事只敲 `nvidia-driver`。

---

## 4. 不要用 `dkms status` 判断是 open 还是专有

- 两种 kmod 在 DKMS 里可能都显示为类似 `nvidia/580.xxx` 的模块名，**仅凭 `dkms status` 容易误判**。
- **建议核验**：

```bash
rpm -qa | grep -E '^kmod-nvidia'
modinfo nvidia | grep -E '^(license|version)'
```

- **open 模块**的 `license` 一般为 **`Dual MIT/GPL`**；专有模块为 **`NVIDIA`**。
- **期望**：已装 **`kmod-nvidia-open-dkms-...`**，且**不应**再出现 **`kmod-nvidia-latest-dkms`**（除非你有意混装，Blackwell 场景应避免）。

---

## 5. DNF Module：「包在目录里，yum 却说找不到」

NVIDIA 的 **`nvidia-driver-local-repo-rhel8-*.rpm`** 会携带 **module 元数据**。常见现象：

- `dnf module list nvidia-driver` 里出现 **`latest-dkms [e]`**（专有流被 enable）等状态；
- 执行 `yum install kmod-nvidia-open-dkms-...` 时报：

  **`All matches were filtered out by modular filtering`**

**原因**：当前启用的 module stream 与 open 包不兼容，dnf 在 **modular 过滤** 下把匹配结果全部剔除，**并非**「RPM 真的不存在」。

**处理思路**（与生产脚本 `nvidia-580.sh` 中逻辑一致）：

```bash
sudo dnf module reset nvidia-driver -y
sudo dnf module enable nvidia-driver:580-open -y   # 版本流与当前 driver 大版本一致，如 580-open
sudo dnf clean all
```

若仍异常，可排查 `/etc/dnf/modules.d/` 下残留状态，或联系基线/镜像是否曾 enable 过 `latest-dkms`。

---

## 6. 仓库形态：local-repo 与外网 CUDA 源

- **`nvidia-driver-local-repo-rhel8-<ver>-1.0-1.x86_64.rpm`**：安装后会在本机生成 **带 repodata 的本地仓库**（路径通常在 `/var/nvidia-driver-local-repo-rhel8-<ver>/`），`yum` 从 **本地 file 源** 解析依赖。
- **与「能否访问外网」无关**：之前能装 550 成功，往往是因为 **内部已提供 local-repo 或等价离线源**，而不是主机一定能连 NVIDIA 官网。
- **排查「有没有 open 包」**时，可只启用 local 仓库，避免其它 repo 干扰：

```bash
yum --disablerepo='*' --enablerepo='nvidia-driver-local*' list available | grep -i open
```

---

## 7. 驱动大版本与容器 / PyTorch 栈

- 实际联调中曾出现：**宿主机 `nvidia-smi` 正常，容器内 PyTorch `torch.cuda.is_available()` 为 false 或 CUDA 初始化报错**，而 **同一套较旧的 nvidia-container-toolkit** 在 **较新驱动分支** 上更容易暴露兼容问题。
- **经验**：生产上优先采用 **与镜像、toolkit 一起验证过的驱动大版本**（例如你们最终落在 **580 LTS + open**），不要仅因「版本号更大」就换到未与全栈联调过的分支。

---

## 8. 容器与 GPU：三类常见问题

### 8.1 Docker 报 `could not select device driver "" with capabilities: [[gpu]]`

- 多为 **Docker 未注册 `nvidia` runtime**（`/etc/docker/daemon.json` 缺少 `runtimes.nvidia`，或修改后未 **`systemctl restart docker`**）。
- **与驱动是否安装成功是两条线**：`nvidia-smi` 在宿主机正常，仍可能出现上述错误。

### 8.2 老版本 `nvidia-container-toolkit`（如 1.2.x）

- 可能出现 **`nvidia-container-runtime-hook` 找不到** 等问题；
- **重装 toolkit 不一定会自动写 `daemon.json`**，需在运维规范或装机脚本中 **固化 runtime 配置 + 重启 docker**。

### 8.3 容器内缺 `/dev/nvidia-uvm` 与 CUDA / PyTorch

- **CUDA 用户态**依赖 **`/dev/nvidia-uvm`** 等节点；仅有 `nvidia0` / `nvidiactl` 时，`nvidia-smi` 有时仍「看似正常」，但 **PyTorch 等会初始化失败**（例如曾遇到的 **error 304** 类现象）。
- **宿主机**侧应确认：`lsmod | grep nvidia_uvm`、`ls -l /dev/nvidia*`。
- **老 toolkit** 下有时需 **`docker run` 显式 `--device /dev/nvidia-uvm` 等**，或在宿主机执行 **`nvidia-modprobe -u -c=0`**（是否写入装机脚本以你们基线为准）。

---

## 9. `nvidia-smi` 与 PyTorch：分层排障

建议顺序：

1. **PCI**：`lspci -nn | grep -i nvidia`（虚拟机还须确认 GPU 已透传）。
2. **宿主机驱动**：`nvidia-smi`、`lsmod`、`/dev/nvidia*`。
3. **容器设备**：容器内 `ls -l /dev/nvidia*`。
4. **容器内驱动通信**：容器内 `nvidia-smi`。
5. **框架**：`python -c "import torch; print(torch.cuda.is_available())"`。

**说明**：容器内 **OpenBLAS 线程** 相关告警有时与 **cgroup / ulimit** 有关，易干扰判断；仍以 **CUDA 返回码、dmesg、libcuda/设备节点** 为主线。

---

## 10. Persistence Mode 与 `nvidia-persistenced`

- **`systemctl enable nvidia-persistenced`** 只保证开机自启，**不保证当前会话已启动**。
- 若希望装完立刻在 `nvidia-smi` 中看到 **Persistence-M: On**，应使用 **`systemctl enable --now nvidia-persistenced`** 或装完后手动 **`systemctl start`**。

---

## 11. 工作站卡与 Fabric Manager

- **RTX PRO 5000** 一类工作站卡 **通常不需要** `nvidia-fabric-manager`（更偏数据中心 NVLink Switch 等场景）。
- 误装可能导致 **服务无法启动、开机失败项**；装机脚本中一般 **不必安装、不必 enable 该服务**。

---

## 12. 与仓库内 `nvidia-580.sh` 的关系

- `nvidia-580.sh` 是当前生产验证过的 **580.126.09 + open + internal RPM + container 四件套** 的自动化入口；其中已包含 **local-repo 安装**、**`dnf module` 锁到 `580-open`**、**open kmod + `nvidia-open`** 等关键步骤。
- **Docker `daemon.json`、老 toolkit 下的显式 device、`nvidia-modprobe`、persistenced 是否 `--now`** 等，若你们基线由**镜像或其它脚本**承担，请以**实际生产配置**为准；本文档用于说明**历史上踩过的坑与对应原理**，避免后人只抄脚本却不懂边界。

---

## 13. 快速检查清单（换机验收可照抄）

```bash
# 内核模块与设备
lsmod | grep nvidia
ls -l /dev/nvidia*

# open 与版本
rpm -qa | grep -E '^kmod-nvidia|nvidia-open'
modinfo nvidia | grep -E '^(license|version)'

# DKMS
dkms status | grep nvidia

# Docker GPU
sudo docker info | grep -i runtime
cat /etc/docker/daemon.json
```

---

## 14. 文档维护

- 若更换 **driver 大版本**（例如 580 → 590），请同步更新：**local-repo 文件名、`dnf module enable` 的 stream 名、`kmod-nvidia-open-dkms` 与 `nvidia-open` 的版本号**。
- 若公司内部 **toolkit 或 Docker 基线** 升级，请重新做一轮：**宿主机 `nvidia-smi` → 容器 `nvidia-smi` → 业务镜像内 `torch.cuda.is_available()`** 的回归。
