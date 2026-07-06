# GPU 僵尸 CUDA 上下文排障指南

> 实战经验：2026-07-03 在 8×H20 集群 (ocp58projgxijhtc-tr013245-0) 上排障的完整记录。
> 驱动版本：550.144.03，CUDA 12.8

---

## 1. 问题现象

- GPU 显存被大量占用（如 39GB/65GB），但 GPU 利用率为 0%
- `nvidia-smi` 能看到 PID，但 `ps -p <PID>` 查不到进程
- nvitop 进程列表显示 "No Such Process" 或 "Unknown"
- 进程已退出，但 CUDA 上下文未被正确释放（孤儿上下文）

---

## 2. 排障工具对比

| 工具 | 安装方式 | 优势 | 注意事项 |
|------|---------|------|----------|
| `nvidia-smi` | 驱动自带 | 最权威，直接查硬件级显存 | 默认表格可能不显示孤儿进程详情 |
| `gpustat` | `pip install gpustat` | 一行一卡，显示**硬件级** `memory.used` | **最可靠**，能看到真实显存占用 |
| `nvitop` | `pip install nvitop` | 交互式界面，实时刷新 | 某些驱动版本下可能**遗漏**孤儿进程的显存统计 |

### 关键发现

**nvitop 可能漏报孤儿进程显存！** 在驱动 550.144.03 下实测：
- `gpustat` 显示 GPU 4/5 各占 39663 MB
- 同时段 `nvitop` 显示 GPU 4/5 仅 3.50 MiB（漏报）
- 后来同步查看时 nvitop 才能正确显示

**结论**：判断显存是否干净，以 `gpustat` 或 `nvidia-smi --query-gpu` 为准。

---

## 3. 诊断命令速查

### 3.1 查看硬件级真实显存占用（最准确）

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
```

### 3.2 查看 GPU 上的 compute 进程

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

# 指定 GPU
nvidia-smi --id=4,5 --query-compute-apps=pid,used_memory --format=csv
```

### 3.3 验证进程是否存活

```bash
ps -p <PID>

# 批量验证
ps -p 112713,112714,110487
```

### 3.4 查看设备文件句柄（找隐藏持有者）

```bash
sudo lsof /dev/nvidia4 /dev/nvidia5
sudo fuser -v /dev/nvidia4 /dev/nvidia5
```

---

## 4. 修复方案（按优先级排序）

### 方案 1：Kill 进程（如果进程还在）

```bash
kill -9 <PID>

# 批量杀掉某 GPU 上所有进程
nvidia-smi --id=4 --query-compute-apps=pid --format=csv,noheader | xargs -I {} kill -9 {}
```

### 方案 2：GPU Reset（进程已不存在时，推荐）

```bash
# 先关 persistence mode（如果开了）
sudo nvidia-smi -i 4,5 -pm 0

# 重置 GPU
sudo nvidia-smi --gpu-reset -i 4
sudo nvidia-smi --gpu-reset -i 5
```

### 方案 3：关闭 MPS（如果使用了 Multi-Process Service）

```bash
echo quit | nvidia-cuda-mps-control
```

### 方案 4：重载内核模块（影响所有 GPU）

```bash
# 注意：会中断所有 GPU 上的工作！
sudo systemctl stop nvidia-persistenced
sudo rmmod nvidia_uvm
sudo rmmod nvidia_drm
sudo rmmod nvidia_modeset
sudo rmmod nvidia

sudo modprobe nvidia
sudo modprobe nvidia_uvm
sudo modprobe nvidia_drm
sudo modprobe nvidia_modeset
sudo systemctl start nvidia-persistenced
```

### 方案 5：重启机器（终极方案）

---

## 5. 完整排障流程图

```
发现 GPU 显存异常占用
    │
    ▼
nvidia-smi --query-gpu=index,memory.used --format=csv  （确认哪些卡有问题）
    │
    ▼
nvidia-smi --id=N --query-compute-apps=pid,used_memory --format=csv  （找到 PID）
    │
    ▼
ps -p <PID>  （验证进程是否存活）
    │
    ├── 进程存在 → kill -9 <PID>
    │
    └── 进程不存在（孤儿上下文）
         │
         ▼
    sudo nvidia-smi --gpu-reset -i N
         │
         ├── 成功 → gpustat 确认显存已释放 ✓
         │
         └── 失败 → 关 pm → 重试 → 仍失败 → 重载内核模块 → 最后重启
```

---

## 6. 预防措施

1. **训练脚本中正确清理资源**：确保异常退出时调用 `torch.cuda.empty_cache()` 和显式 `del model`
2. **使用 try/finally 或 signal handler**：捕获 SIGTERM/SIGKILL 做清理
3. **容器化部署**：容器退出时驱动会自动回收该容器的 GPU 上下文
4. **定期巡检**：用 cron 定期跑 `nvidia-smi --query-gpu` 对比 `--query-compute-apps`，发现不一致即告警

---

## 7. 常见 gpu-reset 报错及解决

| 报错 | 原因 | 解决 |
|------|------|------|
| `GPU not idle` | 还有进程持有设备文件 | `lsof /dev/nvidiaN` 找到并 kill |
| `persistence mode is on` | PM 阻止 reset | `nvidia-smi -i N -pm 0` 后重试 |
| `GPU has fallen off the bus` | 硬件级故障 | 需要重启机器，可能需要换卡 |
| `in use by another client` | 其他驱动组件持有 | 重载内核模块 |
