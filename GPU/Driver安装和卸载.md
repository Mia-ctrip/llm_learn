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

# 检查 NVIDIA 内核模块是否还存在
lsmod | grep nvidia

# 检查 RPM 包是否残留
rpm -qa | grep -i nvidia

# 检查 nvidia-smi 是否还能运行
nvidia-smi 2>&1

# 检查相关文件是否清理
ls -la /usr/src/ | grep nvidia
ls -la /etc/modprobe.d/ | grep nvidia
ls /usr/lib/modules/$(uname -r)/extra/ | grep nvidia

# 检查服务状态
systemctl list-units | grep nvidia

判断标准：
- ✅ lsmod | grep nvidia 无输出
- ✅ rpm -qa | grep nvidia 无输出
- ✅ nvidia-smi 提示找不到命令
- ✅ 上述文件/目录检查无 nvidia 相关内容


# 显卡诊断
# 1. 检查驱动版本                                                                               cat /proc/driver/nvidia/version
                                                                                                
# 2. 检查加载的模块
lsmod | grep nvidia

# 3. 检查模块详情（是open还是专有）
modinfo nvidia | head -20

# 4. 检查DKMS构建状态
dkms status

# 5. 查看安装的内核模块文件
ls -la /lib/modules/$(uname -r)/extra/ | grep nvidia

# 6. 检查dmesg日志
sudo dmesg | grep -i nvidia | tail -30

# 7. 检查安装了哪些nvidia包
rpm -qa | grep nvidia | sort



# 安装完成测试

✅ 基础验证（必须全部通过）

# 1. 驱动加载检查
nvidia-smi
# 预期：显示 GPU 信息，无错误

# 2. 内核模块检查
lsmod | grep nvidia
# 预期：看到 nvidia, nvidia_modeset, nvidia_uvm 等模块

# 3. CUDA 设备节点检查
ls -la /dev/nvidia*
# 预期：看到 /dev/nvidia0, /dev/nvidiactl, /dev/nvidia-uvm 等设备文件

# 4. nvidia-container-cli 检查
nvidia-container-cli info
# 预期：显示 GPU 信息，包括型号、UUID、架构等

# 5. 服务状态检查
systemctl status nvidia-persistenced
systemctl status nvidia-fabricmanager
# 预期：服务正常运行（active）

# 6. 简单的容器测试（方案 A）
sudo docker run --rm \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --cap-add=SYS_ADMIN \
  --security-opt seccomp=unconfined \
  hub.cloud.ctripcorp.com/ml_environments/peta/train/peta-training:0.2.0-torch270-workbench-py310-cuda128-cudnn9-alma92 \
  nvidia-smi
# 预期：在容器内成功显示 GPU 信息

✅ 深度验证（CUDA 可用性）

# 7. PyTorch CUDA 测试（方案 A）
sudo docker run --rm \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --cap-add=SYS_ADMIN \
  --security-opt seccomp=unconfined \
  hub.cloud.ctripcorp.com/ml_environments/peta/train/peta-training:0.2.0-torch270-workbench-py310-cuda128-cudnn9-alma92 \
  python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
# 预期：CUDA available: True, GPU count: 1

✅ 配置文件验证

# 8. 检查 GSP 固件禁用配置
cat /etc/modprobe.d/nvidia-gsp.conf
# 预期：options nvidia NVreg_EnableGpuFirmware=0

# 9. 检查安装的包
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