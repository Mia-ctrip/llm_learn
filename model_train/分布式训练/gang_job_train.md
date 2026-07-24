分布式训练知识点 & 排障技巧总结
一、分布式训练框架架构
1. torchrun vs verl（启动器 vs 训练框架）
组件	角色	职责
torchrun	启动器 / 进程管理器	拉起多个进程（每卡一个），设置分布式环境变量（RANK、WORLD_SIZE、MASTER_ADDR/PORT），管理进程通信
verl	训练框架	真正的训练逻辑：加载模型/数据、FSDP 分片、前反向、保存 checkpoint
执行链路：

torchrun（拉起 N 个进程）
  └── 每个进程执行 verl.trainer.fsdp_sft_trainer
        └── 底层用 PyTorch FSDP 做模型分片
        └── 用 Hydra 解析 config 参数
2. verl 的两种运行模式
SFT → 用 torchrun 启动（fsdp_sft_trainer）
RLHF/PPO/GRPO → 用 ray 启动（main_ppo）
⚠️ SFT 脚本里的 ray start 往往是从 RL 模板复制来的冗余代码，不影响运行
二、关键概念
1. 进程与 rank
rank：全局进程编号（rank 0、1、2…）
local_rank：本机内的进程编号
Root Cause: rank 0：第一个失败的进程；其他 rank 通常是被"连坐"终止的
WORLD_SIZE / size：总进程数，size 1 说明只起了 1 个进程（可能配置错误）
2. 并行度配置
NNODES：节点数（单机=1）
NPROC_PER_NODE：每节点进程数（应等于 GPU 数）
ulysses_sequence_parallel_size：序列并行度，需 ≤ GPU 数且能整除
单机可用 --standalone 自动处理 master 地址/端口
3. num_workers（DataLoader 子进程数）
num_workers 是 DataLoader 的参数，控制用几个 CPU 子进程预加载数据，与 GPU 数量和分布式训练完全无关。

num_workers=4 时：
  主进程（训练）
    ├── worker 0  ← 提前读数据、tokenize
    ├── worker 1
    ├── worker 2
    └── worker 3
    → 主进程直接从队列取现成的 batch，不用等 I/O

num_workers=0 时：
  主进程自己读数据、自己训练（串行，慢但错误可见）

⚠️ 多卡下的放大效应：
  8 卡 + num_workers=4 → 实际多出 8×4=32 个数据加载子进程
  每个训练进程内部各有自己的 DataLoader，各自 fork num_workers 个子进程

设成 0 的唯一目的：牺牲速度，换取错误可见性（worker 里的异常不再被吞掉）
4. 退出码（exitcode）的含义
exitcode	含义
1	Python 抛异常退出
-9 / 负数信号	被系统 kill（常见于 OOM）
三、排障技巧（核心方法论）
技巧 1：Traceback 要"从下往上"看
栈顶（如 main()）= 程序入口，通常不是问题
栈底（最后的 XxxError）= 真正的错误点
别被中间的调用链迷惑
技巧 2：识别"外壳报错" vs “真实报错”
ChildFailedError / worker exited unexpectedly = 外壳，只说明"进程挂了"，不说"为什么挂"
真实原因常被子进程吞掉，需要额外手段暴露
技巧 3：让隐藏的错误暴露出来
# DataLoader 多进程的错误被 worker 吞了 → 设为 0，在主进程加载
data.dataloader_num_workers=0

# 开启详细分布式调试
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export HYDRA_FULL_ERROR=1

# torchrun 打印每个 rank 的输出
torchrun --redirects=3 --tee=3 ...

⚠️ 注意：override 的 key 名必须和 verl 源码中实际读取的一致，否则 Hydra 会报
  ConfigCompositionException: Could not override 'xxx'
正确做法：先 grep -r "num_workers" verl源码目录/ 找到真实 key，再用该 key override
如果找不到对应 config key，直接改源码里 DataLoader 构建处的 num_workers=xxx 为 0
技巧 4：单卡先跑通，再上多卡
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 ...
单卡错误信息更清晰，容易定位。

技巧 5：区分三种“资源不足”
类型	查看命令	说明
CPU 内存 (RAM)	free -h	数据处理/加载占用
GPU 显存 (VRAM)	nvidia-smi	模型/激活值占用
OOM-kill 记录	dmesg -T | grep -i killed	被系统杀掉的证据
⚠️ 内存和显存是两回事，别混淆！

技巧 6：Pod/容器环境下的资源限制全景

Linux Pod 可限制的资源类型：
资源类型	查看命令	说明
PID/线程数	ulimit -u、cat /sys/fs/cgroup/pids/pids.max	最大进程/线程数
文件描述符（句柄）	ulimit -n	最大打开文件数
CPU 内存	ulimit -m、cat /sys/fs/cgroup/memory/memory.limit_in_bytes	可用物理内存上限
虚拟内存	ulimit -v	虚拟地址空间上限
CPU 时间/配额	ulimit -t、cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us	CPU 使用时长或带宽
磁盘 I/O	cgroup io.max	磁盘读写带宽/IOPS
栈大小	ulimit -s	每个线程的栈空间（默认 8MB）
锁内存	ulimit -l	可 mlock 的内存量
GPU 显存	nvidia-smi	硬件固定，不可超

cgroup vs ulimit 的区别：
  ulimit：限制单个用户/进程能用多少资源（每人最多点 5 道菜）
  cgroup：限制一组进程（如一个 Pod）总共能用多少资源（这桌总共最多点 20 道菜）
  ulimit 查看：ulimit -a
  cgroup 查看：/sys/fs/cgroup/ 下对应文件
  ⚠️ ulimit 显示 unlimited 不代表 cgroup 没限制，两层是独立的
  K8s 生产环境中 cgroup 才是真正卡你的那层

实战案例：rayon 线程池创建失败（Resource temporarily unavailable）
  现象：DataLoader worker 内 tokenizer 报 errno 11 (EAGAIN)
  排查：ulimit -u unlimited、cgroup pids 未启用、FD 1048576 够用、RAM 未超
  根因：64个 worker 同时 fork 后瞬时并发创建线程池，内核线程创建路径短暂资源竞争
  解决：export RAYON_NUM_THREADS=1（限制每个 worker 内部 rayon 线程池只建 1 个线程）
  教训：不是“不够”而是“太急”，降低瞬时并发压力即可

四、常见错误类型 & 定位
1. 配置错误（Hydra/OmegaConf）
Key 'xxx' is not in struct / Could not override 'xxx'
原因：脚本用的参数在当前 verl 版本配置里不存在（版本不匹配）
解决：
删掉该参数，或
加 + 号强制新增：+trainer.checkpoint.save_contents=...
根本方案：对齐 verl 版本与脚本

Hydra +/- 前缀区别：
  data.xxx=0    → 覆盖已有 key（key 不存在则报错）
  +data.xxx=0   → 强制新增 key（不管存不存在都写入）
  ⚠️ 加 + 只是让 Hydra 不报错；如果框架代码里根本不读这个 key，加了也没效果
  正确做法：grep 源码确认实际读取的 key 名，用真实 key 去 override（无需 +）
2. 版本错配（最隐蔽）
pip 装的版本 ≠ 实际加载的版本
验证加载的是哪份代码：
python3 -c "import verl; print(verl.__file__); print(verl.__version__)"
Python 加载优先级：当前工作目录 (cwd) > site-packages
解压源码 + cd 进目录 → 优先用源码
pip install -e . → 可编辑安装，全局生效
3. 数据加载错误
DataLoader worker exited unexpectedly
可能原因：字段缺失、格式不符、tokenize 失败、超长序列
定位：num_workers=0 暴露真实错误
4. Flash Attention 警告（可忽略）
Flash Attention 2.0 only supports fp16/bf16, but current dtype is float32
model not initialized on GPU
本质是 Warning 不是 Error：FSDP 先在 CPU 以 fp32 加载，后续自动分片转 bf16
A800 建议显式设 bf16
5. GPU 显存 OOM（CUDA out of memory）
现象：Tried to allocate xx MiB. GPU has total capacity xx GiB of which xx MiB is free
原因：模型参数 + 激活值 + 优化器状态超出单卡显存上限
调优参数（均在 torchrun 命令行通过 Hydra override 指定，无需改源码）：
  data.micro_batch_size_per_gpu=1       ← 降低每卡每次计算的样本数（激活值减半）
  model.fsdp_config.model_dtype=bf16    ← 权重以 bf16 加载（参数显存减半）
  model.fsdp_config.cpu_offload=true    ← 优化器状态卸载到 CPU（用时间换空间）
  data.max_length=65536                 ← 降低序列长度（激活值减半）
优先级：降 micro_batch > 改 bf16 > cpu_offload > 降 max_length
五、pip install -e .（可编辑安装）
cd /path/to/verl_source
pip install -e .
-e = editable，不复制代码，建立链接指向源码
改源码立即生效，无需重装
用于替换 pip 装的旧版本为本地新版源码
六、排障黄金流程（总结）
1. 看到外壳报错（ChildFailed / worker exited）
        ↓
2. 往上翻日志，找真正的 Traceback（从下往上读）
        ↓
3. 如果错误被子进程吞了 → 加参数暴露
   （num_workers=0 / TORCH_DISTRIBUTED_DEBUG=DETAIL / HYDRA_FULL_ERROR=1）
        ↓
4. 确认是哪类问题：
   - 配置错误？→ 检查版本、加 +/删参数
   - 资源不足？→ 分清 RAM/VRAM/OOM-kill
   - 数据问题？→ 检查字段、格式、单独测 Dataset
        ↓
5. 单卡先跑通，再扩到多卡
一句话核心： 分布式报错常是"外壳"，关键是想办法逼出真实的 Traceback，然后从栈底定位真因。