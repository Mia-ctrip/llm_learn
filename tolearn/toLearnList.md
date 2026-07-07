# 待学习清单

> 记录需要学习但还没学的内容

1. Attention Flash为什么能加速推理；
2. **CUDA Forward Compatibility（驱动向后兼容）**  
   - 驱动版本决定最高可运行的 CUDA 版本（nvidia-smi 顶部显示）  
   - 驱动 550.90.07 → 最高支持 CUDA 12.4  
   - 镜像 CUDA 13.0 + PyTorch 2.9.0+cu130 → 驱动太老无法直接跑  
   - **解决**：`export LD_LIBRARY_PATH=/usr/local/cuda/compat:${LD_LIBRARY_PATH}`  
   - 原理：`compat/` 下是 CUDA 13.0 的运行时库，通过老驱动接口做了一层兼容转译  
   - **匹配关系**：R550 系列 → CUDA 12.4 | R560 → 12.6 | R570 → 13.0 
3. Thinking Reasoning这些和大模型有关吗 和attention有关吗 开thinking模式为什么会取得更好的效果？
4. KV cache缓存命中的理论解释，KV cache的实现， 为什么只有KV cahe和Q无关？
5. EP是什么（deepseek提出的）？DP是什么？
6. PB分离的PB是什么？通过attention原理能理解吗？
7. 专家数是什么？
8. MoE架构范式是什么？
9. open api的 TPM RPM分别是什么
10.为什么prompt过长之后会出现模型循环输出？
11. Deepseek的thinking模式是怎么实现的，thinking中的内容也是模型的输出吗？

