# 下一步学习计划

> 基于当前学习进度制定，理论 + 实操结合
> 制定日期：2026-06-18

---

## 当前进度回顾

### 已掌握 ✅

| 模块 | 深度 |
|------|------|
| Transformer 架构（Enc/Dec/三种变体） | 深入 |
| Attention 机制（MHSA/Cross/Masked + 实现细节） | 深入 |
| LLM 推理流程（Prefill + Decode + KV Cache） | 深入 |
| KV Cache 结构与显存计算 | 深入 |
| LM Head / Logits / 输出维度链路 | 深入 |
| Decoder 本质辨析（架构/Head/Loss 三层拆分） | 深入 |
| GPU 基础（Week 1-2 内容） | 已学 |
| PyTorch 训练基础 | 已学 |
| Mini-GPT 项目代码（model/train/generate） | 有代码，待实操 |

### 待学习 ❌（按优先级排序）

| 优先级 | 内容 | 原因 |
|--------|------|------|
| 高 | Flash Attention | 推理加速核心 |
| 高 | GQA/MQA | 直接影响 KV Cache 大小 |
| 高 | 推理脚本跑通（KV 增长实验） | 理论已学，缺实操验证 |
| 中 | MoE 架构 + 专家数 | 理解 DeepSeek 等模型 |
| 中 | Attention is All You Need 论文 | attention_notes 未打勾 |
| 中 | PyTorch 手写 Self-Attention | attention_notes 未打勾 |
| 中 | EP/DP/PB 分离 | 分布式推理相关 |
| 低 | Thinking/Reasoning | 偏应用层 |
| 低 | Open API TPM/RPM | 偏运维 |

---

## 下一步学习方案

### Step 1：跑通推理脚本（纯实操）

**目标**：把 `inference_flow_experiment.py` 实际运行一遍，用真实数据验证理论

**前置**：安装 torch + transformers，下载 GPT-2 模型（约 500MB）

**要观察的重点**：
- [ ] 第 1.5 部分：KV Cache 的实际 shape `(batch, heads, seq_len, d_k)`
- [ ] 第 1.5 部分：KV Cache 存储设备（CPU 时在内存）
- [ ] 第 2 部分：有/无 KV Cache 的速度对比（实际加速比）
- [ ] 第 2 部分：EOS 停止条件是否正常触发
- [ ] 第 4 部分：KV Cache 大小随序列长度的增长（验证公式）
- [ ] 第 5 部分：完整推理流程图

**完成后**：
- 对照公式手算一遍 GPT-2 的 KV Cache 大小，和脚本输出对比
- 记录实际加速比（CPU 上通常 2-5 倍）

**对应文件**：`inference/inference_flow_experiment.py`

---

### Step 2：Flash Attention（理论 + 实操）

**理论部分**：
- [ ] 标准 Attention 的显存瓶颈：N×N attention 分数矩阵（你已经理解）
- [ ] Flash Attention 的 tiling 策略：分块计算，不在 HBM 存完整矩阵
- [ ] IO 感知计算：HBM（慢，容量大）vs SRAM（快，容量小）的读写差异
- [ ] Online Softmax：分块时如何正确计算 softmax（不需要全局最大值）

**实操部分**：
- [ ] 用 `transformers` 开启/关闭 `sdpa`（Scaled Dot Product Attention），对比推理速度
- [ ] 写一个简单脚本：对比标准 attention 和分块 attention 的内存占用
- [ ] 在推理脚本中加入 `attn_implementation="sdpa"` 参数，观察效果

**对应文件**：新建 `inference/flash_attention_experiment.py`

---

### Step 3：Mini-GPT 项目实操（纯实操）

**目标**：亲手从零训练一个 mini GPT，体验完整的训练 → 推理链路

**步骤**：
- [ ] 跑 `train.py`：观察训练数据准备、Loss 下降过程、checkpoint 保存
- [ ] 跑 `generate.py`：用自己训练的模型生成文本（体会自回归生成）
- [ ] **核心实操**：在 `generate.py` 里手动加 KV Cache
  - Prefill：一次性处理 prompt，建立 KV Cache
  - Decode：每次只进 1 个新 token，追加到 Cache
  - 对比有/无 KV Cache 的生成速度
- [ ] 对比：自己写的 KV Cache 和 GPT-2 脚本中 transformers 自动管理的 KV Cache

**对应文件**：`projects/mini_gpt/`

---

### Step 4：GQA/MQA（理论 + 代码）

**理论部分**：
- [ ] MHA → MQA → GQA 的演进逻辑
  - MHA: Q/K/V 头数相同（标准多头）
  - MQA: 所有 Q 头共享 1 组 K/V（KV Cache 最小）
  - GQA: Q 头分组共享 K/V（平衡精度和显存）
- [ ] 为什么 GQA 能减少 KV Cache（你已有 KV Cache 显存公式基础）
- [ ] 哪些模型在用 GQA（LLaMA-2-70B、Qwen2、Mistral）

**实操部分**：
- [ ] 在 `llm_memory_calculator.py` 里扩展 GQA/MQA 的显存计算
- [ ] 对比同一模型 MHA vs GQA 的 KV Cache 大小差异（用表格展示）
- [ ] 计算 LLaMA-2-70B 用 MHA vs GQA 分别需要多少 KV Cache 显存

**对应文件**：`inference/llm_memory_calculator.py`

---

## 长期方向（不急）

| 方向 | 内容 | 前置条件 |
|------|------|----------|
| 分布式推理 | EP/DP/PB 分离 | 理解单卡推理后 |
| 推理框架 | vLLM/TGI 源码学习 | 理解 KV Cache 管理后 |
| 模型量化 | INT8/INT4/GPTQ/AWQ | 理解显存计算后 |
| 论文精读 | Attention is All You Need | Transformer 基础已有 |
