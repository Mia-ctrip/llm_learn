# MTP（Multi-Token Prediction）学习指南

> **适合人群**：了解 LLM 自回归推理基础，希望学习推理加速技术的 AI Infra 工程师  
> **学习时长**：30-45 分钟  
> **前置知识**：了解自回归生成、前向传播、概率采样基础  
> **配合阅读**：[LLM_推理速度学习指南](LLM_推理速度学习指南.md)、[KV_Cache学习指南](KV_Cache学习指南.md)

---

## 1. 什么是 MTP？

### 1.1 核心思想

**MTP = Multi-Token Prediction（多 Token 预测）**

传统自回归（Autoregressive）LLM 每次前向传播只预测 **1 个** 下一个 token：

```
[Hello] → 前向传播 → 预测 [world]
[Hello world] → 前向传播 → 预测 [!]
[Hello world !] → 前向传播 → 预测 [How]
...
生成 N 个 token 需要 N 次前向传播
```

MTP 的目标是在 **一次前向传播中预测多个** token：

```
[Hello] → 一次前向传播 → 同时预测 [world, !, How, are, you]
生成 5 个 token 只需 1 次前向传播（理想情况）
```

### 1.2 为什么 MTP 重要？

| 维度 | 传统自回归 | MTP |
|------|-----------|-----|
| 前向传播次数 | N 次生成 N 个 token | 可降至 N/K 次（K 为预测数） |
| 推理延迟 | 高（memory-bound，受限于显存带宽） | 显著降低 |
| 输出质量 | 基准 | 数学等价，无损 |
| 适用场景 | 通用 | 可预测性强的文本收益更大 |

---

## 2. MTP 的两大路线

MTP 技术分为 **纯推理层面** 和 **训练内置** 两大类，理解它们的区别是关键。

### 2.1 纯推理层面：Speculative Decoding（投机解码）

**任何模型都能用，不需要训练时做任何改动。**

#### 核心架构：大小模型协作

```
Draft Model（小模型，快）：快速猜测 K 个 token
Target Model（大模型，慢）：一次前向传播验证这 K 个 token
```

#### 工作流程

```
Step 1: Draft Model 快速生成 K 个 token
  prompt → [t1, t2, t3, t4, t5]

Step 2: Target Model 一次前向传播，并行算出 K 个位置的概率分布
  位置1: P_target(· | prompt)
  位置2: P_target(· | prompt, t1)
  位置3: P_target(· | prompt, t1, t2)
  位置4: P_target(· | prompt, t1, t2, t3)
  位置5: P_target(· | prompt, t1, t2, t3, t4)

Step 3: 从左到右逐个验证，接受或拒绝
  t1 ✓ 接受    t2 ✓ 接受    t3 ✗ 拒绝！

Step 4: 拒绝后，用 Target Model 已算好的概率分布采样替换 token
  从 P3 的修正分布中采样出 t3'

Step 5: 本轮结束，已产出 [t1, t2, t3']
  从 t3' 开始下一轮 Draft 预测
```

#### 代表方案

- **经典 Speculative Decoding**：外挂一个同分布的小模型（如 Llama-68M draft → Llama-70B target）
- **Self-Speculative**：用模型自身的浅层（Early Exit）做 draft，无需额外模型
- **Prompt Lookup Decoding**：从 prompt 中查找 n-gram 匹配作为 draft，适合 RAG 场景

### 2.2 训练内置 MTP：模型原生支持

**模型在训练阶段就设计了多 token 预测能力。**

#### Medusa

```
模型最后一层 → 解码头 1 → 预测 next token (位置 +1)
             → 解码头 2 → 预测 next+1 token (位置 +2)
             → 解码头 3 → 预测 next+2 token (位置 +3)
             → ...
```

- 在模型上添加多个额外的解码头（head）
- 训练时微调这些 head（base model 可冻结）
- 推理时多个 head 并行输出，一次前向传播得到多步预测

#### EAGLE / EAGLE-2

```
当前 hidden state → 轻量级预测模块 → 预测未来多步的 hidden state
                                        → 映射到 token 概率
```

- 在特征层面（hidden state）训练轻量级预测模块
- 需要额外训练（数据来自模型自身的 hidden states）
- 比 Medusa 更轻量，效果更好

#### 原生 MTP（DeepSeek-V3、GLM 等）

- 模型架构本身就设计了多 token 预测目标
- 训练时以多 token 预测为 loss 目标
- 效果最好，但模型必须从预训练阶段就支持

### 2.3 两大路线对比

| 维度 | 纯推理 Speculative Decoding | 训练内置 MTP |
|------|---------------------------|-------------|
| 所有模型都能用？ | ✅ 是 | ❌ 需要专门训练 |
| 额外显存开销 | 需要 draft model（10%-30%） | 通常只需轻量 head |
| 加速比 | 2x-3x | 3x-5x+ |
| 输出是否等价 | ✅ 数学严格等价 | ✅ 数学严格等价 |
| 典型代表 | vLLM speculative decoding | Medusa, EAGLE, DeepSeek MTP |
| 部署复杂度 | 较低，框架支持即可 | 较高，需要特定模型权重 |

> **类比**：Speculative Decoding 像是给任何车加一个涡轮增压外挂，而原生 MTP 像是出厂就设计好的涡轮增压发动机。

---

## 3. 验证机制详解：Rejection Sampling

### 3.1 核心算法

这是 MTP 最精妙的部分：**推理时没有标准答案，不是判断"对错"，而是用 Target Model 的概率分布来校正 Draft Model 的概率分布。**

#### 概率比较与接受/拒绝

对每个 draft 输出的 token x：

```
Draft 采样出了 token x，概率为 p_draft(x)
Target 对同一个 token 的概率为 p_target(x)

接受概率 = min(1, p_target(x) / p_draft(x))

生成随机数 r ~ Uniform(0, 1)：
  - 如果 r < 接受概率 → 接受这个 token，继续验证下一个
  - 如果 r ≥ 接受概率 → 拒绝这个 token，用修正分布重新采样
```

#### 具体例子

**例1：Target 更认可的 token**
```
Draft 采样了 "the"，p_draft("the") = 0.6
Target 给出        p_target("the") = 0.7

接受概率 = min(1, 0.7/0.6) = min(1, 1.17) = 1.0
→ 100% 接受！因为 target 比 draft 更喜欢这个词
```

**例2：Target 不认可的 token**
```
Draft 采样了 "hello"，p_draft("hello") = 0.3
Target 给出           p_target("hello") = 0.05

接受概率 = min(1, 0.05/0.3) = 0.167
→ 只有 16.7% 概率接受，83.3% 概率被拒绝
→ 拒绝后用修正分布重新采样
```

### 3.2 拒绝后的修正分布

当 draft 的 token 被拒绝时，用 **修正分布** 重新采样：

```
p_修正(x) ∝ max(0, p_target(x) - p_draft(x))
```

**直觉**：只在 target 认为概率更高但 draft 低估了的 token 中重新选择。

### 3.3 为什么能保证输出等价？

这套算法基于 **rejection sampling（拒绝采样）**，数学上证明了：

> 经过 accept/reject 过程后，最终 token 的分布 **严格等于** 直接从 target model 采样的分布。

- **greedy decoding（temperature=0）**：严格完全等价，无任何差异
- **temperature > 0 采样**：理论完全等价，实际中仅有极小的浮点精度差异（可忽略）

### 3.4 拒绝后不是退化为单 token 生成

一个常见的误解是：拒绝后会退化为单 token 自回归。实际上：

```
Draft 猜测了 [t1, t2, t3, t4, t5]
Target 一次前向传播，已经算好了 5 个位置的概率分布

验证到 t3 被拒绝时：
  → Target 在位置3的概率分布 P3 已经算好了
  → 直接从 P3 的修正分布采样出 t3'
  → 不需要额外的前向传播
  → 不需要重新调用 Draft
```

**所有需要的概率信息都在 Target 的那一次前向传播里，拒绝后只需一次采样操作。**

### 3.5 生活类比

```
想象你是一个作家（Target Model），有个实习生（Draft Model）帮你写草稿：

1. 实习生快速写了 5 个词
2. 你逐个读：
   - "这个词我也会这样写" → 概率比实习生高 → 直接保留
   - "这个词我不太会这样用" → 概率比实习生低 → 有概率改掉
   - 改掉时，从"我比实习生更有把握的那些词"里选
3. 最终文章 = 你亲自写的文章（分布等价）
4. 但你只需要"审核"而非"从零写"，所以更快
```

---

## 4. 输出质量：有损耗吗？

### 4.1 理论结论：零损耗

MTP（包括 Speculative Decoding）**在数学上保证输出分布与原始自回归生成完全一致。**

核心保障机制是 rejection sampling 验证：
- 猜对的 token → 直接保留
- 猜错的 token → 丢弃，用 target model 的正确概率重新采样
- 最终分布 = 纯 target model 生成的分布

### 4.2 实际中的微小差异

| 场景 | 等价性 |
|------|--------|
| greedy decoding（temperature=0） | ✅ 严格完全等价 |
| 采样模式（temperature>0） | ✅ 理论完全等价 |
| 实际浮点精度 | ⚠️ 极小差异，可忽略不计 |

### 4.3 真正的代价不在准确性

| 维度 | 代价 |
|------|------|
| 显存 | Speculative Decoding 需要额外加载 draft model |
| 延迟抖动 | Draft 命中率低时，验证开销反而让单次请求变慢 |
| 适用场景 | 高度不可预测的文本（数学推理等），加速比下降 |
| 工程复杂度 | 需要框架支持验证逻辑 |

---

## 5. 不同场景的加速效果

Draft 命中率直接决定加速比，不同文本类型差异显著：

| 生成内容 | Draft 命中率 | 加速效果 | 原因 |
|----------|-------------|----------|------|
| 日常对话、常见文本 | 高（70%-90%） | 3x-5x | 文本可预测性强 |
| 通用知识问答 | 中（50%-70%） | 2x-3x | 大部分是常见表达 |
| 代码生成 | 中低（40%-60%） | 1.5x-2.5x | 语法模式有规律但变量名不确定 |
| 数学推理 | 低（20%-40%） | 1.2x-1.5x | 每步推理难以预测 |
| 创造性写作 | 低（20%-40%） | 1.2x-1.5x | 多样性高，难以猜测 |

**关键洞察**：MTP 提速是用"验证"换来的，不是用"牺牲质量"换来的。如果你的场景是生成可预测性强的文本（对话、翻译、摘要），MTP 是"白捡"的加速。

---

## 6. 在部署框架中的使用

### 6.1 vLLM

```python
# vLLM 中使用 Speculative Decoding
from vllm import LLM, SamplingParams

# 方法1：使用 draft model
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    speculative_model="meta-llama/Llama-2-7b-hf",  # draft model
    num_speculative_tokens=5,
)

# 方法2：使用 ngram prompt lookup（不需要额外模型）
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    speculative_model="[ngram]",
    num_speculative_tokens=5,
    ngram_prompt_lookup_max=4,
)
```

### 6.2 SGLang

```bash
# SGLang 中启用 MTP（针对支持的模型如 DeepSeek-V3）
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 5
```

### 6.3 部署框架通常两种都支持

- **对普通模型**：用 Speculative Decoding（外挂 draft model）
- **对专门训练的模型**：用原生 MTP（如 DeepSeek-V3 的 MTP 模块）

---

## 7. 关键概念速查

| 术语 | 解释 |
|------|------|
| MTP | Multi-Token Prediction，多 Token 预测 |
| Speculative Decoding | 投机解码，用 draft model 猜测 + target model 验证 |
| Draft Model | 小模型，负责快速猜测多个 token |
| Target Model | 大模型，负责一次前向传播验证所有猜测 |
| Rejection Sampling | 拒绝采样，通过 accept/reject 保证输出分布等价 |
| Accept Probability | 接受概率 = min(1, p_target/p_draft) |
| 修正分布 | 拒绝后用于重新采样的分布 ∝ max(0, p_target - p_draft) |
| Early Exit | 用模型浅层输出做 draft，self-speculative 的一种方式 |
| Medusa | 训练多个解码头实现 MTP 的方案 |
| EAGLE | 在特征层面训练轻量预测模块实现 MTP 的方案 |

---

## 8. 总结

```
MTP 的本质：
  用"便宜的方式"先猜多个 token → 用"贵的方式"一次验证 → 接受对的，纠正错的
  = 减少了前向传播次数 = 加速推理

MTP 的保障：
  通过 rejection sampling → 输出分布严格等价 → 零质量损耗

MTP 的代价：
  额外显存（draft model） + 不可预测文本场景加速有限

MTP 的两条路：
  纯推理（任意模型可用） vs 训练内置（效果更好但需专门训练）
```

---

## 📖 延伸阅读

- [Fast Inference from Transformers via Speculative Decoding (2023)](https://arxiv.org/abs/2211.17192) - Speculative Decoding 原论文
- [Medusa: Simple LLM Inference Acceleration with Multiple Decoding Heads (2024)](https://arxiv.org/abs/2401.10774) - Medusa 方案
- [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (2024)](https://arxiv.org/abs/2401.15077) - EAGLE 方案
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) - 原生 MTP 实现
