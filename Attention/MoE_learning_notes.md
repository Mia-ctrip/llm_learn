# MoE（Mixture of Experts）学习笔记

> 前置知识：已理解 GPT/Decoder-Only 架构、Transformer Block 结构
> 状态：待深入学习，先建立概念框架
> 记录日期：2026-06-12

---

## 第一章：MoE 是什么？一句话概括

**把一个大 FFN 拆成多个小 FFN（专家），每次只用其中几个，用更少的计算量达到更大的模型容量。**

```
传统 Transformer Block:                MoE Transformer Block:
┌────────────────────┐                ┌─────────────────────────────┐
│ Causal MHSA        │                │ Causal MHSA                 │
│ (所有参数都参与)    │                │ (所有参数都参与，和传统一样)  │
├────────────────────┤                ├─────────────────────────────┤
│ 一个大 FFN         │                │  Router (门控网络)           │
│ (所有参数都参与)    │                │  ┌────┬────┬────┬────┐      │
│                    │                │  │ E1 │ E2 │ E3 │ E4 │      │
│                    │                │  │FFN │FFN │FFN │FFN │      │
│                    │                │  └────┴────┴────┴────┘      │
│                    │                │  每次只激活 2 个专家          │
└────────────────────┘                └─────────────────────────────┘

传统: 1 个 FFN，参数全用              MoE: 4 个 FFN（专家），只用 2 个
参数量: 大                            总参数量: 更大（4个FFN）
计算量: 大                            每次计算量: 和传统差不多（只用2个）
```

---

## 第二章：为什么需要 MoE？

### 2.1 Scaling Law 的困境

```
传统做法：模型越大 → 效果越好（Scaling Law）

但问题来了：
  模型参数量翻倍 → 推理时每个 token 的计算量也翻倍
  
  GPT-3 (175B): 每个 token 都要经过全部 175B 参数
  → 推理太慢、太贵

能不能"模型很大但计算不多"？
→ MoE 就是答案之一
```

### 2.2 MoE 的核心思想

```
关键洞察：不是每个 token 都需要全部参数

"今天天气真好" → 描述天气，不需要数学/编程相关参数
"def fib(n):"  → 写代码，不需要文学/天气相关参数

→ 让不同的参数（专家）负责不同类型的知识
→ 每个 token 只激活相关的专家
→ 模型总参数大（知识丰富），但每次计算少（速度快）
```

---

## 第三章：MoE 的工作机制

### 3.1 Router（路由/门控网络）

```
Router 就是一个很简单的线性层：

输入: token 的隐藏向量 h (d_model 维)
  ↓
W_gate (d_model × num_experts)    ← 一个矩阵乘法
  ↓
logits (num_experts 维)            ← 每个专家的得分
  ↓
Softmax / TopK                     ← 选出 Top-K 个专家

例：4 个专家，选 2 个
  logits = [0.1, 0.8, 0.05, 0.7]
  softmax → [0.08, 0.42, 0.04, 0.36]
  Top-2 → 专家2 (0.42) 和 专家4 (0.36)
  
  归一化权重: [0.42/(0.42+0.36), 0.36/(0.42+0.36)]
            = [0.54, 0.46]
```

### 3.2 专家的计算与合并

```
假设选中了专家2 和专家4:

h (token 的向量)
  ├→ 专家2.FFN(h) → output₂
  └→ 专家4.FFN(h) → output₄

最终输出 = 0.54 × output₂ + 0.46 × output₄

= 加权求和（和你学过的 Attention 加权求和思想一样！）
```

### 3.3 伪代码

```python
class MoELayer(nn.Module):
    def __init__(self, num_experts, top_k, d_model, d_ff):
        self.num_experts = num_experts
        self.top_k = top_k
        # Router: 一个简单的线性层
        self.router = nn.Linear(d_model, num_experts)
        # 每个专家就是一个独立的 FFN
        self.experts = nn.ModuleList([
            FFN(d_model, d_ff) for _ in range(num_experts)
        ])
    
    def forward(self, h):
        # 1. Router 打分
        logits = self.router(h)                    # (batch, seq_len, num_experts)
        
        # 2. 选 Top-K 个专家
        topk_logits, topk_indices = logits.topk(self.top_k, dim=-1)
        topk_weights = softmax(topk_logits, dim=-1) # 归一化权重
        
        # 3. 计算选中的专家并加权合并
        output = torch.zeros_like(h)
        for i in range(self.top_k):
            expert_idx = topk_indices[..., i]
            weight = topk_weights[..., i]
            # 每个 token 路由到对应的专家
            output += weight * self.experts[expert_idx](h)
        
        return output
```

---

## 第四章：MoE 的关键概念

### 4.1 总参数量 vs 激活参数量

```
以 Mixtral 8x7B（Mistral 出品）为例：

  8 个专家，每个都是 7B 规模的 FFN
  每次激活 2 个专家 (Top-2)

  总参数量: ~47B（8 个专家的 FFN + 共享的 Attention 等）
  每次激活参数量: ~13B（2 个专家 + Attention 等）
  
  → 有 47B 的知识容量，但只有 13B 的计算成本！
  → 这就是 MoE 的价值：大容量 + 低计算
```

### 4.2 专家数（num_experts）

```
专家数 = MoE 层中 FFN 的数量

常见配置：
  Mixtral 8x7B:    8 个专家, Top-2
  DeepSeek-V2:     160 个专家(!), Top-6
  GPT-4 (推测):    可能有 8-16 个专家

专家数越多 → 模型知识越细分 → 路由越重要
但太多专家 → 路由训练困难，负载不均衡
```

### 4.3 负载均衡（Load Balancing）

```
问题：如果 Router 总是选同一批专家怎么办？

极端情况：
  专家1: 被选了 90% 的时间 → 过劳，学到太多东西
  专家2~8: 很少被选 → 闲置，没学到东西
  
  → 其他专家等于白训练（浪费参数和显存）

解决方案：加一个负载均衡 Loss
  鼓励 Router 均匀分配 token 给各专家
  
  aux_loss = 系数 × Σ(专家被选比例 × 专家平均得分)
  
  如果某个专家被选太多 → aux_loss 增大 → 反向传播会抑制
  
  ⚠ 这是 MoE 训练的核心难题之一
```

### 4.4 Expert Parallelism (EP)

```
MoE 推理时的显存问题：
  8 个专家都要加载到 GPU 上（虽然每次只用 2 个）
  → 显存占用很大

EP（Expert Parallelism）：
  把不同的专家放到不同的 GPU 上
  
  GPU 0: 专家 1, 2
  GPU 1: 专家 3, 4
  GPU 2: 专家 5, 6
  GPU 3: 专家 7, 8
  
  token 被路由到哪个专家 → 发送到对应的 GPU 计算
  
  和你已学的并行策略对比：
    DP (Data Parallelism):     每张卡跑完整模型，处理不同数据
    TP (Tensor Parallelism):   一个矩阵拆到多张卡上
    EP (Expert Parallelism):   不同专家放不同卡上 ← MoE 专属
    PP (Pipeline Parallelism): 不同层放不同卡上
```

---

## 第五章：MoE 在哪些模型中使用？

### 5.1 代表性 MoE 模型

| 模型 | 出品 | 专家数 | Top-K | 总参数 | 激活参数 | 年份 |
|------|------|--------|-------|--------|---------|------|
| GPT-4 (推测) | OpenAI | 未公开 | 未公开 | ~1.8T(推测) | ~280B(推测) | 2023 |
| Mixtral 8x7B | Mistral | 8 | 2 | 47B | 13B | 2024 |
| DeepSeek-V2 | DeepSeek | 160 | 6 | 236B | 21B | 2024 |
| DeepSeek-V3 | DeepSeek | 256 | 8 | 671B | 37B | 2025 |
| Qwen1.5-MoE | 阿里 | 60 | 4 | 14B | 2.7B | 2024 |
| DBRX | Databricks | 16 | 2 | 132B | 36B | 2024 |

### 5.2 DeepSeek 的 MoE 为什么值得关注？

```
DeepSeek-V3: 256 个专家，每次激活 8 个
  总参数 671B，激活参数仅 37B
  → 用相对少的算力跑出大模型的效果
  → 训练成本也比同规模 Dense 模型低很多

这就是为什么 DeepSeek 能定价那么低的原因之一：
  推理时只激活 37B 参数 → 计算成本低
  但知识容量接近 671B → 效果好
  → 低成本 + 高质量 = 价格优势
```

---

## 第六章：MoE vs Dense 模型

### 6.1 对比总结

| | Dense 模型 (传统) | MoE 模型 |
|---|---|---|
| **FFN** | 1 个大 FFN，所有参数参与 | N 个小 FFN（专家），只激活 K 个 |
| **计算量** | 和参数量成正比 | 和激活参数量成正比（远小于总参数） |
| **显存** | 参数量 = 显存需求 | 所有专家都要加载 → 显存需求更大 |
| **训练难度** | 相对简单 | 需要负载均衡、路由训练等额外技巧 |
| **推理速度** | 参数量大 → 慢 | 激活参数少 → 快（但通信开销） |
| **知识容量** | 受限于参数量 | 更大（总参数量大） |

### 6.2 MoE 的显存问题

```
你已学的显存公式:
  总显存 = 模型权重 + KV Cache + 运行时

MoE 的显存问题:
  模型权重 = 所有专家的参数都要加载！
  
  Dense 13B: 权重 26GB (FP16)
  MoE 47B (Mixtral): 权重 94GB (FP16) ← 虽然激活只有13B，但全部专家要加载
  
  → MoE 省的是计算量（FLOPs），不是显存！
  → 需要更多显存来装所有专家
  → 这也解释了为什么 HBM 需求越来越大（和你学的 KV Cache 知识串联）
```

---

## 第七章：MoE 和 Attention 的关系

```
MoE 只替换 FFN，不动 Attention！

一个 MoE Transformer Block:
┌─────────────────────────────────┐
│ Causal Multi-Head Attention     │  ← 完全不变！所有参数参与
│ (和传统 Transformer 一模一样)    │
├─────────────────────────────────┤
│ MoE Layer (替换传统 FFN)        │  ← 只有这里变了
│ Router → 选专家 → 加权合并      │
└─────────────────────────────────┘

为什么只替换 FFN？
  FFN 占 Transformer 参数的 ~2/3（你算过：每层 FFN ≈ 4M vs Attention ≈ 2.3M）
  → FFN 是参数大户，拆它的收益最大
  
  Attention 的参数相对较少，而且每个 token 都需要和所有 token 交互
  → 不适合"只让部分参数参与"
```

---

## 第八章：延伸阅读

### 推荐论文

1. **"Mixtral of Experts"** (2024, Mistral)
   - 最清晰的 MoE 模型论文，篇幅短
   - 用 8 个专家达到接近 70B Dense 模型的效果

2. **"DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts"** (2024)
   - 160 个专家，MLA (Multi-head Latent Attention) 等创新
   - DeepSeek 降本增效的核心技术

3. **"Switch Transformers"** (2022, Google)
   - 简化版 MoE（Top-1 路由），学术价值高
   - 提出了很多 MoE 训练技巧

4. **"GShard"** (2020, Google)
   - 最早的大规模 MoE 实践之一

### 推荐文章

5. **"What is Mixture of Experts?"** by Hugging Face Blog
   - 图文并茂，适合入门
6. **"MoE Explained"** by Sebastian Raschka
   - 详细的技术解读

---

## 第九章：概念串联

```
把你学过的知识和 MoE 串起来：

Transformer Block
  ├── Attention (你已学: Q/K/V, Multi-Head, Causal Mask, GQA)
  │     → MoE 不动这部分
  │
  └── FFN (你已学: 两层全连接, 升维→激活→降维)
        → MoE 把 1 个 FFN 变成 N 个，每次用 K 个
        → 你已知的 FFN 结构 (W₁, W₂, 激活函数) 在每个专家内部完全一样

显存计算 (你已学):
  总显存 = 权重 + KV Cache + 运行时
  MoE 的权重 = 所有专家参数之和 → 显存更大
  KV Cache 不受 MoE 影响（KV Cache 来自 Attention，MoE 不动 Attention）

推理优化 (你已学):
  KV Cache 省计算量 → MoE 也省计算量 → 两者叠加
  KV Cache 吃显存 → MoE 也吃显存 → 两者叠加 → HBM 需求暴增
  → 这就是为什么美光/海力士/三星股价上涨（完整闭环！）

训练:
  MoE 的 Router 是额外的训练参数（类似你学的 W_Q/W_K/W_V，也是随机初始化后训练）
  负载均衡 Loss 是额外的 Loss（类似你学的交叉熵 Loss，但作用于 Router）
```

---

## 附录：速查卡

```
┌─────────────────────────────────────────────────┐
│              MoE 速查                            │
├─────────────────────────────────────────────────┤
│                                                  │
│  核心: 多个 FFN（专家），每次只激活几个            │
│  目的: 大模型容量 + 低计算成本                    │
│                                                  │
│  Router: 线性层，决定 token 去哪个专家            │
│  Top-K: 每次激活的专家数（常见 2~8）              │
│  负载均衡: 防止部分专家过劳                       │
│  EP: 不同专家放不同 GPU                           │
│                                                  │
│  替换对象: FFN（不动 Attention）                  │
│  省的是: 计算量 (FLOPs)                          │
│  不省: 显存（所有专家都要加载）                    │
│                                                  │
│  代表模型: Mixtral, DeepSeek-V2/V3, DBRX         │
│                                                  │
│  参数量 ≈ Σ(所有专家参数) + Attention + 其他      │
│  激活量 ≈ K × 单个专家参数 + Attention + 其他     │
└─────────────────────────────────────────────────┘
```
