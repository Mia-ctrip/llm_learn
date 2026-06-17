# GPT / Decoder-Only 架构详解

> 前置知识：已学完原版 Transformer（Encoder-Decoder 架构）
> 本文目标：理解 GPT 系列及现代 LLM（Claude/Qwen/LLaMA）的架构细节
> 学习日期：2026-06-12

---

## 第一章：从 Transformer 到 GPT — 砍掉了什么？

### 1.1 架构对比（一张图看懂）

```
原版 Transformer (Encoder-Decoder):
┌─────────────────────┐     ┌──────────────────────────────┐
│     Encoder × N     │     │        Decoder × N           │
│ ┌─────────────────┐ │     │ ┌──────────────────────────┐ │
│ │ Self-Attention  │ │     │ │ Masked Self-Attention    │ │
│ │  + Add & Norm   │ │     │ │  + Add & Norm            │ │
│ ├─────────────────┤ │     │ ├──────────────────────────┤ │
│ │      FFN        │ │ ──→ │ │ Cross-Attention          │ │
│ │  + Add & Norm   │ │  R  │ │ (Q←Decoder, K/V←Encoder) │ │
│ └─────────────────┘ │     │ │  + Add & Norm            │ │
│                     │     │ ├──────────────────────────┤ │
│ 输入: 源语言         │     │ │      FFN                 │ │
│ "je suis étudiant"  │     │ │  + Add & Norm            │ │
└─────────────────────┘     │ └──────────────────────────┘ │
                            │                              │
                            │ 输入: 目标语言（右移）         │
                            │ "<BOS> i am a"               │
                            └──────────────────────────────┘
                                        ↓
                                   Linear + Softmax


GPT (Decoder-Only):
┌──────────────────────────────┐
│    Transformer Block × N     │
│ ┌──────────────────────────┐ │
│ │ Masked Self-Attention    │ │
│ │  + Add & Norm            │ │
│ ├──────────────────────────┤ │
│ │      FFN                 │ │
│ │  + Add & Norm            │ │
│ └──────────────────────────┘ │
│                              │
│ 输入 + 输出 是同一个序列：     │
│ "今天 天气 真"               │
│ → 预测 "好"                  │
└──────────────────────────────┘
          ↓
     Linear + Softmax
```

### 1.2 砍掉了什么？保留了什么？

| 组件 | 原版 Transformer | GPT | 为什么砍/留？ |
|------|-----------------|-----|-------------|
| Encoder | ✅ 有 | ❌ 没有 | 不需要"理解"另一个序列 |
| Cross-Attention | ✅ 有 | ❌ 没有 | 没有 Encoder 就没有 K/V 的来源 |
| Masked Self-Attention | ✅ 有 | ✅ 有 | 自回归生成必须遮住未来 |
| FFN | ✅ 有 | ✅ 有 | 每个 token 的独立加工 |
| Add & Norm | ✅ 有 | ✅ 有（位置有变） | 残差和归一化仍然需要 |
| Positional Encoding | sin/cos 固定 | 可训练的 RoPE/ALiBi | 现代改进 |

### 1.3 一个关键问题：没有 Encoder，输入从哪来？

```
原版 Transformer:
  Encoder 处理源序列（法语）→ 产生 R
  Decoder 处理目标序列（英语）+ Cross-Attention 读 R
  → 两个不同的序列，两个不同的角色

GPT:
  只有一个序列！
  输入 = 已有的所有 token（prompt + 已生成的词）
  输出 = 下一个 token
  
  "今天天气真" → 预测 "好"
  "今天天气真好" → 预测 "，"
  "今天天气真好，" → 预测 "适合"
  
  输入和输出是同一个序列的不同长度切片
```

---

## 第二章：GPT 单层详细结构

### 2.1 GPT 的 Transformer Block

```
输入 x (N × d_model)
  │
  ├──→ LayerNorm(x)                    ← ⚡ 先 Norm 再 Attention（Pre-Norm）
  │      ↓
  │    Masked Multi-Head Self-Attention ← ⚡ 用因果注意力（Causal Mask）
  │      ↓
  ├──→ + x（残差连接）                   ← h = x + Attention(LayerNorm(x))
  │
  ├──→ LayerNorm(h)                    ← ⚡ 先 Norm 再 FFN
  │      ↓
  │    FFN (MLP)                        ← ⚡ 现代变体用 GELU/SwiGLU 替代 ReLU
  │      ↓
  └──→ + h（残差连接）                   ← output = h + FFN(LayerNorm(h))
  
输出 → 传给下一层
```

### 2.2 与原版 Transformer 的三处关键差异

#### 差异一：LayerNorm 的位置（Pre-Norm vs Post-Norm）

```
原版 Transformer（Post-Norm，先算再归一化）:
  x → SubLayer → + x → LayerNorm → output

GPT / 现代 LLM（Pre-Norm，先归一化再算）:
  x → LayerNorm → SubLayer → + x → output
```

```
Post-Norm（原版）:                    Pre-Norm（GPT/现代LLM）:

    x                                    x
    ↓                                    ↓
  SubLayer                          LayerNorm(x)
    ↓                                    ↓
  + x ←── 残差                        SubLayer
    ↓                                    ↓
  LayerNorm                           + x ←── 残差
    ↓                                    ↓
  output                              output
```

**为什么改？**
- Pre-Norm 训练更稳定：梯度流更平滑，不容易爆炸
- Post-Norm 理论上表达能力更强，但需要更精细的学习率调度
- 现代大模型几乎全部采用 Pre-Norm（GPT-2 起、LLaMA、Qwen 全部是 Pre-Norm）

#### 差异二：激活函数

```
原版 Transformer:  ReLU(x) = max(0, x)
GPT-2:             GELU(x) ≈ x · Φ(x)  （平滑版 ReLU）
LLaMA/Qwen:        SiLU(x) = x · σ(x)  （也叫 Swish）
现代 MoE 模型:      SwiGLU: SiLU(x·W₁) ⊙ (x·W₃)  （门控机制）
```

```
ReLU:   __/      （硬折角，负值直接砍）
GELU:   _/‾       （平滑过渡，负值附近有微小输出）
SiLU:   _/‾       （类似 GELU，但计算更简单）

为什么不用 ReLU？
  ReLU 把负值直接变成 0 → 信息丢失
  GELU/SiLU 保留微小的负值信息 → 模型表达力更强
```

#### 差异三：位置编码

```
原版 Transformer:  sin/cos 固定位置编码（非训练参数）
GPT-2/3:           可训练的位置编码（训练参数）
LLaMA/Qwen/现代:   RoPE（旋转位置编码）或 ALiBi
```

**RoPE 简介**（Rotary Position Embedding，旋转位置编码）：
```
核心思想：把位置信息编码到 Q 和 K 的"旋转角度"里

不是加到输入上（像原版那样 x + pos_encoding）
而是在计算 Q·K 点积时，让点积的结果自然包含"相对位置"信息

数学本质：
  q_i 旋转 i×θ 角度
  k_j 旋转 j×θ 角度
  q_i · k_j = 只取决于 (i - j)，即相对位置

优势：
  1. 天然编码相对位置（而非绝对位置）
  2. 可以外推到训练时没见过的更长序列
  3. 不增加额外参数
```

---

## 第三章：GPT 的完整数据流

### 3.1 推理时（生成一个词）

```
用户输入: "AI is"
         ↓ tokenize
token_ids = [1024, 338]           ← 假设 "AI"=1024, "is"=338
         ↓ Embedding 查表
x = [E[1024], E[338]]            ← (2 × d_model) 矩阵
         ↓ 加位置编码（RoPE 融入 Q/K）

┌───── Transformer Block 1 ─────┐
│ LayerNorm → Masked MHSA → +  │
│ LayerNorm → FFN → +          │
└───────────────────────────────┘
         ↓
┌───── Transformer Block 2 ─────┐
│ LayerNorm → Masked MHSA → +  │
│ LayerNorm → FFN → +          │
└───────────────────────────────┘
         ↓
       ... × N 层 ...
         ↓
┌───── Transformer Block N ─────┐
│ LayerNorm → Masked MHSA → +  │
│ LayerNorm → FFN → +          │
└───────────────────────────────┘
         ↓
      最终 LayerNorm              ← 有些模型最后再加一次 Norm
         ↓
      取最后一行向量 z (d_model 维)  ← 只用最后一个 token 的输出！
         ↓
      z × E^T                     ← ⚡ 和 Embedding 矩阵转置相乘
         ↓
      logits (vocab_size 维)      ← 每个词的得分
         ↓
      Softmax / 采样
         ↓
      下一个 token: "amazing"
```

### 3.2 ⚡ Weight Tying（权重共享）

```
原版 Transformer:
  Embedding 矩阵 E: (vocab_size × d_model)   ← 输入端
  Linear 层 W:      (d_model × vocab_size)    ← 输出端
  两个独立的参数矩阵，各自训练

GPT / 现代 LLM（大部分）:
  Embedding 矩阵 E: (vocab_size × d_model)   ← 唯一的矩阵
  输出层:           z × E^T                   ← 直接用 E 的转置！
  
  参数量减少: vocab_size × d_model 个参数
  例: GPT-2 (vocab=50257, d_model=768):
      省去 50257 × 768 ≈ 3860万 参数

为什么可以共享？
  E 的第 i 行 = "token i 的语义向量"
  输出时 z × E^T 的第 i 个值 = z 和 E[i] 的点积
  点积越大 = z 和 token i 的语义越接近 → 得分越高
  
  本质：输入时"token→向量"，输出时"向量→token"，互为逆过程
```

### 3.3 GPT 的完整参数量估算

```
以 GPT-2 (117M) 为例:
  d_model = 768
  num_layers = 12
  num_heads = 12
  d_k = 768 / 12 = 64
  vocab_size = 50257

每层 Transformer Block:
  Attention:
    W_Q: 768 × 768 = 589,824
    W_K: 768 × 768 = 589,824
    W_V: 768 × 768 = 589,824
    W_O: 768 × 768 = 589,824
    Attention 小计: 2,359,296

  FFN:
    W₁: 768 × 3072 = 2,359,296    （升维到 4×d_model）
    W₂: 3072 × 768 = 2,359,296    （降维回来）
    FFN 小计: 4,718,592

  LayerNorm × 2: 2 × (768 + 768) = 3,072

  每层总计: ≈ 7,080,960 ≈ 7M

12 层总计: 12 × 7M ≈ 85M
Embedding: 50257 × 768 ≈ 38.6M
最终 LayerNorm: 768 × 2 ≈ 1.5K

总计: 85M + 38.6M ≈ 124M
（实际 117M 因为 Weight Tying 省了 Embedding 的一部分）
```

**通用估算公式**：
```
总参数 ≈ 12 × d_model² × num_layers + vocab_size × d_model

简化版：总参数 ≈ 12 × d² × L（当 vocab×d 相对较小时）
```

---

## 第四章：GPT 家族的演进

### 4.1 GPT 系列架构演进

```
GPT-1 (2018, 117M):
  基础 Decoder-Only
  12层, d_model=768
  可训练位置编码
  ReLU → 后面改 GELU

GPT-2 (2019, 1.5B):
  Pre-Norm（关键改进！）
  48层, d_model=1600
  Weight Tying
  50257 vocab (BPE)

GPT-3 (2020, 175B):
  96层, d_model=12288
  96 heads, d_k=128
  架构和 GPT-2 几乎一样，纯粹靠 scale up
  
GPT-4 (2023, 未公开):
  推测: MoE（混合专家）架构
  多模态（文本+图像）
  具体架构未公开
```

### 4.2 现代 LLM 的通用架构（LLaMA/Qwen/DeepSeek 风格）

```
┌────────────────────────────────────────────────┐
│              现代 Decoder-Only LLM              │
│                                                │
│  输入 tokens                                    │
│    ↓                                            │
│  Embedding (vocab × d_model)                    │
│    ↓                                            │
│  ┌────────────────────────────────────────┐     │
│  │  Transformer Block × N（几十到上百层）    │     │
│  │                                        │     │
│  │  RMSNorm(x)          ← 用 RMSNorm      │     │
│  │    ↓                                   │     │
│  │  Causal Multi-Head Attention           │     │
│  │  (with RoPE, GQA)    ← 旋转位置编码    │     │
│  │    ↓                                   │     │
│  │  + x (残差)                            │     │
│  │    ↓                                   │     │
│  │  RMSNorm(h)                            │     │
│  │    ↓                                   │     │
│  │  SwiGLU FFN          ← 门控 FFN       │     │
│  │    ↓                                   │     │
│  │  + h (残差)                            │     │
│  └────────────────────────────────────────┘     │
│    ↓                                            │
│  最终 RMSNorm                                   │
│    ↓                                            │
│  LM Head (d_model → vocab)                      │
│    ↓                                            │
│  下一个 token                                    │
└────────────────────────────────────────────────┘
```

### 4.3 现代 LLM 相比 GPT-2 的额外改进

| 改进 | GPT-2 | LLaMA/Qwen 等现代模型 | 好处 |
|------|-------|---------------------|------|
| **归一化** | LayerNorm | RMSNorm | 计算更快（去掉均值计算） |
| **激活函数** | GELU | SiLU/SwiGLU | 门控机制增强表达力 |
| **位置编码** | 可训练绝对位置 | RoPE | 可外推长序列 |
| **注意力** | MHA (Multi-Head) | GQA (Grouped-Query) | KV Cache 更小 |
| **FFN 结构** | 标准两层 | 门控三层 (SwiGLU) | 更强的非线性 |
| **训练效率** | 标准 | FlashAttention | 注意力计算加速 |
| **Normalization** | Pre-Norm | Pre-Norm | 一样 |

---

## 第五章：GQA — 现代 LLM 对注意力的重要改进

### 5.1 从 MHA → MQA → GQA

```
MHA (Multi-Head Attention, GPT-2/3 用的):
  每个 Head 有独立的 Q, K, V
  8 个 Head = 8 套 Q + 8 套 K + 8 套 V
  KV Cache 大小 = 8 × (K + V)

MQA (Multi-Query Attention):
  每个 Head 有独立的 Q，但 K 和 V 所有 Head 共享
  8 个 Head = 8 套 Q + 1 套 K + 1 套 V
  KV Cache 大小 = 1 × (K + V)  ← 缩小 8 倍！
  缺点：模型质量略有下降

GQA (Grouped-Query Attention, LLaMA-2 70B/Qwen2 等):
  折中方案：把 Head 分组，每组共享一套 K/V
  8 个 Head 分 4 组 = 8 套 Q + 4 套 K + 4 套 V
  KV Cache 大小 = 4 × (K + V)  ← 缩小 2 倍
  质量接近 MHA，显存接近 MQA
```

```
MHA (8 heads, 8 KV heads):
  Head1: Q1 K1 V1
  Head2: Q2 K2 V2
  Head3: Q3 K3 V3
  Head4: Q4 K4 V4
  Head5: Q5 K5 V5
  Head6: Q6 K6 V6
  Head7: Q7 K7 V7
  Head8: Q8 K8 V8
  KV Cache: 8 份

GQA (8 heads, 4 KV heads):
  Head1: Q1 K1 V1 ─┐
  Head2: Q2 K1 V1 ─┘  ← 共享 K1/V1
  Head3: Q3 K2 V2 ─┐
  Head4: Q4 K2 V2 ─┘  ← 共享 K2/V2
  Head5: Q5 K3 V3 ─┐
  Head6: Q6 K3 V3 ─┘  ← 共享 K3/V3
  Head7: Q7 K4 V4 ─┐
  Head8: Q8 K4 V4 ─┘  ← 共享 K4/V4
  KV Cache: 4 份（减半！）

MQA (8 heads, 1 KV head):
  Head1~8: Q1~Q8 K1 V1
  全部共享同一套 K/V
  KV Cache: 1 份（最小！）
```

### 5.2 GQA 对 KV Cache 的影响

```
你之前学的 KV Cache 公式（MHA）:
  KV Cache = 2 × batch × seq_len × num_layers × hidden_size × bytes

GQA 的 KV Cache 公式:
  KV Cache = 2 × batch × seq_len × num_layers × (num_kv_heads × d_k) × bytes
  
  num_kv_heads < num_heads → KV Cache 更小！

例：LLaMA-2 70B:
  num_heads = 64, num_kv_heads = 8（GQA, 8组）
  KV Cache 缩小到 8/64 = 1/8
  
  这就是为什么大模型能用更少的显存处理更长的对话
```

---

## 第六章：Causal Mask 的实现

### 6.1 和原版 Transformer 的 Mask 一样吗？

完全一样！你在 Transformer 笔记里学的 Mask 机制，GPT 一字不改地沿用：

```
4 个 token 的注意力分数矩阵（softmax 前）:

           t1      t2      t3      t4
  t1  [  2.1     -∞      -∞      -∞  ]   ← 只看自己
  t2  [  1.5     3.2     -∞      -∞  ]   ← 看 t1, t2
  t3  [  0.3     1.8     2.9     -∞  ]   ← 看 t1, t2, t3
  t4  [ -0.1     0.5     1.2     3.1 ]   ← 看所有

softmax 后 -∞ → 0:
           t1      t2      t3      t4
  t1  [  1.0      0        0       0  ]
  t2  [  0.15    0.85      0       0  ]
  t3  [  0.05    0.15     0.80     0  ]
  t4  [  0.02    0.05     0.13    0.80]

每行的权重加起来 = 1，且只能"看到"自己和前面的 token
```

### 6.2 GPT 推理时 Mask + KV Cache 的配合

```
推理第 3 步（已生成 t1, t2, t3，正在生成 t4）:

没有 KV Cache:
  输入 [t1, t2, t3, t4]
  算 4×4 的注意力矩阵（大部分被 Mask 遮掉，浪费计算）
  
有 KV Cache:
  t4 只需要算自己的 q4
  q4 和缓存的 [k1, k2, k3, k4] 做点积
  得到 4 个分数 → softmax → 加权 [v1, v2, v3, v4]
  
  ⚡ 不需要 Mask 矩阵了！
  因为 q4 和所有 k 做点积，天然就是"看所有前面的"
  （没有"看未来"的问题，因为未来根本还没生成）

→ 推理时，KV Cache 不仅省计算量，还省去了 Mask 的开销
```

---

## 第七章：GPT 训练流程

### 7.1 训练 vs 推理（和原版 Transformer 对比）

```
原版 Transformer 训练:
  Encoder 输入: 法语 "je suis étudiant"
  Decoder 输入: 英语 "<BOS> i am a"          ← Teacher Forcing
  Decoder 目标: 英语 "i am a student"
  → 两个序列，Cross-Attention 连接

GPT 训练:
  输入: "The cat sat on the"
  目标: "cat sat on the mat"                  ← 右移一位，和原版一样
  
  一次 forward 算所有位置的预测:
    位置1 看到 "The"           → 预测 "cat"   → Loss₁
    位置2 看到 "The cat"       → 预测 "sat"   → Loss₂
    位置3 看到 "The cat sat"   → 预测 "on"    → Loss₃
    位置4 看到 "The cat sat on"→ 预测 "the"   → Loss₄
    位置5 看到 "The cat sat on the" → 预测 "mat" → Loss₅
  
  Loss = 平均(Loss₁~Loss₅)
  → 一个序列，没有 Cross-Attention
```

### 7.2 GPT 的训练数据从哪来？

```
原版 Transformer: 需要平行语料（法语-英语对照）
  → 有监督，数据稀缺

GPT: 只需要大量文本，不需要对照！
  → 训练数据 = 互联网文本 + 书籍 + 代码 + ...
  → 自监督：用文本的前 N-1 个词预测第 N 个词
  → 数据量可以达到数十万亿 token

这就是 GPT 能 scale up 到 175B 参数的关键原因：
  不需要人工标注的平行语料
  任何文本都可以当训练数据（"下一个词预测"是天然的自监督任务）
```

---

## 第八章：一图总结 GPT 的推理全流程

```
用户: "解释什么是AI"
         │
    ┌────▼─────────────────────────────────┐
    │ 1. Tokenize（分词）                    │
    │    "解释什么是AI" → [8421, 3301, 772] │
    └────┬─────────────────────────────────┘
         │
    ┌────▼─────────────────────────────────┐
    │ 2. Embedding 查表                     │
    │    [E[8421], E[3301], E[772]]        │
    │    → (3 × 4096) 矩阵                  │
    └────┬─────────────────────────────────┘
         │
    ┌────▼─────────────────────────────────┐
    │ 3. Prefill（并行处理所有输入 token）   │
    │                                      │
    │  ┌─ Block 1 ──────────────────────┐  │
    │  │ RMSNorm → Causal MHSA(+RoPE)  │  │
    │  │ → +残差 → RMSNorm → SwiGLU    │  │
    │  │ → +残差                         │  │
    │  └────────────────────────────────┘  │
    │            × 32层（举例）              │
    │  ┌─ Block 32 ─────────────────────┐  │
    │  │ RMSNorm → Causal MHSA(+RoPE)  │  │
    │  │ → +残差 → RMSNorm → SwiGLU    │  │
    │  │ → +残差                         │  │
    │  └────────────────────────────────┘  │
    │                                      │
    │  同时：缓存所有 token 的 K/V → KV Cache │
    │  取最后位置输出 → Linear → Softmax     │
    │  → 第一个生成的 token: "人工"          │
    └────┬─────────────────────────────────┘
         │
    ┌────▼─────────────────────────────────┐
    │ 4. Decode（逐 token 生成，循环）       │
    │                                      │
    │  新 token "人工" → Embedding          │
    │  → 只算这一个 token 的 Q              │
    │  → 和 KV Cache 里的所有 K 做点积      │
    │  → 加权所有 V → 得到输出              │
    │  → 追加新 K/V 到缓存                  │
    │  → Linear → Softmax → "智能"         │
    │                                      │
    │  重复: "智能" → ... → "是" → ...      │
    │  直到生成 <EOS> 或达到 max_length      │
    └────┬─────────────────────────────────┘
         │
    ┌────▼─────────────────────────────────┐
    │ 5. 输出                               │
    │    "人工智能是一种模拟人类智能的技术"     │
    └──────────────────────────────────────┘
```

---

## 第九章：和你已学知识的映射表

| 你已学的 Transformer 概念 | 在 GPT 中对应什么 | 有变化吗？ |
|---|---|---|
| Encoder | ❌ 不存在 | 被砍掉 |
| Cross-Attention | ❌ 不存在 | 被砍掉 |
| Masked Self-Attention | 唯一的注意力层 | ✅ 完全一样（Causal Mask） |
| FFN | FFN / MLP | 激活函数从 ReLU → GELU → SwiGLU |
| LayerNorm | RMSNorm 或 LayerNorm | 位置从 Post-Norm → Pre-Norm |
| 残差连接 | 残差连接 | ✅ 完全一样 |
| sin/cos 位置编码 | RoPE 或可训练位置编码 | 现代改进 |
| Embedding 矩阵 E | 完全一样 | ✅ |
| Linear + Softmax | LM Head（常 Weight Tying） | 常共享权重 |
| Q·K^T / √d_k | 完全一样 | ✅ |
| Multi-Head Attention | MHA 或 GQA | 现代用 GQA 省 KV Cache |
| Teacher Forcing | 完全一样 | ✅ |
| 自回归生成 | 完全一样 | ✅ |

---

## 第十章：推荐阅读

### 核心论文（按推荐阅读顺序）

1. **"Language Models are Unsupervised Multitask Learners"** (GPT-2 论文, 2019)
   - OpenAI 出品，篇幅短，清晰描述了 Decoder-Only + Pre-Norm 架构
   - 重点看 Section 2 (Model) 和 Section 3 (Approach)
   - https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

2. **"LLaMA: Open and Efficient Foundation Language Models"** (2023)
   - Meta 出品，描述了现代 LLM 的所有改进（RMSNorm, RoPE, SwiGLU, GQA）
   - 重点看 Section 2 (Pre-training data) 和 Section 3 (Architecture)
   - 这篇论文就是你"现代 LLM 架构"的最佳参考

3. **"GPT-Q: Accurate Quantization for GPT"** 或 **"GQA: Training Generalized Multi-Query Transformer Models"** (2023)
   - 理解 GQA 的原始论文

### 可视化文章

4. **"The Illustrated GPT-2"** by Jay Alammar
   - 和你学 Transformer 时看的 "Illustrated Transformer" 同一个作者
   - 用可视化方式讲解 GPT-2 的推理和训练流程
   - https://jalammar.github.io/illustrated-gpt2/

5. **"Transformer Taxonomy"** by Eugene Yan
   - 对比各种 Transformer 变体的架构差异
   - https://eugeneyan.com/writing/transformers/

---

## 附录：GPT 架构速查卡

```
┌─────────────────────────────────────────────────┐
│           GPT / Decoder-Only LLM 速查            │
├─────────────────────────────────────────────────┤
│                                                  │
│  结构:  N 层 Transformer Block 堆叠              │
│  每层:  RMSNorm → Causal MHSA → 残差            │
│        → RMSNorm → SwiGLU FFN → 残差            │
│                                                  │
│  关键组件:                                        │
│  · Causal Mask: 遮住未来 token                   │
│  · RoPE: 旋转位置编码（编码在 Q/K 的旋转中）      │
│  · GQA: 多个 Q head 共享 K/V head                │
│  · SwiGLU: 门控 FFN 激活                         │
│  · Weight Tying: Embedding 和 LM Head 共享      │
│  · Pre-Norm: 先归一化再计算                       │
│                                                  │
│  推理两阶段:                                      │
│  · Prefill: 并行处理 prompt，建立 KV Cache       │
│  · Decode: 逐 token 生成，读取 KV Cache          │
│                                                  │
│  训练: 自监督（下一个 token 预测）                 │
│  无需 Encoder，无需 Cross-Attention              │
│                                                  │
│  参数量 ≈ 12 × d_model² × num_layers            │
└─────────────────────────────────────────────────┘
```
