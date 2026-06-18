# KV Cache 学习指南

> **适合人群**：了解神经网络基础，但还未学习 Attention 机制的 AI Infra 工程师  
> **学习时长**：30-45 分钟（快速入门） + 2-3 小时（深入理解，需先学完 Attention）  
> **配合学习路线**：Week 5-6（在学完 Week 3-4 Transformer 后深化）

---

## 📚 前置知识速览

### 极简版 Attention 机制

在完全理解 Attention 之前，你只需要知道这些**最小必要概念**：

#### 1. 大模型如何生成文本？

```
输入：请写一篇关于
输出过程：
  第1步：生成 "人工"     → 请写一篇关于 人工
  第2步：生成 "智能"     → 请写一篇关于 人工 智能
  第3步：生成 "的"       → 请写一篇关于 人工 智能 的
  ...
```

**关键特点**：**自回归生成** = 一次生成一个词，每次都要"看懂"前面所有的词。

#### 2. Q、K、V 是什么？

想象你在图书馆找书：

```
你的需求（Query, Q）："我想找关于机器学习的书"
书架标签（Key, K）  ：["数学", "物理", "计算机", "历史"]
书的内容（Value, V） ：[《微积分》, 《力学》, 《算法导论》, 《世界史》]

匹配过程：
Q 与每个 K 计算相似度 → "计算机" 匹配度最高 → 取对应的 V → 《算法导论》
```

在大模型中：
- **Query (Q)**：当前词想要找什么信息？
- **Key (K)**：每个词的"索引特征"（用来匹配）
- **Value (V)**：每个词的"实际内容"（用来理解）

**Attention 计算**（简化版）：
```python
# 1. 计算相似度
scores = Q @ K.T  # Q 与所有 K 的匹配分数

# 2. 归一化（得到注意力权重）
weights = softmax(scores)  # [0.1, 0.05, 0.8, 0.05] 表示关注程度

# 3. 加权求和
output = weights @ V  # 根据权重混合所有 V
```

#### 3. 为什么是 "KV" Cache？

因为在生成过程中：
- **Q**：每次生成新词时都会变（当前词的需求是新的）
- **K 和 V**：已生成的词的 K 和 V 不会变（历史信息固定）

**所以只需要缓存 K 和 V！**

---

## 🧠 KV Cache 原理

### 核心思想

**避免对历史 token 的重复计算**，把已经计算过的 Key 和 Value 存起来。

### 没有 KV Cache 的生成过程

```python
# 假设输入："翻译：Hello" → 输出："你好"

# 第1步：生成 "你"
input_tokens = ["翻译", ":", "Hello"]  # 3个词
# 对每个词计算 Q, K, V（3次计算）
# Attention 计算涉及 3×3 的矩阵运算
output_token_1 = "你"

# 第2步：生成 "好"
input_tokens = ["翻译", ":", "Hello", "你"]  # 4个词
# 对每个词计算 Q, K, V（4次计算）← 前3个词重复计算了！
# Attention 计算涉及 4×4 的矩阵运算
output_token_2 = "好"

# 总计算量：3 + 4 = 7 次词处理
```

### 有 KV Cache 的生成过程

```python
# 第1步：生成 "你"
input_tokens = ["翻译", ":", "Hello"]
# 计算所有词的 K, V，并缓存
kv_cache = {
    "翻译": (K1, V1),
    ":"   : (K2, V2),
    "Hello": (K3, V3)
}
output_token_1 = "你"

# 第2步：生成 "好"
new_token = "你"
# 只计算新词的 K, V
K_new, V_new = compute(new_token)
# 从缓存中读取历史的 K, V
K_all = [K1, K2, K3, K_new]  # ← 前3个直接读缓存！
V_all = [V1, V2, V3, V_new]
output_token_2 = "好"

# 总计算量：3 + 1 = 4 次词处理（节省了 43%）
```

### 伪代码实现

```python
class LLMWithKVCache:
    def __init__(self):
        self.kv_cache = []  # [(K_layer1, V_layer1), (K_layer2, V_layer2), ...]
    
    def generate(self, input_ids, max_new_tokens=100):
        """带 KV Cache 的生成"""
        
        # === Prefill 阶段 ===
        # 处理输入 prompt，初始化 KV Cache
        for layer_idx in range(self.num_layers):
            Q, K, V = self.compute_qkv(input_ids, layer_idx)
            
            # 缓存 K 和 V
            self.kv_cache.append((K, V))
            
            # Attention 计算
            output = self.attention(Q, K, V)
        
        # 预测第一个新 token
        next_token = self.predict_next_token(output)
        
        # === Decode 阶段 ===
        for step in range(max_new_tokens - 1):
            input_ids = [next_token]  # 只有一个新 token
            
            for layer_idx in range(self.num_layers):
                # 只计算新 token 的 Q, K, V
                Q_new, K_new, V_new = self.compute_qkv(input_ids, layer_idx)
                
                # 从缓存中读取历史 K, V
                K_cached, V_cached = self.kv_cache[layer_idx]
                
                # 拼接：历史 + 新的
                K_all = torch.cat([K_cached, K_new], dim=1)  # [batch, seq_len+1, dim]
                V_all = torch.cat([V_cached, V_new], dim=1)
                
                # 更新缓存
                self.kv_cache[layer_idx] = (K_all, V_all)
                
                # Attention（Q 只有1个，K 和 V 有 seq_len+1 个）
                output = self.attention(Q_new, K_all, V_all)
            
            next_token = self.predict_next_token(output)
        
        return generated_tokens
```

### KV Cache 的实际数据结构（深入）

#### 容器类型

```python
# 新版 transformers (≥4.36)：DynamicCache 对象
past_key_values = outputs.past_key_values  # DynamicCache 实例

# 旧版 transformers：tuple of tuples
past_key_values = ((K₀, V₀), (K₁, V₁), ..., (K₁₁, V₁₁))

# 访问方式统一：按层索引
past_key_values[0]  → (K₀, V₀)  # 第 0 层
past_key_values[1]  → (K₁, V₁)  # 第 1 层
```

#### K/V tensor 的 4 维形状

```
每层 K shape: (batch, num_heads, seq_len, d_k)
              = (1,    12,        10,      64)  # GPT-2 Small 示例

重要认知：
  12 个头的 K 数据打包在一个 tensor 里，不是 12×12=144 组！
  每层只有 1 个 K tensor 和 1 个 V tensor，头信息嵌在第 2 维度

访问单个头：
  K[层0, 头0, :, :] → (seq_len, d_k) = (10, 64)
  → 这就是第 0 层、头 0 给所有 10 个 token 计算的 K 向量
```

#### KV Cache 是追加（append），不是求和（sum）

```
❌ 误解：sum += new_token_kv  → 所有 token 的 K/V 加起来变成一个向量
✅ 正确：cache.append(new_kv) → 每个 token 的 K/V 独立保存，seq_len 维度不断增长

如果"加起来"，10 个 token 的 K 向量就变成了 1 个向量，每个 token 的个体信息全部丢失了。
但 Attention 计算时需要和每个 token 的 K 单独做点积，所以必须独立保存。

类比：KV Cache 像一个笔记本，每页存一个 token 的 K/V，Decode 每次新增一页。
```

#### Decode 阶段 KV Cache 的动态增长

```
Prefill 结束后：
  K shape: (batch, heads, seq_len=10, d_k)  ← 10 个 token

Decode Step 1: 新 token → K_new 追加 → (batch, heads, seq_len=11, d_k)
Decode Step 2: 新 token → K_new 追加 → (batch, heads, seq_len=12, d_k)
...
Decode Step N: 新 token → K_new 追加 → (batch, heads, seq_len=10+N, d_k)

→ seq_len 这个维度在不断增长，其他维度都不变
→ 最终大小 = 2 × batch × num_layers × num_heads × (seq_len + output_len) × d_k × bytes
```

#### KV Cache 存储在哪个设备

```
KV Cache 是 torch tensor，和模型在同一个设备上：
  模型在 CPU → KV Cache 在内存（RAM）
  模型在 GPU → KV Cache 在显存（VRAM）

检查方式：past_key_values[0][0].device

这正是 KV Cache 成为大模型推理瓶颈的核心原因：
  长序列时 KV Cache 会占用大量显存，甚至超出显存容量
  → 这就是为什么有 PagedAttention (vLLM) 等显存管理技术
```


---

## 📊 有无 KV Cache 的对比

### 计算量对比

假设生成 N 个新 token，输入 prompt 长度为 M：

| 指标 | 无 KV Cache | 有 KV Cache | 节省 |
|------|------------|-------------|------|
| **Prefill 阶段** | 处理 M 个 token | 处理 M 个 token | 0% |
| **Decode 第1步** | 处理 M+1 个 token | 处理 1 个 token | **M/(M+1)** |
| **Decode 第2步** | 处理 M+2 个 token | 处理 1 个 token | **(M+1)/(M+2)** |
| **Decode 第N步** | 处理 M+N 个 token | 处理 1 个 token | **(M+N-1)/(M+N)** |
| **总计算量** | M + (M+1) + ... + (M+N) <br>≈ **N×M + N²/2** | M + N×1 <br>≈ **M + N** | **~98%**（当 N 大时） |

**实际案例**：
```
输入 prompt：1000 tokens
生成长度：100 tokens

无 KV Cache：
  1000 + 1001 + 1002 + ... + 1100 = 105,050 次 token 处理

有 KV Cache：
  1000 + 100 = 1,100 次 token 处理

节省：(105,050 - 1,100) / 105,050 = 98.95%
```

### 显存占用对比

```python
# 显存组成
无 KV Cache：
  显存 = 模型参数 + 当前批次的激活值

有 KV Cache：
  显存 = 模型参数 + 当前批次的激活值 + KV Cache

# KV Cache 显存公式
KV_Cache_显存 = 2 × batch_size × seq_len × num_layers × hidden_size × bytes_per_element

解释：
- 2：K 和 V 各一份
- batch_size：批量大小
- seq_len：序列长度（会随着生成不断增长！）
- num_layers：Transformer 层数
- hidden_size：隐藏层维度
- bytes_per_element：数据类型（FP16=2字节，FP32=4字节）
```

**实际案例**（GPT-3 规模模型）：
```python
batch_size = 1
seq_len = 2048  # 生成了2048个token
num_layers = 96
hidden_size = 12288
dtype = FP16  # 2 bytes

KV_Cache = 2 × 1 × 2048 × 96 × 12288 × 2
          = 9,663,676,416 bytes
          ≈ 9.0 GB

# 一个对话就占用 9GB！这就是为什么长对话会 OOM
```

### 速度对比

```
            无 KV Cache              有 KV Cache
Prefill    |████████| 1.2s         |████████| 1.2s
Decode 1   |████████| 1.2s         |█| 0.05s  ← 快 24 倍！
Decode 2   |████████| 1.2s         |█| 0.05s
Decode 3   |████████| 1.2s         |█| 0.05s
...
总耗时     120s (100 tokens)       6.2s (100 tokens)

首 Token 延迟（TTFT）：
  无 KV Cache: 2.4s
  有 KV Cache: 1.25s (节省 48%)

每 Token 延迟：
  无 KV Cache: ~1.2s
  有 KV Cache: ~0.05s (快 24 倍)
```

---

## ⚙️ 推理阶段的工作流程

### 完整推理流程图

```
用户请求：翻译：Hello world
         ↓
┌──────────────────────────────────────────────────┐
│  Phase 1: Prefill（预填充阶段）                     │
├──────────────────────────────────────────────────┤
│  输入：["翻译", ":", "Hello", "world"] (4 tokens)   │
│                                                  │
│  1. 并行处理所有 token                            │
│     token1 → Q1, K1, V1 ┐                        │
│     token2 → Q2, K2, V2 ├─→ Attention           │
│     token3 → Q3, K3, V3 │                        │
│     token4 → Q4, K4, V4 ┘                        │
│                                                  │
│  2. 初始化 KV Cache                              │
│     cache = [(K1,V1), (K2,V2), (K3,V3), (K4,V4)]│
│                                                  │
│  3. 预测第一个输出 token                         │
│     → "你"                                       │
│                                                  │
│  耗时：较长（但只有一次）                         │
│  特点：算力密集（可以并行）                       │
└──────────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────────┐
│  Phase 2: Decode（解码阶段）                       │
├──────────────────────────────────────────────────┤
│  循环生成每个新 token：                           │
│                                                  │
│  Iteration 1:                                    │
│    新 token: "你"                                │
│    ├─ 计算 Q5, K5, V5                           │
│    ├─ 读缓存: K1-4, V1-4  ← 关键优化！           │
│    ├─ Attention(Q5, K1-5, V1-5)                 │
│    ├─ 更新缓存: append (K5, V5)                 │
│    └─ 预测: "好"                                 │
│                                                  │
│  Iteration 2:                                    │
│    新 token: "好"                                │
│    ├─ 计算 Q6, K6, V6                           │
│    ├─ 读缓存: K1-5, V1-5  ← 关键优化！           │
│    ├─ Attention(Q6, K1-6, V1-6)                 │
│    ├─ 更新缓存: append (K6, V6)                 │
│    └─ 预测: <EOS>（结束）                        │
│                                                  │
│  耗时：每步很快（~0.05s/token）                   │
│  特点：内存带宽密集（逐个生成）                    │
└──────────────────────────────────────────────────┘
         ↓
    输出："你好"
```

### 关键概念

**Prefill vs Decode**：

| 阶段 | 输入 | 计算特点 | KV Cache 操作 | 瓶颈 |
|------|------|---------|--------------|------|
| **Prefill** | 整个 prompt | 并行处理多个 token | 初始化缓存 | 算力（GPU 计算） |
| **Decode** | 单个新 token | 串行生成（依赖上一个 token） | 读取 + 追加 | 内存带宽（读写缓存） |

**为什么分两个阶段？**

- Prefill：输入已知，可以一次性并行处理，充分利用 GPU 并行能力
- Decode：生成过程，每个 token 依赖前一个，只能串行

---

## 🔌 API 层面的应用

### Prompt Caching 机制

#### 什么是 Prompt Caching？

把 **KV Cache** 存储在服务器端，在多次请求间复用。

```
请求1: System + 长文档 + 问题1
       → 计算 KV Cache → 缓存到服务器（有效期5分钟）

请求2: System + 长文档 + 问题2
       → 检测到前缀相同 → 直接读缓存 → 只计算"问题2"
```

#### 主流 API 的 Prompt Caching 对比

| 提供商 | 功能名称 | 缓存时长 | 最小缓存长度 | 费用折扣 |
|--------|---------|---------|-------------|---------|
| **Anthropic** | Prompt Caching | 5 分钟 | 1024 tokens | 缓存读取 90% off<br>缓存写入 25% 额外费用 |
| **OpenAI** | Prompt Caching<br>(自动) | ~5-10 分钟 | 自动检测 | 缓存命中 50% off |
| **Google** | Context Caching | 1 小时（可调） | 32K tokens | 缓存存储按时间计费 |

### 缓存命中规则

#### Anthropic API 示例

```python
import anthropic

client = anthropic.Anthropic()

# 请求1：建立缓存
response1 = client.messages.create(
    model="claude-sonnet-4",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "你是专业的代码审查助手...",  # 1000 tokens
            "cache_control": {"type": "ephemeral"}  # ← 标记缓存
        }
    ],
    messages=[
        {"role": "user", "content": "审查这段代码：\ndef foo(): pass"}
    ]
)

# 检查缓存使用情况
print(response1.usage)
# {
#   "input_tokens": 1050,
#   "cache_creation_tokens": 1000,  # ← 创建缓存
#   "cache_read_tokens": 0,
#   "output_tokens": 200
# }

# 请求2（5分钟内）：命中缓存
response2 = client.messages.create(
    model="claude-sonnet-4",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "你是专业的代码审查助手...",  # 完全相同！
            "cache_control": {"type": "ephemeral"}
        }
    ],
    messages=[
        {"role": "user", "content": "审查这段代码：\ndef bar(): return 1"}
    ]
)

print(response2.usage)
# {
#   "input_tokens": 55,
#   "cache_creation_tokens": 0,
#   "cache_read_tokens": 1000,  # ← 命中缓存！
#   "output_tokens": 180
# }
```

#### 缓存命中的条件

✅ **必须满足的条件**：

1. **前缀完全匹配**（逐字节比对）
   ```python
   # ✅ 会命中
   prefix1 = "请分析这段代码\n\n" + long_code
   prefix2 = "请分析这段代码\n\n" + long_code  # 完全相同
   
   # ❌ 不会命中
   prefix1 = "请分析这段代码\n\n" + long_code
   prefix2 = "请分析这段代码\n" + long_code  # 少了一个\n
   ```

2. **在缓存时效内**（通常 5 分钟）

3. **模型和参数相同**
   - 同一个模型（如 `claude-sonnet-4`）
   - `temperature` 等采样参数可以不同

4. **缓存边界标记正确**
   ```python
   # Anthropic 需要显式标记
   "cache_control": {"type": "ephemeral"}
   
   # OpenAI 自动检测（但不保证）
   ```

❌ **会破坏缓存的操作**：

```python
# 1. 在缓存部分前插入动态内容
messages = [
    {"role": "system", "content": f"时间：{time.now()}"},  # ← 每次都变
    {"role": "system", "content": long_prompt}  # ← 无法缓存
]

# 2. 修改缓存部分的任何字符
system_prompt = "你是助手" + random_emoji  # ← 每次不同

# 3. 超过缓存时效
# 第一次请求后，等待 6 分钟再请求 → 缓存失效
```

### 成本优化最佳实践

#### 1. 固定内容前置

```python
# ❌ 差：动态内容在前
messages = [
    {"role": "user", "content": f"时间：{time.now()}"},  # 每次变
    {"role": "user", "content": long_document}  # 无法缓存
]

# ✅ 好：固定内容在前
messages = [
    {"role": "system", "content": system_rules,  # 固定
     "cache_control": {"type": "ephemeral"}},
    {"role": "user", "content": long_document,  # 固定
     "cache_control": {"type": "ephemeral"}},
    {"role": "user", "content": f"时间：{time.now()}\n问题：..."}  # 动态
]
```

#### 2. 批量请求间隔控制

```python
# 批量处理时，控制节奏
documents = [doc1, doc2, doc3, ...]  # 100个文档

for doc in documents:
    response = client.messages.create(
        system=[{"type": "text", "text": fixed_rules,  # 每次都缓存
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": f"分析：{doc}"}]
    )
    
    # 不要让任务间隔超过缓存时效
    time.sleep(1)  # ✅ 1秒（远小于5分钟）
    # time.sleep(400)  # ❌ 超过5分钟，缓存失效
```

#### 3. 监控缓存命中率

```python
def track_cache_efficiency(responses):
    """统计缓存效率"""
    total_input = 0
    cache_hits = 0
    cache_creation = 0
    
    for resp in responses:
        usage = resp.usage
        total_input += usage.input_tokens
        cache_hits += usage.cache_read_tokens
        cache_creation += usage.cache_creation_tokens
    
    hit_rate = cache_hits / (total_input + cache_creation) * 100
    
    print(f"缓存命中率: {hit_rate:.1f}%")
    print(f"总输入: {total_input:,} tokens")
    print(f"缓存命中: {cache_hits:,} tokens")
    print(f"缓存创建: {cache_creation:,} tokens")
    
    # 计算成本节省
    # 假设：输入 $3/1M，缓存读取 $0.3/1M（90% off）
    original_cost = (total_input + cache_creation) * 3 / 1_000_000
    actual_cost = (total_input * 3 + cache_creation * 3.75 + cache_hits * 0.3) / 1_000_000
    savings = (1 - actual_cost / original_cost) * 100
    
    print(f"成本节省: {savings:.1f}%")
```

#### 4. 多轮对话优化

```python
# ✅ 推荐：保持对话历史，利用缓存
conversation = [
    {"role": "system", "content": system_prompt,
     "cache_control": {"type": "ephemeral"}},
]

for user_input in user_inputs:
    conversation.append({"role": "user", "content": user_input})
    
    response = client.messages.create(
        model="claude-sonnet-4",
        messages=conversation  # 整个历史会被缓存
    )
    
    conversation.append({"role": "assistant", "content": response.content})

# ❌ 避免：每次都重新开始（浪费缓存）
for user_input in user_inputs:
    response = client.messages.create(
        messages=[{"role": "user", "content": user_input}]
    )
```

#### 5. 实际案例：文档批量分析

```python
# 场景：分析100份合同，每份10页
fixed_rules = """
你是专业的法律文档分析师...
【5000 字审查规则】
"""

total_cost_without_cache = 100 * (5000 + 10000) * 3 / 1_000_000 = $4.50
total_cost_with_cache = (
    5000 * 3.75 / 1_000_000  # 第一次创建缓存
    + 99 * 5000 * 0.3 / 1_000_000  # 99次命中缓存
    + 100 * 10000 * 3 / 1_000_000  # 100次文档输入
) = $0.01875 + $0.1485 + $3.00 = $3.17

节省：(4.50 - 3.17) / 4.50 = 29.6%
```

---

## ❓ 常见问题

### Q1: KV Cache 越大越好吗？

**不是**。KV Cache 占用显存，会影响：

```
显存分配：
├─ 模型参数（固定）：10GB
├─ KV Cache（动态增长）：0 → 1GB → 2GB → ...
└─ 可用于批处理的显存：剩余

KV Cache 过大 → 能并行处理的请求数减少 → 吞吐量下降
```

**实践建议**：
- **长对话场景**：允许大 KV Cache（如 32K context）
- **高吞吐场景**：限制 KV Cache 大小（如 4K context），服务更多并发请求

### Q2: 为什么长对话会越来越慢？

```
对话长度：100 tokens  → KV Cache: 0.5GB  → 速度: 50 tokens/s
对话长度：1000 tokens → KV Cache: 5GB    → 速度: 30 tokens/s  ← 变慢
对话长度：4000 tokens → KV Cache: 20GB   → OOM（显存不足）
```

**原因**：
1. KV Cache 随序列长度线性增长
2. Attention 计算复杂度是 O(N²)（N 是序列长度）
3. 内存带宽成为瓶颈（从 HBM 读取大量 KV）

**解决方案**（Week 5-6 深入学习）：
- **PagedAttention**（vLLM）：像操作系统分页管理 KV Cache
- **Sliding Window Attention**：只保留最近的 N 个 token
- **稀疏 Attention**：不是所有 token 都参与 Attention

### Q3: Prompt Caching 和 KV Cache 是同一个东西吗？

**本质相同**，但应用层面不同：

| 概念 | 层面 | 范围 | 生命周期 |
|------|------|------|---------|
| **KV Cache** | 推理引擎内部 | 单次请求内 | 请求结束即释放 |
| **Prompt Caching** | API 服务层 | 跨请求共享 | 5-60 分钟（可配置） |

```
单次请求内（KV Cache）:
  Prefill → 缓存 → Decode 逐步追加 → 请求结束释放

跨请求（Prompt Caching）:
  请求1 → 缓存到服务器
  请求2 → 从服务器加载 → 继续使用
```

### Q4: 为什么 API 要额外收"缓存写入费"？

```
成本分析：
1. 计算成本：生成 KV Cache 需要 GPU 计算
2. 存储成本：缓存存储在高速内存（GPU HBM/系统内存）
3. 管理成本：缓存查找、匹配、过期管理

Anthropic 的定价：
- 缓存创建：1.25x 输入价格（多收25%）← 覆盖计算+存储
- 缓存读取：0.1x 输入价格（便宜90%）  ← 只需内存读取
```

**是否划算？**

```python
# 假设：1000 tokens 的 system prompt，使用 10 次

无缓存：
  成本 = 10 × 1000 × $3/1M = $0.03

有缓存：
  成本 = 1 × 1000 × $3.75/1M   # 第1次创建
       + 9 × 1000 × $0.3/1M    # 9次读取
       = $0.00375 + $0.0027
       = $0.00645
  
节省：(0.03 - 0.00645) / 0.03 = 78.5% ✅

# 结论：使用2次以上就回本
```

### Q5: 如何判断我的应用适合用 Prompt Caching？

**适合的场景**：

✅ 有大量固定前缀：
- System prompt 很长（>1000 tokens）
- 文档批量处理（相同规则）
- 多轮对话（历史上下文复用）
- Few-shot examples（固定的示例）

✅ 请求频繁：
- 每5分钟内有多次请求
- 用户交互密集

**不适合的场景**：

❌ 每次请求内容都完全不同  
❌ 请求间隔 > 缓存时效  
❌ 输入很短（<500 tokens，缓存收益小）

---

## 📖 延伸阅读

### 推荐资源

1. **官方文档**
   - [Anthropic Prompt Caching 文档](https://docs.anthropic.com/claude/docs/prompt-caching)
   - [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching)

2. **技术博客**
   - [vLLM PagedAttention 原理](https://vllm.ai/) - Week 6 会学到
   - [Hugging Face: KV Cache 实现解析](https://huggingface.co/docs/transformers/main/en/llm_optims)

3. **论文**
   - "Attention is All You Need" (2017) - Transformer 原论文
   - "FlashAttention" (2022) - Attention 计算优化
   - "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023)

### 相关学习指南（本仓库）

- `LLM_显存计算学习指南.md` - 深入理解 KV Cache 显存占用
- `LLM_推理速度学习指南.md` - 推理优化全景
- `AI_Infra工程师学习路线.md` - Week 5-6 会深入 KV Cache 优化

---

## ✅ 学习 Checklist

### 现在就能理解的（无需 Attention 基础）

```
□ 理解自回归生成：一次生成一个词
□ 理解 KV Cache 的动机：避免重复计算
□ 理解 Prefill 和 Decode 两阶段
□ 会计算 KV Cache 的计算量节省
□ 会计算 KV Cache 的显存占用
□ 理解 Prompt Caching 的成本优化逻辑
□ 会在 API 调用中利用缓存（固定前缀前置）
□ 会监控缓存命中率
```

### 学完 Week 3-4 Attention 后回来深化

```
□ 理解 Q @ K.T 的数学含义
□ 理解为什么是 K 和 V 被缓存（不是 Q）
□ 理解 Multi-Head Attention 中的 KV Cache
□ 理解 Causal Masking 如何影响缓存
□ 实现一个带 KV Cache 的 Attention 层
□ 可视化 KV Cache 的增长过程
□ 理解 FlashAttention 对 KV Cache 的优化
```

### 学完 Week 5-6 推理优化后深化

```
□ 使用 Hugging Face transformers 的 use_cache 参数
□ 理解 vLLM 的 PagedAttention 如何管理 KV Cache
□ 测量 KV Cache 对推理速度的影响
□ 实现 Prefix Caching（共享 KV Cache）
□ 理解 Continuous Batching 如何利用 KV Cache
□ 优化长对话场景的显存占用
□ 实现 KV Cache 的量化（INT8/INT4）
□ 测试不同 max_length 对吞吐量的影响
```

---

**学习建议**：

1. **现在**：理解概念，会用 API 优化成本
2. **Week 3-4**：回来补充 Attention 相关数学原理
3. **Week 5-6**：实践推理优化，测量性能指标

祝学习顺利！有问题随时查阅本指南。🚀
