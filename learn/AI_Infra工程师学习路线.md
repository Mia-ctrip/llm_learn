# 🎯 AI Infra工程师的LLM学习路线

> 面向训练/推理平台开发、运维、技术支持和Agent开发

---

## 🎭 角色定位

**你是**: AI Infra工程师
**你的工作**: 
- 开发运维训练/推理平台
- 技术支持和答疑
- 性能监控和优化
- Agent应用开发

**你需要的知识**:
```
                    你需要深入的 ↓
            ┌────────────────────────────┐
            │    LLM应用层 (Agent)        │
            │  - Prompt Engineering      │
            │  - RAG / Function Calling  │
            │  - Agent框架               │
            └────────────────────────────┘
                         ↑
            ┌────────────────────────────┐
            │    LLM架构层                │
            │  - Transformer原理         │ ← 必须深入理解
            │  - 注意力机制               │
            │  - 位置编码                 │
            └────────────────────────────┘
                         ↑
            ┌────────────────────────────┐
            │    框架和工具层              │
            │  - PyTorch基础              │ ← 需要理解
            │  - Hugging Face            │
            │  - vLLM / TGI              │
            └────────────────────────────┘
                         ↑
            ┌────────────────────────────┐
            │    GPU底层 (基座)           │
            │  - GPU如何计算              │ ← 必须深入理解
            │  - 显存管理                 │
            │  - 性能指标                 │
            │  - CUDA基础                 │
            └────────────────────────────┘

            快速略过 ↓ (只需了解概念)
            ┌────────────────────────────┐
            │    传统ML/NLP               │
            │  - 词嵌入 (知道是什么)       │
            │  - RNN/LSTM (知道痛点)      │
            │  - 传统NLP (跳过)           │
            └────────────────────────────┘
```

---

## 🗺️ 重新设计的学习路线

### 总时长: 6-8周

```
Week 1-2: GPU基座层 (夯实底层基础)
  └─ GPU如何运行神经网络
  └─ 显存管理和监控
  └─ PyTorch基础和CUDA

Week 3-4: Transformer架构 (LLM核心)
  └─ 注意力机制
  └─ Transformer完整架构
  └─ 为什么Transformer替代了RNN

Week 5-6: LLM实战 (推理和部署)
  └─ Hugging Face生态
  └─ 模型加载和推理
  └─ 性能优化 (量化、批处理、KV Cache)

Week 7-8: Agent开发 (应用层)
  └─ Prompt Engineering
  └─ RAG / Function Calling
  └─ Agent框架 (LangChain/AutoGen)
```

---

## 📅 详细学习计划

## Week 1-2: GPU基座层 🔥

**为什么先学这个？**
- 你已经理解了神经网络的数学原理
- 现在需要理解它如何在GPU上运行
- 这是理解性能优化的基础

### Week 1: GPU计算原理

#### Day 1-2: GPU vs CPU

**核心问题**: 神经网络的计算为什么用GPU而不是CPU？

```python
# CPU: 顺序执行
for i in range(1000000):
    result[i] = a[i] * b[i]  # 一次算一个

# GPU: 并行执行
result = a * b  # 同时算1000000个!
```

**学习任务**:
```
□ 理解GPU的并行计算架构
  - CUDA核心的概念
  - 为什么矩阵乘法适合GPU
  
□ 实验: 对比CPU vs GPU速度
  import torch
  
  # CPU
  a = torch.randn(10000, 10000)
  b = torch.randn(10000, 10000)
  %time c = torch.mm(a, b)  # 慢
  
  # GPU
  a = a.cuda()
  b = b.cuda()
  %time c = torch.mm(a, b)  # 快!

□ 阅读: 
  - NVIDIA GPU架构入门 (官方文档)
  - PyTorch的CUDA教程
```

#### Day 3-4: 显存管理

**核心问题**: 模型在GPU上如何使用显存？

```python
# 显存占用的组成部分
显存 = 模型参数 + 激活值 + 梯度 + 优化器状态 + KV Cache

# 你已经理解了参数和KV Cache (从你的LLM显存计算项目)
# 现在理解激活值和梯度
```

**学习任务**:
```
□ 动手: 监控显存使用
  import torch
  
  # 查看显存
  torch.cuda.memory_allocated()  # 已分配
  torch.cuda.memory_reserved()   # 保留的
  torch.cuda.max_memory_allocated()  # 峰值
  
  # 清空缓存
  torch.cuda.empty_cache()

□ 实验: 观察前向/反向传播的显存变化
  model = YourModel().cuda()
  
  # 前向传播
  print("Before forward:", torch.cuda.memory_allocated())
  output = model(input)
  print("After forward:", torch.cuda.memory_allocated())  # 增加了激活值
  
  # 反向传播
  loss.backward()
  print("After backward:", torch.cuda.memory_allocated())  # 增加了梯度

□ 理解: 为什么训练比推理需要更多显存？
  训练 = 参数 + 激活值 + 梯度 + 优化器状态
  推理 = 参数 + 激活值 (+ KV Cache)
```

#### Day 5-7: PyTorch基础

**你不需要**: 从头学PyTorch的所有功能
**你需要**: 理解PyTorch如何操作GPU和显存

**学习任务**:
```
□ 核心API (2天)
  # Tensor操作
  x = torch.tensor([1, 2, 3])
  x = x.cuda()  # 移到GPU
  x = x.cpu()   # 移回CPU
  
  # 模型操作
  model.cuda()  # 整个模型移到GPU
  model.eval()  # 推理模式 (不计算梯度)
  model.train() # 训练模式
  
  with torch.no_grad():  # 推理时节省显存
      output = model(input)

□ 实践: 改写你的MLP (1天)
  # 把你的 model_train_v2.py 改成PyTorch版本
  import torch
  import torch.nn as nn
  
  class NeuralNetwork(nn.Module):
      def __init__(self, layer_sizes):
          super().__init__()
          layers = []
          for i in range(len(layer_sizes)-1):
              layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
              if i < len(layer_sizes)-2:
                  layers.append(nn.Sigmoid())
          self.network = nn.Sequential(*layers)
      
      def forward(self, x):
          return self.network(x)
  
  # 训练
  model = NeuralNetwork([2, 4, 3, 1]).cuda()  # 移到GPU!
  optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
  
  for epoch in range(1000):
      output = model(X.cuda())  # 数据也要移到GPU
      loss = criterion(output, y.cuda())
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

□ 对比: NumPy实现 vs PyTorch实现
  - 速度差异 (GPU加速)
  - 代码简洁度
  - 自动求导 vs 手动反向传播
```

### Week 2: 性能监控和优化

#### Day 1-3: 性能指标

**核心问题**: 如何判断GPU利用率和性能？

**学习任务**:
```
□ nvidia-smi深度解析 (你应该已经熟悉，但要深入)
  watch -n 1 nvidia-smi
  
  关键指标:
  - GPU-Util: GPU计算核心利用率 (应该>80%)
  - Memory-Usage: 显存占用 (应该60-90%)
  - Temperature: 温度 (应该<85°C)
  - Power: 功耗 (接近TDP是好的)

□ nvitop / nvidia-smi dmon
  # 更详细的监控工具
  pip install nvitop
  nvitop

□ PyTorch Profiler
  from torch.profiler import profile, ProfilerActivity
  
  with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
      model(input)
  
  print(prof.key_averages().table())
  # 看到每层的时间和显存占用

□ 实践: 诊断性能瓶颈
  场景1: GPU-Util只有30%
    → 可能原因: 数据加载慢 (CPU瓶颈)
    → 解决: 增加DataLoader的num_workers
  
  场景2: 显存占用很低但训练慢
    → 可能原因: batch_size太小
    → 解决: 增大batch_size
  
  场景3: 显存OOM
    → 解决: 梯度累积、混合精度、模型并行
```

#### Day 4-7: 实战项目

**项目: GPU性能监控工具**

```python
# gpu_monitor.py
"""
训练/推理平台的GPU监控工具

功能:
1. 实时监控GPU利用率和显存
2. 记录性能日志
3. 检测异常 (OOM、低利用率等)
4. 生成性能报告
"""

import torch
import time
import nvidia_smi

class GPUMonitor:
    def __init__(self):
        nvidia_smi.nvmlInit()
        self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    
    def get_metrics(self):
        """获取GPU指标"""
        # 显存
        mem_info = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
        mem_used_gb = mem_info.used / 1024**3
        mem_total_gb = mem_info.total / 1024**3
        mem_util = (mem_info.used / mem_info.total) * 100
        
        # 利用率
        util = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle)
        gpu_util = util.gpu
        
        # 温度
        temp = nvidia_smi.nvmlDeviceGetTemperature(
            self.handle, nvidia_smi.NVML_TEMPERATURE_GPU
        )
        
        # 功耗
        power = nvidia_smi.nvmlDeviceGetPowerUsage(self.handle) / 1000  # W
        
        return {
            'mem_used_gb': mem_used_gb,
            'mem_total_gb': mem_total_gb,
            'mem_util': mem_util,
            'gpu_util': gpu_util,
            'temperature': temp,
            'power_w': power
        }
    
    def detect_issues(self, metrics):
        """检测性能问题"""
        issues = []
        
        if metrics['gpu_util'] < 30:
            issues.append("警告: GPU利用率过低 (可能CPU瓶颈)")
        
        if metrics['mem_util'] > 95:
            issues.append("警告: 显存接近上限")
        
        if metrics['temperature'] > 85:
            issues.append("警告: GPU温度过高")
        
        return issues
    
    def monitor_training(self, model, dataloader, epochs=1):
        """监控训练过程"""
        log = []
        
        for epoch in range(epochs):
            for batch in dataloader:
                start_time = time.time()
                
                # 训练步骤 (简化)
                # output = model(batch)
                # loss.backward()
                
                # 记录指标
                metrics = self.get_metrics()
                metrics['timestamp'] = time.time()
                metrics['step_time'] = time.time() - start_time
                
                log.append(metrics)
                
                # 检测问题
                issues = self.detect_issues(metrics)
                if issues:
                    for issue in issues:
                        print(f"[{time.strftime('%H:%M:%S')}] {issue}")
        
        return log
    
    def generate_report(self, log):
        """生成性能报告"""
        import numpy as np
        
        gpu_utils = [m['gpu_util'] for m in log]
        mem_utils = [m['mem_util'] for m in log]
        
        print("=" * 50)
        print("GPU性能报告")
        print("=" * 50)
        print(f"平均GPU利用率: {np.mean(gpu_utils):.1f}%")
        print(f"平均显存利用率: {np.mean(mem_utils):.1f}%")
        print(f"峰值显存: {np.max([m['mem_used_gb'] for m in log]):.1f}GB")
        print(f"平均温度: {np.mean([m['temperature'] for m in log]):.1f}°C")
        print(f"平均功耗: {np.mean([m['power_w'] for m in log]):.0f}W")


# 使用示例
monitor = GPUMonitor()

# 实时监控
while True:
    metrics = monitor.get_metrics()
    print(f"GPU: {metrics['gpu_util']:>3}% | "
          f"Mem: {metrics['mem_used_gb']:.1f}/{metrics['mem_total_gb']:.1f}GB | "
          f"Temp: {metrics['temperature']}°C")
    time.sleep(1)
```

**这个项目的价值**:
- 直接用于你的推理平台监控
- 理解GPU指标的实际意义
- 为技术支持提供诊断工具

---

## Week 3-4: Transformer架构 🔥

**为什么这是重点？**
- 所有现代LLM都基于Transformer
- 理解Transformer = 理解LLM的80%
- 不需要深入RNN/LSTM，它们已经被替代了

### Week 3: 注意力机制

#### Day 1-3: 从问题到Attention

**为什么需要Attention？**

```python
# RNN的问题 (你只需要知道，不需要深入实现)
句子: "The cat sat on the mat"

RNN处理:
"The" → hidden1
"cat" → hidden2 (记住"The")
"sat" → hidden3 (记住"The cat"，但"The"的信息已经衰减了!)
...
"mat" → hidden6 (几乎忘记了"cat"是什么)

问题: 信息会随着距离衰减 (长期依赖问题)

# Attention的解决方案
Attention让模型在处理"mat"时，可以直接看回"cat"!

"mat" can attend to:
  "The"  (权重 0.1)
  "cat"  (权重 0.8) ← 重点关注!
  "sat"  (权重 0.1)
  "on"   (权重 0.0)
  "the"  (权重 0.0)
```

**学习任务**:
```
□ Day 1: 理解Attention直觉
  - 阅读: "Attention is All You Need" 论文的摘要和图1
  - 阅读: Jay Alammar的 "Illustrated Transformer"
  - 理解: Query, Key, Value的类比
    - Query: 我想找什么？
    - Key: 这是什么？
    - Value: 实际的内容

□ Day 2-3: 实现Self-Attention
  import torch
  import torch.nn.functional as F
  
  def self_attention(X, d_k):
      """
      最简单的Self-Attention实现
      
      X: (seq_len, d_model) 输入序列
      """
      # 生成Q, K, V (简化版，实际需要学习的矩阵)
      Q = X  # (seq_len, d_model)
      K = X
      V = X
      
      # 计算注意力分数
      scores = torch.mm(Q, K.T) / (d_k ** 0.5)  # (seq_len, seq_len)
      
      # Softmax归一化
      attention_weights = F.softmax(scores, dim=-1)
      
      # 加权求和
      output = torch.mm(attention_weights, V)  # (seq_len, d_model)
      
      return output, attention_weights
  
  # 测试
  seq_len = 5
  d_model = 8
  X = torch.randn(seq_len, d_model)
  
  output, weights = self_attention(X, d_model)
  
  print("注意力权重矩阵:")
  print(weights)
  # 每一行是一个词对所有词的注意力分布
```

#### Day 4-7: Multi-Head Attention

**为什么需要多头？**

```python
# 单头Attention的局限
"The cat sat on the mat"

单头只能关注一种关系:
"sat" → "cat" (主语)

# 多头Attention
Head 1: "sat" → "cat" (找主语)
Head 2: "sat" → "on" (找介词)
Head 3: "sat" → "mat" (找宾语)

多个头学习不同的关系模式!
```

**学习任务**:
```
□ 实现Multi-Head Attention
  class MultiHeadAttention(torch.nn.Module):
      def __init__(self, d_model, num_heads):
          super().__init__()
          self.num_heads = num_heads
          self.d_k = d_model // num_heads
          
          # 投影矩阵
          self.W_q = torch.nn.Linear(d_model, d_model)
          self.W_k = torch.nn.Linear(d_model, d_model)
          self.W_v = torch.nn.Linear(d_model, d_model)
          self.W_o = torch.nn.Linear(d_model, d_model)
      
      def forward(self, X):
          batch_size, seq_len, d_model = X.shape
          
          # 投影到Q, K, V
          Q = self.W_q(X)  # (batch, seq_len, d_model)
          K = self.W_k(X)
          V = self.W_v(X)
          
          # 拆分成多头
          Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k)
          K = K.view(batch_size, seq_len, self.num_heads, self.d_k)
          V = V.view(batch_size, seq_len, self.num_heads, self.d_k)
          
          # 转置: (batch, num_heads, seq_len, d_k)
          Q = Q.transpose(1, 2)
          K = K.transpose(1, 2)
          V = V.transpose(1, 2)
          
          # 计算注意力
          scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
          attn_weights = F.softmax(scores, dim=-1)
          output = torch.matmul(attn_weights, V)
          
          # 合并多头
          output = output.transpose(1, 2).contiguous()
          output = output.view(batch_size, seq_len, d_model)
          
          # 输出投影
          output = self.W_o(output)
          
          return output

□ 可视化注意力权重
  - 看不同head关注的模式
  - 理解为什么需要多头
```

### Week 4: 完整Transformer

#### Day 1-3: Transformer Block

**学习任务**:
```
□ 实现完整的Transformer Block
  class TransformerBlock(torch.nn.Module):
      def __init__(self, d_model, num_heads, d_ff):
          super().__init__()
          
          # Multi-Head Attention
          self.attention = MultiHeadAttention(d_model, num_heads)
          
          # Feed-Forward Network
          self.ffn = torch.nn.Sequential(
              torch.nn.Linear(d_model, d_ff),
              torch.nn.ReLU(),
              torch.nn.Linear(d_ff, d_model)
          )
          
          # Layer Normalization
          self.ln1 = torch.nn.LayerNorm(d_model)
          self.ln2 = torch.nn.LayerNorm(d_model)
      
      def forward(self, x):
          # Self-Attention + Residual + Norm
          attn_out = self.attention(x)
          x = self.ln1(x + attn_out)  # 残差连接!
          
          # Feed-Forward + Residual + Norm
          ffn_out = self.ffn(x)
          x = self.ln2(x + ffn_out)
          
          return x

□ 理解每个组件的作用
  - Multi-Head Attention: 捕捉词之间的关系
  - Feed-Forward: 对每个位置独立变换
  - Residual Connection: 帮助梯度传播
  - Layer Norm: 稳定训练
```

#### Day 4-5: 位置编码

**为什么需要？**

```python
# Attention本身没有位置信息!
"cat sat" 和 "sat cat" 的Attention结果一样

# 位置编码
def positional_encoding(seq_len, d_model):
    """
    为每个位置生成唯一的编码
    """
    position = torch.arange(seq_len).unsqueeze(1)  # (seq_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * 
                         -(math.log(10000.0) / d_model))
    
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
    pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
    
    return pe

# 添加到输入
X = token_embeddings + positional_encoding(seq_len, d_model)
```

#### Day 6-7: GPT架构

**从Transformer到GPT**:

```python
class GPT(torch.nn.Module):
    """
    简化的GPT架构 (只有Decoder)
    """
    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        super().__init__()
        
        # Token Embedding + Position Embedding
        self.token_embedding = torch.nn.Embedding(vocab_size, d_model)
        self.position_embedding = torch.nn.Embedding(1024, d_model)  # max_len=1024
        
        # Transformer Blocks
        self.blocks = torch.nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff=4*d_model)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.ln_f = torch.nn.LayerNorm(d_model)
        self.lm_head = torch.nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)  # (batch, seq_len, d_model)
        pos_ids = torch.arange(seq_len, device=input_ids.device)
        pos_emb = self.position_embedding(pos_ids)
        
        x = token_emb + pos_emb
        
        # Transformer Blocks
        for block in self.blocks:
            x = block(x)
        
        # 输出
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        
        return logits


# 使用
vocab_size = 50000
d_model = 768
num_heads = 12
num_layers = 12

model = GPT(vocab_size, d_model, num_heads, num_layers).cuda()

# 推理
input_ids = torch.tensor([[1, 2, 3, 4]]).cuda()  # "The cat sat"
logits = model(input_ids)  # 预测下一个词的概率分布

print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
print(f"显存占用: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
```

**关键理解**:
- GPT = 堆叠的Transformer Decoder
- 自回归生成: 一次预测一个词
- Causal Masking: 只能看到前面的词

---

## Week 5-6: LLM实战

**从玩具模型到真实LLM**

### Week 5: Hugging Face生态

#### Day 1-3: 模型加载和推理

**学习任务**:
```
□ Hugging Face基础
  from transformers import AutoTokenizer, AutoModelForCausalLM
  import torch
  
  # 加载模型
  model_name = "Qwen/Qwen2.5-7B-Instruct"
  
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForCausalLM.from_pretrained(
      model_name,
      torch_dtype=torch.float16,  # 半精度
      device_map="auto"  # 自动分配到GPU
  )
  
  # 推理
  prompt = "What is machine learning?"
  inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
  
  outputs = model.generate(
      inputs.input_ids,
      max_new_tokens=100,
      temperature=0.7,
      top_p=0.9
  )
  
  print(tokenizer.decode(outputs[0]))

□ 监控显存和性能
  # 结合Week 1的知识
  import torch
  
  print(f"模型参数: {model.num_parameters() / 1e9:.1f}B")
  print(f"显存占用: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
  
  # 推理速度
  import time
  start = time.time()
  outputs = model.generate(inputs.input_ids, max_new_tokens=100)
  elapsed = time.time() - start
  print(f"生成速度: {100 / elapsed:.1f} tokens/s")

□ 理解tokenizer
  text = "Hello, world!"
  tokens = tokenizer.tokenize(text)
  token_ids = tokenizer.encode(text)
  
  print(f"文本: {text}")
  print(f"Tokens: {tokens}")
  print(f"Token IDs: {token_ids}")
```

#### Day 4-7: 性能优化

**学习任务**:
```
□ Day 4: 量化 (INT8 / INT4)
  from transformers import AutoModelForCausalLM, BitsAndBytesConfig
  
  # INT8量化
  model = AutoModelForCausalLM.from_pretrained(
      model_name,
      load_in_8bit=True,
      device_map="auto"
  )
  
  print(f"量化后显存: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
  # 应该减少约50%!
  
  # 对比推理速度和质量

□ Day 5: 批处理
  # 单个请求 vs 批量请求
  prompts = ["Question 1", "Question 2", "Question 3"]
  
  # 批量tokenize
  inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
  
  # 批量生成
  outputs = model.generate(inputs.input_ids, max_new_tokens=50)
  
  # 对比吞吐量

□ Day 6-7: KV Cache
  # 理解KV Cache的作用
  # (结合你的显存计算知识)
  
  # 不使用KV Cache (慢)
  outputs = model.generate(inputs.input_ids, use_cache=False)
  
  # 使用KV Cache (快)
  outputs = model.generate(inputs.input_ids, use_cache=True)
  
  # 监控显存变化
  # KV Cache大小 ≈ batch_size × seq_len × layers × hidden_size × 2
```

### Week 6: 推理框架

#### Day 1-3: vLLM

**为什么学vLLM？**
- 生产环境最常用的推理框架
- 你会需要运维和优化vLLM服务

**学习任务**:
```
□ 安装和基础使用
  pip install vllm
  
  from vllm import LLM, SamplingParams
  
  # 初始化
  llm = LLM(
      model="Qwen/Qwen2.5-7B-Instruct",
      tensor_parallel_size=1,  # GPU数量
      gpu_memory_utilization=0.9  # 显存利用率
  )
  
  # 推理
  prompts = ["Hello!", "What is AI?"]
  sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
  
  outputs = llm.generate(prompts, sampling_params)
  
  for output in outputs:
      print(output.outputs[0].text)

□ 性能对比
  # Hugging Face vs vLLM
  # 对比吞吐量和延迟

□ PagedAttention理解
  - vLLM的核心优化
  - KV Cache的内存管理
```

#### Day 4-7: 推理服务搭建

**实战项目: 简单的推理API服务**

```python
# llm_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import torch

app = FastAPI()

# 初始化模型
model = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    gpu_memory_utilization=0.9
)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7

class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    time_seconds: float

@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    import time
    start = time.time()
    
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )
    
    outputs = model.generate([request.prompt], sampling_params)
    generated_text = outputs[0].outputs[0].text
    
    elapsed = time.time() - start
    
    return GenerateResponse(
        text=generated_text,
        tokens_generated=len(outputs[0].outputs[0].token_ids),
        time_seconds=elapsed
    )

@app.get("/health")
def health():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "gpu_memory_gb": torch.cuda.memory_allocated() / 1e9
    }

@app.get("/metrics")
def metrics():
    """性能指标"""
    return {
        "gpu_util": get_gpu_util(),  # 从Week 1的监控工具
        "memory_used_gb": torch.cuda.memory_allocated() / 1e9,
        "requests_processed": request_counter
    }

# 运行: uvicorn llm_server:app --host 0.0.0.0 --port 8000
```

**价值**:
- 直接用于你的推理平台
- 理解推理服务的架构
- 为技术支持提供基础

---

## Week 7-8: Agent开发 🤖

**从LLM到Agent**

### Week 7: Prompt Engineering

#### Day 1-2: Prompt基础

**学习任务**:
```
□ 基本Prompt结构
  # System + User
  messages = [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is machine learning?"}
  ]
  
  # 使用chat template
  prompt = tokenizer.apply_chat_template(messages, tokenize=False)

□ Few-Shot Learning
  prompt = """
  Classify the sentiment:
  
  Text: "I love this!"
  Sentiment: Positive
  
  Text: "This is terrible."
  Sentiment: Negative
  
  Text: "It's okay, I guess."
  Sentiment:
  """

□ Chain-of-Thought
  prompt = """
  Question: Roger has 5 tennis balls. He buys 2 more. How many does he have?
  
  Let's think step by step:
  1. Roger starts with 5 balls
  2. He buys 2 more balls
  3. Total = 5 + 2 = 7 balls
  
  Answer: 7 balls
  
  Question: Sarah has 3 apples. She gives 1 to Tom. How many does she have?
  
  Let's think step by step:
  """
```

#### Day 3-7: Agent框架

**学习LangChain基础**:

```python
# agent_simple.py
from langchain.llms import HuggingFacePipeline
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

# 加载模型
llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    device=0
)

# 加载工具
tools = load_tools(["python_repl", "wikipedia"], llm=llm)

# 初始化Agent
memory = ConversationBufferMemory(memory_key="chat_history")

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# 运行
response = agent.run("What is 25 * 4 + 10?")
print(response)
```

### Week 8: 高级Agent

#### Day 1-3: Function Calling

**学习任务**:
```
□ 理解Function Calling
  tools = [
      {
          "type": "function",
          "function": {
              "name": "get_weather",
              "description": "Get current weather",
              "parameters": {
                  "type": "object",
                  "properties": {
                      "location": {"type": "string"}
                  },
                  "required": ["location"]
              }
          }
      }
  ]
  
  # LLM输出
  # {"name": "get_weather", "arguments": {"location": "Beijing"}}

□ 实现Function Calling Agent
  def get_weather(location):
      # 实际API调用
      return f"Weather in {location}: Sunny, 25°C"
  
  # Agent loop
  while True:
      response = llm.generate(prompt)
      
      if response.is_function_call:
          result = execute_function(response.function_name, response.arguments)
          prompt += f"\\nFunction result: {result}"
      else:
          return response.text
```

#### Day 4-7: RAG (检索增强生成)

**实战项目: 文档问答Agent**

```python
# rag_agent.py
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# 1. 加载文档
loader = TextLoader("your_docs.txt")
documents = loader.load()

# 2. 分割文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
splits = text_splitter.split_documents(documents)

# 3. 创建向量数据库
embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_documents(splits, embeddings)

# 4. 创建RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=your_llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# 5. 问答
question = "What is the GPU utilization target?"
answer = qa_chain.run(question)
print(answer)
```

---

## 📊 学习检查清单

### Week 1-2: GPU基座层
```
□ 理解GPU并行计算原理
□ 会用torch.cuda监控显存
□ 能诊断GPU性能问题
□ 实现了GPU监控工具
□ 把MLP改写成PyTorch版本
```

### Week 3-4: Transformer
```
□ 理解Self-Attention机制
□ 实现了Multi-Head Attention
□ 理解Transformer Block结构
□ 实现了简单的GPT模型
□ 理解为什么Transformer替代RNN
```

### Week 5-6: LLM实战
```
□ 会用Hugging Face加载模型
□ 理解量化的原理和实践
□ 会用vLLM做推理
□ 搭建了简单的推理API服务
□ 能监控和优化推理性能
```

### Week 7-8: Agent
```
□ 掌握Prompt Engineering技巧
□ 会用LangChain框架
□ 理解Function Calling
□ 实现了RAG Agent
□ 能开发简单的Agent应用
```

---

## 🎯 学习资源

### 必读
1. **"Attention is All You Need"** - Transformer原论文
2. **Jay Alammar的Illustrated系列**
   - Illustrated Transformer
   - Illustrated GPT-2
3. **Hugging Face文档** - transformers库
4. **vLLM文档** - 推理优化

### 视频
1. **Andrej Karpathy - "Let's build GPT"**
2. **NVIDIA - GPU架构介绍**

### 实践
1. **你的项目**: 继续完善显存计算和监控工具
2. **Kaggle**: LLM相关竞赛
3. **开源项目**: 贡献到vLLM或LangChain

---

## 💡 关键建议

### 1. 学习重点分配
- 30% GPU底层 (基座)
- 40% Transformer和LLM (核心)
- 30% Agent开发 (应用)
- 0% 传统ML细节 (跳过)

### 2. 不要深入的东西
- ❌ RNN/LSTM的详细数学推导
- ❌ 词嵌入训练算法
- ❌ 传统NLP方法
- ❌ BERT的预训练细节

### 3. 要深入的东西
- ✅ GPU如何运行神经网络
- ✅ Attention机制的数学和实现
- ✅ Transformer的每个组件
- ✅ 显存和性能优化
- ✅ Agent的架构和工具

### 4. 实践优先
- 每学一个概念,立刻写代码验证
- 用你的GPU监控工具观察实际指标
- 把学到的知识应用到推理平台

---

## 🚀 第一周行动计划

```
Day 1-2: GPU并行计算
□ 实验: CPU vs GPU矩阵乘法
□ 阅读: CUDA编程基础
□ 理解: 为什么神经网络适合GPU

Day 3-4: 显存监控
□ 用torch.cuda API监控显存
□ 观察前向/反向传播的显存变化
□ 理解: 训练 vs 推理的显存差异

Day 5-7: PyTorch实践
□ 安装PyTorch + CUDA
□ 改写你的MLP到PyTorch
□ 对比: NumPy vs PyTorch性能
□ 开始GPU监控工具项目
```

---

**准备好开始了吗? 从Day 1开始,一周后回来汇报进度! 🚀**

有问题随时问我,特别是关于:
- GPU性能优化
- 推理平台架构
- Agent开发技巧
