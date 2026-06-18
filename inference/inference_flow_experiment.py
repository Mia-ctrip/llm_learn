"""
LLM 推理全流程实验：Prefill + Decode + KV Cache

目标：用真实模型跑一遍推理，亲手感受 Prefill 和 Decode 两个阶段
前置知识：已理解 GPT 架构（多层 Decoder Block: Causal MHSA + FFN）

实验内容：
  1. 手动实现 Prefill 和 Decode，理解每一步发生了什么
  2. 对比有无 KV Cache 的速度差异
  3. 观察 KV Cache 随序列增长的变化

依赖安装：
  pip install transformers torch

运行：
  python inference_flow_experiment.py
"""

import torch
import time

# ============================================================
# 第零部分：准备工作
# ============================================================

print("=" * 70)
print("LLM 推理全流程实验")
print("=" * 70)

# 使用 GPT-2 small（117M 参数，不需要 GPU 也能跑）
from transformers import GPT2LMHeadModel, GPT2Tokenizer
try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None  # 旧版 transformers 没有这个类


def extract_kv_list(cache):
    """从 KV Cache 对象中提取 (key, value) tensor 列表，兼容所有版本"""
    # 方式1：tuple/list of tuples（旧版 transformers）
    if isinstance(cache, (tuple, list)):
        return [(k, v) for k, v in cache]
    # 方式2：DynamicCache，尝试多种属性名
    for attr_k, attr_v in [('key_cache', 'value_cache'), ('keys', 'values'), ('_key_cache', '_value_cache')]:
        if hasattr(cache, attr_k) and hasattr(cache, attr_v):
            ks = getattr(cache, attr_k)
            vs = getattr(cache, attr_v)
            return list(zip(ks, vs))
    # 方式3：新版 DynamicCache 用 layers 属性
    if hasattr(cache, 'layers'):
        result = []
        for layer in cache.layers:
            result.append((layer.keys, layer.values))
        return result
    # 方式4：支持 __getitem__（按层索引）
    try:
        result = []
        for i in range(len(cache)):
            item = cache[i]
            result.append((item[0], item[1]))
        return result
    except (IndexError, TypeError, KeyError):
        pass
    raise RuntimeError(f"无法从 KV Cache 类型 {type(cache)} 中提取数据")

print("\n[准备] 加载 GPT-2 模型 (117M 参数)...")
model_name = "gpt2"  # 如果下载慢可以换成 "gpt2-medium" (345M)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()  # 推理模式（关闭 Dropout 等训练专用组件）

# 打印模型结构信息
config = model.config
print(f"  d_model (hidden_size): {config.n_embd}")
print(f"  num_layers:            {config.n_layer}")
print(f"  num_heads:             {config.n_head}")
print(f"  d_k:                   {config.n_embd // config.n_head}")
print(f"  vocab_size:            {config.vocab_size}")
print(f"  总参数量:              {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")


# ============================================================
# 第一部分：Prefill 阶段 — 处理输入 prompt
# ============================================================

print("\n" + "=" * 70)
print("第一部分：Prefill 阶段（一次性处理所有输入 token）")
print("=" * 70)

prompt = "The quick brown fox jumps over the lazy dog and then"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
print(f"\n输入文本: \"{prompt}\"")
print(f"Token IDs: {input_ids[0].tolist()}")
print(f"Token 数量: {input_ids.shape[1]}")

# Prefill：一次性把所有 token 送进模型
print("\n[Prefill] 一次性并行处理所有输入 token...")
start = time.perf_counter()
with torch.no_grad():
    # use_cache=True 让模型返回 past_key_values（即 KV Cache）
    outputs = model(input_ids, use_cache=True)
    logits = outputs.logits          # (1, seq_len, vocab_size)
    past_key_values = outputs.past_key_values  # KV Cache！
prefill_time = time.perf_counter() - start

print(f"  Prefill 耗时: {prefill_time*1000:.1f} ms")
print(f"  logits shape: {logits.shape}  ← (batch=1, seq_len={input_ids.shape[1]}, vocab_size={config.vocab_size})")

# 用通用辅助函数提取 KV Cache 信息
kv_pairs = extract_kv_list(past_key_values)
num_kv_layers = len(kv_pairs)
k_shape = kv_pairs[0][0].shape
v_shape = kv_pairs[0][1].shape
kv_cache_size = sum(
    k.numel() * k.element_size() + v.numel() * v.element_size()
    for k, v in kv_pairs
)
print(f"  KV Cache 层数: {num_kv_layers}  ← 每层 Decoder Block 一份")
print(f"  每层 KV Cache shape: K={k_shape}, V={v_shape}")

# 取最后一个 token 的 logits，预测下一个词
next_token_logits = logits[:, -1, :]  # 只用最后一行！
next_token_id = torch.argmax(next_token_logits, dim=-1)
next_token_text = tokenizer.decode(next_token_id)
print(f"\n  预测下一个 token: \"{next_token_text}\" (id={next_token_id.item()})")

# 统计 KV Cache 大小（已在上面计算）
print(f"\n  KV Cache 总大小: {kv_cache_size / 1024 / 1024:.2f} MB")


# ============================================================
# 第 1.5 部分：深入观察 KV Cache 的内部结构
# ============================================================

print("\n" + "=" * 70)
print("第 1.5 部分：深入观察 KV Cache 内部结构")
print("=" * 70)

# 1. KV Cache 的容器类型
print(f"\n[1] KV Cache 容器类型:")
print(f"  type: {type(past_key_values).__name__}")
print(f"  → DynamicCache（新版 transformers ≥4.36）")
print(f"  → 旧版返回的是 tuple of tuples: ((K₀,V₀), (K₁,V₁), ...)")

# 2. 按层索引访问
print(f"\n[2] 按层索引访问（共 {num_kv_layers} 层）:")
for layer_idx in [0, 1, num_kv_layers - 1]:
    k_layer, v_layer = kv_pairs[layer_idx]
    print(f"  层 {layer_idx:>2}: K shape={k_layer.shape}, V shape={v_layer.shape}")
print(f"  ... 省略中间层，结构完全相同")

# 3. K/V tensor 的 4 维结构解析
print(f"\n[3] K/V tensor 的 4 维结构解析:")
k0 = kv_pairs[0][0]  # 第 0 层的 K tensor
print(f"  K₀.shape = {k0.shape}")
print(f"  拆解: (batch={k0.shape[0]}, num_heads={k0.shape[1]}, seq_len={k0.shape[2]}, d_k={k0.shape[3]})")
print(f"  → {k0.shape[1]} 个头的 K 数据打包在一个 tensor 里，不是 12×12=144 组！")
print(f"  → 每层只有 1 个 K tensor 和 1 个 V tensor，头信息嵌在第 2 维度")

# 4. 访问单个头的 K/V
print(f"\n[4] 访问第 0 层、第 0 个头的 K:")
k0_head0 = k0[0, 0, :, :]  # (seq_len, d_k)
print(f"  K[层0, 头0, :, :].shape = {k0_head0.shape}")
print(f"  → ({input_ids.shape[1]} tokens, {k0.shape[3]} 维向量)")
print(f"  → 这就是第 0 层、头 0 给所有 {input_ids.shape[1]} 个 token 计算的 K 向量")

# 5. KV Cache 存储在哪个设备
print(f"\n[5] KV Cache 存储设备:")
print(f"  K₀.device = {k0.device}")
model_device = next(model.parameters()).device
print(f"  模型 device = {model_device}")
print(f"  → KV Cache 和模型在同一个设备上（都是 {model_device}）")
print(f"  → 模型在 CPU → KV Cache 在内存；模型在 GPU → KV Cache 在显存")

# 6. 单头 K 的内容预览
print(f"\n[6] 单头 K 内容预览（第 0 层、头 0、前 2 个 token、前 8 维）:")
print(f"  token 0: {k0[0, 0, 0, :8].tolist()}")
print(f"  token 1: {k0[0, 0, 1, :8].tolist()}")
print(f"  → 每个 token 都有自己独立的 K 向量（64 维，这里只显示前 8 维）")

# 7. KV Cache 的内存占用拆解
print(f"\n[7] KV Cache 内存占用拆解:")
batch, heads, seq_len, d_k = k0.shape
dtype_bytes = k0.element_size()
per_layer = 2 * batch * heads * seq_len * d_k * dtype_bytes  # × 2 因为 K 和 V
print(f"  每层: 2(K+V) × {batch}(batch) × {heads}(heads) × {seq_len}(seq_len) × {d_k}(d_k) × {dtype_bytes}(bytes) = {per_layer/1024:.2f} KB")
print(f"  共 {num_kv_layers} 层: {per_layer * num_kv_layers / 1024 / 1024:.2f} MB")
print(f"  公式: 2 × batch × num_layers × num_heads × seq_len × d_k × bytes_per_element")


# ============================================================
# 第二部分：Decode 阶段 — 逐 token 生成
# ============================================================

print("\n" + "=" * 70)
print("第二部分：Decode 阶段（逐 token 生成，每次只算 1 个新 token）")
print("=" * 70)

eos_id = tokenizer.eos_token_id
max_new_tokens = 20  # 安全上限，演示用不需要太长
print(f"\n停止条件：生成 <EOS>（id={eos_id}）或达到 max_new_tokens={max_new_tokens}")
print(f"  ⚠ 实际部署中 max_new_tokens 可以设置很大（如 2048），这里为了演示简洁")

# --- 方式 A：有 KV Cache（正确做法）---
print("\n[方式 A] 有 KV Cache：每次只算 1 个新 token")

generated_ids = [next_token_id.item()]
current_past = past_key_values  # 复用 Prefill 的 KV Cache

total_decode_time_with_cache = 0
num_decode_steps = 0
for step in range(max_new_tokens):
    new_input = torch.tensor([[generated_ids[-1]]])  # 只输入 1 个新 token！
    
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(new_input, past_key_values=current_past, use_cache=True)
    decode_time = time.perf_counter() - start
    total_decode_time_with_cache += decode_time
    
    logits = outputs.logits[:, -1, :]
    next_id = torch.argmax(logits, dim=-1).item()
    generated_ids.append(next_id)
    current_past = outputs.past_key_values
    num_decode_steps += 1
    
    token_text = tokenizer.decode([next_id])
    full_text = tokenizer.decode(generated_ids)
    print(f"  Step {step+1}: \"{token_text}\" | 耗时 {decode_time*1000:.1f}ms | 序列: \"{full_text}\"")
    
    if next_id == eos_id:
        print(f"  → 生成 <EOS>，停止！")
        break

# KV Cache 增长后的大小
kv_pairs_final = extract_kv_list(current_past)
kv_cache_final = sum(
    k.numel() * k.element_size() + v.numel() * v.element_size()
    for k, v in kv_pairs_final
)

print(f"\n  Decode 总耗时 (有 KV Cache): {total_decode_time_with_cache*1000:.1f} ms")
print(f"  生成 token 数: {num_decode_steps}")
print(f"  平均每个 token: {total_decode_time_with_cache/num_decode_steps*1000:.1f} ms")
print(f"  KV Cache 最终大小: {kv_cache_final / 1024 / 1024:.2f} MB")


# --- 方式 B：无 KV Cache（暴力做法，每次都重新算所有 token）---
print("\n[方式 B] 无 KV Cache：每次都重新算所有 token")

generated_ids_no_cache = [next_token_id.item()]
total_decode_time_no_cache = 0
num_decode_steps_no_cache = 0

for step in range(max_new_tokens):
    # 把 prompt + 已生成的所有 token 一起输入（没有 KV Cache，全部重算）
    full_input = torch.cat([input_ids, torch.tensor([generated_ids_no_cache])], dim=1)
    
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(full_input, use_cache=False)  # 不算 KV Cache
    decode_time = time.perf_counter() - start
    total_decode_time_no_cache += decode_time
    
    logits = outputs.logits[:, -1, :]
    next_id = torch.argmax(logits, dim=-1).item()
    generated_ids_no_cache.append(next_id)
    num_decode_steps_no_cache += 1
    
    token_text = tokenizer.decode([next_id])
    print(f"  Step {step+1}: \"{token_text}\" | 耗时 {decode_time*1000:.1f}ms | 输入 {full_input.shape[1]} tokens")
    
    if next_id == eos_id:
        print(f"  → 生成 <EOS>，停止！")
        break

print(f"\n  Decode 总耗时 (无 KV Cache): {total_decode_time_no_cache*1000:.1f} ms")
print(f"  生成 token 数: {num_decode_steps_no_cache}")
print(f"  平均每个 token: {total_decode_time_no_cache/num_decode_steps_no_cache*1000:.1f} ms")


# ============================================================
# 第三部分：对比总结
# ============================================================

print("\n" + "=" * 70)
print("第三部分：对比总结")
print("=" * 70)

speedup = total_decode_time_no_cache / total_decode_time_with_cache
print(f"""
┌─────────────────────┬──────────────────┬──────────────────┐
│                     │   有 KV Cache     │   无 KV Cache     │
├─────────────────────┼──────────────────┼──────────────────┤
│ Decode 总耗时        │ {total_decode_time_with_cache*1000:>8.1f} ms      │ {total_decode_time_no_cache*1000:>8.1f} ms      │
│ 生成 token 数        │ {num_decode_steps:>8}          │ {num_decode_steps_no_cache:>8}          │
│ 平均每个 token       │ {total_decode_time_with_cache/num_decode_steps*1000:>8.1f} ms      │ {total_decode_time_no_cache/num_decode_steps_no_cache*1000:>8.1f} ms      │
│ 加速比              │ {speedup:>8.1f}x         │                  │
└─────────────────────┴──────────────────┴──────────────────┘

有 KV Cache 比无 KV Cache 快 {speedup:.1f} 倍！

总推理时间:
  Prefill:  {prefill_time*1000:.1f} ms（一次性处理 {input_ids.shape[1]} 个输入 token）
  Decode:   {total_decode_time_with_cache*1000:.1f} ms（逐 token 生成 {num_decode_steps} 个 token）
  总计:     {(prefill_time + total_decode_time_with_cache)*1000:.1f} ms

对应你学过的指标:
  TTFT (首 token 延迟) ≈ Prefill 时间 = {prefill_time*1000:.1f} ms
  TPOT (每个 token 耗时) ≈ Decode 平均时间 = {total_decode_time_with_cache/num_decode_steps*1000:.1f} ms/token
""")


# ============================================================
# 第四部分：KV Cache 随序列长度的增长
# ============================================================

print("=" * 70)
print("第四部分：KV Cache 随序列长度的增长")
print("=" * 70)

# 生成不同长度的序列，观察 KV Cache 增长
test_lengths = [10, 50, 100, 200, 500]
print(f"\n{'序列长度':>10} | {'KV Cache 大小':>15} | {'Decode 耗时':>12}")
print("-" * 50)

for target_len in test_lengths:
    # 构造指定长度的输入（重复 prompt）
    long_input = tokenizer.encode(prompt * (target_len // len(input_ids[0]) + 1), return_tensors="pt")
    long_input = long_input[:, :target_len]
    
    with torch.no_grad():
        out = model(long_input, use_cache=True)
    
    kv_pairs = extract_kv_list(out.past_key_values)
    kv_size = sum(
        k.numel() * k.element_size() + v.numel() * v.element_size()
        for k, v in kv_pairs
    )
    
    # 测一次 Decode
    new_input = torch.tensor([[0]])
    start = time.perf_counter()
    with torch.no_grad():
        model(new_input, past_key_values=out.past_key_values, use_cache=True)
    dt = time.perf_counter() - start
    
    print(f"{target_len:>10} | {kv_size/1024/1024:>10.2f} MB     | {dt*1000:>8.1f} ms")

print(f"""
观察：
  1. KV Cache 大小与序列长度成正比（你笔记里的公式: 2 × batch × seq_len × d_model × layers × bytes）
  2. Decode 每步耗时随序列增长而增加（需要读取更大的 KV Cache）
  3. 这就是为什么长对话会越来越慢！

验证公式：
  理论 KV Cache = 2 × 1(batch) × seq_len × {config.n_embd}(d_model) × {config.n_layer}(layers) × 4(bytes, FP32)
  seq_len=100 时: 2 × 1 × 100 × {config.n_embd} × {config.n_layer} × 4 = {2 * 1 * 100 * config.n_embd * config.n_layer * 4 / 1024 / 1024:.2f} MB
  （注：GPT-2 用的是缓存的 key 维度 n_embd/n_head * n_head = n_embd）
""")


# ============================================================
# 第五部分：流程总结图
# ============================================================

print("=" * 70)
print("完整推理流程（串起你学过的所有概念）")
print("=" * 70)

print("""
用户输入: "The quick brown fox"
         │
    ┌────▼─────────────────────────────────────────┐
    │ 1. Tokenize                                   │
    │    "The quick brown fox" → [464, 1820, 3877, 14142] │
    └────┬─────────────────────────────────────────┘
         │
    ┌────▼─────────────────────────────────────────┐
    │ 2. Embedding 查表                              │
    │    [E[464], E[1820], E[3877], E[14142]]       │
    │    → (4 × 768) 矩阵                           │
    └────┬─────────────────────────────────────────┘
         │
    ┌────▼─────────────────────────────────────────┐
    │ 3. Prefill（一次性并行处理所有 4 个 token）     │
    │                                              │
    │    对每个 token 同时计算:                       │
    │    Block 1: RMSNorm → Causal MHSA → +残差     │
    │            → RMSNorm → FFN → +残差             │
    │    Block 2: ...                               │
    │    ... × 12 层                                 │
    │                                              │
    │    ⚡ 同时：缓存每层每个 token 的 K/V → KV Cache │
    │                                              │
    │    取最后一行 logits[0, -1, :]                  │
    │    → argmax → 第一个生成的 token: " jumps"     │
    │                                              │
    │    耗时：较长（但只跑一次）                      │
    │    对应指标：TTFT（首 token 延迟）               │
    └────┬─────────────────────────────────────────┘
         │
    ┌────▼─────────────────────────────────────────┐
    │ 4. Decode（逐 token 串行生成）                  │
    │                                              │
    │    Step 1:                                    │
    │      新 token " jumps" → Embedding (1 × 768)  │
    │      → 只算这 1 个 token 的 Q                  │
    │      → 和 KV Cache 里所有 K 做点积              │
    │      → softmax → 加权所有 V                    │
    │      → 追加新 K/V 到缓存                        │
    │      → Linear → Softmax → " over"             │
    │                                              │
    │    Step 2: " over" → 同样的流程 → " the"       │
    │    Step 3: " the" → ... → " fence"            │
    │    ...                                       │
    │    直到生成 EOS 或达到 max_length               │
    │                                              │
    │    耗时：每步很快，但要跑 N 次                    │
    │    对应指标：TPOT（每个 token 耗时）              │
    │    瓶颈：内存带宽（每步都要读一遍模型权重+KV Cache） │
    └────┬─────────────────────────────────────────┘
         │
    ┌────▼─────────────────────────────────────────┐
    │ 5. 输出                                       │
    │    " jumps over the fence"                    │
    └──────────────────────────────────────────────┘

关键理解：
  Prefill: 输入多个 token → 并行计算 → 算力密集 → GPU 利用率高
  Decode:  每次 1 个 token → 串行生成 → 带宽密集 → GPU 利用率低
  
  没有 KV Cache: 每次 Decode 都要重算所有历史 token → O(N²)
  有 KV Cache:   每次只算 1 个新 token → O(N)
  → 这就是你在实验里看到的加速比！
""")
