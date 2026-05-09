#!/usr/bin/env python3
"""
Qwen2-72B模型权重检查工具
用于理解模型的实际结构和显存占用
"""

from safetensors import safe_open
import json
from pathlib import Path

# 模型路径
MODEL_PATH = Path("/home/powerop/work/model_serving/models/qwen2-72b")

def load_config():
    """加载模型配置"""
    with open(MODEL_PATH / "config.json") as f:
        return json.load(f)

def calculate_params(shape):
    """计算参数量"""
    result = 1
    for dim in shape:
        result *= dim
    return result

def bytes_to_gb(bytes_val):
    """字节转GB"""
    return bytes_val / (1024**3)

def inspect_weights():
    """检查模型权重"""

    print("=" * 80)
    print("Qwen2-72B 模型权重分析")
    print("=" * 80)

    # 加载配置
    config = load_config()
    print("\n【模型配置】")
    print(f"  隐藏层维度 (hidden_size): {config['hidden_size']}")
    print(f"  层数 (num_hidden_layers): {config['num_hidden_layers']}")
    print(f"  注意力头数 (num_attention_heads): {config['num_attention_heads']}")
    print(f"  KV头数 (num_key_value_heads): {config['num_key_value_heads']}")
    print(f"  词汇表大小 (vocab_size): {config['vocab_size']}")
    print(f"  最大序列长度 (max_position_embeddings): {config['max_position_embeddings']}")
    print(f"  数据类型: {config['torch_dtype']}")

    # 检查第一个权重文件
    first_file = MODEL_PATH / "model-00001-of-00037.safetensors"

    print(f"\n【权重文件分析】(第一个文件)")
    print(f"  文件: {first_file.name}")
    print(f"  大小: {bytes_to_gb(first_file.stat().st_size):.2f} GB")

    print("\n【权重层结构】(前20层)")
    total_params = 0
    layer_count = 0

    with safe_open(first_file, framework="pt") as f:
        for i, key in enumerate(f.keys()):
            if i >= 20:  # 只显示前20层
                print(f"  ... (共 {len(f.keys())} 层)")
                break

            tensor = f.get_tensor(key)
            shape = tuple(tensor.shape)
            params = calculate_params(shape)
            total_params += params
            layer_count += 1

            # 计算该层的显存占用 (bfloat16 = 2字节)
            memory_mb = (params * 2) / (1024**2)

            print(f"  {key}")
            print(f"    形状: {shape}")
            print(f"    参数量: {params:,}")
            print(f"    显存: {memory_mb:.2f} MB")
            print()

    print(f"\n【第一个文件统计】")
    print(f"  总参数量: {total_params:,}")
    print(f"  显存占用: {bytes_to_gb(total_params * 2):.2f} GB")

def calculate_inference_memory():
    """计算推理显存需求"""

    config = load_config()

    print("\n" + "=" * 80)
    print("推理显存计算")
    print("=" * 80)

    # 模型权重
    total_params = 72_000_000_000  # 72B
    weight_memory = bytes_to_gb(total_params * 2)  # bfloat16

    print(f"\n【模型权重】")
    print(f"  总参数: {total_params:,}")
    print(f"  bfloat16显存: {weight_memory:.2f} GB")

    # KV Cache计算
    print(f"\n【KV Cache显存】(最重要!)")

    scenarios = [
        ("短对话", 1, 512),
        ("普通对话", 1, 2048),
        ("长文本", 1, 8192),
        ("最大长度", 1, 32768),
        ("批处理x4", 4, 2048),
    ]

    for name, batch_size, seq_len in scenarios:
        # KV Cache公式: 2 × batch × seq_len × layers × hidden_size × bytes
        kv_cache = (
            2 * batch_size * seq_len *
            config['num_hidden_layers'] *
            config['hidden_size'] *
            2  # bfloat16
        )
        kv_cache_gb = bytes_to_gb(kv_cache)

        # 激活值约占2-4GB
        activation_gb = 2 + (batch_size - 1) * 0.5

        # 总显存
        total_gb = weight_memory + kv_cache_gb + activation_gb

        print(f"\n  场景: {name}")
        print(f"    batch_size: {batch_size}")
        print(f"    sequence_length: {seq_len}")
        print(f"    KV Cache: {kv_cache_gb:.2f} GB")
        print(f"    激活值: {activation_gb:.2f} GB")
        print(f"    总显存: {total_gb:.2f} GB")

        # GPU建议
        if total_gb <= 40:
            gpu_suggestion = "1 × A100 40GB"
        elif total_gb <= 80:
            gpu_suggestion = "1 × A100 80GB 或 2 × A100 40GB"
        elif total_gb <= 160:
            gpu_suggestion = f"需要 {int(total_gb / 40) + 1} × A100 40GB"
        else:
            gpu_suggestion = f"需要 {int(total_gb / 80) + 1} × A100 80GB"

        print(f"    GPU建议: {gpu_suggestion}")

def explain_key_concepts():
    """解释关键概念"""

    print("\n" + "=" * 80)
    print("关键概念速查")
    print("=" * 80)

    concepts = {
        "hidden_size": "每个token的向量维度,影响模型表达能力",
        "num_hidden_layers": "Transformer层数,影响模型理解深度",
        "num_attention_heads": "注意力头数,从多个角度理解输入",
        "vocab_size": "词汇表大小,模型认识的token总数",
        "batch_size": "一次处理的样本数,影响吞吐量和显存",
        "sequence_length": "输入序列的实际长度,影响KV Cache显存",
        "KV Cache": "存储注意力机制的Key和Value,显存占用大户",
    }

    for term, explanation in concepts.items():
        print(f"\n  {term}:")
        print(f"    {explanation}")

def main():
    """主函数"""
    try:
        # 检查权重
        inspect_weights()

        # 计算显存
        calculate_inference_memory()

        # 解释概念
        explain_key_concepts()

        print("\n" + "=" * 80)
        print("分析完成!")
        print("=" * 80)

    except Exception as e:
        print(f"\n错误: {e}")
        print("\n提示:")
        print("  1. 确保已安装 safetensors: pip install safetensors")
        print("  2. 确保模型路径正确")

if __name__ == "__main__":
    main()
