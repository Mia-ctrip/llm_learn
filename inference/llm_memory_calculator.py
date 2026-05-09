#!/usr/bin/env python3
"""
LLM 显存计算器

使用说明：
python3 llm_memory_calculator.py --param-billion 7 --precision fp16 --batch-size 1 --seq-len 2000
"""

import argparse
import sys

# 常见模型的参数
MODEL_CONFIGS = {
    'qwen2.5-7b': {'param_billion': 7, 'hidden_size': 3584, 'num_layers': 28},
    'qwen2.5-14b': {'param_billion': 14, 'hidden_size': 5120, 'num_layers': 40},
    'qwen2.5-72b': {'param_billion': 72, 'hidden_size': 8192, 'num_layers': 80},
    'qwen3-7b': {'param_billion': 7, 'hidden_size': 3584, 'num_layers': 28},
    'qwen3-70b': {'param_billion': 70, 'hidden_size': 8192, 'num_layers': 80},
    'mistral-7b': {'param_billion': 7, 'hidden_size': 4096, 'num_layers': 32},
    'llama3-8b': {'param_billion': 8, 'hidden_size': 4096, 'num_layers': 32},
    'llama3-70b': {'param_billion': 70, 'hidden_size': 8192, 'num_layers': 80},
}

# 精度配置
PRECISION_BYTES = {
    'fp32': 4,
    'fp16': 2,
    'bf16': 2,
    'int8': 1,
    'fp8': 1,
}

# GPU 显存规格
GPU_SPECS = {
    't4': 16,
    'v100': 32,
    'a100-40g': 40,
    'a100-80g': 80,
    'l20-24g': 24,
    'l20-48g': 48,
    'h20': 96,
    'h100': 80,
}

def calculate_memory(param_billion, hidden_size, num_layers, precision, batch_size, seq_len):
    """
    计算推理所需的总显存

    Args:
        param_billion: 参数量（单位：十亿）
        hidden_size: 隐藏层维度
        num_layers: 层数
        precision: 精度（'fp32', 'fp16', 'bf16', 'int8'）
        batch_size: 批大小
        seq_len: 序列长度

    Returns:
        dict: 包含各部分显存信息
    """

    bytes_per_param = PRECISION_BYTES.get(precision, 2)

    # 1. 权重显存
    weight_memory = param_billion * bytes_per_param

    # 2. KV Cache 显存
    # KV Cache = 2 × batch × seq_len × hidden_size × num_layers × bytes / 1B
    kv_cache_memory = (2 * batch_size * seq_len * hidden_size * num_layers * bytes_per_param) / (1024 * 1024 * 1024)

    # 3. 运行时显存（约为权重的 15%）
    runtime_memory = weight_memory * 0.15

    # 总显存
    total_memory = weight_memory + kv_cache_memory + runtime_memory

    return {
        'weight': weight_memory,
        'kv_cache': kv_cache_memory,
        'runtime': runtime_memory,
        'total': total_memory,
        'bytes_per_param': bytes_per_param,
    }

def recommend_gpu(required_memory):
    """
    根据所需显存推荐 GPU
    """
    # 预留 20% 安全边际
    required_with_margin = required_memory / 0.8

    recommendations = []

    for gpu_name, gpu_memory in sorted(GPU_SPECS.items(), key=lambda x: x[1]):
        if gpu_memory >= required_with_margin:
            utilization = (required_memory / gpu_memory) * 100
            recommendations.append({
                'gpu': gpu_name,
                'memory': gpu_memory,
                'utilization': utilization,
            })

    return recommendations

def print_result(model_name, config, param_billion, precision, batch_size, seq_len, memory_info):
    """打印计算结果"""
    print("=" * 80)
    print("LLM 推理显存计算结果")
    print("=" * 80)
    print()

    print(f"📊 配置参数：")
    print(f"   模型：{model_name}")
    print(f"   参数量：{param_billion}B")
    print(f"   精度：{precision.upper()}")
    print(f"   Batch Size：{batch_size}")
    print(f"   Sequence Length：{seq_len} tokens")
    print()

    print(f"💾 显存分解：")
    print(f"   ├─ 模型权重：{memory_info['weight']:.2f} GB")
    print(f"   ├─ KV Cache：{memory_info['kv_cache']:.2f} GB")
    print(f"   ├─ 运行时开销：{memory_info['runtime']:.2f} GB")
    print(f"   └─ 总计：{memory_info['total']:.2f} GB")
    print()

    total_mem = memory_info['total']
    print(f"🎯 GPU 推荐（由小到大）：")

    recommendations = recommend_gpu(total_mem)

    if recommendations:
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"   {i}. {rec['gpu']:15} ({rec['memory']:3} GB)  -  利用率 {rec['utilization']:5.1f}%")
    else:
        print(f"   ⚠️  找不到合适的单卡方案")
        print(f"   需要 {total_mem:.1f} GB，超过所有单卡容量")

        # 推荐多卡方案
        for num_cards in [2, 4, 8]:
            for gpu_name, gpu_memory in sorted(GPU_SPECS.items(), key=lambda x: x[1], reverse=True):
                total_gpu_memory = gpu_memory * num_cards
                if total_gpu_memory >= total_mem * 1.2:
                    print(f"   建议：{num_cards} × {gpu_name} ({total_gpu_memory} GB)")
                    break

    print()
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description='LLM 推理显存计算工具')

    # 模型选择
    parser.add_argument('--model', type=str, default='qwen2.5-14b',
                        choices=list(MODEL_CONFIGS.keys()),
                        help='预定义模型（或使用 --param-billion 自定义）')
    parser.add_argument('--param-billion', type=float, default=None,
                        help='参数量（十亿）- 优先级高于 --model')
    parser.add_argument('--hidden-size', type=int, default=None,
                        help='隐藏层维度（需要和 --param-billion 一起使用）')
    parser.add_argument('--num-layers', type=int, default=None,
                        help='层数（需要和 --param-billion 一起使用）')

    # 推理配置
    parser.add_argument('--precision', type=str, default='fp16',
                        choices=list(PRECISION_BYTES.keys()),
                        help='精度类型')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch Size')
    parser.add_argument('--seq-len', type=int, default=2000,
                        help='序列长度')

    args = parser.parse_args()

    # 获取模型配置
    if args.param_billion:
        # 自定义参数
        param_billion = args.param_billion
        hidden_size = args.hidden_size or 3584  # 默认值
        num_layers = args.num_layers or int(param_billion / 0.65)  # 粗略估计
        model_name = f"Custom {param_billion}B"
    else:
        # 使用预定义模型
        model_name = args.model
        config = MODEL_CONFIGS[args.model]
        param_billion = config['param_billion']
        hidden_size = config['hidden_size']
        num_layers = config['num_layers']

    # 计算显存
    memory_info = calculate_memory(
        param_billion=param_billion,
        hidden_size=hidden_size,
        num_layers=num_layers,
        precision=args.precision,
        batch_size=args.batch_size,
        seq_len=args.seq_len
    )

    # 打印结果
    print_result(
        model_name=model_name,
        config=None,
        param_billion=param_billion,
        precision=args.precision,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        memory_info=memory_info
    )

if __name__ == '__main__':
    main()
