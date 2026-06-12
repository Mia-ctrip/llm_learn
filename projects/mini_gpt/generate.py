"""
Mini-GPT 文本生成脚本
=====================

加载训练好的模型，进行交互式文本生成。

用法：
    python generate.py                          # 交互式
    python generate.py --prompt "First Citizen"  # 指定 prompt
    python generate.py --checkpoint model.pt     # 指定模型文件
"""

import torch
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from model import MiniGPT
from train import CharTokenizer, generate


def load_model(checkpoint_path: str, device: str = 'cpu'):
    """从 checkpoint 加载模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['model_config']
    
    # 重建模型
    model = MiniGPT(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        max_len=config['max_len'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 重建 tokenizer
    tokenizer = CharTokenizer.__new__(CharTokenizer)
    tokenizer.char_to_id = checkpoint['tokenizer_char_to_id']
    tokenizer.id_to_char = {int(k): v for k, v in checkpoint['tokenizer_id_to_char'].items()}
    tokenizer.vocab_size = len(tokenizer.char_to_id)
    tokenizer.pad_token = '<PAD>'
    
    print(f"模型加载成功！")
    print(f"  参数量: {model.get_num_params():,}")
    print(f"  最终训练 Loss: {checkpoint['final_loss']:.4f}")
    
    return model, tokenizer


def interactive_generate(model, tokenizer, device):
    """交互式文本生成"""
    print("\n" + "=" * 60)
    print("Mini-GPT 交互式生成")
    print("=" * 60)
    print("输入文本 prompt，模型会接着写。输入 'quit' 退出。")
    print("参数: temperature=0.8, top_k=40, max_new_tokens=200")
    print("-" * 60)
    
    while True:
        try:
            prompt = input("\n> Prompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if prompt.lower() in ('quit', 'exit', 'q'):
            break
        
        if not prompt:
            print("请输入一些文本")
            continue
        
        generated = generate(
            model, tokenizer, prompt,
            max_new_tokens=200,
            temperature=0.8,
            top_k=40,
            device=device,
        )
        
        print(f"\n{generated}")


def main():
    parser = argparse.ArgumentParser(description="Mini-GPT 文本生成")
    parser.add_argument('--checkpoint', type=str, default=None, help='模型 checkpoint 路径')
    parser.add_argument('--prompt', type=str, default=None, help='生成 prompt')
    parser.add_argument('--max-tokens', type=int, default=200, help='最大生成 token 数')
    parser.add_argument('--temperature', type=float, default=0.8, help='温度参数')
    parser.add_argument('--top-k', type=int, default=40, help='Top-K 采样')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 查找 checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        default_path = os.path.join(os.path.dirname(__file__), 'mini_gpt_checkpoint.pt')
        if os.path.exists(default_path):
            checkpoint_path = default_path
        else:
            print(f"找不到模型文件。请先运行 train.py 训练模型。")
            print(f"或者用 --checkpoint 指定模型路径。")
            return
    
    model, tokenizer = load_model(checkpoint_path, device)
    
    if args.prompt:
        generated = generate(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )
        print(generated)
    else:
        interactive_generate(model, tokenizer, device)


if __name__ == "__main__":
    main()
