"""
从 HuggingFace 下载中文 Wikipedia 语料，生成 train.txt / eval.txt

用法:
    python prepare_data.py                          # 默认 5000 条训练, 500 条评估
    python prepare_data.py --rows 2000              # 2000 条训练, 200 条评估
    python prepare_data.py --train 10000 --eval 1000 # 自定义数量

依赖:
    pip install datasets
"""

import os
import argparse

_DIR = os.path.dirname(os.path.abspath(__file__))


def prepare_zhwiki(train_rows=5000, eval_rows=500):
    """
    下载 shibing624/zhwiki 语料，分割为训练集和评估集
    
    参数:
        train_rows: 训练集条数 (默认 5000)
        eval_rows:  评估集条数 (默认 500)
    
    返回:
        (train_path, eval_path): 生成的文件路径
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("=" * 50)
        print("需要安装 datasets 库:")
        print("  pip install datasets")
        print("=" * 50)
        raise

    total = train_rows + eval_rows
    print(f"正在下载 shibing624/zhwiki（共取 {total} 条，其中训练 {train_rows} 条，评估 {eval_rows} 条）...")

    dataset = load_dataset("shibing624/zhwiki", split="train", streaming=True)

    texts = []
    for i, example in enumerate(dataset):
        if i >= total:
            break
        texts.append(example["text"])
        if (i + 1) % 1000 == 0:
            print(f"  已下载 {i + 1} / {total} 条...")

    train_texts = texts[:train_rows]
    eval_texts = texts[train_rows:train_rows + eval_rows]

    train_path = os.path.join(_DIR, "train.txt")
    eval_path = os.path.join(_DIR, "eval.txt")

    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_texts))
    print(f"✅ 训练数据已保存: {train_path}（{len(train_texts)} 条，约 {os.path.getsize(train_path) / 1024:.1f} KB）")

    with open(eval_path, "w", encoding="utf-8") as f:
        f.write("\n".join(eval_texts))
    print(f"✅ 评估数据已保存: {eval_path}（{len(eval_texts)} 条，约 {os.path.getsize(eval_path) / 1024:.1f} KB）")

    print(f"\n💡 现在可以运行 train() 开始训练了")

    return train_path, eval_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载中文维基百科语料")
    parser.add_argument("--train", type=int, default=5000, help="训练集条数 (默认: 5000)")
    parser.add_argument("--eval", type=int, default=500, help="评估集条数 (默认: 500)")
    parser.add_argument("--rows", type=int, default=None, help="快捷设置: 训练=rows, 评估=rows//10")
    args = parser.parse_args()

    if args.rows is not None:
        train_rows = args.rows
        eval_rows = max(args.rows // 10, 100)
    else:
        train_rows = args.train
        eval_rows = args.eval

    prepare_zhwiki(train_rows, eval_rows)
