"""
张量广播 - 5分钟搞懂
"""
import torch
import numpy as np

print("=" * 60)
print("张量广播 (Broadcasting) - 白话讲解")
print("=" * 60)

# ============================================================================
# 场景1: 给每个学生的成绩加10分
# ============================================================================
print("\n场景1: 给每个学生的成绩加10分")
print("-" * 60)

scores = torch.tensor([85, 90, 78, 92])  # 4个学生的成绩
bonus = 10  # 标量

# NumPy/PyTorch 会自动把 10 "复制"成 [10, 10, 10, 10]
new_scores = scores + bonus

print(f"原成绩: {scores}")
print(f"加分:   {bonus}")
print(f"新成绩: {new_scores}")
print("\n>>> 你没写循环，但PyTorch自动帮每个学生都加了10分")

# ============================================================================
# 场景2: 给每门课的成绩都加10分（二维）
# ============================================================================
print("\n场景2: 给每门课的成绩都加上不同的调整分")
print("-" * 60)

# 4个学生，3门课
grades = torch.tensor([
    [80, 85, 90],  # 学生1
    [75, 80, 85],  # 学生2
    [90, 95, 88],  # 学生3
    [70, 75, 80]   # 学生4
])

# 每门课的调整分不同：数学+5，英语+10，物理+3
adjustments = torch.tensor([5, 10, 3])  # shape: (3,)

# 广播：把 (3,) 自动扩展成 (4, 3)
# adjustments 在每一行都复制一遍
adjusted_grades = grades + adjustments

print(f"原成绩 (4个学生, 3门课):\n{grades}")
print(f"\n调整分 (每门课不同):\n{adjustments}")
print(f"\n调整后成绩:\n{adjusted_grades}")
print("\n>>> PyTorch自动把 [5,10,3] 复制成4行，每个学生都用上了")

# ============================================================================
# 场景3: 理解广播的规则
# ============================================================================
print("\n场景3: 广播的核心规则")
print("-" * 60)

print("\n规则：从右往左对齐维度，缺的维度自动填1，然后复制")
print()

# 例子1: 标量广播
a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])  # shape: (2, 3)
b = 10                          # shape: () 标量
print(f"a shape: {a.shape}  ->  (2, 3)")
print(f"b shape: scalar     ->  (2, 3)  # 自动扩展")
print(f"a + b:\n{a + b}\n")

# 例子2: 向量广播
a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])  # shape: (2, 3)
b = torch.tensor([10, 20, 30])  # shape: (3,)
print(f"a shape: {a.shape}      ->  (2, 3)")
print(f"b shape: {b.shape}      ->  (1, 3)  # 补一个维度")
print(f"                            (2, 3)  # 然后在第一维复制")
print(f"a + b:\n{a + b}\n")

# 例子3: 列向量广播
a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])  # shape: (2, 3)
b = torch.tensor([[10],
                  [20]])       # shape: (2, 1)
print(f"a shape: {a.shape}      ->  (2, 3)")
print(f"b shape: {b.shape}     ->  (2, 3)  # 在第二维复制")
print(f"a + b:\n{a + b}\n")

# ============================================================================
# 场景4: 实战 - Attention中的广播
# ============================================================================
print("\n场景4: Attention机制中的广播")
print("-" * 60)

# 简化的Attention计算
seq_len = 4
d_k = 8

# Q @ K^T 得到注意力分数
scores = torch.randn(seq_len, seq_len)  # (4, 4)
d_k_sqrt = torch.tensor(d_k ** 0.5)     # 标量

# 除以 sqrt(d_k) - 这里用到广播！
scaled_scores = scores / d_k_sqrt

print(f"注意力分数 shape: {scores.shape}")
print(f"sqrt(d_k): {d_k_sqrt.item():.2f}")
print(f"缩放后的分数 shape: {scaled_scores.shape}")
print("\n>>> 一个标量除以(4,4)的矩阵，自动广播")

# ============================================================================
# 场景5: 常见错误
# ============================================================================
print("\n场景5: 什么时候会出错？")
print("-" * 60)

try:
    a = torch.tensor([[1, 2, 3],
                      [4, 5, 6]])  # (2, 3)
    b = torch.tensor([10, 20])     # (2,)
    c = a + b  # 会报错！
except RuntimeError as e:
    print(f"[X] 错误: {e}")
    print("\n原因分析:")
    print("  a shape: (2, 3)")
    print("  b shape:    (2)  # 对齐到最右边")
    print("           ------")
    print("           (2, 3) vs (1, 2)  # 3 != 2，无法广播！")

print("\n正确做法:")
a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])  # (2, 3)
b = torch.tensor([10, 20]).unsqueeze(1)  # (2, 1) - 加一个维度
c = a + b
print(f"  b.unsqueeze(1) shape: {b.shape}")
print(f"  结果:\n{c}")

# ============================================================================
# 场景6: 快速检查广播是否合法
# ============================================================================
print("\n场景6: 快速判断能否广播")
print("-" * 60)

def can_broadcast(shape1, shape2):
    """检查两个形状是否能广播"""
    # 从右往左对齐
    for dim1, dim2 in zip(reversed(shape1), reversed(shape2)):
        if dim1 != dim2 and dim1 != 1 and dim2 != 1:
            return False
    return True

examples = [
    ((2, 3), (3,)),      # ✓ (2,3) + (1,3) = (2,3)
    ((2, 3), (2, 1)),    # ✓ (2,3) + (2,1) = (2,3)
    ((2, 3), (2,)),      # ✗ (2,3) + (1,2) 不匹配
    ((4, 1, 5), (3, 5)), # ✓ (4,1,5) + (1,3,5) = (4,3,5)
]

for s1, s2 in examples:
    result = "[OK] 可以" if can_broadcast(s1, s2) else "[NO] 不行"
    print(f"  {s1} + {s2}  ->  {result}")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 60)
print("总结")
print("=" * 60)

summary = """
广播的本质：自动复制数据，让不同形状能运算

核心规则（3步）：
  1. 从右往左对齐维度
  2. 缺的维度当作1
  3. 维度是1的地方自动复制

什么时候能广播？
  [OK] 维度相等
  [OK] 其中一个维度是1
  [NO] 维度不等且都不是1

实战建议：
  1. 遇到维度错误，print出两个tensor的shape
  2. 在纸上画出形状，从右往左对齐
  3. 用 .unsqueeze() 或 .reshape() 调整维度

记住：
  - 广播不是魔法，就是"自动复制"
  - 看懂别人代码里的广播：print出shape
  - 调试shape错误：一步步打印中间结果的shape
"""

print(summary)

print("\n现在运行这个脚本，看看实际效果！")
print("=" * 60)
