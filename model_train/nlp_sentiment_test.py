import numpy
from model_train_v2 import NeuralNetwork

"""
简单的情感分析任务：判断评论是正面(1)还是负面(0)

注意：这是一个极其简化的例子，用于演示MLP在NLP上的能力和局限
"""

# ===== 步骤1: 构建词汇表 =====
# 这是我们的"词典"
vocabulary = {
    'good': 0, 'great': 1, 'excellent': 2, 'amazing': 3, 'love': 4,
    'bad': 5, 'terrible': 6, 'awful': 7, 'hate': 8, 'poor': 9,
    'product': 10, 'movie': 11, 'service': 12, 'food': 13, 'experience': 14
}

vocab_size = len(vocabulary)

def text_to_bow(text, vocabulary):
    """
    文本转为词袋模型（Bag of Words）

    这是最简单的文本表示方法，但会丢失词序信息
    例如："good movie" → [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    """
    bow = numpy.zeros(len(vocabulary))
    words = text.lower().split()

    for word in words:
        if word in vocabulary:
            bow[vocabulary[word]] = 1  # 词出现标记为1

    return bow

# ===== 步骤2: 准备训练数据 =====
train_texts = [
    # 正面评论
    "good movie",
    "great product",
    "excellent service",
    "amazing food",
    "love this experience",
    "good product",
    "great movie",

    # 负面评论
    "bad movie",
    "terrible product",
    "awful service",
    "hate this food",
    "poor experience",
    "bad service",
    "terrible movie"
]

train_labels = [
    1, 1, 1, 1, 1, 1, 1,  # 正面
    0, 0, 0, 0, 0, 0, 0   # 负面
]

# 转换为向量
X_train = numpy.array([text_to_bow(text, vocabulary) for text in train_texts])
y_train = numpy.array(train_labels).reshape(-1, 1)

print("=" * 60)
print("情感分析任务：判断评论是正面(1)还是负面(0)")
print("=" * 60)
print(f"\n词汇表大小: {vocab_size}")
print(f"训练样本数: {len(train_texts)}")
print(f"\n示例文本: '{train_texts[0]}'")
print(f"转换为向量: {X_train[0]}")
print(f"标签: {y_train[0][0]}")

# ===== 步骤3: 训练模型 =====
print("\n" + "=" * 60)
print("开始训练...")
print("=" * 60)

# 创建神经网络：输入15维（词汇表大小）→ 8个神经元 → 1维输出
nn = NeuralNetwork([vocab_size, 8, 1])
nn.train(X_train, y_train, epochs=5000, learning_rate=0.5, verbose=True)

# ===== 步骤4: 测试模型 =====
print("\n" + "=" * 60)
print("测试：已见过的评论")
print("=" * 60)

for i in range(len(train_texts)):
    text = train_texts[i]
    x = text_to_bow(text, vocabulary).reshape(1, -1)
    prediction = nn.predict(x)[0][0]
    actual = train_labels[i]

    result = "[OK]" if (prediction > 0.5) == actual else "[FAIL]"
    print(f"{result} '{text:25s}' -> Prediction: {prediction:.3f}, Actual: {actual}")

# ===== 步骤5: 泛化测试（未见过的组合）=====
print("\n" + "=" * 60)
print("泛化测试：未见过的评论组合")
print("=" * 60)

test_cases = [
    ("excellent movie", 1),     # 应该是正面
    ("bad product", 0),          # 应该是负面
    ("great food", 1),           # 应该是正面
    ("awful experience", 0),     # 应该是负面
    ("love this product", 1),    # 应该是正面
    ("hate this service", 0),    # 应该是负面
]

for text, expected in test_cases:
    x = text_to_bow(text, vocabulary).reshape(1, -1)
    prediction = nn.predict(x)[0][0]

    result = "[OK]" if (prediction > 0.5) == expected else "[FAIL]"
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    print(f"{result} '{text:25s}' -> {sentiment} (score: {prediction:.3f})")

# ===== 步骤6: 展示模型的局限性 =====
print("\n" + "=" * 60)
print("模型的局限性演示：词序被忽略")
print("=" * 60)

limitation_cases = [
    "good bad movie",           # 矛盾的词
    "bad good product",         # 词序颠倒
    "not good",                 # 否定词（未在词汇表中）
]

print("\nThese cases confuse the model (word order is ignored):")
for text in limitation_cases:
    x = text_to_bow(text, vocabulary).reshape(1, -1)
    prediction = nn.predict(x)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    print(f"  '{text:25s}' -> {sentiment} (score: {prediction:.3f})")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
[+] Your MLP model CAN do simple text classification!
[+] It learned to classify positive/negative words correctly
[+] It can generalize to unseen word combinations

[-] But it has serious limitations:
   - Cannot see word order ("good bad" = "bad good")
   - Cannot understand negation ("not good" treated as "good")
   - Cannot handle complex semantics

[*] For better NLP, you need:
   - Word embeddings (Word2Vec, GloVe) instead of bag-of-words
   - RNN/LSTM to understand sequences
   - Transformers (BERT, GPT) for context understanding
""")
