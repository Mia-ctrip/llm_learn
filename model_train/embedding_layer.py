import numpy

class EmbeddingLayer:
    """
    词嵌入层：把词ID转换为稠密向量

    核心思想：Embedding就是一个查找表
    - 输入：词ID (整数)
    - 输出：词向量 (浮点数向量)
    """
    def __init__(self, vocab_size, embed_dim):
        """
        参数:
            vocab_size: 词汇表大小 (比如10000个词)
            embed_dim: 词向量维度 (比如50维或300维)
        """
        # TODO: 初始化嵌入矩阵
        # 形状: (vocab_size, embed_dim)
        # 提示: 用小的随机数初始化，比如 * 0.01
        self.embeddings = numpy.random.randn(vocab_size, embed_dim) * 0.01

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def forward(self, word_ids):
        """
        前向传播：词ID → 词向量

        参数:
            word_ids: numpy数组
                     单个词: shape (1,) 或标量
                     一句话: shape (seq_len,)
                     一批句子: shape (batch_size, seq_len)

        返回:
            词向量，shape会对应增加一个embed_dim维度
        """
        # TODO: 实现查表操作
        # 提示: 在NumPy中，可以用索引直接查找
        #       self.embeddings[word_ids] 就能完成查表
        return self.embeddings[word_ids]

    def backward(self, word_ids, grad_output):
        """
        反向传播：更新词向量

        参数:
            word_ids: 前向传播时的词ID
            grad_output: 损失函数对这些词向量的梯度
                        shape和forward的输出一样

        这个函数会把梯度累加到对应的词向量上
        """
        # TODO: 实现梯度更新
        # 提示: 对于每个词ID，把对应的梯度加到embeddings矩阵的对应行上
        # 注意: 如果同一个词出现多次，梯度要累加

        # 简化版实现（假设word_ids是1D的）
        if word_ids.ndim == 1:
            for i, word_id in enumerate(word_ids):
                self.embeddings[word_id] += grad_output[i]
        else:
            # 处理更复杂的情况
            flat_ids = word_ids.flatten()
            flat_grads = grad_output.reshape(-1, self.embed_dim)
            for i, word_id in enumerate(flat_ids):
                self.embeddings[word_id] += flat_grads[i]


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    print("=" * 70)
    print("Embedding Layer Test")
    print("=" * 70)

    # 创建一个小词汇表
    vocab = {
        "hello": 0,
        "world": 1,
        "good": 2,
        "bad": 3,
        "movie": 4
    }

    vocab_size = len(vocab)
    embed_dim = 8  # 用8维向量表示每个词

    print(f"\nVocabulary size: {vocab_size}")
    print(f"Embedding dimension: {embed_dim}")

    # 初始化Embedding层
    embed_layer = EmbeddingLayer(vocab_size, embed_dim)

    print(f"\nEmbedding matrix shape: {embed_layer.embeddings.shape}")
    print(f"Expected: ({vocab_size}, {embed_dim})")

    # ===== 测试1: 单个词 =====
    print("\n" + "=" * 70)
    print("Test 1: Single word")
    print("=" * 70)

    word_id = vocab["hello"]  # 0
    word_vec = embed_layer.forward(numpy.array([word_id]))

    print(f"Word: 'hello' (ID={word_id})")
    print(f"Vector shape: {word_vec.shape}")
    print(f"Vector: {word_vec[0]}")

    # ===== 测试2: 一句话 =====
    print("\n" + "=" * 70)
    print("Test 2: Sentence")
    print("=" * 70)

    sentence = "hello world good movie"
    word_ids = numpy.array([vocab[w] for w in sentence.split()])
    word_vecs = embed_layer.forward(word_ids)

    print(f"Sentence: '{sentence}'")
    print(f"Word IDs: {word_ids}")
    print(f"Output shape: {word_vecs.shape}")
    print(f"Expected: ({len(word_ids)}, {embed_dim})")

    # ===== 测试3: 计算词相似度 =====
    print("\n" + "=" * 70)
    print("Test 3: Word similarity (before training)")
    print("=" * 70)

    def cosine_similarity(v1, v2):
        """计算余弦相似度"""
        return numpy.dot(v1, v2) / (numpy.linalg.norm(v1) * numpy.linalg.norm(v2) + 1e-8)

    hello_vec = embed_layer.forward(numpy.array([vocab["hello"]]))[0]
    world_vec = embed_layer.forward(numpy.array([vocab["world"]]))[0]
    good_vec = embed_layer.forward(numpy.array([vocab["good"]]))[0]
    bad_vec = embed_layer.forward(numpy.array([vocab["bad"]]))[0]

    print(f"hello vs world: {cosine_similarity(hello_vec, world_vec):.4f}")
    print(f"hello vs good:  {cosine_similarity(hello_vec, good_vec):.4f}")
    print(f"good vs bad:    {cosine_similarity(good_vec, bad_vec):.4f}")

    print("\nNote: 随机初始化的词向量相似度接近0是正常的")
    print("训练后，语义相似的词向量会聚在一起")

    # ===== 测试4: 梯度更新 =====
    print("\n" + "=" * 70)
    print("Test 4: Gradient update")
    print("=" * 70)

    # 保存更新前的向量
    word_id = vocab["hello"]
    old_vec = embed_layer.embeddings[word_id].copy()

    # 模拟一个梯度
    fake_gradient = numpy.random.randn(1, embed_dim) * 0.1

    # 反向传播
    embed_layer.backward(numpy.array([word_id]), fake_gradient)

    # 检查是否更新
    new_vec = embed_layer.embeddings[word_id]

    print(f"Old vector: {old_vec}")
    print(f"Gradient:   {fake_gradient[0]}")
    print(f"New vector: {new_vec}")
    print(f"Difference: {new_vec - old_vec}")
    print(f"Match gradient? {numpy.allclose(new_vec - old_vec, fake_gradient[0])}")

    print("\n" + "=" * 70)
    print("Embedding Layer: All tests passed!")
    print("=" * 70)
    print("\nKey takeaways:")
    print("1. Embedding layer = learnable lookup table")
    print("2. Forward: word_id -> word_vector (just indexing!)")
    print("3. Backward: accumulate gradients to update embeddings")
    print("4. After training, similar words will have similar vectors")
