# 大模型评估指标学习笔记

## 一、核心认知：两种 Eval 要分清

| 类型 | 时机 | 目的 | 指标 |
|------|------|------|------|
| **训练中的 Eval** | 训练过程中 | 监控训练健康度、判断过拟合与收敛 | Loss、Perplexity |
| **Benchmark 评测** | 训练完成后 | 衡量模型真实能力，对外展示 | Accuracy、Pass@k、BLEU、Elo 等 |


---

## 二、大模型评估指标体系

大模型任务形态多样,没有像传统分类(AUC/F1)那样统一的指标,需按任务类型选择。

### 1. 选择题类（知识、推理）
- **指标**:Accuracy(准确率)
- **代表 Benchmark**:MMLU、C-Eval、CMMLU、AGIEval
- **评估方式**:
  - 方式一:生成下一个 token,看是 A/B/C/D
  - 方式二:对比 4 个选项的 logits,取概率最高

### 2. 数学/推理类
- **指标**:Accuracy
- **代表 Benchmark**:GSM8K、MATH
- **评估方式**:从输出中提取最终答案,与 ground truth 比对

### 3. 代码生成类
- **指标**:Pass@k
- **代表 Benchmark**:HumanEval、MBPP、LiveCodeBench
- **评估方式**:生成 k 份代码,只要有一份通过全部单元测试就算对

### 4. 文本生成类(翻译、摘要)
- **BLEU / ROUGE / METEOR**:基于 n-gram 重合度
- **BERTScore**:基于语义向量相似度

### 5. 语言建模
- **Perplexity (PPL)**:困惑度,越低越好,表示模型对文本的"意外程度"越小

### 6. 开放对话(最难评估)
- **人工评测**:Chatbot Arena 的 **Elo 评分**(用户盲测投票)
- **LLM-as-a-Judge**:用 GPT-4 等强模型当裁判(MT-Bench、AlpacaEval)
- **Win Rate**:A vs B 的胜率对比

---

## 三、Eval 的输入输出示例

### 示例 1:MMLU(选择题)

**输入 Prompt**:
```
The following is a multiple choice question about biology.

Question: What is the powerhouse of the cell?
A. Nucleus
B. Mitochondria
C. Ribosome
D. Golgi apparatus
Answer:
```

**输出**:A/B/C/D 中某一个

**评估伪代码**:
```python
for sample in dataset:
    prompt = format_prompt(sample)
    pred = model.generate(prompt)
    if pred == sample.answer:
        correct += 1
accuracy = correct / total
```

### 示例 2:HumanEval(代码)

**输入**:函数签名 + docstring
```python
def add(a, b):
    """Return the sum of a and b."""
```

**输出**:模型补全函数体 → 运行单元测试 → 通过即对

---

## 四、代码能力评估专题

### LiveCodeBench
- **背景**:UC Berkeley 等 2024 推出,主打**抗数据污染**
- **特点**:
  - 持续从 LeetCode、AtCoder、Codeforces 抓取新题
  - 题目带时间戳,可按"模型训练截止日期之后"筛选
- **评测 4 个维度**:
  1. Code Generation(生成代码)
  2. Self-Repair(修 bug)
  3. Code Execution(预测代码输出)
  4. Test Output Prediction(预测测试输出)
- **指标**:Pass@1

### Codeforces
- **性质**:真实的全球算法竞赛平台(非 LLM 专用 benchmark)
- **评估方式**:让模型做真实比赛题 → 折算 Elo Rating → 对标人类水平
- **Rating 分段**:
  - Newbie < 1200
  - Expert 1600–1900
  - Master 2100–2300
  - Grandmaster 2400+
  - Legendary Grandmaster 3000+
- **典型成绩参考**:
  - GPT-4: ~300–400(Newbie)
  - o1-preview: ~1800(Expert)
  - o3: ~2700(接近 Grandmaster)
  - DeepSeek-R1: ~2000

### 两者对比

| 维度 | LiveCodeBench | Codeforces |
|------|---------------|------------|
| 性质 | 专为 LLM 设计 | 真实人类竞赛 |
| 指标 | Pass@1 | Elo Rating |
| 难度 | 中等偏难 | 跨度极大 |
| 抗污染 | 时间窗口筛选 | 每周新题 |
| 解读 | "正确率 40%" | "相当于 Expert 人类" |

### 其他代码 Benchmark

| 名称 | 特点 |
|------|------|
| HumanEval | 164 道 Python 函数题,经典但已饱和 |
| MBPP | 974 道入门 Python 题 |
| **SWE-Bench** | 真实 GitHub issue 修复,考察**工程能力** |
| BigCodeBench | 复杂 API 调用场景 |
| APPS | 10000 道编程题,有难度分级 |
| CRUXEval | 代码执行/推理 |

---

## 五、主流评估 Benchmark 速查表

| 领域 | Benchmark | 指标 |
|------|-----------|------|
| 综合知识 | MMLU、C-Eval、CMMLU、AGIEval | Accuracy |
| 数学推理 | GSM8K、MATH | Accuracy |
| 代码生成 | HumanEval、MBPP、LiveCodeBench | Pass@k |
| 代码工程 | SWE-Bench | Resolve Rate |
| 算法竞赛 | Codeforces | Elo Rating |
| 常识推理 | HellaSwag、ARC、WinoGrande | Accuracy |
| 长文本 | LongBench、RULER | 综合指标 |
| 对话能力 | MT-Bench、AlpacaEval | LLM-as-Judge 评分 |
| 人类盲测 | Chatbot Arena | Elo |
| 中文综合 | SuperCLUE | 综合得分 |

---

## 六、评估趋势与思考

1. **简单题已饱和**:HumanEval 等早期 benchmark 区分度下降
2. **抗污染是刚需**:LiveCodeBench、Codeforces 这类"活"的评测更受认可
3. **工程能力受重视**:SWE-Bench 等真实场景评测兴起
4. **开放问答靠投票**:主流是 Chatbot Arena(Elo)+ LLM-as-Judge
5. **单一指标不可靠**:需多维度交叉验证,警惕刷榜

---

## 七、关键结论

1. 训练 eval loss ≠ 模型能力,只是训练健康度指标
2. 跑分 = 在标准测试集做推理,按任务用不同指标(Accuracy / Pass@k / BLEU / Elo)
3. Eval 输入通常是**精心设计的 prompt**,输出是生成的 token 序列
4. 开放任务无标准答案 → 靠**人类投票** 或 **强模型当裁判**
5. 看榜单要看**多个 benchmark 综合表现**,不能只信一个数字