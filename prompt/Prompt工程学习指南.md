# Prompt 工程学习指南

> 核心逻辑：理解模型如何处理输入 → 规避输入陷阱 → 主动引导生成 → 系统化技法

---

## 一、底层机制：为什么 Prompt 的写法会影响结果

理解这一层，后面所有"技巧"才有根基，不会沦为玄学。

**Prefill 阶段决定一切上下文**
整个 prompt 在 Prefill 阶段一次性被模型处理，所有 token 建立 KV Cache，Decode 阶段生成的每个 token 都通过 attention 向这份 KV Cache 查询。  
因此：**prompt 的每个 token 都是真实的计算资源消耗，也是真实的语义信号来源。**

**Attention 是分布式的，不是均匀的**  
Softmax 归一化导致 attention weight 总和为 1，序列越长，单个 token 获得的平均权重越低。这是"位置效应"所有现象的根源。

---

## 二、输入陷阱：哪些写法在静默消耗 & 干扰

### 2.1 格式字符占 Token

| 字符 | Token 消耗 |
|---|---|
| 单个空格 | 通常合并进相邻词，额外消耗极小 |
| 换行 `\n` | 独立 token，1个 |
| 空行 `\n\n` | 1~2 个 token |
| Markdown 符号 `**`, `##`, `-` | 各 1~2 个 token |

**结论**：格式空白本身消耗小，但大量堆砌时在 token budget 紧张的场景值得注意。

### 2.2 位置效应：Lost in the Middle

**实验来源**：Liu et al., 2023 *"Lost in the Middle: How Language Models Use Long Contexts"*

```
attention weight 在长文本中的分布：
开头  ████████  高（被所有后续 token attend）
中间  ██        低（信号被稀释）
结尾  ██████    较高（距生成位置近）
```

**机制解释**：
- 开头 token 在 causal attention 中被所有后续 token 引用，积累效应最强
- 结尾 token 与生成位置相对距离最近，在 RoPE 等位置编码中内积天然更大
- 中间内容受两端夹击，attention weight 被稀释

**实践推论**：

| 内容类型 | 推荐位置 |
|---|---|
| 角色定义、最高优先级规则 | **开头**（System Prompt） |
| 背景知识、参考文档 | 中间（影响有限，RAG 内容放这里） |
| 当前任务描述、用户 Query | **结尾**（紧靠生成位置） |

---

## 三、表达引导：如何写才能有效传递意图

### 3.1 强调手段的有效性对比

| 手段 | 有效性 | 原因 |
|---|---|---|
| Markdown 加粗 `**text**` | 条件有效 | instruction-tuned 模型学过 Markdown 语义，但本质是统计相关 |
| 显式文字描述 `"注意：以下规则必须严格遵守"` | 稳定有效 | 语义明确，不依赖模型对符号的理解 |
| 大写 `MUST` | 部分有效 | 英文 instruction-tuned 模型对大写敏感，中文场景效果弱 |

**结论**：与其依赖格式符号，不如用明确的自然语言表达优先级。

### 3.2 Markdown 格式 vs 纯文本

Markdown 的价值不在于"视觉效果"，在于**给模型传递结构层级信号**。

- **使用 Markdown**：适合 instruction-tuned 模型，`##` 标题帮助模型理解信息层次，并倾向于输出结构化内容
- **使用纯文本**：适合 base model 或对输出格式无要求的场景，避免符号噪声

---

## 四、技法体系：系统化引导模型推理

### 4.1 基础技法

**Few-shot Prompting**（GPT-3 论文，2020）
```
示例1: [输入] → [输出]
示例2: [输入] → [输出]
当前:  [输入] → ?
```
机制：示例 token 在 Prefill 阶段建立 KV Cache，模型 attend 到示例 pattern 后模仿输出。

---

**Chain-of-Thought (CoT)**（Wei et al., 2022）
```
# 加入思考过程
Q: ...先推导...因此答案是X
```
机制：强制模型在 Decode 阶段生成中间推理 token，这些 token 进入 KV Cache 后成为后续生成的"工作记忆"，相当于将隐式推理外显化。

---

**Zero-shot CoT**
在问题末尾加：`"让我们一步步思考。"` / `"Let's think step by step."`  
最小成本激活慢思考路径。

### 4.2 进阶技法

| 技法 | 核心思想 | 适用场景 |
|---|---|---|
| Self-Consistency | 多次采样后投票取一致答案 | 数学/逻辑推理（牺牲速度换准确） |
| Tree of Thoughts (ToT) | 树状搜索多条推理路径，剪枝保留最优 | 复杂规划问题 |
| ReAct | 交替 Reason（思考）+ Act（调工具） | Agent 场景 |
| Role Prompting | 赋予模型专家身份 | 专业领域输出质量提升 |

### 4.3 结构化输出约束

当 prompt 需要控制输出格式时，直接用 JSON Schema 描述期望结构，比自然语言描述格式可靠得多。

---

## 五、工程化：超越手写 Prompt

当任务复杂、需要持续优化时，手写 prompt 的天花板很低。

**DSPy（Stanford）**  
核心理念：把 prompt 当成可优化的参数，用数据驱动代替人工调试。

```python
import dspy

class QA(dspy.Signature):
    """根据问题给出准确回答"""
    question = dspy.InputField()
    answer = dspy.OutputField()

# 框架自动优化 prompt，不需要手写
```

适用场景：有评估数据集、需要反复迭代优化的生产级任务。

---

## 六、学习路径

```
第一阶段（机制理解）
  → 读本文第一、二章
  → 结合 Transformer attention 机制理解位置效应

第二阶段（技法掌握）
  → OpenAI Prompt Engineering Guide（官方，30分钟）
  → Anthropic Prompt Engineering Guide（理论更深）
  → 找一个实际任务，把 CoT / Few-shot 都动手试一遍

第三阶段（工程化）
  → DSPy 文档 + 跑通一个 Optimizer 示例
  → 读 CoT 原始论文（Wei et al., 2022）
```

---

## 参考

- Brown et al., 2020 — *Language Models are Few-Shot Learners*（GPT-3，Few-shot 来源）
- Wei et al., 2022 — *Chain-of-Thought Prompting Elicits Reasoning in LLMs*
- Liu et al., 2023 — *Lost in the Middle: How Language Models Use Long Contexts*
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic Prompt Engineering](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)
