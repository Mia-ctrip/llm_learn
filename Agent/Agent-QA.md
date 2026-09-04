# Agent架构范式
Agent 构建范式
│
├── 1. 基于对话 (Conversation-based)    
                                              ← ReAct
├── 2. 基于图 (Graph-based)  流程显式建模为有向图，LLM 只负责节点内的决策
                                             ←  LangGraph 
├── 3. 基于代码执行 (Code-based ) LLM 生成代码，代码执行结果反馈给 LLM
                                                    ← Shown in project using claude
├── 4. 多智能体 (Multi-Agent)
 
└── 5. 基于规划 (Plan-based)  先完整规划，再按计划执行，分离"思考"和"行动"

# 三种agent架构范式
| | Function-calling loop | 经典 ReAct | FC + Reasoning |
|---|---|---|---|
| **工具调用** | 结构化 Tool Call | 文本 Action | 结构化 Tool Call |
| **推理是否显式** | 不要求 | 是 | 是，但通常结构化/精简 |
| **循环依据** | tool call / final | Thought → Action → Observation | reasoning + tool result |
| **Runtime 复杂度** | 低 | 较高，需要 parser | 中 |
| **稳定性** | 高 | 较低 | 高 |
| **可观测性** | 中 | 高 | 高 |
| **Token 消耗** | 低 | 高 | 中 |
| **适合生产** | 很高 | 较低 | 很高 |