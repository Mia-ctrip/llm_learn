# Agent Skill 编写、优化、安装与发布指南

## 1. Skill 的基本概念

Skill 可以理解为一份提供给 Agent 的**可复用能力说明书**。

它通常用于告诉 Codex、Claude Code 或其他兼容 Agent Skills 标准的 Agent：

- 什么情况下应该调用这个 Skill
- 应该执行什么流程
- 需要读取哪些参考资料
- 可以执行哪些脚本
- 最终应该输出什么
- 如何判断任务是否完成

推荐优先按照 **Agent Skills 开放标准** 编写 Skill，从而尽量实现：

```text
同一份 Skill
    ↓
Codex
Claude Code
其他兼容 Agent Skills 的 Agent
```

核心原则：

> Skill 本体尽量保持平台无关，Codex / Claude Code 的差异主要放在安装路径和平台专属配置层。

---

# 2. 一个最小可用 Skill 的目录结构

最简单的 Skill 只需要：

```text
my-skill/
└── SKILL.md
```

其中：

```text
SKILL.md
```

是唯一必需文件。

不强制要求：

```text
README.md
manifest.json
package.json
config.yaml
```

因此，一个 Skill 最小可以只有一个文件。

---

# 3. 推荐的完整 Skill 目录结构

复杂 Skill 推荐采用：

```text
my-skill/
├── SKILL.md
├── references/
│   └── ...
├── scripts/
│   └── ...
├── assets/
│   └── ...
└── agents/
    └── openai.yaml
```

各目录职责如下。

| 文件 / 目录 | 是否必须 | 用途 |
|---|---|---|
| `SKILL.md` | 必须 | Skill 主入口、核心说明 |
| `references/` | 可选 | API 文档、业务规则、长篇参考资料 |
| `scripts/` | 可选 | Python、Shell、JS 等可执行脚本 |
| `assets/` | 可选 | 模板、Schema、静态资源 |
| `agents/openai.yaml` | 可选 | OpenAI / Codex 相关扩展配置 |

推荐职责划分：

```text
SKILL.md
负责：
- 什么时候调用
- 核心流程
- 输入要求
- 输出要求
- 调用哪些 reference
- 调用哪些 script
- Validation

references/
负责：
- 大量知识
- API 文档
- 业务规则
- 示例
- 长篇说明

scripts/
负责：
- 可以确定执行的逻辑
- 数据处理
- Validation
- 文件转换
- 自动化操作

assets/
负责：
- 模板
- Schema
- 示例文件
- 静态资源
```

---

# 4. SKILL.md 必须包含什么

`SKILL.md` 文件最前面必须包含 YAML Frontmatter。

最小格式：

```markdown
---
name: my-skill
description: Use when ...
---
```

至少需要：

```yaml
name
description
```

---

# 5. name 编写规范

例如 Skill 目录：

```text
code-review/
└── SKILL.md
```

那么：

```yaml
name: code-review
```

推荐满足：

```text
小写字母
数字
-
```

例如：

```yaml
name: code-review
name: creating-api-client
name: validating-json-schema
name: analyzing-logs
```

不要使用：

```yaml
name: CodeReview
name: code_review
name: code review
name: my--skill
```

推荐采用：

```text
动词 / 动名词 + 对象
```

例如：

```text
creating-skills
reviewing-code
analyzing-logs
generating-api-docs
validating-responses
debugging-services
```

相比：

```text
skill-helper
code-tool
debug-utils
```

前者更容易让 Agent 理解 Skill 的用途。

---

# 6. description 是 Skill 最重要的字段之一

Agent 通常不会一开始就读取所有 Skill 的完整内容。

大致流程：

```text
Agent 启动
    ↓
读取 Skills 的
name + description
    ↓
根据当前任务判断是否匹配
    ↓
匹配
    ↓
加载完整 SKILL.md
    ↓
需要时继续读取 references / scripts
```

因此：

> description 决定 Agent 是否能够发现你的 Skill。

一个 Skill 即使正文写得很好，如果 description 写得很差，也可能永远不会被自动调用。

---

# 7. description 推荐写法

推荐格式：

```yaml
description: Use when <明确的触发场景>.
```

例如：

```yaml
description: Use when creating new skills, editing existing skills, or verifying skills before deployment.
```

或者：

```yaml
description: Use when debugging intermittent test failures, race conditions, or timing-dependent behavior.
```

description 应该主要描述：

```text
什么时候使用
```

而不是：

```text
具体怎么执行
```

---

# 8. description 不推荐写法

不要把 Skill 的完整执行流程写进 description。

例如：

```yaml
description: Use when reviewing code. First inspect all files, then run tests, then analyze architecture, then generate a report.
```

这种写法的问题是：

Agent 可能只根据 description 执行，而没有真正读取 Skill 正文。

更好的写法：

```yaml
description: Use when reviewing code changes for correctness, maintainability, regressions, or implementation quality.
```

具体执行步骤放在正文。

---

# 9. Skill Discovery Optimization

Skill 编写时要考虑：

```text
Agent 怎么找到这个 Skill？
```

推荐在 description 和正文中覆盖 Agent 可能搜索的关键词。

例如测试问题：

```text
flaky
race condition
timeout
hang
freeze
timing
intermittent
```

例如 Debug Skill：

```text
debug
root cause
exception
error
failure
stack trace
logs
unexpected behavior
regression
```

Skill 名称和 description 本质上也承担类似：

```text
搜索索引
```

的作用。

---

# 10. SKILL.md 是否必须有固定章节

不必须。

下面已经是一个格式上合法的 Skill：

```markdown
---
name: my-skill
description: Use when ...
---

Do something.
```

Agent Skills 标准并没有强制要求：

```text
Overview
Inputs
Outputs
Instructions
Examples
```

这些章节。

但是为了稳定性和可维护性，建议自己制定统一模板。

---

# 11. 推荐的 SKILL.md 结构

推荐长期统一采用：

```markdown
---
name: <skill-name>
description: Use when <trigger conditions>.
---

# <Skill Name>

## Overview

## When to Use

## When Not to Use

## Inputs

## Instructions

## Output

## Validation

## Edge Cases

## Common Mistakes
```

不是每一个 Skill 都必须拥有所有章节。

简单 Skill 可以减少章节。

复杂 Skill 再逐步扩展。

---

# 12. 最简单 Skill 模板

目录：

```text
<skill-name>/
└── SKILL.md
```

`SKILL.md`：

```markdown
---
name: <skill-name>
description: <description>
---
```

这就是最小模板。

---

# 13. 推荐的标准 Skill 模板

```markdown
---
name: <skill-name>
description: Use when <trigger conditions>.
---

# <Skill Name>

## Overview

<核心能力说明>

## When to Use

<适用场景>

## When Not to Use

<不适用场景>

## Inputs

<需要哪些输入>

## Instructions

1. <Step 1>
2. <Step 2>
3. <Step 3>

## Output

<输出格式及要求>

## Validation

<完成任务前必须检查什么>

## Edge Cases

<特殊情况处理>

## Common Mistakes

<容易出现的问题>
```

---

# 14. 不要把所有内容都塞进 SKILL.md

不推荐：

```text
SKILL.md
3000 行
```

推荐：

```text
SKILL.md
├── 核心规则
├── 核心流程
└── reference 跳转

references/
├── api.md
├── business-rules.md
└── examples.md
```

例如：

```markdown
## Instructions

When evaluating API compatibility, read:

`references/api-compatibility.md`

When validating output, run:

`scripts/validate.py`
```

这样可以实现：

```text
Progressive Disclosure
```

即：

```text
先加载少量 Skill 信息
        ↓
任务匹配
        ↓
加载 SKILL.md
        ↓
真正需要时
        ↓
才加载 Reference
```

从而减少 Context Token 消耗。

---

# 15. 哪些内容应该放 references

适合：

```text
完整 API 文档
大量业务规则
几十个示例
错误码列表
Schema 文档
产品规范
技术规范
领域知识
```

例如：

```text
my-skill/
├── SKILL.md
└── references/
    ├── api.md
    ├── error-codes.md
    ├── schema.md
    └── business-rules.md
```

---

# 16. 哪些内容应该放 scripts

如果某个步骤可以由程序确定执行：

> 优先考虑 Script，而不是让 LLM 每次重新推理。

例如：

```text
JSON Validation
Schema Validation
CSV 处理
格式转换
Lint
文件扫描
代码检查
重复计算
数据解析
```

目录：

```text
scripts/
├── validate.py
├── parse.py
└── convert.py
```

Skill 中写：

```markdown
Run:

`scripts/validate.py`

before returning the final result.
```

---

# 17. Skill 应该保持多大

推荐：

```text
SKILL.md 尽量简洁
```

尤其是频繁被调用的 Skill。

原则：

```text
核心规则 → SKILL.md

大量知识 → references

确定性逻辑 → scripts
```

不要为了“看起来完整”不断扩充正文。

Skill 的目标不是成为一本书。

而是：

```text
让 Agent 稳定完成任务。
```

---

# 18. 编写 Skill 时最重要的设计问题

写 Skill 前应该回答：

### Trigger

```text
Agent 什么时候应该想到这个 Skill？
```

### Scope

```text
这个 Skill 负责什么？
不负责什么？
```

### Input

```text
执行前需要什么信息？
```

### Procedure

```text
Agent 应该按照什么顺序执行？
```

### Output Contract

```text
最终输出是什么结构？
```

### Validation

```text
怎么判断已经做对？
```

### Failure Handling

```text
信息缺失或者执行失败怎么办？
```

---

# 19. 一个 Skill 最好只负责一种明确能力

推荐：

```text
reviewing-java-code
```

而不是：

```text
super-developer-assistant
```

推荐：

```text
generating-api-tests
```

而不是：

```text
backend-everything
```

好的 Skill 通常满足：

```text
一个 Skill
=
一个明确能力
+
一个明确 Trigger
+
一个稳定 Procedure
+
一个明确 Output
+
一个 Validation
```

---

# 20. Skill 完成后如何检查格式

可以使用 Agent Skills validator：

```bash
skills-ref validate ./my-skill
```

主要检查：

```text
YAML frontmatter
name
description
目录结构
Agent Skills 格式
```

推荐：

```text
写 Skill
    ↓
Validate
    ↓
测试
    ↓
优化
```

---

# 21. Codex 中用于创建和优化 Skill 的工具

Codex 中可以使用：

```text
$skill-creator
```

它适合：

```text
创建 Skill
修改 Skill
Review Skill
优化 Skill
修复 Skill
优化 Trigger
优化 Description
调整目录
增加 Reference
增加 Script
```

例如：

```text
$skill-creator

Review ./my-skill.

Optimize it for:
- trigger precision
- Agent Skills compliance
- progressive disclosure
- token efficiency
- Codex and Claude Code compatibility
- deterministic output
- validation
- edge cases
```

---

# 22. Superpowers 中的 writing-skills

另外一个适合 Skill 工程化优化的 Skill：

```text
superpowers:writing-skills
```

用途：

```text
创建新 Skill
修改现有 Skill
测试 Skill
验证 Skill
部署前优化
```

其核心理念是：

```text
把 Skill 开发当成 TDD。
```

---

# 23. Skill 的 TDD 开发方式

普通软件开发：

```text
RED
测试失败

GREEN
实现功能

REFACTOR
优化实现
```

Skill 也可以采用：

```text
RED
在没有 Skill 的情况下
给 Agent 一个测试场景
观察 Agent 怎么犯错

        ↓

GREEN
编写 Skill
重新运行同样场景
检查 Agent 是否正确执行

        ↓

REFACTOR
寻找新的误解或漏洞
修改 Skill
重新测试
```

核心思想：

> 如果你从未观察 Agent 在没有 Skill 的情况下如何失败，就很难判断 Skill 到底解决了什么问题。

---

# 24. 不同 Skill 应该怎么测试

## Rules / Discipline Skill

例如：

```text
TDD
Code Review
Verification
Security Rules
```

重点测试：

```text
压力场景
边界情况
Agent 是否绕过规则
```

---

## Technique Skill

例如：

```text
Debugging
Root Cause Analysis
Log Analysis
```

重点测试：

```text
Agent 是否能正确应用方法
Agent 是否能处理不同场景
```

---

## Reference Skill

例如：

```text
API Reference
Business Rules
Framework Documentation
```

重点测试：

```text
Agent 能否找到正确信息
Agent 能否正确使用 Reference
```

---

## Output Skill

例如：

```text
生成 JSON
生成 API 文档
生成报告
```

重点测试：

```text
输出结构是否稳定
有没有遗漏字段
格式是否符合 Contract
```

---

# 25. 推荐的 Skill 开发流程

完整流程：

```text
确定 Skill 目标
        ↓
设计 Trigger
        ↓
设计测试 Case
        ↓
先让 Agent 在没有 Skill 时执行
        ↓
记录失败行为
        ↓
编写 SKILL.md
        ↓
skills-ref validate
        ↓
运行相同测试
        ↓
使用 $skill-creator Review
        ↓
使用 writing-skills 优化
        ↓
继续 Eval
        ↓
修复问题
        ↓
发布 / 安装
```

可以压缩成：

```text
Design
↓
Test
↓
Write
↓
Validate
↓
Evaluate
↓
Refactor
↓
Install
```

---

# 26. Codex 如何安装个人 Skill

Codex 用户级 Skill 推荐放在：

```text
~/.agents/skills/
```

例如：

```text
~/.agents/skills/
└── my-skill/
    └── SKILL.md
```

Windows 通常对应：

```text
%USERPROFILE%\.agents\skills\
```

例如：

```text
C:\Users\<username>\.agents\skills\my-skill\
```

---

# 27. Codex 项目级 Skill

项目中可以：

```text
project/
└── .agents/
    └── skills/
        └── my-skill/
            └── SKILL.md
```

优点：

```text
可以提交 Git
团队成员自动共享
Skill 与项目版本绑定
```

适合：

```text
项目开发规范
项目架构规则
项目测试流程
内部 API 使用规则
```

---

# 28. Codex Skill 如何调用

通常有两种方式。

## 自动调用

Agent 根据：

```text
name
description
当前任务
```

自行判断是否加载 Skill。

流程：

```text
Prompt
↓
匹配 Skill description
↓
加载 SKILL.md
↓
执行
```

---

## 显式调用

可以直接：

```text
$my-skill
```

然后继续描述任务。

例如：

```text
$reviewing-java-code

Review the changes in this branch.
```

---

# 29. Claude Code 如何安装个人 Skill

Claude Code 用户级 Skill：

```text
~/.claude/skills/
```

例如：

```text
~/.claude/skills/
└── my-skill/
    └── SKILL.md
```

---

# 30. Claude Code 项目级 Skill

目录：

```text
project/
└── .claude/
    └── skills/
        └── my-skill/
            └── SKILL.md
```

适合提交到 Git。

---

# 31. Claude Code 如何调用 Skill

Claude Code 可以根据 description 自动触发。

也可以显式调用：

```text
/my-skill
```

例如：

```text
/reviewing-java-code
```

---

# 32. 如何同时支持 Codex 和 Claude Code

推荐只维护一份 Skill Source。

例如：

```text
agent-skills/
├── reviewing-java-code/
├── debugging-api/
├── creating-tests/
└── generating-docs/
```

然后分别同步到：

```text
Codex
~/.agents/skills/

Claude Code
~/.claude/skills/
```

最终：

```text
agent-skills/
        │
        ├──────────────┐
        ↓              ↓
Codex             Claude Code
        ↓              ↓
.agents/skills    .claude/skills
```

这样只有：

```text
一个 Source of Truth
```

---

# 33. 推荐的 Git Repository 结构

如果以后会持续开发大量 Skills，推荐单独维护：

```text
my-agent-skills/
├── README.md
├── skills/
│   ├── skill-a/
│   │   ├── SKILL.md
│   │   ├── references/
│   │   └── scripts/
│   │
│   ├── skill-b/
│   │   └── SKILL.md
│   │
│   └── skill-c/
│       └── SKILL.md
│
└── scripts/
    ├── install-codex.sh
    ├── install-claude.sh
    └── validate-all.sh
```

甚至可以写统一安装脚本：

```text
Source Repository
       ↓
install
       ↓
┌──────────────┐
│              │
↓              ↓
Codex        Claude
```

---

# 34. 自己使用 Skill 不需要“发布”

如果只是自己使用：

```text
复制 Skill 目录
```

到：

Codex：

```text
~/.agents/skills/
```

Claude Code：

```text
~/.claude/skills/
```

即可。

不需要：

```text
npm publish
上传 Marketplace
创建 Package
注册服务
```

---

# 35. 团队内部发布

项目内部 Skill 最简单的发布方式：

```text
Git Repository
```

直接提交：

```text
.agents/skills/
```

或者：

```text
.claude/skills/
```

团队成员：

```text
git clone
```

之后即可使用。

---

# 36. 对外发布 Skill

目前比较通用的方式是：

```text
GitHub Repository
```

例如：

```text
github.com/xxx/my-agent-skills
```

里面：

```text
skills/
├── skill-a/
├── skill-b/
└── skill-c/
```

其他人可以：

```text
Clone
Copy
Install
```

---

# 37. Codex Skill Installer

Codex 可以通过：

```text
$skill-installer
```

安装 Skills。

因此公开 Skill Repository 后，可以进一步做成方便安装的 Skill 集合。

---

# 38. 更正式的分发方式：Plugin

如果未来不仅有 Skill，还包含：

```text
MCP
Scripts
Assets
多个 Skills
外部服务
```

可以考虑 Plugin。

例如：

```text
Plugin
├── Skill A
├── Skill B
├── Skill C
├── MCP
├── Scripts
└── Assets
```

这时候：

```text
Skill
```

是能力单位，

而：

```text
Plugin
```

更像一个完整能力包。

---

# 39. Skill 和 Plugin 的关系

可以简单理解：

```text
Skill
=
Agent 的一项能力说明书
```

而：

```text
Plugin
=
一整套 Agent 能力包
```

一个 Plugin 可以拥有：

```text
多个 Skills
+
Tools
+
MCP
+
Resources
+
Configuration
```

---

# 40. 推荐的 Skill 优化 Pipeline

以后编写完一个 Skill，可以固定执行：

```text
1. 编写 Skill

2. skills-ref validate

3. 使用 $skill-creator review

4. 设计真实测试 Prompt

5. 执行测试

6. 使用 superpowers:writing-skills

7. 修正 Trigger / Procedure / Output

8. 再次测试

9. 安装到 Codex / Claude Code

10. 实际使用过程中持续收集失败 Case
```

---

# 41. Skill Review Checklist

每次完成 Skill 后，可以检查：

## Structure

- [ ] 是否存在 `SKILL.md`
- [ ] `name` 是否正确
- [ ] `description` 是否存在
- [ ] Skill 目录名称是否规范
- [ ] 大量内容是否拆进 `references/`
- [ ] 确定性逻辑是否应该放入 `scripts/`

## Discovery

- [ ] description 是否明确说明 When to Use
- [ ] 是否包含用户可能使用的关键词
- [ ] 是否避免把完整执行流程写进 description
- [ ] Skill 名是否清楚表达能力

## Scope

- [ ] Skill 是否只有一个主要职责
- [ ] 是否明确 When to Use
- [ ] 是否明确 When Not to Use
- [ ] 是否存在和其他 Skill 的职责冲突

## Procedure

- [ ] Agent 是否知道第一步做什么
- [ ] 执行顺序是否明确
- [ ] 是否明确什么时候读取 reference
- [ ] 是否明确什么时候执行 script

## Output

- [ ] 输出格式是否明确
- [ ] 必需字段是否明确
- [ ] 是否定义完成标准

## Validation

- [ ] Agent 返回结果之前是否需要检查
- [ ] 是否有 Script 可以做确定性 Validation
- [ ] 是否覆盖 Edge Cases

## Testing

- [ ] 是否有真实测试 Case
- [ ] 是否观察过没有 Skill 时 Agent 的行为
- [ ] 是否观察过使用 Skill 后 Agent 的行为
- [ ] 是否针对失败行为修改过 Skill

---

# 42. 最推荐遵守的十条 Skill 编写原则

## Rule 1

```text
一个 Skill 只解决一个核心问题。
```

## Rule 2

```text
description 负责 Trigger，不负责完整 Workflow。
```

## Rule 3

```text
Skill 名称应该表达 Agent 正在做什么。
```

## Rule 4

```text
SKILL.md 保持精简。
```

## Rule 5

```text
大量知识放 references。
```

## Rule 6

```text
确定性工作放 scripts。
```

## Rule 7

```text
告诉 Agent 什么时候读取哪个 reference。
```

## Rule 8

```text
Output 必须有明确 Contract。
```

## Rule 9

```text
任务完成前应该有 Validation。
```

## Rule 10

```text
Skill 必须通过真实 Prompt 测试，而不只是人工阅读。
```

---

# 43. 推荐最终工程模式

长期开发 Skills 时推荐形成：

```text
Git Repository
      │
      ↓
Agent Skills Source
      │
      ├── Skill A
      ├── Skill B
      ├── Skill C
      └── Skill D
      │
      ↓
Validation
      │
      ↓
Evaluation
      │
      ↓
Install Script
      │
 ┌────┴────┐
 ↓         ↓
Codex    Claude Code
```

其中：

```text
Git Repository
```

作为唯一：

```text
Source of Truth
```

避免同时手工维护：

```text
~/.agents/skills
```

和：

```text
~/.claude/skills
```

中的两个版本。

---

# 44. 一句话总结

一套质量较高的 Agent Skill，可以概括为：

```text
Good Skill
=
Precise Trigger
+
Clear Scope
+
Stable Procedure
+
Progressive Disclosure
+
Deterministic Scripts
+
Explicit Output Contract
+
Validation
+
Real-world Evaluation
```

而推荐开发流程是：

```text
Design
↓
Baseline Test
↓
Write SKILL.md
↓
Validate
↓
$skill-creator Review
↓
writing-skills Evaluation
↓
Refactor
↓
Install
↓
Real-world Feedback
```

最终目标不是：

> 写一份很长、很详细的 Prompt。

而是：

> 让 Agent 在正确的时间发现这个 Skill，并以稳定、可重复、可验证的方式完成某一种任务。