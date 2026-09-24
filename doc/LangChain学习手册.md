# LangChain 零基础学习手册

> **适用版本**：LangChain v1.0+（2024 年 10 月后的新 API）  
> **目标读者**：完全没有 LangChain 基础的 Python 开发者  
> **学完你将能够**：独立构建 RAG 应用、Agent 智能体、结构化数据提取等 LLM 应用

---

## 目录

1. [前置知识速览](#1-前置知识速览)
2. [版本注意事项](#2-版本注意事项)
3. [学习路线总览](#3-学习路线总览)
4. [基础概念深度解析](#4-基础概念深度解析)
5. [环境安装与配置](#5-环境安装与配置)
6. [核心教学](#6-核心教学)
   - Step 1：第一次调用大模型
   - Step 2：Prompt 模板——给模型写"标准信函"
   - Step 3：输出解析——把模型回复变成程序能用的数据
   - Step 4：LCEL 管道——用 `|` 把零件串起来
   - Step 5：结构化输出——让模型严格返回 JSON
   - Step 6：RAG 检索增强生成——让模型"开卷考试"
   - Step 7：Tool 工具——给模型装上"手和脚"
   - Step 8：Agent 智能体——让模型自己决定用什么工具
   - Step 9：LangGraph——构建复杂多步骤工作流
7. [实战项目](#7-实战项目)
8. [调试与评估](#8-调试与评估)
9. [进阶方向](#9-进阶方向)
10. [30 天学习计划](#10-30-天学习计划)

---

## 1. 前置知识速览

> 这一章帮你在 10 分钟内建立对核心术语的直觉。后续正文中首次出现的专有术语会用括号再次简短解释。

### 1.1 Token（词元）

**生活类比**：想象你在用一本特殊的字典拆句子。英文里 "unhappiness" 可能被拆成 "un" + "happi" + "ness" 三个词元；中文里 "今天天气不错" 可能被拆成 "今天" + "天气" + "不" + "错" 四个词元。

**技术定义**：Token 是大语言模型处理文本的最小单位。模型不是逐字或逐词读文本，而是先把文本切分成 token，再逐个理解。

**为什么你需要关心它**：
- 大模型按 token 数量计费（输入 + 输出都算钱）
- 每个模型有"上下文窗口"限制（比如 128K token），超过了模型就"记不住"前面的内容
- 1 个英文单词 ≈ 1.3 token；1 个中文字 ≈ 1.5-2 token

### 1.2 Embedding（向量嵌入）

**生活类比**：想象你有一个超级坐标系。每段文字在这个坐标系里都有一个"位置"——意思相近的文字，位置也相近。比如"猫"和"猫咪"会挨得很近，而"猫"和"汽车"则相隔很远。

**技术定义**：Embedding 是把文本（词、句子、段落）转换为一串数字（向量，如 `[0.12, -0.45, 0.78, ...]`，通常有 768 或 1536 个数字）。这些数字编码了文本的语义信息。

**为什么需要它**：计算机不懂文字，但懂数学。有了 embedding，计算机就能用"计算两个向量的距离"来判断"两段文字是否语义相近"。这是 RAG 搜索的基础。

### 1.3 向量数据库（Vector Database）

**生活类比**：普通数据库像一本按拼音排序的字典——你得知道确切的词才能查到。向量数据库更像一个"按意思分区"的图书馆——你说"我想找关于太空冒险的书"，它能把《星际穿越》《火星救援》都找出来，即使这些书名里没有"太空"两个字。

**技术定义**：专门存储和检索 embedding 向量的数据库，支持"相似度搜索"（找出与目标向量最接近的 N 个向量）。常见产品：Chroma、FAISS、Pinecone、Milvus。

### 1.4 Prompt Engineering（提示词工程）

**生活类比**：就像给一个新来的实习生布置任务——你说"帮我写个东西"，他大概率写不出你要的；但你说"帮我写一封给客户的道歉邮件，语气要诚恳但不卑微，200 字以内，重点解释延迟发货的原因"，他就能交出让满意的活儿。

**技术定义**：通过精心设计输入给大模型的文本（prompt），来引导模型产生期望输出的技术。包括角色设定、示例提供（few-shot）、格式要求等策略。

### 1.5 Function Calling / Tool Use（函数调用 / 工具使用）

**生活类比**：大模型本身就像一个知识渊博但被关在房间里的人——它能聊天、写作、分析，但不能上网查天气、不能帮你订机票、不能读你电脑上的文件。Function Calling 就是给这个人一部电话，让他可以"打电话"叫外卖（调用外部工具）。

**技术定义**：让大模型在对话中识别出"我需要调用某个外部函数"，然后输出结构化的调用请求（函数名 + 参数），由你的程序实际执行函数后，再把结果告诉模型。

### 1.6 Structured Output（结构化输出）

**生活类比**：普通对话就像让人"随便聊聊"，结构化输出就像给人一张表格让他"按格式填"。

**技术定义**：强制大模型的回复严格遵循你定义的数据格式（通常是 JSON Schema 或 Pydantic 模型），而不是自由文本。这样你的程序可以直接解析返回值，无需担心格式错误。

### 1.7 RAG（检索增强生成，Retrieval-Augmented Generation）

**生活类比**：大模型本身像一个"闭卷考试"的学生——只能凭记忆回答，可能记错或过时。RAG 就是让模型变成"开卷考试"——先从资料库里翻到相关段落，再结合这些段落来回答。

**技术定义**：RAG = 检索（Retrieval）+ 生成（Generation）。流程：用户提问 → 在向量数据库中搜索相关文档片段 → 把搜索到的片段 + 用户问题一起发给大模型 → 大模型基于这些"参考资料"生成回答。

**为什么需要它**：
- 大模型的知识有截止日期，RAG 可以接入最新数据
- 大模型不了解你的私有数据（公司文档、产品手册），RAG 可以
- 减少"幻觉"（模型编造不存在的信息）

### 1.8 Agent（智能体）

**生活类比**：普通的大模型对话像"一问一答"的客服。Agent 更像一个有自主判断力的助理——你告诉他"帮我安排下周的出差"，他会自己决定：先查日历看哪天有空 → 搜航班信息 → 比价 → 查酒店 → 最终给你一个完整方案。

**技术定义**：Agent = 大模型（作为"大脑"进行推理和决策）+ 工具集（可执行的动作）+ 循环逻辑（观察结果 → 思考 → 再行动，直到任务完成）。

### 1.9 LLM vs Chat Model

**生活类比**：LLM（大语言模型）像一台"文字接龙机"——给它一段文字，它续写下一段。Chat Model（对话模型）像一个"聊天伙伴"——它理解多轮对话的上下文，知道谁说了什么。

**技术定义**：
- LLM（如早期的 GPT-3）：输入纯文本，输出纯文本，没有对话概念
- Chat Model（如 GPT-4、Claude）：输入消息列表（system/user/assistant），输出 assistant 消息

**在 LangChain 中**：现在几乎所有教程都使用 Chat Model，旧的纯文本 LLM 接口已逐步弃用。

---

## 2. 版本注意事项

### ⚠️ 极其重要：LangChain 在 2024 年经历了重大 API 重构

| 旧 API（已废弃 ❌） | 新 API（v1.0+ ✅） | 说明 |
|---|---|---|
| `from langchain.chat_models import ChatOpenAI` | `from langchain_openai import ChatOpenAI` 或 `init_chat_model()` | 模型类移入独立包 |
| `LLMChain(llm=..., prompt=...)` | `prompt \| llm \| parser`（LCEL 管道） | 链式调用用 `\|` 运算符 |
| `initialize_agent(tools, llm)` | `create_react_agent()` 或 LangGraph | Agent 构建方式彻底改变 |
| `AgentExecutor(agent=..., tools=...)` | LangGraph `StateGraph` | 执行引擎重构 |
| `ConversationBufferMemory` | LangGraph `Checkpointer` | 记忆管理新方案 |
| `llm.predict()` / `llm()` | `llm.invoke()` / `llm.stream()` | 统一调用接口 |

### 网上教程的"坑"

如果你搜索 LangChain 教程，**80% 以上的中文教程使用的是旧 API**。判断方法：
- 看到 `LLMChain`、`initialize_agent`、`AgentExecutor` → 旧教程，跳过
- 看到 `init_chat_model`、`create_react_agent`、LCEL `|` 管道 → 新教程，可以参考

### 本手册使用的包版本

```
langchain >= 0.3.0
langchain-core >= 0.3.0
langchain-openai >= 0.2.0
langgraph >= 0.2.0
langsmith >= 0.1.0
```

---

## 3. 学习路线总览

```
┌─────────────────────────────────────────────────────────┐
│  Phase 1: 基础调用（Step 1-3）                            │
│  学会：调模型 → 写 Prompt → 解析输出                       │
├─────────────────────────────────────────────────────────┤
│  Phase 2: 组合能力（Step 4-6）                            │
│  学会：LCEL 管道 → 结构化输出 → RAG 检索增强               │
├─────────────────────────────────────────────────────────┤
│  Phase 3: 自主智能（Step 7-9）                            │
│  学会：工具定义 → Agent 决策 → LangGraph 工作流            │
├─────────────────────────────────────────────────────────┤
│  Phase 4: 实战落地                                        │
│  项目：智能客服 / 文档问答 / 数据分析助手                   │
└─────────────────────────────────────────────────────────┘
```

**建议学习节奏**：每个 Step 花 1-2 小时，包含阅读 + 动手敲代码 + 完成"试一试"练习。

---

## 4. 基础概念深度解析

### 4.1 Runnable——LangChain 的"万能接口"

**生活类比**：想象一个标准插座。不管你插手机充电器、台灯还是电脑，插座的接口都是一样的（两个孔 + 一个地线）。在 LangChain 里，Runnable 就是这个"标准插座"——所有组件（模型、Prompt 模板、输出解析器、工具）都实现了这个接口，所以它们可以随意组合。

**技术定义**：`Runnable` 是 LangChain 的核心协议（Protocol/接口），任何实现了它的对象都具备以下标准方法：

| 方法 | 作用 | 类比 |
|------|------|------|
| `.invoke(input)` | 处理单个输入，返回单个输出 | 寄一封信，等一个回信 |
| `.stream(input)` | 处理单个输入，逐步返回输出（流式） | 打电话，对方一个字一个字说 |
| `.batch([inputs])` | 并行处理多个输入 | 同时寄多封信 |
| `.ainvoke(input)` | 异步版 invoke | 用 await 寄信 |
| `.astream(input)` | 异步版 stream | 用 await 打电话 |

**为什么需要它**：如果每个组件都有自己的调用方式（有的叫 `.run()`，有的叫 `.predict()`，有的叫 `.generate()`），组合它们就是噩梦。统一的 Runnable 接口让任何两个组件都能无缝衔接。

**最简代码示例**：

```python
from langchain_openai import ChatOpenAI

# 创建一个聊天模型实例——它就是一个 Runnable
model = ChatOpenAI(model="gpt-4o-mini")

# invoke() 是最常用的方法：发送一条消息，获得一个回复
response = model.invoke("用一句话解释什么是 Python")

# response 的类型是 AIMessage（AI 消息对象）
print(response.content)
# 预期输出类似：
# "Python 是一种高级编程语言，以简洁易读的语法著称，广泛用于 Web 开发、数据分析和人工智能领域。"
```

**试一试 🎯**：
1. 把 `.invoke()` 换成 `.stream()`，观察输出有什么不同（提示：你会看到文字一个个蹦出来）
2. 用 `.batch()` 同时问 3 个不同的问题，看看返回值的结构

---

### 4.2 LCEL——用 `|` 管道把组件串起来

**生活类比**：想象一条工厂流水线。原材料从一头进去，经过"切割站 → 打磨站 → 喷漆站 → 质检站"，最后成品从另一头出来。LCEL 就是让你用 `|` 符号搭建这样的流水线：

```
原材料 | 切割 | 打磨 | 喷漆 | 质检 → 成品
prompt | model | parser              → 结构化数据
```

**技术定义**：LCEL（LangChain Expression Language，LangChain 表达式语言）是一种声明式的链构建语法。用 `|` 运算符（Python 的 `__or__` 魔法方法）把多个 Runnable 连接成一个 `RunnableSequence`。

**为什么需要它**：
- **可读性**：一眼看出数据流向，从左到右
- **自动获得流式/批量/异步能力**：只要管道中每个组件都是 Runnable，整条管道自动支持 `.stream()`、`.batch()`、`.ainvoke()`
- **自动重试和错误处理**：框架层面统一处理

**如果不这样写会怎样**：你就得手动嵌套调用，代码变成"洋葱式"嵌套：

```python
# ❌ 不用 LCEL 的写法——难以阅读和维护
prompt_value = prompt_template.invoke({"topic": "Python"})
model_output = model.invoke(prompt_value)
result = output_parser.invoke(model_output)

# ✅ 用 LCEL 的写法——清晰的数据流
chain = prompt_template | model | output_parser
result = chain.invoke({"topic": "Python"})
```

**最简代码示例**：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 组件1：Prompt 模板（定义发给模型的"信纸格式"）
prompt = ChatPromptTemplate.from_template("用 {style} 的风格解释 {topic}")

# 组件2：模型（处理请求的"大脑"）
model = ChatOpenAI(model="gpt-4o-mini")

# 组件3：输出解析器（从模型回复中提取纯文本）
parser = StrOutputParser()

# 用 | 把三个组件串成一条链
chain = prompt | model | parser

# 调用这条链——传入模板需要的变量
result = chain.invoke({"style": "像对5岁小孩说话", "topic": "量子力学"})
print(result)
# 预期输出类似：
# "量子力学就像是一个超级神奇的魔法世界！在这个世界里，
#  一个小球可以同时在这里和那里，就像你同时在看电视和在睡觉一样..."
```

**试一试 🎯**：
1. 在链的末尾再加一个组件：`chain = prompt | model | parser | (lambda x: x.upper())`，看看会发生什么
2. 把 `invoke` 换成 `stream`，用 `for chunk in chain.stream(...)` 循环打印，体验流式输出

---

### 4.3 Chain（链）——组合模式的统称

**生活类比**：如果 Runnable 是"单个工位"，那 Chain 就是"一条完整的流水线"。在 LangChain 里，任何时候你用 `|` 把多个 Runnable 串起来，得到的就是一条 Chain（内部类型叫 `RunnableSequence`）。

**技术定义**：Chain 不是一个具体的类，而是一个概念——任何由多个 Runnable 组合而成的可执行单元都可以称为"链"。常见的链模式：

| 链类型 | 结构 | 用途 |
|--------|------|------|
| 简单链 | `prompt \| model \| parser` | 一问一答 |
| 路由链 | `prompt \| model \| router \| branch_a/branch_b` | 根据条件走不同分支 |
| 并行链 | `{"joke": prompt_a \| model, "poem": prompt_b \| model}` | 同时生成多种输出 |
| RAG 链 | `retriever \| format_docs \| prompt \| model \| parser` | 检索 + 生成 |

---

### 4.4 Prompt Template（提示词模板）

**生活类比**：想象你开了一家餐厅，菜单上每道菜的"点单话术"都有固定格式："我要一份 [辣度] 的 [菜名]，不要 [忌口]。" Prompt Template 就是这个"填空模板"。

**技术定义**：预定义的文本模板，包含可变占位符（用 `{变量名}` 标记），运行时填入具体值，生成最终发给模型的 prompt。

**为什么需要它**：
- 复用：同一个模板可以用不同参数调用无数次
- 维护：修改 prompt 只需改模板，不用改业务代码
- 清晰：代码阅读者一眼看出"哪些是固定的指令，哪些是动态的输入"

---

### 4.5 Output Parser（输出解析器）

**生活类比**：模型返回的原始回复是一个"信封"（AIMessage 对象），里面有信纸（content）、邮戳（usage_metadata）等。Output Parser 就是"拆信员"——按你的要求从信封里取出你要的东西：纯文本？JSON 数据？特定格式的列表？

**技术定义**：Output Parser 接收模型的输出（AIMessage），将其转换为程序可以直接使用的数据类型（字符串、字典、Pydantic 对象等）。

**常用解析器**：

| 解析器 | 输入 | 输出 | 用途 |
|--------|------|------|------|
| `StrOutputParser` | AIMessage | `str` | 只要纯文本 |
| `JsonOutputParser` | AIMessage | `dict` | 解析 JSON |
| `PydanticOutputParser` | AIMessage | Pydantic 对象 | 强类型解析 |

---

### 4.6 Memory（记忆）与对话历史

**生活类比**：没有记忆的大模型就像一个"金鱼脑"的客服——每次对话都从零开始，不记得你之前说过什么。Memory 就是给这个客服一个"笔记本"，记录之前所有对话，下次聊天前先翻翻笔记。

**技术定义**：Memory 是在多轮对话中保存和传递历史消息的机制。在 LangChain v1.0+ 中，推荐使用 LangGraph 的 `Checkpointer` 来管理对话状态（而不是旧的 `ConversationBufferMemory`）。

**新方案（LangGraph Checkpointer）**：
```python
from langgraph.checkpoint.memory import MemorySaver

# 创建一个"记忆保存器"
checkpointer = MemorySaver()

# 在 graph 中使用它
graph = builder.compile(checkpointer=checkpointer)

# 每次调用时传入 thread_id，系统自动记住这个对话的历史
config = {"configurable": {"thread_id": "user-001"}}
graph.invoke({"messages": [("user", "我叫小明")]}, config)
graph.invoke({"messages": [("user", "我叫什么？")]}, config)
# 模型会回答："你叫小明"
```

---

### 4.7 Retriever（检索器）

**生活类比**：想象你在一个巨大的图书馆里找资料。Retriever 就是那个"图书管理员"——你告诉他你要找什么主题，他从书架上把最相关的几本书抽出来递给你。

**技术定义**：Retriever 是一个接收查询字符串（query）、返回相关文档列表的组件。最常见的实现是基于向量相似度搜索：把查询转为 embedding → 在向量数据库中找最接近的文档 embedding → 返回对应文档。

**工作流程**：
```
用户问题 "Python 怎么读文件？"
    ↓
Retriever 把问题转为向量
    ↓
在向量数据库中搜索最相似的文档片段
    ↓
返回: ["Python 中可以用 open() 函数...", "with 语句可以自动关闭文件..."]
    ↓
这些片段被塞进 Prompt，交给大模型参考回答
```

---

### 4.8 Callback（回调）

**生活类比**：想象你在餐厅点了一份大餐。Callback 就像服务员在烹饪过程中不断向你汇报："菜开始炒了"、"已经装盘了"、"马上上桌"。你不需要一直盯着厨房，但可以随时了解进度。

**技术定义**：Callback 是 LangChain 的事件钩子系统。你可以在链的执行过程中注册回调函数，当特定事件发生时（模型开始生成、生成完成、出错等）自动触发。

**主要用途**：
- **日志记录**：记录每次调用的输入输出
- **流式传输**：实时把 token 推送到前端
- **监控追踪**：配合 LangSmith 做性能分析
- **调试**：看到链内部每一步的中间结果

---

### 4.9 Tool（工具）

**生活类比**：大模型就像一个被困在玻璃房间里的超级大脑——它能思考、能说话，但不能动手做事。Tool 就是你递给它的"工具包"：一个计算器、一个搜索引擎、一个数据库查询器...让它的大脑能够"延伸"到现实世界。

**技术定义**：Tool 是一个带有名称、描述和输入 schema 的函数包装器。大模型通过阅读 Tool 的描述来决定何时调用它、传什么参数。LangChain 会用模型返回的参数实际执行函数，再把结果传回给模型。

**一个工具由三部分组成**：
1. **名称**（name）：模型用来"点名"调用哪个工具
2. **描述**（description）：告诉模型"这个工具能干什么，什么时候该用"
3. **输入 Schema**（args_schema）：告诉模型"调用时需要传什么参数"

---

### 4.10 Agent（智能体）

**生活类比**：如果说 Chain（链）是"写死的流水线"——数据固定地从 A 流到 B 再到 C，那 Agent 就是"有自主判断力的工人"——它会看看当前情况，自己决定接下来该用哪个工具、该走哪条路。

**技术定义**：Agent = LLM（推理引擎）+ Tools（可用工具）+ Loop（循环执行）。Agent 的核心逻辑是 ReAct 模式：
```
Thought: 我需要查一下今天的天气
Action: 调用 weather_tool(city="北京")
Observation: 今天北京晴，25°C
Thought: 用户还问了穿什么，我根据天气给建议
Action: 返回最终答案
```

**Chain vs Agent 的区别**：
| | Chain（链） | Agent（智能体） |
|---|---|---|
| 执行路径 | 固定、预定义 | 动态、模型决定 |
| 适用场景 | 流程明确的简单任务 | 需要判断和灵活应对的复杂任务 |
| 可预测性 | 高 | 较低（但更灵活） |
 成本 | 低（通常 1 次模型调用） | 高（可能多次循环调用） |

---

## 5. 环境安装与配置

### 5.1 前提条件

在开始之前，确保你的电脑上已经安装了：
- **Python 3.9 或更高版本**（推荐 3.11+）
- **pip**（Python 包管理器，通常随 Python 一起安装）
- 一个代码编辑器（推荐 VS Code 或 PyCharm）

检查 Python 版本：
```bash
python --version
# 预期输出类似：Python 3.11.5
```

### 5.2 创建虚拟环境（强烈推荐）

**为什么要用虚拟环境**：想象你有一间专门的工作室，里面只放做当前项目需要的工具。虚拟环境就是这间工作室——它隔离了不同项目的依赖，避免"A 项目需要 1.0 版，B 项目需要 2.0 版"的冲突。

```bash
# 创建虚拟环境（会在当前目录下生成一个 .venv 文件夹）
python -m venv .venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 激活成功后，命令行前面会出现 (.venv) 标识
```

### 5.3 安装 LangChain 相关包

```bash
# 核心包：LangChain 本体 + OpenAI 集成
pip install langchain langchain-openai langchain-core

# LangGraph：用于构建 Agent 和复杂工作流
pip install langgraph

# 社区包：包含各种第三方集成（Chroma 向量库等）
pip install langchain-community

# RAG 相关：文本分割器
pip install tiktoken

# 向量数据库（本地运行，无需注册云服务）
pip install chromadb

# 结构化输出需要的数据验证库
pip install pydantic

# 可选：LangSmith 追踪和调试
pip install langsmith
```

### 5.4 配置 API Key

大模型不是免费午餐——你需要一个 API Key（类似"门票"）来调用它们。

**方式一：环境变量（推荐）**

```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-你的密钥"

# macOS/Linux
export OPENAI_API_KEY="sk-你的密钥"
```

**方式二：.env 文件（项目级别）**

在项目根目录创建 `.env` 文件：
```
OPENAI_API_KEY=sk-你的密钥
LANGSMITH_API_KEY=ls-你的密钥（可选）
LANGSMITH_TRACING=true（可选，开启调试追踪）
```

然后在代码中加载：
```python
from dotenv import load_dotenv
load_dotenv()  # 自动读取 .env 文件中的环境变量
```

> ⚠️ **安全提醒**：永远不要把 API Key 直接写在代码里或提交到 Git！把 `.env` 加入 `.gitignore`。

### 5.5 验证安装是否成功

创建文件 `test_setup.py`：

```python
"""验证 LangChain 环境是否安装成功"""
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

# 创建模型实例
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 发送一条测试消息
response = model.invoke("请回复'安装成功'四个字")

print(response.content)
# 预期输出：安装成功
```

运行：
```bash
python test_setup.py
```

如果看到"安装成功"，恭喜你，环境搭建完毕！

### 5.6 关于模型选择

本手册默认使用 OpenAI 的模型，但 LangChain 支持几乎所有主流大模型。如果你想用其他模型：

```python
# 方式一：直接实例化特定模型类
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o-mini")

# 方式二：使用 init_chat_model（推荐，更灵活）
from langchain.chat_models import init_chat_model

# OpenAI
model = init_chat_model("gpt-4o-mini", model_provider="openai")

# Anthropic Claude
model = init_chat_model("claude-3-5-sonnet-latest", model_provider="anthropic")

# 本地 Ollama 模型（免费！）
model = init_chat_model("llama3", model_provider="ollama")
```

**`init_chat_model()` 方法详解**：
- **作用**：根据模型名称和提供商，自动创建对应的聊天模型实例
- **参数**：
  - 第一个参数（`model`）：模型名称字符串，如 `"gpt-4o-mini"`
  - `model_provider`：模型提供商，如 `"openai"`、`"anthropic"`、`"ollama"`
  - `temperature`：控制随机性，0 = 确定性最高，1 = 最随机（可选）
  - 其他关键字参数会传递给底层模型类
- **返回值**：一个实现了 Runnable 接口的聊天模型对象

---

## 6. 核心教学

> 以下每个 Step 都遵循：**概念讲解 → 代码实现 → 预期输出 → 试一试练习** 的结构。
> 建议按顺序学习，每完成一个 Step 的练习再进入下一个。

---

### Step 1：第一次调用大模型

#### 1.1 本节目标

学会用最少的代码让大模型回答你的问题。这是所有 LangChain 应用的起点。

#### 1.2 核心概念：ChatModel

你可以把 ChatModel 想象成一个"远程顾问"：
- 你发一条消息给它（invoke）
- 它思考后回复你一条消息
- 每次调用都是独立的（它不会自动记住之前的对话）

#### 1.3 基础调用

```python
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

# === 创建模型实例 ===
# model: 指定使用哪个模型（gpt-4o-mini 便宜且够用于学习）
# temperature: 0 表示"尽量确定性回答"，1 表示"更有创意/随机"
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# === 最简单的调用：传入一个字符串 ===
# invoke() 接收一个字符串时，会自动包装成 HumanMessage
response = model.invoke("什么是机器学习？请用一句话回答。")

# === 查看返回值 ===
# response 是一个 AIMessage 对象，不只是一个字符串！
print(f"类型: {type(response)}")
print(f"内容: {response.content}")
print(f"Token 用量: {response.usage_metadata}")
```

**预期输出**：
```
类型: <class 'langchain_core.messages.ai.AIMessage'>
内容: 机器学习是人工智能的一个分支，它让计算机能够从数据中自动学习模式和规律，而无需被明确编程。
Token 用量: {'input_tokens': 22, 'output_tokens': 38, 'total_tokens': 60}
```

#### 1.4 使用消息列表（多角色对话）

**为什么需要消息列表**：大模型的对话格式不只是一条文本——它区分"系统指令"、"用户消息"和"AI 历史回复"。你可以把它想象成一个剧本：
- `system`：导演给演员的角色设定（"你是一个专业的营养师"）
- `human`：用户的台词
- `ai`：AI 的台词（用于提供对话历史）

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# === 使用消息列表调用 ===
messages = [
    # SystemMessage: 设定 AI 的"人设"和行为规则
    SystemMessage(content="你是一个专业的Python教师，回答要简洁，代码示例要有注释。"),
    # HumanMessage: 用户的提问
    HumanMessage(content="Python 中列表和元组有什么区别？"),
]

response = model.invoke(messages)
print(response.content)
```

**预期输出**：
```
列表(list)和元组(tuple)的主要区别：
1. 可变性：列表可修改，元组不可修改
2. 语法：列表用[]，元组用()
3. 性能：元组略快，占用内存更少
4. 用途：列表用于同类数据集合，元组用于异构数据记录

```python
# 列表 - 可以增删改
fruits = ["苹果", "香蕉"]
fruits.append("橙子")  # OK

# 元组 - 创建后不可修改
point = (3, 4)
# point[0] = 5  # 报错！TypeError
```
```

#### 1.5 流式输出（逐字打印效果）

**为什么需要流式**：如果模型回复很长，用 `invoke()` 要等全部生成完才能看到结果（可能等 10+ 秒）。`stream()` 让你像看打字一样实时看到内容逐步出现——这就是 ChatGPT 网页版的"逐字打印"效果。

```python
# === stream() 返回一个迭代器，每次产出一小块文本 ===
print("AI 正在思考：", end="", flush=True)

for chunk in model.stream("写一首关于编程的四行小诗"):
    # chunk 也是 AIMessage 对象，但 content 只有一小段
    print(chunk.content, end="", flush=True)

print()  # 最后换行
```

**预期输出**（文字会逐步出现，而不是等全部完成）：
```
AI 正在思考：键盘声声夜色深，
Bug 潜藏在代码林。
一朝调试通幽径，
满屏绿灯慰我心。
```

#### 1.6 `invoke()` vs `stream()` vs `batch()` 对比

```python
# invoke(): 一问一答，返回完整的 AIMessage
result = model.invoke("1+1等于几？")
print(result.content)  # "1+1等于2"

# stream(): 一问一答，但逐步返回（适合实时展示）
for chunk in model.stream("1+1等于几？"):
    print(chunk.content, end="")  # 逐字打印

# batch(): 多问多答，并行处理（适合批量任务）
results = model.batch(["1+1等于？", "2+2等于？", "3+3等于？"])
for r in results:
    print(r.content)
# 输出三行：2、4、6（顺序可能和输入一致）
```

#### 试一试 🎯

1. **基础练习**：修改 temperature 为 0.9，多次运行同一个问题，观察每次回答是否不同
2. **消息列表**：在 messages 中加入一条 AIMessage 作为"对话历史"，然后问一个需要上下文才能回答的问题。例如：先让 AI 说"我最喜欢的颜色是蓝色"，然后问"我喜欢什么颜色？"
3. **流式输出**：写一个程序，用 stream() 让模型写一个 200 字的故事，并在终端中实时显示

---

### Step 2：Prompt 模板——给模型写"标准信函"

#### 2.1 本节目标

学会使用 Prompt Template 来管理你的提示词，而不是每次都在代码里硬编码字符串。

#### 2.2 为什么需要 Prompt Template？

想象你是一个公司的客服主管，你要给 100 个客服写工作指令。你不会每次都从头写——你会做一个模板：

```
亲爱的客服 {名字}：
你负责的产品线是 {产品}。
当客户问到 {常见问题} 时，请这样回答：{标准答案}
```

Prompt Template 的作用完全一样：**把 prompt 中不变的部分固定下来，把需要变化的部分变成参数**。

#### 2.3 ChatPromptTemplate 基本用法

```python
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# === 方式一：from_template（单条消息模板）===
# 适合简单场景：只有一条用户消息
prompt = ChatPromptTemplate.from_template(
    "请将以下文本翻译成{language}：\n\n{text}"
)

# invoke() 传入字典，key 对应模板中的 {变量名}
# 返回值是 ChatPromptValue 对象（包含格式化好的消息列表）
formatted = prompt.invoke({"language": "英文", "text": "今天天气真好"})
print(formatted.to_messages())
# 预期输出：[HumanMessage(content='请将以下文本翻译成英文：\n\n今天天气真好')]

# === 方式二：from_messages（多角色消息模板）===
# 适合复杂场景：需要 system + human + 历史对话
prompt = ChatPromptTemplate.from_messages([
    # ("角色", "模板内容") 的元组列表
    ("system", "你是一位{profession}，用{style}的风格回答问题。"),
    ("human", "{question}"),
])

formatted = prompt.invoke({
    "profession": "儿童科学老师",
    "style": "生动有趣",
    "question": "为什么天空是蓝色的？"
})

# 把格式化后的消息发给模型
response = model.invoke(formatted)
print(response.content)
```

**预期输出**：
```
你知道吗？天空之所以是蓝色的，是因为阳光里藏着彩虹的所有颜色！
当阳光穿过大气层时，蓝色光因为"个子小"（波长短），
特别容易被空气分子"弹来弹去"（散射），
所以无论你朝哪个方向看，都能看到被弹得到处都是的蓝光！
这就像在操场上扔了一大把蓝色弹力球，到处都是！
```

#### 2.4 `ChatPromptTemplate.from_messages()` 方法详解

| 参数 | 说明 |
|------|------|
| 输入 | 一个列表，每个元素是 `("role", "template_string")` 的元组 |
| role 可选值 | `"system"`、`"human"`、`"ai"`、`"placeholder"` |
| template_string | 支持 `{变量名}` 占位符 |
| 返回值 | `ChatPromptTemplate` 对象，调用 `.invoke(dict)` 后返回 `ChatPromptValue` |

#### 2.5 加入 Few-shot 示例（给模型看"范文"）

**为什么需要 few-shot**：有时候光说"按这个格式输出"模型不一定听话。但如果你给它看几个例子（就像教小孩写字先给字帖临摹），它就学得很快。

```python
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

# === 定义示例 ===
examples = [
    {"input": "我今天很开心", "output": "正面"},
    {"input": "这部电影太烂了", "output": "负面"},
    {"input": "还行吧，一般般", "output": "中性"},
]

# 定义每个示例的格式
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

# 把示例包装成 few-shot 模板
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# 最终的 prompt：先放示例，再放用户的实际问题
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个情感分析器，只回答'正面'、'负面'或'中性'。"),
    few_shot_prompt,  # 这里插入 few-shot 示例
    ("human", "{text}"),
])

# 调用
response = model.invoke(final_prompt.invoke({"text": "这家餐厅的菜太好吃了！"}))
print(response.content)
# 预期输出：正面
```

#### 2.6 MessagesPlaceholder——动态插入对话历史

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# MessagesPlaceholder 是一个"占位符"，运行时你往里塞历史消息
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的助手，请记住对话内容。"),
    MessagesPlaceholder(variable_name="history"),  # 这里会插入历史消息
    ("human", "{input}"),
])

# 模拟对话历史
from langchain_core.messages import HumanMessage, AIMessage
history = [
    HumanMessage(content="我叫张三"),
    AIMessage(content="你好张三！有什么可以帮你的？"),
]

# 调用时传入 history
response = model.invoke(prompt.invoke({
    "history": history,
    "input": "我叫什么名字？"
}))
print(response.content)
# 预期输出：你叫张三！
```

#### 试一试 🎯

1. **模板练习**：创建一个 Prompt Template，接收 `city`（城市名）和 `days`（天数）两个参数，让模型生成一份旅行攻略
2. **Few-shot 练习**：给模型 3 个"把中文数字转为阿拉伯数字"的示例，然后测试 "三百二十五" 能否正确转换为 "325"
3. **角色扮演**：用 system message 让模型扮演一个"毒舌程序员"，然后问它"我的代码写得好吗？"

---

### Step 3：输出解析——把模型回复变成程序能用的数据

#### 3.1 本节目标

学会把模型返回的自由文本，转换成 Python 程序可以直接使用的数据类型（字符串、字典、列表等）。

#### 3.2 为什么需要输出解析？

模型返回的不是普通字符串——它是一个 `AIMessage` 对象。如果你的程序需要：
- 只拿到文本内容 → 用 `StrOutputParser`
- 把文本解析成 JSON → 用 `JsonOutputParser`
- 把文本解析成强类型对象 → 用 `PydanticOutputParser`

**类比**：模型像一个"快递包裹"（AIMessage），Output Parser 就是"拆包员"——按你的要求取出里面的东西，放到正确的容器里。

#### 3.3 StrOutputParser——提取纯文本

```python
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_template("列出3种{category}，每行一个")
parser = StrOutputParser()

# === 不用 parser ===
raw_response = model.invoke(prompt.invoke({"category": "编程语言"}))
print(type(raw_response))  # <class 'langchain_core.messages.ai.AIMessage'>
print(raw_response.content)  # 需要手动 .content 才能拿到文本

# === 用 parser ===
# parser 接收 AIMessage，返回纯 str
text_response = parser.invoke(raw_response)
print(type(text_response))  # <class 'str'>
print(text_response)
# 预期输出：
# Python
# JavaScript
# Java
```

#### 3.4 JsonOutputParser——解析 JSON

**场景**：你让模型返回 JSON 格式的数据（比如提取文章中的关键信息），需要把文本解析成 Python 字典。

```python
from langchain_core.output_parsers import JsonOutputParser

# JsonOutputParser 会自动从模型回复中提取 JSON
parser = JsonOutputParser()

prompt = ChatPromptTemplate.from_template(
    "分析这本书的信息，以 JSON 格式返回，包含 title、author、year 三个字段。\n"
    "书名：{book_name}"
)

# 手动组合（下一步会学用 | 管道自动串联）
formatted_prompt = prompt.invoke({"book_name": "三体"})
model_response = model.invoke(formatted_prompt)
result = parser.invoke(model_response)

print(type(result))  # <class 'dict'>
print(result)
# 预期输出：
# {'title': '三体', 'author': '刘慈欣', 'year': 2008}

# 现在你可以像普通字典一样使用它
print(f"书名：{result['title']}")
print(f"作者：{result['author']}")
```

#### 3.5 PydanticOutputParser——强类型解析（进阶）

**为什么用 Pydantic**：JsonOutputParser 返回的是普通字典，如果你拼错了 key（比如写了 `titel` 而不是 `title`），Python 不会报错，只会在运行时出问题。Pydantic 定义了一个"数据模型"（类似一份表格模板），解析时自动验证格式是否正确。

```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

# === 第一步：定义你想要的数据结构 ===
class MovieReview(BaseModel):
    """电影评价的结构化数据"""
    title: str = Field(description="电影名称")
    rating: float = Field(description="评分，1-10分")
    summary: str = Field(description="一句话总结")
    recommend: bool = Field(description="是否推荐观看")

# === 第二步：创建解析器 ===
parser = PydanticOutputParser(pydantic_object=MovieReview)

# === 第三步：构建 prompt（parser 会自动生成格式说明）===
prompt = ChatPromptTemplate.from_template(
    "请评价这部电影。\n{format_instructions}\n\n电影：{movie_name}"
)

# get_format_instructions() 返回一段告诉模型"你应该输出什么格式"的文字
formatted = prompt.invoke({
    "movie_name": "肖申克的救赎",
    "format_instructions": parser.get_format_instructions()
})

response = model.invoke(formatted)
result = parser.invoke(response)

# result 是一个 MovieReview 对象，可以用 . 访问属性
print(f"电影：{result.title}")
print(f"评分：{result.rating}")
print(f"总结：{result.summary}")
print(f"推荐：{result.recommend}")
# 预期输出：
# 电影：肖申克的救赎
# 评分：9.7
# 总结：一部关于希望与自由的经典之作，展现了人性的坚韧。
# 推荐：True
```

#### 3.6 各种 Parser 的对比总结

| Parser | 返回类型 | 适用场景 | 容错性 |
|--------|---------|---------|--------|
| `StrOutputParser` | `str` | 只需要文本内容 | 高（不会失败） |
| `JsonOutputParser` | `dict` | 需要 JSON 数据 | 中（模型可能输出格式错误的 JSON） |
| `PydanticOutputParser` | Pydantic 对象 | 需要强类型验证 | 中（格式错误会抛异常） |
| `ListOutputParser` | `list[str]` | 需要列表 | 中 |
| `CommaSeparatedListOutputParser` | `list[str]` | 逗号分隔的列表 | 中 |

#### 试一试 🎯

1. **StrOutputParser**：构建一个链，让模型写一段话，然后用 parser 拿到纯字符串并统计字数
2. **JsonOutputParser**：让模型从一段新闻文本中提取 `{"人物": [...], "事件": "...", "时间": "..."}`
3. **PydanticOutputParser**：定义一个 `BookInfo` 模型（含 title、author、pages、genre 字段），让模型填充《百年孤独》的信息

---

### Step 4：LCEL 管道——用 `|` 把零件串起来

#### 4.1 本节目标

学会用 LCEL（LangChain Expression Language）的 `|` 语法，把 Prompt、Model、Parser 组装成一条完整的"流水线"，一次调用就完成全部处理。

#### 4.2 为什么这一步很重要？

在 Step 1-3 中，你可能注意到了这样的代码模式：
```python
# 每次都要手动三步调用——繁琐且容易出错
formatted = prompt.invoke(input_dict)
response = model.invoke(formatted)
result = parser.invoke(response)
```

LCEL 让你用一行代码完成同样的事：
```python
# 组装一次，到处使用
chain = prompt | model | parser
result = chain.invoke(input_dict)
```

**更重要的好处**：组装好的 chain 自动获得：
- `chain.stream()` — 流式输出
- `chain.batch()` — 批量处理
- `chain.ainvoke()` — 异步调用

你不需要为每个功能单独写代码！

#### 4.3 基本管道：prompt | model | parser

```python
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# === 定义三个组件 ===
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}，用简洁的语言回答问题。"),
    ("human", "{question}"),
])
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

# === 用 | 组装成链 ===
# 数据流向：输入字典 → prompt填充 → 模型生成 → parser提取文本
chain = prompt | model | parser

# === 调用链 ===
# 传入的字典 key 必须匹配 prompt 模板中的变量名
result = chain.invoke({
    "role": "技术顾问",
    "question": "什么是微服务架构？"
})

print(type(result))  # <class 'str'> —— 直接是字符串！
print(result)
# 预期输出：
# 微服务架构是将一个大型应用拆分成多个小型、独立部署的服务，
# 每个服务负责一个特定功能，通过 API 通信协作。
```

#### 4.4 流式管道——整条链都支持 stream

```python
# 同样一条链，用 stream() 就能逐字输出
# 这是因为 LCEL 会自动把流式能力"传递"给整条链
for chunk in chain.stream({"role": "诗人", "question": "写一句关于代码的诗"}):
    print(chunk, end="", flush=True)
    # chunk 是纯字符串（因为链的末端是 StrOutputParser）

print()
# 预期输出（逐字出现）：
# 代码如诗行行间，逻辑似水潺潺流。
```

#### 4.5 RunnablePassthrough 和 RunnableLambda——管道中的"加工站"

**问题场景**：有时候你需要在管道中间做一些数据变换（比如把字典的一个 key 提取出来，或者对字符串做处理），这时候就需要 RunnableLambda。

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# RunnableLambda: 把普通 Python 函数变成管道中的一个环节
# 比如：把字符串转成大写
upper_case = RunnableLambda(lambda x: x.upper())

chain = prompt | model | parser | upper_case
result = chain.invoke({"role": "助手", "question": "Say hello"})
print(result)
# 预期输出：HELLO! (全大写)

# RunnablePassthrough: "原样传递"——什么都不做，把输入原封不动地传到下一步
# 常用于并行分支中保留原始输入
```

#### 4.6 并行管道——同时做多件事

```python
from langchain_core.runnables import RunnableParallel

# 场景：对同一个输入，同时让模型写一首诗和一个笑话
poem_prompt = ChatPromptTemplate.from_template("用一句诗描述{topic}")
joke_prompt = ChatPromptTemplate.from_template("讲一个关于{topic}的冷笑话")

# RunnableParallel 接收一个字典，每个 value 是一条独立的链
# 所有链会并行执行，最终返回一个字典
parallel_chain = RunnableParallel({
    "poem": poem_prompt | model | parser,
    "joke": joke_prompt | model | parser,
})

result = parallel_chain.invoke({"topic": "程序员"})
print(f"诗：{result['poem']}")
print(f"笑话：{result['joke']}")
# 预期输出：
# 诗：键盘声里岁月长，一行代码一星光。
# 笑话：为什么程序员分不清万圣节和圣诞节？因为 Oct 31 == Dec 25。
```

#### 4.7 管道中的错误处理

```python
from langchain_core.runnables import RunnableLambda

# with_fallbacks: 如果主链失败，自动尝试备用方案
# 场景：主模型挂了，自动切换到备用模型
from langchain_openai import ChatOpenAI

primary_model = ChatOpenAI(model="gpt-4o-mini")
backup_model = ChatOpenAI(model="gpt-3.5-turbo")

# with_fallbacks 让链具备"容灾"能力
robust_model = primary_model.with_fallbacks([backup_model])

chain = prompt | robust_model | parser
# 如果 gpt-4o-mini 调用失败，会自动尝试 gpt-3.5-turbo
```

#### 试一试 🎯

1. **基本管道**：构建一条链 `prompt | model | parser`，输入一个城市名，输出该城市的三句话介绍
2. **并行管道**：用 RunnableParallel 同时对一个话题生成"标题"、"摘要"和"关键词"
3. **流式体验**：把第 1 题的链改用 `.stream()` 调用，观察输出效果
4. **加工站**：在链的末尾加一个 RunnableLambda，把模型输出的文本统计字数并返回 "共 N 字"

---

### Step 5：结构化输出——让模型严格返回 JSON

#### 5.1 本节目标

学会用 `with_structured_output()` 方法，让模型**保证**返回你定义的 JSON 结构，而不是"祈祷"模型按格式输出。

#### 5.2 为什么不能只靠 prompt 要求格式？

你可以在 prompt 里写"请以 JSON 格式返回"，但模型经常：
- 在 JSON 前后加上多余的文字（"好的，这是你要的 JSON："）
- 用 markdown 代码块包裹（```json ... ```）
- 漏掉某个字段或类型错误（把数字写成字符串）

`with_structured_output()` 利用模型的原生 Function Calling（函数调用）能力，从**模型解码层面**强制输出合法 JSON。

**类比**：普通 prompt 要求格式 = 你口头告诉实习生"记得用 Excel 表格"；with_structured_output = 你直接给他一份锁定了格式的 Excel 模板，他只能在格子里填内容。

#### 5.3 使用 Pydantic 定义输出结构

```python
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# === 第一步：用 Pydantic 定义你想要的数据结构 ===
# BaseModel 是 Pydantic 的基类，用于定义"数据模型"
# Field() 中的 description 会告诉模型这个字段该填什么
class CityInfo(BaseModel):
    """城市信息"""
    name: str = Field(description="城市名称")
    country: str = Field(description="所属国家")
    population: int = Field(description="人口数量（万）")
    famous_for: list[str] = Field(description="著名特色，至少3项")
    best_season: str = Field(description="最佳旅游季节")

# === 第二步：创建模型并绑定结构化输出 ===
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# with_structured_output() 方法详解：
#   参数：一个 Pydantic BaseModel 子类
#   作用：告诉模型"你必须返回符合这个结构的 JSON"
#   返回：一个新的 Runnable，调用后直接返回 Pydantic 对象（不是字符串！）
structured_model = model.with_structured_output(CityInfo)

# === 第三步：调用 ===
# 传入普通字符串或消息列表都行
result = structured_model.invoke("介绍一下东京")

# result 直接就是一个 CityInfo 对象！不需要手动解析！
print(f"城市：{result.name}")
print(f"国家：{result.country}")
print(f"人口：{result.population}万")
print(f"特色：{result.famous_for}")
print(f"最佳季节：{result.best_season}")
```

**预期输出**：
```
城市：东京
国家：日本
人口：1396万
特色：['樱花', '动漫文化', '寿司料理', '东京塔']
最佳季节：春季（3-4月）
```

#### 5.4 结构化输出 + Prompt 模板 + LCEL 管道

```python
from langchain_core.prompts import ChatPromptTemplate

# 结构化输出也能完美融入 LCEL 管道
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个旅游顾问，提供准确的城市信息。"),
    ("human", "请介绍{city}这座城市。"),
])

# 管道：prompt → structured_model（自动返回 CityInfo 对象）
chain = prompt | structured_model

result = chain.invoke({"city": "巴黎"})
print(result)
# 预期输出：
# CityInfo(name='巴黎', country='法国', population=215,
#          famous_for=['埃菲尔铁塔', '卢浮宫', '法式料理'],
#          best_season='春季（4-6月）')
```

#### 5.5 复杂嵌套结构

```python
from typing import Optional
from pydantic import BaseModel, Field

# 支持嵌套结构——模型也能正确填充
class Ingredient(BaseModel):
    """食材"""
    name: str = Field(description="食材名称")
    amount: str = Field(description="用量")

class Recipe(BaseModel):
    """菜谱"""
    dish_name: str = Field(description="菜名")
    difficulty: str = Field(description="难度：简单/中等/困难")
    time_minutes: int = Field(description="烹饪时间（分钟）")
    ingredients: list[Ingredient] = Field(description="食材清单")
    steps: list[str] = Field(description="烹饪步骤")
    tips: Optional[str] = Field(default=None, description="小贴士")

structured_model = model.with_structured_output(Recipe)
result = structured_model.invoke("教我做番茄炒蛋")

print(f"菜名：{result.dish_name}")
print(f"难度：{result.difficulty}")
print(f"时间：{result.time_minutes}分钟")
print("食材：")
for ing in result.ingredients:
    print(f"  - {ing.name}: {ing.amount}")
print("步骤：")
for i, step in enumerate(result.steps, 1):
    print(f"  {i}. {step}")
```

**预期输出**：
```
菜名：番茄炒蛋
难度：简单
时间：15分钟
食材：
  - 番茄: 2个
  - 鸡蛋: 3个
  - 盐: 适量
  - 糖: 1小勺
  - 葱花: 少许
步骤：
  1. 番茄切块，鸡蛋打散加少许盐搅匀
  2. 热锅凉油，倒入蛋液炒至凝固盛出
  3. 锅中再加少许油，放入番茄翻炒出汁
  4. 加入糖和盐调味，倒回鸡蛋翻炒均匀
  5. 撒上葱花出锅
```

#### 5.6 with_structured_output() vs PydanticOutputParser 对比

| | `with_structured_output()` | `PydanticOutputParser` |
|---|---|---|
| 实现原理 | 利用模型原生 Function Calling | 在 prompt 中注入格式说明 + 事后解析 |
| 可靠性 | 非常高（模型解码层面强制） | 中等（模型可能不遵守） |
| 需要手动解析 | 不需要，直接返回对象 | 需要 `parser.invoke(response)` |
| 适用模型 | 支持 Function Calling 的模型 | 任何模型 |
| 推荐程度 | ⭐⭐⭐⭐⭐（首选） | ⭐⭐⭐（备用） |

#### 试一试 🎯

1. **基础练习**：定义一个 `PersonCard` 模型（姓名、年龄、职业、一句话简介），让模型填充"爱因斯坦"的信息
2. **列表输出**：定义一个 `MovieList` 模型，包含 `movies: list[MovieItem]`，让模型推荐 5 部科幻电影
3. **结合管道**：构建一条完整链 `prompt | structured_model`，输入一段英文文本，输出结构化的翻译结果（含原文、译文、关键词列表）

---

### Step 6：RAG 检索增强生成——让模型"开卷考试"

#### 6.1 本节目标

学会构建一个完整的 RAG 系统：把你的文档"喂"给向量数据库，然后让用户提问时，系统先搜索相关文档片段，再让模型基于这些片段回答问题。

#### 6.2 RAG 全流程概览

```
╔═══════════════════════════════════════════════════╗
║  第一阶段：文档准备（离线/一次性）                    ║
║                                                   ║
║  原始文档 → 分割成小块 → 计算 Embedding → 存入向量库  ║
╚═══════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════╗
║  第二阶段：查询回答（在线/每次提问）                    ║
║                                                   ║
║  用户提问 → 计算问题 Embedding → 搜索相似文档        ║
║         → 把文档 + 问题一起发给模型 → 生成回答        ║
╚═══════════════════════════════════════════════════╝
```

#### 6.3 第一阶段：文档加载与分割

```python
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# === 加载文档 ===
# DocumentLoader 负责把各种格式的文件读入为 Document 对象
# 这里用最简单的纯文本文件做演示
loader = TextLoader("my_knowledge.txt", encoding="utf-8")
documents = loader.load()

print(f"加载了 {len(documents)} 个文档")
print(f"文档内容前100字：{documents[0].page_content[:100]}")

# === 分割文档 ===
# 为什么要分割？因为模型的上下文窗口有限，而且搜索时我们希望精确到"段落"级别
# RecursiveCharacterTextSplitter 会递归地按照 段落→句子→字符 的优先级分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # 每个块最大 500 字符
    chunk_overlap=50,     # 相邻块重叠 50 字符（避免在句子中间切断导致信息丢失）
    separators=["\n\n", "\n", "。", "，", " ", ""]  # 分割优先级
)

chunks = text_splitter.split_documents(documents)
print(f"分割成 {len(chunks)} 个文本块")
print(f"第一块内容：{chunks[0].page_content[:100]}...")
```

#### 6.4 第一阶段（续）：计算 Embedding 并存入向量数据库

```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# === 创建 Embedding 模型 ===
# OpenAIEmbeddings 把文本转为向量（一串数字）
# 这个操作需要调用 OpenAI API（会消耗少量费用）
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# === 创建向量数据库并存入文档 ===
# Chroma 是一个轻量级的本地向量数据库，适合学习和小规模项目
# from_documents() 会自动：1) 对每个 chunk 计算 embedding 2) 存入数据库
vectorstore = Chroma.from_documents(
    documents=chunks,           # 要存入的文档块列表
    embedding=embeddings,       # 用哪个 embedding 模型
    persist_directory="./chroma_db"  # 持久化存储路径（下次启动不用重新导入）
)

print(f"已存入 {vectorstore._collection.count()} 个向量")
# 预期输出：已存入 15 个向量（具体数字取决于你的文档）
```

#### 6.5 第二阶段：检索 + 生成

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# === 创建检索器 ===
# as_retriever() 把向量数据库变成一个 Retriever（检索器）
# Retriever 是一个 Runnable，可以直接用在 LCEL 管道中！
retriever = vectorstore.as_retriever(
    search_type="similarity",       # 搜索类型：相似度搜索
    search_kwargs={"k": 3}          # 返回最相似的 3 个文档块
)

# 测试检索器
docs = retriever.invoke("Python 的优缺点是什么？")
for i, doc in enumerate(docs):
    print(f"\n--- 检索结果 {i+1} ---")
    print(doc.page_content[:200])
# 预期输出：显示3段与"Python优缺点"相关的文档内容

# === 构建 RAG 链 ===
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

# RAG 专用的 Prompt：告诉模型"根据提供的参考资料回答"
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个知识问答助手。请严格根据以下参考资料回答用户的问题。"
               "如果资料中没有相关信息，请说'根据现有资料无法回答'。\n\n"
               "参考资料：\n{context}"),
    ("human", "{question}"),
])

# 辅助函数：把检索到的文档列表格式化为字符串
def format_docs(docs):
    """把 Document 对象列表拼接成一段文本"""
    return "\n\n".join(doc.page_content for doc in docs)

# 组装 RAG 管道
# RunnablePassthrough.assign() 的作用：保留原始输入，同时添加新字段
rag_chain = (
    # 输入是 {"question": "..."}
    # retriever 会根据 question 搜索文档
    # assign 把搜索结果作为 "context" 字段加入字典
    RunnablePassthrough.assign(
        context=lambda x: format_docs(retriever.invoke(x["question"]))
    )
    | rag_prompt   # 填充模板（需要 context 和 question 两个变量）
    | model        # 模型生成回答
    | parser       # 提取纯文本
)

# === 提问 ===
answer = rag_chain.invoke({"question": "Python 适合做什么？"})
print(answer)
# 预期输出（基于你的文档内容）：
# 根据参考资料，Python 适合以下领域：
# 1. Web 开发（Django、Flask 框架）
# 2. 数据分析和可视化（Pandas、Matplotlib）
# 3. 人工智能和机器学习（TensorFlow、PyTorch）
# 4. 自动化脚本和运维工具
```

#### 6.6 关键方法详解

| 方法 | 作用 | 参数 | 返回值 |
|------|------|------|--------|
| `vectorstore.as_retriever()` | 把向量库转为检索器 | `search_type`, `search_kwargs` | Retriever (Runnable) |
| `retriever.invoke(query)` | 根据查询搜索相关文档 | 字符串查询 | `list[Document]` |
| `Chroma.from_documents()` | 创建向量库并导入文档 | `documents`, `embedding` | Chroma 实例 |
| `text_splitter.split_documents()` | 分割文档为小块 | `list[Document]` | `list[Document]` |
| `RunnablePassthrough.assign()` | 保留输入并添加新字段 | `key=callable` | Runnable |

#### 6.7 如果不用 RAG 会怎样？

```python
# 不用 RAG：直接问模型（模型只能凭记忆回答）
direct_chain = ChatPromptTemplate.from_template("{question}") | model | parser
answer = direct_chain.invoke({"question": "我们公司的年假政策是什么？"})
print(answer)
# 预期输出：抱歉，我不知道你们公司的具体年假政策...
# （模型不可能知道你的私有信息！）

# 用 RAG：模型会参考你导入的公司文档来回答
# answer = rag_chain.invoke({"question": "我们公司的年假政策是什么？"})
# 预期输出：根据员工手册，公司年假政策为：工龄1-3年享5天...
```

#### 试一试 🎯

1. **基础 RAG**：创建一个 `my_notes.txt` 文件，写入你自己的学习笔记（至少 500 字），然后构建 RAG 系统回答关于笔记的问题
2. **检索质量**：修改 `search_kwargs={"k": 3}` 中的 k 值（1、3、5、10），观察检索结果和最终回答的质量变化
3. **多文档 RAG**：加载 2-3 个不同的文本文件到同一个向量库，测试跨文档检索能力

---

### Step 7：Tool 工具——给模型装上"手和脚"

#### 7.1 本节目标

学会定义工具（Tool），让大模型能够"调用外部函数"来完成任务。这是构建 Agent 的前置知识。

#### 7.2 过渡说明：从 RAG 到工具调用

在 Step 6 中，你学会了让模型"开卷考试"（检索文档后回答）。但 RAG 只能"读"——它不能让模型"做事"。

想象一下这个场景：
- 用户问："北京今天多少度？" → RAG 无法回答（天气实时变化）
- 用户问："帮我算一下 1234 * 5678" → 模型可能算错（LLM 不擅长数学）
- 用户问："搜索一下最新的 Python 3.13 新特性" → 模型知识有截止日期

**工具就是解决这些问题的钥匙**——让模型可以调用天气 API、计算器、搜索引擎等外部功能。

#### 7.3 用 @tool 装饰器定义工具（最简单的方式）

```python
from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool

# === 用 @tool 装饰器把一个普通 Python 函数变成 LangChain 工具 ===
# @tool 会自动从函数签名和 docstring 中提取：
#   - 工具名称 = 函数名
#   - 工具描述 = docstring
#   - 参数 schema = 函数参数的类型注解

@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。当用户询问天气情况时使用这个工具。"""
    # 实际项目中这里会调用天气 API，现在用模拟数据演示
    weather_data = {
        "北京": "晴，25°C，北风 2 级",
        "上海": "多云，28°C，东南风 3 级",
        "深圳": "阵雨，32°C，南风 2 级",
    }
    return weather_data.get(city, f"抱歉，暂无{city}的天气数据")

@tool
def calculate(expression: str) -> str:
    """计算数学表达式。当用户需要做数学计算时使用这个工具。
    输入应该是一个合法的数学表达式，如 '2 + 3 * 4'。"""
    try:
        # 注意：实际项目中不要用 eval，这里仅为教学演示
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"

# === 查看工具的元信息 ===
print(f"工具名：{get_weather.name}")
print(f"描述：{get_weather.description}")
print(f"参数 Schema：{get_weather.args_schema.model_json_schema()}")
# 预期输出：
# 工具名：get_weather
# 描述：查询指定城市的当前天气。当用户询问天气情况时使用这个工具。
# 参数 Schema：{'properties': {'city': {'title': 'City', 'type': 'string'}},
#              'required': ['city'], 'title': 'get_weatherSchema', 'type': 'object'}
```

#### 7.4 把工具绑定到模型（bind_tools）

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# === bind_tools() 方法详解 ===
# 作用：告诉模型"你有这些工具可以用"
# 参数：一个工具列表 [tool1, tool2, ...]
# 返回：一个新的模型实例（绑定了工具信息）
# 原理：工具的名称、描述、参数schema 会被转换为模型的 function calling 格式
model_with_tools = model.bind_tools([get_weather, calculate])

# === 调用绑定了工具的模型 ===
response = model_with_tools.invoke("北京今天天气怎么样？")

# 检查模型的返回
print(f"普通内容：{response.content}")  # 通常为空字符串
print(f"工具调用：{response.tool_calls}")  # 模型决定调用哪个工具
# 预期输出：
# 普通内容：
# 工具调用：[{'name': 'get_weather', 'args': {'city': '北京'}, 'id': 'call_abc123'}]
```

**重要理解**：模型并不会自己执行工具！它只是返回一个"调用请求"（我想调用 get_weather，参数是 city='北京'）。实际执行需要你的代码来做。

#### 7.5 手动执行工具调用（理解底层原理）

```python
# 这段代码演示"工具调用循环"的底层逻辑（Agent 会自动做这件事）

# 第一步：用户提问，模型决定调用工具
response = model_with_tools.invoke("北京今天天气怎么样？")

# 第二步：检查模型是否要调用工具
if response.tool_calls:
    tool_call = response.tool_calls[0]  # 取第一个工具调用
    print(f"模型想调用: {tool_call['name']}, 参数: {tool_call['args']}")
    
    # 第三步：手动执行工具
    # 实际中，Agent 框架会自动完成这一步
    tool_map = {"get_weather": get_weather, "calculate": calculate}
    tool_result = tool_map[tool_call['name']].invoke(tool_call['args'])
    print(f"工具返回: {tool_result}")
    
    # 第四步：把工具结果告诉模型，让它生成最终回答
    from langchain_core.messages import ToolMessage
    tool_message = ToolMessage(
        content=tool_result,
        tool_call_id=tool_call['id']  # 必须对应之前的调用 ID
    )
    
    # 把完整对话（用户问题 + 工具调用 + 工具结果）发给模型
    final_response = model_with_tools.invoke([
        response,          # AI 的工具调用请求
        tool_message,      # 工具的执行结果
    ])
    print(f"最终回答: {final_response.content}")

# 预期输出：
# 模型想调用: get_weather, 参数: {'city': '北京'}
# 工具返回: 晴，25°C，北风 2 级
# 最终回答: 北京今天是晴天，气温25°C，北风2级，适合外出活动。
```

#### 7.6 工具设计最佳实践

写好工具的**描述**比写好代码更重要！因为模型是通过读描述来决定"什么时候用这个工具"的。

```python
# ❌ 差的工具定义——描述太模糊
@tool
def search(query: str) -> str:
    """搜索"""
    pass

# ✅ 好的工具定义——描述清晰、告诉模型什么时候该用
@tool
def search_documents(query: str, max_results: int = 5) -> str:
    """在公司内部文档库中搜索信息。
    
    当用户询问公司政策、产品规格、历史数据等内部信息时使用。
    不适用于查询实时信息（如天气、股价）。
    
    Args:
        query: 搜索关键词，尽量具体
        max_results: 最多返回几条结果，默认 5 条
    """
    pass
```

#### 试一试 🎯

1. **定义工具**：创建一个 `get_stock_price(symbol: str)` 工具（可以用模拟数据），描述要清晰
2. **多工具协作**：问模型"北京今天天气怎么样？如果超过 30 度帮我计算 30 * 1.8 + 32 转换为华氏度"，观察模型是否会连续调用两个工具
3. **错误处理**：在工具函数中加入 try-except，测试当传入无效参数时工具能否优雅地返回错误信息

---

### Step 8：Agent 智能体——让模型自己决定用什么工具

#### 8.1 本节目标

学会用 LangGraph 的 `create_react_agent()` 构建一个能够自主决策、循环调用工具的 Agent。

#### 8.2 从"手动工具调用"到"自动 Agent"

在 Step 7 中，你手动写了"检查 tool_calls → 执行工具 → 把结果返回模型"的循环。但实际场景中：
- 模型可能需要调用多个工具（先查天气，再查航班，再查酒店）
- 每次调用的工具可能不同
- 可能需要多轮循环才能完成任务

**Agent 就是把这个循环自动化了**——你只需定义工具，Agent 会自己决定调用顺序和次数。

**类比**：
- Step 7（手动工具调用）= 你亲自当"调度员"，每次告诉工人下一步做什么
- Step 8（Agent）= 你雇了一个"项目经理"，他自己看情况分配任务

#### 8.3 使用 create_react_agent 构建 Agent

```python
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# === 定义工具 ===
@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。当用户询问天气情况时使用。"""
    weather_data = {
        "北京": "晴，25°C，北风 2 级",
        "上海": "多云，28°C，东南风 3 级",
        "广州": "阵雨，33°C，南风 2 级",
    }
    return weather_data.get(city, f"暂无{city}的天气数据")

@tool
def calculate(expression: str) -> str:
    """计算数学表达式。当需要精确数学计算时使用。"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"

@tool
def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """获取当前时间。当用户询问现在几点或今天日期时使用。"""
    from datetime import datetime
    import zoneinfo
    now = datetime.now(zoneinfo.ZoneInfo(timezone))
    return now.strftime("%Y年%m月%d日 %H:%M:%S")

# === 创建 Agent ===
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# create_react_agent() 方法详解：
#   参数 1 (model): 作为"大脑"的 LLM
#   参数 2 (tools): 可用工具列表
#   返回: 一个编译好的 LangGraph 图（可直接 .invoke()）
#   内部逻辑: 自动实现 "Thought → Action → Observation" 循环
agent = create_react_agent(
    model=model,
    tools=[get_weather, calculate, get_current_time],
)

# === 调用 Agent ===
# 输入格式：{"messages": [(角色, 内容), ...]}
result = agent.invoke({
    "messages": [("user", "北京现在几点？天气怎么样？")]
})

# 查看 Agent 的完整思考过程
for message in result["messages"]:
    print(f"[{message.type}]: {message.content[:100]}")
    if hasattr(message, 'tool_calls') and message.tool_calls:
        for tc in message.tool_calls:
            print(f"  → 调用工具: {tc['name']}({tc['args']})")

# 获取最终回答
final_answer = result["messages"][-1].content
print(f"\n最终回答: {final_answer}")
```

**预期输出**：
```
[human]: 北京现在几点？天气怎么样？
[ai]: 
  → 调用工具: get_current_time({'timezone': 'Asia/Shanghai'})
  → 调用工具: get_weather({'city': '北京'})
[tool]: 2024年12月15日 14:30:25
[tool]: 晴，25°C，北风 2 级
[ai]: 北京现在是2024年12月15日下午14:30，天气晴朗，气温25°C，北风2级。

最终回答: 北京现在是2024年12月15日下午14:30，天气晴朗，气温25°C，北风2级。
```

#### 8.4 给 Agent 添加系统提示（设定人设）

```python
# 通过 state_modifier 参数给 Agent 添加 system prompt
from langchain_core.messages import SystemMessage

agent = create_react_agent(
    model=model,
    tools=[get_weather, calculate, get_current_time],
    # state_modifier 可以是一个 SystemMessage 或一个函数
    state_modifier=SystemMessage(content=(
        "你是一个智能生活助手。回答要简洁友好。"
        "如果需要使用工具，请主动使用，不要猜测答案。"
        "如果多个问题需要不同的工具，请一次性调用所有需要的工具。"
    ))
)

result = agent.invoke({
    "messages": [("user", "帮我算一下 (15 + 27) * 3 等于多少")]
})
print(result["messages"][-1].content)
# 预期输出：(15 + 27) * 3 = 126
```

#### 8.5 Agent 的多步推理（观察 Agent 思考过程）

```python
# Agent 会自动处理需要多步推理的复杂任务
result = agent.invoke({
    "messages": [("user", "比较北京和上海的天气，哪个城市更热？热多少度？")]
})

# 打印完整推理链
print("=== Agent 推理过程 ===")
for msg in result["messages"]:
    if msg.type == "human":
        print(f"👤 用户: {msg.content}")
    elif msg.type == "ai":
        if msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"🤔 AI 决定调用: {tc['name']}({tc['args']})")
        if msg.content:
            print(f"🤖 AI 回答: {msg.content}")
    elif msg.type == "tool":
        print(f"🔧 工具返回: {msg.content}")

# 预期输出：
# === Agent 推理过程 ===
# 👤 用户: 比较北京和上海的天气，哪个城市更热？热多少度？
# 🤔 AI 决定调用: get_weather({'city': '北京'})
# 🤔 AI 决定调用: get_weather({'city': '上海'})
# 🔧 工具返回: 晴，25°C，北风 2 级
# 🔧 工具返回: 多云，28°C，东南风 3 级
# 🤔 AI 决定调用: calculate({'expression': '28 - 25'})
# 🔧 工具返回: 3
# 🤖 AI 回答: 上海比北京更热，上海28°C，北京25°C，上海比北京热3度。
```

#### 试一试 🎯

1. **添加新工具**：给 Agent 添加一个 `translate(text: str, target_lang: str)` 工具，测试它能否自动在需要翻译时调用
2. **复杂任务**：问 Agent "现在几点了？如果超过下午 3 点，帮我计算距离晚上 6 点还有多少分钟"
3. **失败情况**：问一个所有工具都无法回答的问题，观察 Agent 如何应对

---

### Step 9：LangGraph——构建复杂多步骤工作流

#### 9.1 本节目标

学会使用 LangGraph 的 `StateGraph` 构建自定义工作流，包括条件分支、循环、记忆（Checkpointer）等高级功能。

#### 9.2 为什么需要 LangGraph（而不是只用 create_react_agent）？

`create_react_agent` 是一个"通用型 Agent"——它的决策完全由模型自由发挥。但实际项目中，你可能需要：
- **固定流程**：先验证用户身份 → 再查询数据 → 最后生成报告
- **条件分支**：如果金额 > 10000 走人工审批，否则自动通过
- **人工干预**：某些步骤需要人类确认后才能继续
- **持久记忆**：记住之前对话的内容

LangGraph 让你像画流程图一样定义工作流，精确控制每一步的逻辑。

**类比**：
- `create_react_agent` = 给员工一个目标，让他自由发挥
- `LangGraph StateGraph` = 给员工一份详细的 SOP 流程图，每步都有明确规则

#### 9.3 LangGraph 核心概念

| 概念 | 作用 | 类比 |
|------|------|------|
| `State` | 工作流的"数据背包"，在各节点间传递 | 流水线上的托盘 |
| `Node` | 一个处理步骤（函数） | 流水线上的工位 |
| `Edge` | 节点之间的连接（固定/条件） | 传送带 |
| `Conditional Edge` | 根据条件走不同分支 | 分拣员（看包裹标签决定送哪条线） |
| `Checkpointer` | 保存状态，支持记忆和恢复 | 自动存档点 |

#### 9.4 第一个 LangGraph 工作流

```python
from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# === 第一步：定义 State（工作流的"数据背包"）===
# State 是一个 TypedDict，定义了在工作流中流动的数据结构
class ChatState(TypedDict):
    """对话状态"""
    # Annotated[list, add_messages] 的含义：
    #   - 类型是 list（消息列表）
    #   - add_messages 是一个"reducer"函数，定义了新消息如何合并到列表中
    #   - 效果：每个节点返回新消息时，会自动追加到列表（而不是覆盖）
    messages: Annotated[list, add_messages]

# === 第二步：定义 Node（工作流中的处理步骤）===
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def chatbot_node(state: ChatState) -> dict:
    """聊天节点：把当前所有消息发给模型，获得回复"""
    response = model.invoke(state["messages"])
    # 返回一个字典，key 必须匹配 State 中的字段
    # add_messages reducer 会自动把这个新消息追加到 messages 列表
    return {"messages": [response]}

# === 第三步：构建图 ===
builder = StateGraph(ChatState)

# 添加节点：builder.add_node(名称, 函数)
builder.add_node("chatbot", chatbot_node)

# 添加边：定义执行顺序
builder.add_edge(START, "chatbot")   # 从起点进入 chatbot 节点
builder.add_edge("chatbot", END)     # chatbot 执行完就结束

# === 第四步：编译图 ===
graph = builder.compile()

# === 第五步：运行 ===
result = graph.invoke({
    "messages": [HumanMessage(content="你好！用一句话介绍你自己")]
})

for msg in result["messages"]:
    print(f"[{msg.type}]: {msg.content}")
# 预期输出：
# [human]: 你好！用一句话介绍你自己
# [ai]: 你好！我是AI助手，可以回答问题、提供建议和协助完成各种任务。
```

#### 9.5 条件分支——根据情况走不同路径

```python
from typing import Literal

# 场景：根据用户问题的类型，走不同的处理路径
class RouterState(TypedDict):
    messages: Annotated[list, add_messages]
    route: str  # 记录路由决策

def classify_node(state: RouterState) -> dict:
    """分类节点：判断用户问题属于哪个类别"""
    last_message = state["messages"][-1].content
    # 这里用简单规则演示，实际中可以用模型来分类
    if any(word in last_message for word in ["代码", "编程", "Python", "bug"]):
        route = "tech"
    elif any(word in last_message for word in ["天气", "温度", "下雨"]):
        route = "weather"
    else:
        route = "general"
    return {"route": route}

def tech_node(state: RouterState) -> dict:
    """技术问题专家节点"""
    tech_model = model.bind(
        system="你是一个资深程序员，回答要包含代码示例。"
    ) if hasattr(model, 'bind') else model
    response = model.invoke(state["messages"] + [
        AIMessage(content="")  # 占位
    ])
    return {"messages": [AIMessage(content=f"[技术专家] {response.content}")]}

def general_node(state: RouterState) -> dict:
    """通用问答节点"""
    response = model.invoke(state["messages"])
    return {"messages": [AIMessage(content=f"[通用助手] {response.content}")]}

# 路由函数：根据 state 决定下一步去哪个节点
def route_decision(state: RouterState) -> Literal["tech", "general"]:
    """条件边的判断函数——返回下一个节点的名称"""
    if state["route"] == "tech":
        return "tech"
    return "general"

# 构建带条件分支的图
builder = StateGraph(RouterState)
builder.add_node("classify", classify_node)
builder.add_node("tech", tech_node)
builder.add_node("general", general_node)

builder.add_edge(START, "classify")
# add_conditional_edges: 根据函数返回值决定走哪条边
builder.add_conditional_edges("classify", route_decision)
builder.add_edge("tech", END)
builder.add_edge("general", END)

graph = builder.compile()

# 测试
result = graph.invoke({"messages": [HumanMessage(content="Python 怎么读取 JSON 文件？")]})
print(result["messages"][-1].content)
# 预期输出：[技术专家] 你可以使用 json 模块...
```

#### 9.6 Checkpointer——给 Agent 加上记忆

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

@tool
def search_knowledge(query: str) -> str:
    """搜索知识库获取信息。"""
    return f"知识库中关于'{query}'的信息：这是一个模拟的搜索结果。"

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# === 创建带记忆的 Agent ===
# MemorySaver 是一个内存中的 checkpointer
# 它会保存每个 thread（对话线程）的完整状态
checkpointer = MemorySaver()

agent = create_react_agent(
    model=model,
    tools=[search_knowledge],
    checkpointer=checkpointer,  # 关键：传入 checkpointer
)

# === 第一次对话 ===
# thread_id 标识一个对话线程，相同 thread_id 的消息会共享记忆
config = {"configurable": {"thread_id": "conversation-001"}}

result1 = agent.invoke(
    {"messages": [HumanMessage(content="我叫张三，我是一名 Python 开发者")]},
    config=config
)
print(result1["messages"][-1].content)
# 预期：你好张三！很高兴认识你...

# === 第二次对话（同一个 thread_id）===
result2 = agent.invoke(
    {"messages": [HumanMessage(content="我叫什么？我是做什么的？")]},
    config=config
)
print(result2["messages"][-1].content)
# 预期：你叫张三，你是一名 Python 开发者。
# Agent 记住了之前的对话！

# === 新的对话线程（不同 thread_id）===
config2 = {"configurable": {"thread_id": "conversation-002"}}
result3 = agent.invoke(
    {"messages": [HumanMessage(content="我叫什么？")]},
    config=config2
)
print(result3["messages"][-1].content)
# 预期：抱歉，我不知道你的名字...
# 新线程没有之前的记忆
```

#### 9.7 带工具的完整 Agent + 记忆

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

@tool
def get_weather(city: str) -> str:
    """查询城市天气"""
    return {"北京": "晴 25°C", "上海": "雨 20°C"}.get(city, "未知")

@tool
def add_reminder(content: str, time: str) -> str:
    """设置提醒事项。content 是提醒内容，time 是提醒时间。"""
    return f"已设置提醒：{time} - {content}"

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
checkpointer = MemorySaver()

agent = create_react_agent(
    model=model,
    tools=[get_weather, add_reminder],
    checkpointer=checkpointer,
    state_modifier="你是一个智能助手，可以查天气和设提醒。回答简洁。"
)

config = {"configurable": {"thread_id": "user-xiaoming"}}

# 多轮对话演示
conversations = [
    "北京明天天气怎么样？",
    "如果下雨的话，提醒我明天带伞",
    "我刚才让你设了什么提醒？",  # 测试记忆
]

for user_msg in conversations:
    result = agent.invoke(
        {"messages": [HumanMessage(content=user_msg)]},
        config=config
    )
    print(f"👤: {user_msg}")
    print(f"🤖: {result['messages'][-1].content}\n")

# 预期输出：
# 👤: 北京明天天气怎么样？
# 🤖: 北京明天晴天，25°C。
#
# 👤: 如果下雨的话，提醒我明天带伞
# 🤖: 根据查询结果，北京明天是晴天，不会下雨，所以不需要设带伞提醒。
#
# 👤: 我刚才让你设了什么提醒？
# 🤖: 你刚才说如果明天下雨就提醒你带伞，但因为明天是晴天，所以没有设置提醒。
```

#### 试一试 🎯

1. **基本图**：用 StateGraph 构建一个两步工作流：第一步让模型生成一个标题，第二步让模型根据标题写一段摘要
2. **条件分支**：构建一个"情感路由"图——先判断用户消息的情感（正面/负面），正面走"鼓励节点"，负面走"安慰节点"
3. **记忆测试**：创建一个带 Checkpointer 的 Agent，进行 3 轮对话，测试它是否记住了第 1 轮的信息

---

## 7. 实战项目

> 以下三个项目从易到难，建议至少完成一个。每个项目都综合了前面学到的多个知识点。

### 项目一：智能文档问答助手（难度：⭐⭐）

**场景**：你有一堆技术文档，想要一个能回答问题的助手。

**涉及知识点**：RAG、LCEL、Prompt Template、向量数据库

```python
"""
智能文档问答助手
功能：加载文档 → 分割 → 向量化 → 检索 → 回答
"""
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# === 配置 ===
MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4

# === 第一步：加载和分割文档 ===
def load_and_split(file_path: str):
    """加载文档并分割成小块"""
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ 加载并分割了 {len(chunks)} 个文本块")
    return chunks

# === 第二步：创建向量数据库 ===
def create_vectorstore(chunks):
    """把文档块向量化并存入 Chroma"""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./qa_vectorstore"
    )
    print(f"✅ 向量数据库创建完成，共 {vectorstore._collection.count()} 条记录")
    return vectorstore

# === 第三步：构建 RAG 链 ===
def build_rag_chain(vectorstore):
    """构建完整的 RAG 问答链"""
    model = ChatOpenAI(model=MODEL_NAME, temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "你是一个专业的技术文档助手。请严格根据提供的参考资料回答问题。\n"
         "规则：\n"
         "1. 只使用参考资料中的信息\n"
         "2. 如果资料不足以回答，明确说明\n"
         "3. 回答要简洁清晰，可以用列表格式\n\n"
         "参考资料：\n{context}"),
        ("human", "{question}"),
    ])
    
    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | model
        | StrOutputParser()
    )
    return chain

# === 第四步：交互式问答 ===
def main():
    # 加载文档（替换为你自己的文件路径）
    chunks = load_and_split("my_docs.txt")
    vectorstore = create_vectorstore(chunks)
    rag_chain = build_rag_chain(vectorstore)
    
    print("\n📚 文档问答助手已就绪！输入 'quit' 退出。\n")
    
    while True:
        question = input("👤 你: ").strip()
        if question.lower() in ('quit', 'exit', 'q'):
            print("👋 再见！")
            break
        if not question:
            continue
        
        answer = rag_chain.invoke({"question": question})
        print(f"🤖 助手: {answer}\n")

if __name__ == "__main__":
    main()
```

**运行效果示例**：
```
✅ 加载并分割了 23 个文本块
✅ 向量数据库创建完成，共 23 条记录

📚 文档问答助手已就绪！输入 'quit' 退出。

👤 你: Python 中怎么处理异常？
🤖 助手: 根据文档，Python 中处理异常的方式包括：
1. try/except 语句捕获异常
2. finally 块确保清理代码执行
3. 自定义异常类继承 Exception
...
```

---

### 项目二：多工具 Agent 助手（难度：⭐⭐⭐）

**场景**：一个能查天气、做计算、搜索知识的综合助手，带记忆功能。

**涉及知识点**：Agent、Tool、LangGraph、Checkpointer

```python
"""
多工具 Agent 助手
功能：天气查询 + 数学计算 + 知识搜索 + 多轮记忆
"""
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime
import json

# === 定义工具集 ===

@tool
def get_weather(city: str, date: str = "today") -> str:
    """查询指定城市的天气信息。
    
    Args:
        city: 城市名称，如"北京"、"上海"
        date: 日期，默认"today"表示今天
    """
    # 模拟天气数据（实际项目中调用真实 API）
    weather_db = {
        "北京": {"temp": 22, "condition": "晴", "humidity": "35%", "wind": "北风3级"},
        "上海": {"temp": 26, "condition": "多云", "humidity": "65%", "wind": "东南风2级"},
        "广州": {"temp": 31, "condition": "雷阵雨", "humidity": "80%", "wind": "南风2级"},
    }
    info = weather_db.get(city)
    if info:
        return json.dumps({"city": city, "date": date, **info}, ensure_ascii=False)
    return f"暂无{city}的天气数据"

@tool
def calculate(expression: str) -> str:
    """执行数学计算。支持基本运算和常用数学函数。
    
    Args:
        expression: 数学表达式，如 "2**10" 或 "3.14 * 5**2"
    """
    import math
    allowed_names = {"abs": abs, "round": round, "min": min, "max": max,
                     "sqrt": math.sqrt, "log": math.log, "sin": math.sin,
                     "cos": math.cos, "pi": math.pi, "e": math.e}
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"{expression} = {result}"
    except Exception as err:
        return f"计算错误: {err}"

@tool
def get_datetime(timezone: str = "Asia/Shanghai") -> str:
    """获取当前日期和时间。
    
    Args:
        timezone: 时区，默认中国时区
    """
    try:
        import zoneinfo
        now = datetime.now(zoneinfo.ZoneInfo(timezone))
    except Exception:
        now = datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M:%S (%A)")

@tool
def save_note(title: str, content: str) -> str:
    """保存一条笔记。当用户说"记一下"或"帮我记住"时使用。
    
    Args:
        title: 笔记标题
        content: 笔记内容
    """
    # 实际项目中可以存入数据库
    note = {"title": title, "content": content, "time": datetime.now().isoformat()}
    with open("notes.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(note, ensure_ascii=False) + "\n")
    return f"已保存笔记：{title}"

# === 创建 Agent ===
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
checkpointer = MemorySaver()

agent = create_react_agent(
    model=model,
    tools=[get_weather, calculate, get_datetime, save_note],
    checkpointer=checkpointer,
    state_modifier=(
        "你是一个智能生活助手。规则：\n"
        "1. 需要实时信息时主动使用工具\n"
        "2. 数学计算必须用 calculate 工具，不要自己算\n"
        "3. 回答简洁友好\n"
        "4. 记住用户之前说过的信息"
    )
)

# === 交互式对话 ===
def chat_loop():
    config = {"configurable": {"thread_id": "main-conversation"}}
    print("🤖 智能助手已启动！输入 'quit' 退出\n")
    
    while True:
        user_input = input("👤 你: ").strip()
        if user_input.lower() in ('quit', 'exit', 'q'):
            break
        if not user_input:
            continue
        
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        )
        
        # 显示工具调用过程（可选）
        for msg in result["messages"]:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"  🔧 调用: {tc['name']}({tc['args']})")
        
        print(f"🤖 助手: {result['messages'][-1].content}\n")

if __name__ == "__main__":
    chat_loop()
```

---

### 项目三：结构化数据提取流水线（难度：⭐⭐⭐）

**场景**：从非结构化文本（新闻、评论、简历）中自动提取结构化数据。

**涉及知识点**：Structured Output、Pydantic、LCEL、Batch 处理

```python
"""
结构化数据提取流水线
功能：批量从文本中提取结构化信息
"""
from dotenv import load_dotenv
load_dotenv()

from typing import Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# === 定义输出结构 ===
class NewsArticle(BaseModel):
    """新闻文章的结构化信息"""
    headline: str = Field(description="新闻标题")
    summary: str = Field(description="50字以内的摘要")
    entities: list[str] = Field(description="涉及的人名/组织名")
    category: str = Field(description="分类：科技/财经/体育/娱乐/其他")
    sentiment: str = Field(description="情感倾向：正面/负面/中性")
    keywords: list[str] = Field(description="3-5个关键词")

# === 构建提取链 ===
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_model = model.with_structured_output(NewsArticle)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的新闻分析师，从文本中提取结构化信息。"),
    ("human", "请分析以下新闻：\n\n{text}"),
])

extraction_chain = prompt | structured_model

# === 批量处理 ===
news_texts = [
    "苹果公司今日发布新款 MacBook Pro，搭载 M4 芯片，性能提升 40%。CEO 库克表示这是 Mac 历史上最大的性能飞跃。",
    "国足在世预赛中 0:3 不敌日本队，全场射门仅 2 次。主教练赛后表示需要时间重建球队信心。",
    "特斯拉宣布在上海建设第二座超级工厂，预计投资 50 亿美元，创造 5000 个就业岗位。",
]

print("📊 批量新闻分析结果：\n")
results = extraction_chain.batch([{"text": t} for t in news_texts])

for i, article in enumerate(results, 1):
    print(f"--- 新闻 {i} ---")
    print(f"标题: {article.headline}")
    print(f"摘要: {article.summary}")
    print(f"实体: {article.entities}")
    print(f"分类: {article.category} | 情感: {article.sentiment}")
    print(f"关键词: {article.keywords}")
    print()
```

**预期输出**：
```
📊 批量新闻分析结果：

--- 新闻 1 ---
标题: 苹果发布搭载M4芯片的新款MacBook Pro
摘要: 苹果发布新MacBook Pro，M4芯片带来40%性能提升
实体: ['苹果公司', '库克']
分类: 科技 | 情感: 正面
关键词: ['苹果', 'MacBook Pro', 'M4芯片', '性能提升']

--- 新闻 2 ---
标题: 国足世预赛0:3负于日本
摘要: 国足世预赛大比分落败，全场表现低迷
实体: ['国足', '日本队']
分类: 体育 | 情感: 负面
关键词: ['国足', '世预赛', '日本', '失利']
```

---

## 8. 调试与评估

### 8.1 LangSmith——你的 AI 应用"X 光机"

**生活类比**：如果你的 LLM 应用是一个"黑箱子"，LangSmith 就是给它装了个透明外壳——你可以看到内部每一步发生了什么、花了多少时间、消耗了多少 token。

**配置方法**：
```bash
# 在 .env 文件中添加
LANGSMITH_API_KEY=ls-你的密钥
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=my-first-project
```

注册 LangSmith：访问 https://smith.langchain.com 注册账号（免费层已够学习使用）。

**配置后效果**：你的每次 `invoke()`、`stream()` 调用都会自动被记录，包括：
- 每一步的输入输出
- Token 消耗和费用
- 延迟时间
- 错误信息

### 8.2 常见错误及解决方法

| 错误信息 | 原因 | 解决方法 |
|---------|------|----------|
| `AuthenticationError` | API Key 无效或未设置 | 检查 `.env` 文件和环境变量 |
| `RateLimitError` | 请求太频繁 | 加 `time.sleep()` 或升级付费计划 |
| `OutputParserException` | 模型输出格式不符合预期 | 改用 `with_structured_output()` |
| `InvalidRequestError: model not found` | 模型名称写错 | 检查模型名是否正确 |
| `ContextWindowExceeded` | 输入太长 | 减少文档块大小或换用更大窗口的模型 |
| `ImportError: cannot import name` | 包版本不对 | `pip install --upgrade langchain langchain-openai` |

### 8.3 调试技巧

```python
# 技巧 1：设置 verbose 模式查看详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 技巧 2：用 .with_config() 给链加上标签，方便在 LangSmith 中追踪
chain = (prompt | model | parser).with_config(
    tags=["qa-chain", "production"],
    metadata={"version": "1.0"}
)

# 技巧 3：单独测试每个组件
# 先测 prompt 是否正确填充
print(prompt.invoke({"question": "测试"}).to_messages())

# 再测模型是否能正常返回
print(model.invoke("hello"))

# 最后测 parser 是否能正确解析
print(parser.invoke(model.invoke("hello")))
```

### 8.4 评估你的 RAG 系统

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 用 LLM 来评估 RAG 的回答质量（LLM-as-Judge）
eval_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个评估专家。评估回答的质量，给出 1-5 分和理由。"),
    ("human", "问题：{question}\n参考答案：{reference}\n实际回答：{answer}\n\n请评分："),
])

eval_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
eval_chain = eval_prompt | eval_model | StrOutputParser()

score = eval_chain.invoke({
    "question": "Python 怎么读文件？",
    "reference": "使用 open() 函数配合 with 语句",
    "answer": "可以用 open() 打开文件，建议用 with 语句自动关闭"
})
print(score)
# 预期输出：4/5 - 回答正确涵盖了核心要点...
```

---

## 9. 进阶方向

学完本手册后，你可以根据兴趣选择以下方向深入：

### 9.1 进阶技术路线

| 方向 | 关键技术 | 适用场景 |
|------|---------|----------|
| 多模态应用 | 图片/音频输入、GPT-4V | 图片描述、文档 OCR |
| 多 Agent 协作 | LangGraph 多节点图 | 复杂工作流、团队协作 |
| 本地部署 | Ollama + 开源模型 | 数据隐私、无网络环境 |
| 生产部署 | LangServe + FastAPI | 将链变成 REST API |
| 高级 RAG | Multi-query、Reranking、HyDE | 提升检索质量 |
| 评估体系 | LangSmith Evaluation | 自动化测试 LLM 应用 |

### 9.2 推荐学习资源

| 资源 | 类型 | 说明 |
|------|------|------|
| [LangChain 官方文档](https://python.langchain.com) | 文档 | 最新 API 参考 |
| [LangGraph 文档](https://langchain-ai.github.io/langgraph/) | 文档 | Agent 和工作流 |
| [LangChain Academy](https://academy.langchain.com) | 视频课 | 官方免费课程 |
| [LangSmith 文档](https://docs.smith.langchain.com) | 文档 | 调试和监控 |
| DeepLearning.AI 短课 | 视频课 | Andrew Ng 合作的 LangChain 课程 |

### 9.3 从学习到实战的建议

1. **先跑通再优化**：不要追求完美架构，先让功能跑起来
2. **用便宜模型开发**：开发调试时用 gpt-4o-mini，上线时再换 gpt-4o
3. **Prompt 是核心**：80% 的效果差异来自 prompt 质量，而不是代码架构
4. **加监控再上线**：LangSmith 追踪是必须的，否则出了问题无法定位
5. **处理边界情况**：模型返回空、超时、格式错误——都要有备选方案

---

## 10. 30 天学习计划

### 第 1 周：基础打牢

| 天数 | 内容 | 交付物 |
|------|------|--------|
| Day 1 | 环境搭建 + Step 1（模型调用） | 跑通第一个 invoke/stream |
| Day 2 | Step 2（Prompt Template） | 完成 3 个“试一试”练习 |
| Day 3 | Step 3（Output Parser） | 用 PydanticOutputParser 提取结构化数据 |
| Day 4 | Step 4（LCEL 管道） | 构建一个包含并行分支的链 |
| Day 5 | Step 5（结构化输出） | 完成新闻提取小项目 |
| Day 6-7 | 复习 + 自由练习 | 用所学知识做一个小工具 |

### 第 2 周：RAG 深入

| 天数 | 内容 | 交付物 |
|------|------|--------|
| Day 8 | Step 6（RAG 基础） | 跑通文档加载→分割→存储→检索全流程 |
| Day 9 | RAG 优化：chunk_size、overlap 调参 | 对比不同参数的检索质量 |
| Day 10 | 多文档加载（PDF、网页） | 加载 3 种不同格式的文档 |
| Day 11 | 实战项目一：文档问答助手 | 完成可交互的问答系统 |
| Day 12 | 配置 LangSmith + 调试技巧 | 能看懂 trace 日志 |
| Day 13-14 | 复习 + 优化项目一 | 添加错误处理和用户友好提示 |

### 第 3 周：Agent 与工具

| 天数 | 内容 | 交付物 |
|------|------|--------|
| Day 15 | Step 7（Tool 定义） | 定义 3 个自定义工具 |
| Day 16 | Step 8（Agent 基础） | 跑通 create_react_agent |
| Day 17 | Agent 多步推理 + 记忆 | 完成多轮对话 Agent |
| Day 18 | Step 9（LangGraph StateGraph） | 构建带条件分支的工作流 |
| Day 19 | 实战项目二：多工具助手 | 完成带记忆的交互式助手 |
| Day 20-21 | 复习 + 优化项目二 | 添加更多工具和更好的错误处理 |

### 第 4 周：综合实战

| 天数 | 内容 | 交付物 |
|------|------|--------|
| Day 22 | 实战项目三：数据提取流水线 | 批量提取结构化数据 |
| Day 23 | 学习 Multi-Agent 协作 | 两个 Agent 分工合作的示例 |
| Day 24 | 学习 LangServe 部署 | 把链变成 REST API |
| Day 25 | 自选项目：确定方向和设计 | 项目设计文档 |
| Day 26-28 | 自选项目开发 | 完成核心功能 |
| Day 29 | 测试和优化 | 处理边界情况、添加监控 |
| Day 30 | 总结和分享 | 写一篇学习总结或 demo 展示 |

---

## 附录：常用 API 速查表

| 功能 | 代码 | 返回值 |
|------|------|--------|
| 创建模型 | `ChatOpenAI(model="gpt-4o-mini")` | ChatModel (Runnable) |
| 通用创建模型 | `init_chat_model("gpt-4o-mini", model_provider="openai")` | ChatModel (Runnable) |
| 单次调用 | `model.invoke("你好")` | AIMessage |
| 流式调用 | `model.stream("你好")` | Iterator[AIMessageChunk] |
| 批量调用 | `model.batch(["你好", "世界"])` | list[AIMessage] |
| 创建 Prompt | `ChatPromptTemplate.from_template("你好{ name}")` | ChatPromptTemplate |
| 结构化输出 | `model.with_structured_output(MyModel)` | Runnable → MyModel |
| 绑定工具 | `model.bind_tools([tool1, tool2])` | Runnable |
| 创建 Agent | `create_react_agent(model, tools)` | CompiledGraph |
| 向量库 | `Chroma.from_documents(docs, embeddings)` | Chroma |
| 检索器 | `vectorstore.as_retriever(search_kwargs={"k": 3})` | Retriever |
| 文本分割 | `RecursiveCharacterTextSplitter(chunk_size=500)` | TextSplitter |
| 构建图 | `StateGraph(MyState)` | StateGraph Builder |
| 编译图 | `builder.compile(checkpointer=saver)` | CompiledGraph |
| 管道组装 | `prompt \| model \| parser` | RunnableSequence |
| 并行执行 | `RunnableParallel({"a": chain_a, "b": chain_b})` | Runnable |

---

> **写在最后**：学习 LangChain 最重要的原则是——**动手敲代码**。看 10 遍不如自己写 1 遍。每个 Step 的“试一试”练习一定要做，遇到困难先查官方文档，再搜索解决方案。祝你学习顺利！🚀
