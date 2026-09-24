# AI Agent 框架系统性调研报告

> 调研时间：2026 年 9 月  
> 数据基准：以 2025 年 10 月—2026 年 8 月各主流框架公开信息为主，GitHub Star 数为 2026 年 8 月中旬快照  
> 目标读者：需要为团队做 AI Agent 技术选型的架构师、Tech Lead、产品负责人

---

## 1. AI Agent 框架的基本概念

### 1.1 什么是 AI Agent

AI Agent（智能体）是**以大语言模型为决策核心，能够自主感知环境、规划任务、调用工具、执行动作并根据反馈迭代的软件系统**。它与传统聊天机器人的核心差异在于：

| 特性 | 传统 LLM 应用（Chat） | AI Agent |
|---|---|---|
| 执行模式 | 单轮/多轮问答，被动响应 | 主动规划、循环执行、自主决策 |
| 与外部世界的交互 | 无或极少 | 通过 Tool Use 调用 API、执行代码、读写文件、操作数据库 |
| 状态管理 | 上下文窗口内 | 短期记忆 + 长期记忆 + 持久化 Checkpoint |
| 错误处理 | 由用户重试 | 自动反思（Reflection）、重试、回退 |
| 生命周期 | 一次请求即结束 | 可跨小时/天/周的长时任务 |

### 1.2 什么是 AI Agent 框架

Agent 框架是对"构建 Agent 所需通用能力"的抽象和封装，通常提供：**Agent 循环（Reason→Act→Observe）、工具注册与调度、状态与记忆管理、多 Agent 协作原语、可观测性、评估与安全护栏**等基础设施。开发者在其上编写业务逻辑，无需从零实现 LLM 交互协议、工具调用解析、状态持久化等重复劳动。

### 1.3 与传统工作流编排系统的区别

- **传统工作流（Airflow / n8n / Temporal）**：流程**由开发者预先确定**，节点是确定性函数，路径可枚举。
- **Agent 框架**：流程**部分由 LLM 运行时决定**（动态路由、工具选择、下一步动作），存在不可预测性，需要 Checkpoint、Reflection、Guardrail 等能力兜底。
- **两者的融合趋势**：2025 年后主流框架（LangGraph、CrewAI Flows、LlamaIndex Workflows、Microsoft Agent Framework、Google ADK）都在提供"确定性图 + LLM 节点"的混合模式——**关键路径确定化，模糊决策交给 LLM**，这也是当前企业级 Agent 落地的最佳实践。

---

## 2. 核心能力拆解

一个成熟 Agent 框架通常覆盖以下 9 类能力：

| 能力域 | 说明 | 代表实现 |
|---|---|---|
| **任务规划（Planning）** | 将复杂目标拆解为子任务，ReAct / Plan-and-Execute / Reflexion 等范式 | LangGraph 条件边、CrewAI Hierarchical Process、MetaGPT SOP、Google ADK 动态路由 |
| **工具调用（Tool Use）** | 函数式工具、REST API、代码执行、MCP Server 接入 | 所有主流框架均已支持 **MCP（Model Context Protocol）**，成为事实标准 |
| **记忆管理（Memory）** | 短期对话记忆、长期向量记忆、语义/情节/程序性记忆分层 | LangGraph Store、LlamaIndex Memory Blocks、Mem0、Zep、Letta（原 MemGPT） |
| **上下文管理（Context）** | Token 预算、上下文压缩、检索片段拼接、多轮摘要 | LlamaIndex、Haystack、LangChain Context Compression |
| **RAG 集成** | 文档解析、切分、Embedding、向量库、混合检索、Rerank | LlamaIndex（业界最强）、Dify、RAGFlow、Haystack |
| **多智能体协作** | Supervisor / Handoff / GroupChat / Sequential / Hierarchical | AutoGen GroupChat、CrewAI Crew、OpenAI Agents SDK Handoffs、Google A2A 协议 |
| **状态管理（State）** | 显式状态容器、Checkpoint 持久化、断点续跑 | LangGraph Checkpointer（PostgreSQL/Redis）、MAF Session State、Temporal-backed PydanticAI |
| **可观测性（Observability）** | Trace、Token 计量、成本监控、失败聚类、Prompt 版本管理 | LangSmith、Langfuse（MIT，OTel 原生）、Arize Phoenix、Datadog LLM Observability、AgentOps |
| **评估与安全（Eval & Guardrails）** | LLM-as-Judge、离线数据集、在线打分、输入/输出护栏、PII 脱敏、人工审核 | DeepEval、RAGAS、Braintrust、OpenAI Guardrails、NeMo Guardrails、LangGraph HITL |

**关键趋势（2025-2026）**：
1. **MCP 成为工具接入标准**：Anthropic 提出、OpenAI/Google/Microsoft 全部跟进；框架不再比拼"内置多少工具"，而是比拼 MCP 生态接入质量。
2. **A2A（Agent-to-Agent）协议兴起**：Google 主导、Linux Foundation 托管，v1.0.0 已发布，用于**跨框架、跨厂商的 Agent 互联**，与 MCP（Agent→Tool）形成互补。
3. **Durable Execution（持久化执行）**：PydanticAI 2.x、LangGraph Platform、Temporal 集成让 Agent 具备"跨 API 故障/进程重启存活"能力。
4. **可观测性下沉为标配**：OpenTelemetry GenAI 语义约定统一 Trace 格式，LangSmith/Langfuse 双雄格局。

---

## 3. 主流框架概览

### 3.1 LangChain / LangGraph（LangChain Inc.）

- **定位**：LangChain 是 LLM 应用**组件库**，LangGraph 是其**图状态机编排引擎**，两者共同构成生产级 Agent 事实标准。
- **关键节点**：
  - 2024 年 2 月 LangGraph 开源；2025 年 8 月 0.7 版原生支持 Redis/PostgreSQL Checkpointer、子图、MCP。
  - **2025 年 10 月 LangChain v1.0.0 + LangGraph 1.0 正式发布**，API 冻结至 2.0，移除旧 Chains/Agents（迁至 `langchain-classic`），Python/JS-TS 能力对齐。
  - 2026 年 3 月 LangGraph 1.1：Pydantic 强类型、事件流协议统一、Deep Agents 开源示例。
  - 2026 年 6 月 1.2.x：动态图修改、HITL 状态回滚、错误边界。
- **数据（2026-08）**：LangChain ~144k stars，LangGraph ~40k stars，LangGraph 月下载量 ~3450 万次。
- **配套**：LangSmith（可观测性/评估 SaaS）、LangGraph Platform（云托管、多租户、流式 API）。
- **优势**：生态最完整、生产就绪度最高、精确控制流程、Checkpoint 断点续跑、HITL 内建。
- **劣势**：学习曲线陡峭（图/状态/条件边概念多）、抽象层深、过度封装争议。

### 3.2 LlamaIndex（LlamaIndex Inc.）

- **定位**：**RAG-first 的 Agent 框架**，2024 年从纯 RAG 库演进为 Workflows + Agents 双引擎。
- **核心**：
  - **Workflows**：事件驱动、异步优先，`@step` 装饰器定义步骤，通过类型化事件通信，支持循环/并行/条件/暂停恢复。
  - **AgentWorkflow / Document Agents**：多 Agent 编排、每文档独立 Agent、Router Agent。
  - **LlamaCloud / LlamaParse**：企业级文档解析（业界公认最强之一）。
- **语言**：Python + TypeScript（`llamaindex` npm）。
- **优势**：RAG 能力天花板、数据连接器（LlamaHub）丰富、文档处理一站式。
- **劣势**：多 Agent 编排能力弱于 LangGraph/CrewAI，通用 Agent 场景需要与外部框架混用。

### 3.3 AutoGen / AG2 / Microsoft Agent Framework（微软）

**演化史**：

| 时间 | 事件 |
|---|---|
| 2023 | 微软研究院发布 AutoGen（对话驱动多 Agent 框架） |
| 2025 年 1 月 | AutoGen v0.4 重构：分层架构、异步 Actor 模型 |
| 2025 年末 | **AutoGen 进入维护模式**（只修 Bug 和安全，不加新特性） |
| 2025 年 | 社区 Fork 出 **AG2**（原班创始团队主导），定位"开源 AgentOS"，最新 0.9.8 (2025-08) |
| 2026 年 4 月 3 日 | **Microsoft Agent Framework (MAF) GA**：AutoGen + Semantic Kernel 合并继任者，Python + .NET 双栈，深度绑定 Azure AI Foundry，原生 MCP + A2A |
| 2026 年 7 月 | MAF Go SDK 发布 |

**Semantic Kernel**：微软企业级 SDK（C#/Python/Java），1.42.0 (2026-05)，正逐步迁移到 MAF；插件/规划器模型，与 Azure/M365 集成优秀。

**选型建议**：新项目**不要再从 AutoGen 起步**；.NET/Azure 团队直接选 MAF；喜欢 AutoGen 对话范式且需开源可控的选 AG2。

### 3.4 CrewAI（CrewAI Inc.）

- **定位**：**角色驱动的多 Agent 协作框架**，模拟人类团队（Role + Goal + Backstory）。
- **数据**：GitHub ~52.8k stars，月下载量 ~520 万，2026-06 版本 1.15.1，社区认证开发者超 10 万。
- **核心**：
  - **Crews**：Agent 团队，Sequential / Hierarchical 两种协作模式。
  - **Flows**（2024 末新增）：事件驱动的确定性工作流，弥补 Crews 无法处理循环/精细状态的短板。
  - **AMP Suite**（企业版）：可观测性、部署、监控。
- **优势**：**学习曲线最平缓**，50 行代码跑通多 Agent；不依赖 LangChain；文档友好。
- **劣势**：状态隐式管理（进程重启丢失）、生产可观测性偏弱、A2A 支持缺失、复杂非线性流程受限。

### 3.5 OpenAI Agents SDK（OpenAI）

- **定位**：OpenAI 官方轻量 Agent 框架，Swarm（2024 实验项目）的生产级继任者。
- **发布**：2025 年 3 月发布 Python 版，2026 年 3 月发布独立 JS/TS 版（`@openai/agents`）。
- **五个核心原语**：Agents、Handoffs（任务移交）、Guardrails、Sessions、Tracing。
- **版本迭代（Python）**：0.1.0 (2025-11) → 0.17.7 (2026-06-24)，累计 70+ 版本，**当前迭代最快的 Agent SDK**。
  - 0.10.0 Guardrails；0.11.0 RealtimeAgent（语音）；0.14.0 SandboxAgent（隔离执行）；0.15.0 provider-agnostic（100+ LLM）。
- **优势**：极简、贴近原生 Python、Handoff 原语优雅、内建 Tracing、支持 MCP、语音/实时能力领先。
- **劣势**：企业级治理能力仍在完善，复杂多分支场景表达力弱于 LangGraph。
- **数据**：~29k stars。

### 3.6 smolagents（Hugging Face）

- **定位**：极简 Agent 框架，**核心逻辑仅约 1000 行 Python**，Apache 2.0/MIT。
- **核心理念**：**CodeAgent — "代码即行动"**，让 LLM 直接编写并执行 Python 代码块，而不是 JSON 工具调用，可读性和灵活性双高。
- **能力**：支持 100+ 模型（通过 LiteLLM）、多模态、E2B/Modal/Docker 沙箱、层级 Manager-Worker 多 Agent。
- **优势**：与 HuggingFace Hub 深度集成、开源权重模型首选、行为透明可调试。
- **劣势**：企业治理能力有限，不适合作为大型多 Agent 编排框架。
- **适合**：教学、PoC、单 Agent 自动化、研究型任务。

### 3.7 Dify（LangGenius / 苏州语灵）

- **定位**：**开源 LLMOps + Agentic Workflow 平台**，可视化编排，非纯代码框架。
- **数据**：GitHub **155k+ stars（2026 年中，国内开源 AI 项目 Star 王）**；累计应用 100 万+；服务 2000+ 团队、280+ 企业客户；2026 年 3 月完成 3000 万美元 Pre-A 融资（红杉中国领投）。
- **版本**：v1.17.x（2026 年中），1.14.1（2026-05）聚焦安全加固、私有化优化。
- **能力**：Workflow 画布、Prompt IDE、RAG 管道、模型管理、可观测性、插件市场、Agent 节点。
- **协议**：Dify Open Source License（基于 Apache 2.0，附加多租户/Logo 保留等限制条款，商用需细读）。
- **优势**：私有化部署最成熟的开源方案、可视化门槛低、中文社区活跃、模型无关。
- **劣势**：复杂逻辑受画布表达力限制、企业级功能需付费版、代码扩展不如原生框架灵活。

### 3.8 Coze / 扣子（字节跳动）

- **定位**：**零代码/低代码 Bot 开发平台**，深度绑定字节生态（豆包、抖音、飞书、剪映）。
- **版本**：3.0（2026-06）支持多人多 Agent 协作、项目空间。
- **形态**：SaaS 为主（coze.cn / coze.com），部分开源（`coze-studio`）。
- **优势**：**上手最快（15 分钟搭出客服 Agent）**、渠道分发能力强、中文对话体验一流、插件生态丰富。
- **劣势**：闭源、数据出域风险、深度定制受限、企业级私有化能力弱于 Dify。
- **适合**：内容/营销/客服类 Bot、非技术团队快速验证。

### 3.9 MetaGPT（FoundationAgents）

- **定位**：**"软件公司 SOP"多 Agent 框架**，一句话需求 → PRD/架构/代码/测试。
- **数据**：**69.8k stars**（2026-01 数据，仓库已迁至 `FoundationAgents/MetaGPT`），ICLR 2024/2025 双 Oral。
- **配套**：MGX（MetaGPT X）商业化产品（2025-02）、Data Interpreter、AFlow、SPO 等研究论文。
- **优势**：软件工程场景最成熟、支持所有主流 LLM、可离线（Ollama+Qwen）运行。
- **劣势**：领域较窄（偏软件开发），通用 Agent 场景需二次改造。

### 3.10 CAMEL-AI

- **定位**：**研究型多 Agent 框架**，源自 NeurIPS 2023 论文《Communicative Agents for "Mind" Exploration》。
- **核心**：Role-Playing + Inception Prompting，AI User ↔ AI Assistant 自主对话完成任务。
- **数据**：`camel-ai/camel` **17.7k stars**，Apache 2.0，PyPI `camel-ai` 0.2.90（2026-03）。
- **生态**：OWL（多 Agent 任务自动化，**20k+ stars**）、OASIS（社会模拟）。
- **优势**：研究/仿真/数据生成场景独占鳌头、论文引用高、社区活跃。
- **劣势**：产品化能力弱，生产环境需大量包装。

### 3.11 PydanticAI（Pydantic 团队）

- **定位**：**"FastAPI for Agents"**，类型安全优先。
- **版本**：1.0 (2025-09) → 2.x beta（2026-06）。
- **能力**：Pydantic 校验结构化输出、依赖注入、Durable Execution（跨 API 故障保存进度）、类型提示驱动的图系统、OpenTelemetry 内建、支持 30+ 模型（OpenAI/Anthropic/Gemini/DeepSeek/Grok/Cohere/Mistral）。
- **优势**：Python 类型安全最佳体验、代码量少于 LangGraph、与 FastAPI 微服务天然契合。
- **劣势**：多 Actor 复杂分支/Checkpoint 场景仍需 LangGraph；生态较新（ThoughtWorks Radar 2025-11 移出主榜但仍是 Assess→Trial 路径上的稳健选项）。

### 3.12 其他重要框架

| 框架 | 定位 | 关键信息 |
|---|---|---|
| **Google ADK** | 谷歌代码优先 Agent 框架 | 2025-04 Cloud Next 发布；2.2.0 GA (2026-06)；Python/TS/Go/Java/Kotlin；**A2A 协议原生**；20k+ stars；Vertex AI Agent Engine 一键托管 |
| **Mastra** | TypeScript 优先全栈 Agent 框架 | Gatsby 创始团队 2024-10 创建；**27.4k stars**；140+ 包（Storage/Voice/Auth/Deployer/Playground）；Vercel/Netlify/Cloudflare 部署首选 |
| **Vercel AI SDK** | 前端友好 TS Agent SDK | v6 起 Agents 稳定，MCP/Tool Loop/Realtime API 齐备，Next.js 生态无缝 |
| **AgentScope** | 阿里达摩院分布式多 Agent | 1.0.20 (2026-05)，v1.0 论文 arXiv 2508.16279，高密度抽象、生产级分布式 |
| **Strands Agents** | AWS 开源、模型驱动 | Python + TS，Bedrock 一等公民，OTel Trace，1.x 活跃 |
| **Haystack（deepset）** | 管道式 RAG/Agent | 企业文档智能场景强，显式 Pipeline |
| **Claude Agent SDK** | Anthropic 官方 | 8k stars，与 Claude Code 集成，Artifacts、嵌套子 Agent、`/cd` fallback model |
| **AG2** | AutoGen 社区 Fork | 0.9.8 (2025-08)，AgentOS 定位，节奏放缓 |
| **Letta（MemGPT）** | 记忆型 Agent | 长期记忆能力业界标杆 |
| **Agno** | 轻量高性能 Python Agent | 强调速度与简洁 |

---

## 4. 横向对比分析

### 4.1 主表：主流框架能力矩阵（2026-08 数据）

| 框架 | 语言 | GitHub Stars | 最新版 | 编排范式 | 状态持久化 | 多 Agent | MCP | A2A | 可观测性 | 学习曲线 | 生产就绪 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **LangGraph** | Py / TS | ~40k（LangChain 144k） | 1.2.x (2026-06) | 图状态机 | Checkpointer（Redis/PG）原生 | 强（子图/Supervisor） | 原生 | 通过集成 | LangSmith 深度 | 陡峭 | 极高 |
| **CrewAI** | Python | ~52.8k | 1.15.1 (2026-06) | Role+Crew / Flows | 内存为主，Enterprise 增强 | 强（Sequential/Hierarchical/Handoff） | 社区适配 | 缺失 | 基础日志+AMP | 平缓 | 中-高 |
| **Microsoft Agent Framework** | Py / .NET / Go | 新（AutoGen+SK 合并） | GA (2026-04) | Agent + Workflow | Session State 原生 | 强 | 原生 | 原生 | OTel/Azure Monitor | 中等 | 极高 |
| **Semantic Kernel** | C#/Py/Java | ~25k | 1.42.0 (2026-05) | Plugin + Planner | 中 | 中 | 支持 | 支持 | Azure 集成 | 中等 | 高（迁 MAF 中） |
| **AG2 (AutoGen Fork)** | Python | ~45k（AutoGen 原库） | 0.9.8 (2025-08) | GroupChat | 有限 | 强（对话式） | 社区插件 | 弱 | 基础 | 中等 | 中 |
| **OpenAI Agents SDK** | Py / TS | ~29k | 0.17.7 / 0.11.4 | Handoff Chain | Sessions（含 Sandbox） | 中（Handoff+Agents-as-Tools） | 一等公民 | 弱 | 内建 Tracing | 平缓 | 高 |
| **LlamaIndex** | Py / TS | ~40k | 持续迭代 | Workflows（事件驱动） | Context/Storage 组件 | 中 | 支持 | 弱 | 内建 Tracing | 中等 | 高（RAG 场景极高） |
| **PydanticAI** | Python | ~15k+ | 2.x beta | 类型驱动 + Graph | Durable Execution | 中 | 支持 | 弱 | OTel 内建 | 平缓（Python 熟手） | 高 |
| **smolagents** | Python | ~17k | 持续迭代 | CodeAgent 循环 | 轻量 | Manager-Worker | 支持 | 无 | 基础 | 极平缓 | 中（PoC 优先） |
| **MetaGPT** | Python | ~69.8k | 0.8.x | SOP + Role | 序列化 | 强（软件公司 SOP） | 有限 | 无 | 基础 | 中等 | 中（垂直场景高） |
| **CAMEL-AI** | Python | ~17.7k | 0.2.90 (2026-03) | Role-Playing | 有限 | 强（研究导向） | 支持 | 无 | 基础 | 中等 | 低-中 |
| **Google ADK** | Py/TS/Go/Java/Kotlin | ~20k | 2.2.0 GA (2026-06) | 事件驱动 Runner | Session/Memory Service | 强（层级+路由） | 原生 | **原生（主推）** | Cloud Trace | 中等 | 极高（GCP 场景） |
| **Mastra** | TypeScript | ~27.4k | 1.35+ | Workflow + Agent | Storage 抽象 | 中 | 支持 | 弱 | 内建 Playground | 平缓（TS 熟手） | 高 |
| **Dify** | Py/TS（平台） | ~155k | v1.17.x | 可视化画布 | 平台管理 | 中 | 插件 | 弱 | 平台内建 | 极低 | 高（私有化标杆） |
| **Coze 扣子** | 平台 | 部分开源 | 3.0 (2026-06) | 可视化画布 | 平台管理 | 中 | 插件 | 无 | 平台内建 | 极低 | SaaS 高，私有化受限 |

*注：Star 数为 2026-08 前后公开数据快照，实时数据以 GitHub 为准。*

### 4.2 分维度深度对比

#### 生态成熟度
- **顶级**：LangChain/LangGraph（700+ 集成）、Dify（100 万+ 应用）、LlamaIndex（LlamaHub 数据连接器）。
- **快速追赶**：CrewAI、OpenAI Agents SDK、Google ADK。
- **垂直强势**：MetaGPT（软件工程）、CAMEL（研究仿真）、Haystack（企业搜索）。

#### 开发效率（原型速度）
Coze ≈ Dify > CrewAI ≈ smolagents ≈ OpenAI Agents SDK > Mastra > PydanticAI > LangGraph > AutoGen/AG2 > MetaGPT。

#### 学习成本（越低越好）
Coze / Dify（零代码）< CrewAI（角色直观）< smolagents（1000 行核心）< OpenAI Agents SDK（贴近原生）< PydanticAI（Python 类型熟手友好）< Mastra < LlamaIndex < Google ADK < MAF < LangGraph（图/状态/条件边）< MetaGPT/CAMEL（研究概念多）。

#### 可扩展性
- **代码级最强**：LangGraph、MAF、Google ADK、PydanticAI（类型系统 + 依赖注入）。
- **插件生态最强**：Dify（插件市场）、LangChain（Community 包）、Mastra（140+ 包）。
- **协议级开放**：Google ADK（A2A 原生）、MAF（MCP+A2A）、OpenAI Agents SDK（MCP 一等）。

#### 可维护性
- 强类型语言/框架（PydanticAI、MAF .NET、Google ADK Java）在大型团队协作中优势明显。
- LangGraph 的显式状态图便于 Code Review 与回归测试。
- CrewAI/AutoGen 隐式状态和对话式流程在长期演进中容易变成"黑盒"。

#### 部署复杂度

| 层级 | 代表 | 复杂度 |
|---|---|---|
| 托管 SaaS | Coze、LangGraph Platform、Vertex AI Agent Engine、Azure AI Foundry | 极低 |
| 一键容器 | Dify（Docker Compose 2C4G 起）、FastGPT | 低 |
| 自建服务 | LangGraph、CrewAI、MAF、OpenAI Agents SDK、PydanticAI | 中 |
| 分布式集群 | AgentScope、MetaGPT + 消息队列 | 高 |

#### 调试能力
- **可视化调试最强**：LangSmith（LangGraph 图可视化、状态回溯、单步重试）、Dify（v1.5 起实时工作流调试）、Mastra Playground。
- **代码级调试**：PydanticAI（Logfire 同门）、smolagents（Python 代码可读）。
- **弱项**：CrewAI、AG2、CAMEL 主要靠日志。

#### 测试能力
- **框架内建**：PydanticAI（`pydantic_ai.agent` 测试友好）、LangGraph（`langsmith` 数据集评估）、Google ADK（Evaluator）。
- **第三方评估栈**：DeepEval 4.1+（14.7k stars）、RAGAS 0.2+（13.3k stars）、Braintrust、Langfuse Evaluations、Arize Phoenix。

#### 社区活跃度（2026-08 快照）

| 框架 | Stars | 提交频率 | 贡献者 | 备注 |
|---|---|---|---|---|
| Dify | 155k+ | 极高 | 大量 | 国内开源 AI Star 王 |
| LangChain | 144k | 极高 | 数千 | 生态核心 |
| MetaGPT | 69.8k | 中 | 中 | 学术驱动 |
| CrewAI | 52.8k | 高 | 中 | 认证开发者 10 万+ |
| AutoGen | 45k+ | 低（维护模式） | 中 | 已停新特性 |
| LangGraph | 40k | 极高（双周） | 大量 | 生产标准 |
| LlamaIndex | 40k+ | 高 | 大量 | RAG 王者 |
| OpenAI Agents SDK | 29k | 极高（半年 70+ 版本） | 增长中 | 增速第一 |
| Mastra | 27.4k | 极高（16k+ commits） | ~520 | TS 生态第一 |
| Semantic Kernel | 25k | 中 | 大量 | 迁 MAF 中 |
| Google ADK | 20k+ | 高 | 增长中 | 官方主推 |
| CAMEL / OWL | 17.7k / 20k+ | 中 | 中 | 研究导向 |
| smolagents | 17k+ | 中 | 中 | HF 背书 |
| PydanticAI | 15k+ | 高 | ~520 | Pydantic 团队 |

#### 企业适用性
- **强合规/大企业首选**：Microsoft Agent Framework（Azure）、Google ADK（GCP）、Dify 企业版（私有化）、LangGraph Platform。
- **金融/医疗等严监管**：LangGraph + LangSmith（审计 Trace）、PydanticAI（类型护栏）、MAF（企业治理）。
- **中小企业快速落地**：Dify、Coze、CrewAI。

#### 编程语言支持

| 语言 | 首选框架 |
|---|---|
| **Python** | LangGraph、CrewAI、LlamaIndex、PydanticAI、OpenAI Agents SDK、smolagents、MetaGPT、CAMEL、AgentScope |
| **TypeScript / JS** | Mastra、Vercel AI SDK、LangGraph.js、LlamaIndex.TS、OpenAI Agents SDK (JS)、Dify（前端） |
| **.NET / C#** | Microsoft Agent Framework、Semantic Kernel |
| **Java / Kotlin** | Semantic Kernel for Java、Google ADK、Spring AI、LangChain4j |
| **Go** | Google ADK-Go v2.0（2026-06 GA）、MAF Go SDK（2026-07）、Eino（字节） |

---

## 5. 典型应用场景与推荐框架

| 场景 | 首选 | 备选 | 关键理由 |
|---|---|---|---|
| **智能客服** | LangGraph / Dify | MAF、Coze | 需要 Checkpoint、HITL 转人工、可审计 |
| **企业知识库 RAG** | LlamaIndex / Dify / RAGFlow | Haystack、LangChain | 文档解析、混合检索、Rerank |
| **数据分析 Agent** | LangGraph + 代码执行 | Google ADK、PydanticAI | 长时任务、动态代码生成 |
| **自动化办公 / RPA 增强** | Dify / Coze / Mastra | n8n + LangChain | 可视化编排、渠道分发 |
| **代码助手 / 软件工程 Agent** | MetaGPT / Claude Agent SDK | OpenAI Agents SDK + Sandbox、CAMEL-OWL | 领域 SOP、沙箱执行 |
| **深度研究 / Deep Research** | LangGraph Deep Agents / LlamaIndex | CAMEL、smolagents | 子 Agent 嵌套、任务规划 |
| **多 Agent 仿真 / 学术研究** | CAMEL-AI / AgentScope | MetaGPT、AutoGen | Role-Playing、大规模仿真 |
| **实时语音 Agent** | OpenAI Agents SDK RealtimeAgent | LiveKit + Mastra、Google ADK | 语音打断、Turn 管理 |
| **金融/医疗合规场景** | LangGraph + LangSmith / MAF | PydanticAI + Logfire | 全链路 Trace、审计、类型护栏 |
| **国内私有化部署** | Dify / FastGPT | LangGraph 自建、AgentScope | 中文生态、数据主权 |
| **GCP / Azure 云原生** | Google ADK / MAF | 相应云 SDK | 与云 IAM/监控/托管深度绑定 |
| **TypeScript 全栈产品** | Mastra / Vercel AI SDK | LangGraph.js | Edge/Serverless 部署、前后端一体 |

---

## 6. 选型建议

### 6.1 决策树（简化版）

```
是否是"零代码/业务人员搭建"？
├─ 是 → 国内选 Coze 扣子 / 需要私有化选 Dify
└─ 否 ↓

主要语言栈？
├─ .NET / Azure → Microsoft Agent Framework（不再推荐 AutoGen/SK 单独使用）
├─ TypeScript / 全栈 Web → Mastra 或 Vercel AI SDK
├─ Go → Google ADK-Go v2 / MAF Go
├─ Java → Spring AI / Google ADK / Semantic Kernel Java
└─ Python ↓

场景是"重 RAG / 文档密集"？
├─ 是 → LlamaIndex（Workflows + Agents）
└─ 否 ↓

需要"生产级 + 长时任务 + 精细控制 + 断点续跑"？
├─ 是 → LangGraph（+ LangSmith）
└─ 否 ↓

追求"极简类型安全、单/少 Agent 微服务"？
├─ 是 → PydanticAI
└─ 否 ↓

追求"最快原型 + 角色分工直观"？
├─ 是 → CrewAI
└─ 否 ↓

追求"官方轻量 + Handoff 语义 + 语音/实时"？
├─ 是 → OpenAI Agents SDK
└─ 否 ↓

研究 / 仿真 / 学术？
├─ 是 → CAMEL-AI / AgentScope / MetaGPT
└─ 教学/PoC/极简 → smolagents
```

### 6.2 按团队规模 & 项目阶段

| 情境 | 推荐 | 理由 |
|---|---|---|
| **单人 / 快速原型（≤2 周）** | CrewAI 或 smolagents 或 Coze/Dify | 上手最快，50 行代码见效果 |
| **小团队 MVP（1-3 人，1-2 月）** | PydanticAI / OpenAI Agents SDK / Mastra（TS） | 类型安全、代码可控、依赖轻 |
| **中型生产应用（3-10 人）** | LangGraph + LangSmith 或 LlamaIndex + Langfuse | 精确控制、可观测、可评估 |
| **企业级系统（10+ 人 / 严合规）** | MAF（Azure 栈） / Google ADK（GCP 栈） / LangGraph Platform / Dify 企业版（私有化） | 治理、审计、SLA、厂商背书 |
| **多智能体协作系统** | LangGraph（Supervisor/Swarm） / CrewAI（Hierarchical） / MAF / Google ADK + A2A / AgentScope | 编排能力、跨 Agent 通信 |
| **低代码 / 业务侧共建** | Dify（私有化） / Coze（SaaS） | 可视化画布、非技术人员可参与 |
| **国内团队 + 数据不出域** | Dify / FastGPT / AgentScope（阿里） / Eino（字节 Go 栈） | 中文生态、私有化成熟、合规 |
| **国际团队 / 云原生** | LangGraph / MAF / Google ADK / OpenAI Agents SDK | 与三大云深度绑定，SaaS 化能力完善 |
| **Python 团队无框架经验** | CrewAI（入门）→ LangGraph（进阶） | 学习路径平滑 |
| **Python + LangChain 已有经验** | 直接 LangGraph | 无缝衔接 |

### 6.3 反模式（不要这样做）

1. **不要在 2026 年新项目上从 AutoGen 起步**——已维护模式，应选 MAF 或 AG2。
2. **不要用内存状态跑生产**——CrewAI 默认内存存储，进程重启丢状态；生产务必接入 Redis/PostgreSQL Checkpointer。
3. **不要给单 Agent 挂超过 15 个工具**——工具选择困难、Token 消耗爆炸、错误率上升；应做工具分组或 Router Agent。
4. **不要在没有 Trace 的情况下上线**——出问题无法定位；LangSmith/Langfuse 至少接一个。
5. **不要过早追求"最复杂框架"**——需求不明确时先 CrewAI/smolagents 验证，再迁 LangGraph/MAF。
6. **不要忽视 MCP 生态适配**——2026 年后不支持 MCP 的框架将被工具生态边缘化。

---

## 7. 风险与注意事项

### 7.1 技术风险

| 风险 | 表现 | 缓解策略 |
|---|---|---|
| **幻觉（Hallucination）** | Agent 编造工具输出、虚构事实 | 强制 Grounding（RAG）、结构化输出校验（Pydantic）、Guardrails、LLM-as-Judge 评估 |
| **工具调用失败** | 参数格式错、API 超时、限流 | Schema 自动生成、重试策略、Circuit Breaker、Fallback 工具、Sandbox 隔离 |
| **状态失控 / 死循环** | Agent 陷入无限反思或工具循环 | 最大步数限制、Token 预算、图超时（LangGraph 1.2 强化）、成本熔断 |
| **上下文爆炸** | Token 超限、成本失控 | 上下文压缩、摘要、检索片段裁剪、长短期记忆分层 |
| **多 Agent 协作发散** | 讨论无收敛、Token 消耗大 | Supervisor 模式、终止条件、消息类型化（AutoGen 结构化对话） |
| **难以评估** | 无客观指标，靠感觉 | DeepEval / RAGAS / LangSmith Evaluations / Braintrust 建立离线数据集 + 在线打分 |
| **调试困难** | 黑盒决策链 | 全链路 Trace（OTel GenAI）、图可视化（LangSmith）、单步回放 |
| **权限与安全** | Prompt Injection、越权工具调用、数据泄露 | 最小权限原则、工具白名单、Guardrails 输入/输出过滤、PII 脱敏、Human-in-the-Loop 审批 |
| **成本不可控** | Token 消耗指数增长 | 每步成本上限、模型分层（简单任务用小模型）、缓存、Provider 层 spend control |
| **合规风险** | 数据出域、审计缺失 | 私有化部署（Dify/自建）、审计日志、数据主权条款 |

### 7.2 生态风险

1. **框架快速迭代带来的破坏性变更**：LangChain 1.0 移除旧 Chains/Agents、AutoGen 停更、SK 迁 MAF——**版本锁定 + 迁移预案是必须的**。
2. **厂商锁定**：OpenAI Agents SDK 虽 provider-agnostic 但最优体验仍绑定 OpenAI；MAF 深绑 Azure；ADK 深绑 GCP。选型时评估"退出成本"。
3. **协议标准仍在演进**：MCP 已成事实标准，A2A 刚 v1.0；跨框架互操作尚未完全成熟。
4. **国内合规**：数据出境、生成内容审核、算法备案——建议国内业务优先私有化开源方案（Dify/FastGPT/AgentScope）。

### 7.3 组织风险

- **过度工程化**：把简单问答包装成"多 Agent 系统"，维护成本远大于收益。
- **技能缺口**：LangGraph/MAF 学习曲线陡峭，团队需要预留 2-4 周 ramp-up。
- **责任边界模糊**：Agent 决策出错时业务/工程/合规的责任划分需要提前定义。

---

## 8. 信息来源与时效性

### 8.1 调研时间范围
- **主体数据采集时间**：2026 年 9 月（本次调研）
- **信息覆盖时间窗**：2024 年 1 月 — 2026 年 9 月，重点在 2025 年 10 月以后（LangChain/LangGraph 1.0、MAF GA、OpenAI Agents SDK 快速迭代等关键节点之后）
- **GitHub Star / 版本快照时间**：2026-08-15 前后

### 8.2 关键信息来源

**官方文档 / 博客**
- LangChain 官方博客与 Release Notes（LangGraph 1.0 / 1.1 / 1.2 更新）
- Microsoft Learn — Agent Framework Overview
- Microsoft AutoGen Blog
- OpenAI Agents SDK 官方文档
- LlamaIndex 官方文档
- Dify 官方博客（100k Stars 里程碑公告、1.5 实时调试、1.14.1 安全加固）
- Google ADK 官方文档与 A2A Protocol
- Langfuse 官网 / LangChain LLM Observability Tools 评测
- Hugging Face smolagents 文档
- Pydantic 官网 / Pydantic AI 文档
- Mastra 官方文档
- CAMEL-AI GitHub / camel-ai.org

**第三方评测与研究**
- ThoughtWorks Technology Radar（Pydantic AI、Google ADK 条目，2025-11 / 2026-06）
- AWS Prescriptive Guidance — Agentic AI Frameworks（LlamaIndex 章节）
- NVIDIA NeMo Agent Toolkit 文档（框架无关集成说明）
- AgentMail《The 9 Best AI Agent Frameworks in 2026》
- learnagent.org《Agent 框架 2026 更新追踪》
- the-agent-report.com《AI Agent Frameworks in Mid-2026: The Complete Landscape》
- promptquorum.com AutoGen / CAMEL Review 2026
- Alice Labs 2026 年 8 月评测（100+ 团队样本）
- 国内评测：CSDN、掘金、腾讯云开发者、头条号等多篇 2026 年横评文章

**社区数据**
- GitHub API 快照（2026-08-15，通过第三方 Top AI Skills 汇总）
- PyPI 下载量数据（LangGraph 3450 万/月、CrewAI 520 万/月）
- npm 下载量（Mastra 周下载数据）

### 8.3 数据时效性声明

- **Star 数与版本号**具有强时效性，本报告以 2026 年 8 月中旬为基准；使用前请到相应 GitHub 仓库核对最新数据。
- **AutoGen → MAF 演化**、**LangChain v1.0 破坏性变更**、**OpenAI Agents SDK 版本节奏**是过去 12 个月最重大的三个变化，任何早于 2025 年 10 月的选型文章都需重新审视。
- **国内平台**（Coze、Dify、百炼、千帆）迭代速度快，功能对比请以最近 3 个月的官方更新为准。
- **协议标准**（MCP、A2A）仍在演进，本报告基于 MCP 已被主流框架广泛支持、A2A v1.0 已发布的事实。

---

## 附录 A：一句话结论（TL;DR）

> **2026 年 AI Agent 框架格局已经清晰**：
> - **生产级复杂流程** → LangGraph（Python）/ Microsoft Agent Framework（.NET/Azure）/ Google ADK（GCP）
> - **快速原型 / 角色协作** → CrewAI
> - **RAG 与文档智能** → LlamaIndex / Dify
> - **类型安全 Python 微服务** → PydanticAI
> - **官方轻量 + Handoff + 语音** → OpenAI Agents SDK
> - **TypeScript 全栈** → Mastra / Vercel AI SDK
> - **零代码 / 私有化平台** → Dify（国内首选） / Coze（SaaS 快速）
> - **软件工程垂直** → MetaGPT
> - **研究 / 仿真** → CAMEL-AI / AgentScope
> - **教学 / 极简 PoC** → smolagents
>
> **不要再选**：AutoGen（维护模式，用 MAF 或 AG2 替代）、Swarm（已被 OpenAI Agents SDK 替代）、旧版 LangChain Chains/Agents（1.0 已移除）。
>
> **必配基础设施**：MCP（工具接入）+ LangSmith/Langfuse（可观测性）+ DeepEval/RAGAS（评估）+ Guardrails（安全护栏）。
