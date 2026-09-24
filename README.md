# LangChain 学习项目

个人 LangChain 学习代码，包含基础对话、Prompt 模板、工具调用等示例。

## 文件说明

| 文件 | 说明 |
|---|---|
| `01_基础对话.py` | 最小可运行示例：单轮/多轮对话 |
| `02_Prompt模板.py` | Prompt 模板 + 输出解析（字符串/JSON） |
| `03_工具调用.py` | Agent 自动决策使用工具 |
| `model_service.py` | 模型服务封装（配置读取、切换模型、Prompt 构建） |
| `getConfig.py` | TOML 配置读写工具 |
| `agentTools/` | 自定义工具（天气、计算等） |
| `doc/` | LangChain 学习手册、AI Agent 框架调研报告 |

## 配置

1. 复制 `config.toml.example` 为 `config.toml`
2. 填入你的 API Key 和网关地址

```toml
[model_config]
api_key = "sk-your-api-key-here"
base_url = "https://your-gateway.example.com/v1"
current_model = "bailian/qwen3.8-max"
model_list = ["bailian/qwen3.7-max", "bailian/qwen3.8-max"]
```

## 依赖

```bash
pip install langchain langchain-openai langchain-core loguru toml
```

## 注意

- `config.toml` 已加入 `.gitignore`，不会上传
- 代码中示例的 API Key 和地址均为占位符，请替换为自己的配置
