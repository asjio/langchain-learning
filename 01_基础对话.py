"""基础对话 - LangChain 最小可运行示例"""
from langchain_openai import ChatOpenAI

# 初始化模型
model = ChatOpenAI(
    model="bailian/qwen3.8-max",
    api_key="sk-your-api-key-here",
    base_url="https://your-gateway.example.com/v1",
    streaming=True,
    model_kwargs={"stream_options": {"include_usage": True}},
)

# 单轮对话
response = model.invoke("用一句话解释什么是人工智能")
print("AI:", response.content)

# 多轮对话（带历史）
from langchain_core.messages import HumanMessage, AIMessage

messages = [
    HumanMessage(content="我叫小明"),
    AIMessage(content="你好小明，很高兴认识你！"),
    HumanMessage(content="我叫什么名字？"),
]
response = model.invoke(messages)
print("AI:", response.content)  # 应该记得"小明"
