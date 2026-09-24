"""Prompt 模板 + 输出解析 - 让模型按格式返回"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

model = ChatOpenAI(
    model="bailian/qwen3.8-max",
    api_key="sk-your-api-key-here",
    base_url="https://your-gateway.example.com/v1",
    streaming=True,
    model_kwargs={"stream_options": {"include_usage": True}},
)

# 方式1: 简单字符串输出
prompt = ChatPromptTemplate.from_template("讲一个关于{topic}的冷知识，限50字")
chain = prompt | model | StrOutputParser()
result = chain.invoke({"topic": " Python "})
print("冷知识:", result)

# 方式2: JSON 格式输出（结构化）
json_prompt = ChatPromptTemplate.from_template(
    """提取以下文本的关键信息，返回JSON格式：
    文本：{text}
    
    返回格式：
    {{"人物": "...", "地点": "...", "事件": "..."}}
    """
)
json_chain = json_prompt | model | JsonOutputParser()
result = json_chain.invoke({"text": "张三昨天在北京天安门看了升旗仪式"})
print("提取结果:", result)
print("类型:", type(result))  # <class 'dict'>
