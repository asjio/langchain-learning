from langchain_openai import ChatOpenAI
from loguru import logger as log
# OpenAI兼容接口，直接用ChatOpenAI
model = ChatOpenAI(
    model="bailian/qwen3.8-max",
    api_key="sk-your-api-key-here",
    base_url="https://your-gateway.example.com/v1",
    streaming=True,
    model_kwargs={"stream_options": {"include_usage": True}},
)

# 流式测试：先输出思考内容，再输出正式回复
print("=== 开始 ===")
for chunk in model.stream("1+1为什么等于2？"):
    # 思考内容（推理模型的reasoning_content）
    reasoning = chunk.additional_kwargs.get("reasoning_content")
    if reasoning:
        log.info(chunk.additional_kwargs)
        # print(reasoning, end="", flush=True)
    # 正式回复内容
    if chunk.content:
        pass
        # print(chunk.content, end="", flush=True)
print("\n=== 结束 ===")
