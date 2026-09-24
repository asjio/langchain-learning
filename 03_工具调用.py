"""工具调用 - Agent 自动决策使用工具"""
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

# 定义工具


@tool
def get_weather(city: str) -> str:
    """查询城市天气"""
    # 模拟数据，真实场景调 API
    weathers = {"北京": "晴天 25°C", "杭州": "小雨 22°C", "上海": "多云 24°C"}
    return weathers.get(city, f"{city}天气未知")


@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        return str(eval(expression))
    except:
        return "计算失败"


# 初始化模型
model = ChatOpenAI(
    model="bailian/qwen3.8-max",
    api_key="sk-your-api-key-here",
    base_url="https://your-gateway.example.com/v1",
    streaming=True,
    model_kwargs={"stream_options": {"include_usage": True}},
)

# 创建 Agent（自动绑定工具+执行循环）
agent = create_agent(
    model=model,
    tools=[get_weather, calculate],
    system_prompt="你是一个 helpful assistant，可以使用工具查询天气和计算",
)

# 测试：模型自动决定调哪个工具
result = agent.invoke(
    {"messages": [{"role": "user", "content": "这个role是干嘛的呢？其他的问题不用回答 就关注这一条就可以.杭州今天天气怎么样？然后帮我算一下 128*32"}]})
print("最终回复:", result["messages"][-1].content)

# 查看中间过程（Agent 的思考链）
print("\n--- 执行过程 ---")
for msg in result["messages"]:
    print(f"{msg.type}: {msg.content[:100] if msg.content else msg}")
