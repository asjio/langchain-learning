from langchain.tools import tool


@tool
def get_weather(city: str) -> str:
    """获取指定城市天气
    Args:
        city:城市名称,如: "杭州","北京"
    """
    weather_data = {
        "杭州": "晴，25°C，湿度 60%",
        "北京": "多云，18°C，湿度 45%",
        "上海": "小雨，22°C，湿度 80%",
    }

    return weather_data.get(city, f"未找到{city}的天气数据")


@tool
def calculate(expression: str) -> str:
    """执行数学计算。支持加减乘除等基本运算。

    Args:
        expression: 数学表达式，如 "3 * 7 + 2"
    """
    try:
        # 安全地计算数学表达式
        result = eval(expression, {"__builtins__": {}}, {})
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"
