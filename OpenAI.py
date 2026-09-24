from langchain.chat_models import init_chat_model
from langchain.tools import tool
from getConfig import get_config

api_key = get_config("model_config", "api_key")
base_url = get_config("model_config", "base_url")
model_name = get_config("model_config", "current_model")

model = init_chat_model(
    model_name,
    model_provider="openai",
    temperature=0.3,
    api_key=api_key,
    base_url=base_url,
    streaming=True,
    model_kwargs={"stream_options": {"include_usage": True}},
)


response = model.invoke("你好")
print(response.content)
