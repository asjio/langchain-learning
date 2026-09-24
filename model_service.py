
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from loguru import logger as log
from getConfig import add_config, get_config, set_config
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser


class modelService():

    def __init__(self) -> None:

        # self.model = get_config("model_config", "current_model")
        # self.client = OpenAI(
        #     api_key=get_config("model_config", "api_key"),
        #     base_url=get_config("model_config", "base_url"),
        # )
        self.model = get_config("model_config", "current_model")
        self.client = init_chat_model(
            self.model,
            model_provider="openai",
            temperature=0.3,
            api_key=get_config('model_config', 'api_key'),
            base_url=get_config('model_config', 'base_url'),
            streaming=True,
            model_kwargs={"stream_options": {"include_usage": True}},
        )

    def switch_model(self, modelName: str):
        """切换模型
        Args:
            modelName: 模型名称
        Return: bool
        """
        if modelName in get_config("model_config", "model_list"):
            log.info("模型切换,初始化模型配置")
            self.model = set_config("model_config", "current_model", modelName)
            return True
        else:
            log.error(f'修改模型失败,原因:模型列表中不存在这个"{modelName}"模型')
            return False, f'模型列表中不存在这个模型{modelName},修改失败'

    def add_model(self, modelName: str):
        try:
            res = add_config("model_config", "model_list", modelName)
            if res:
                log.info(f'添加{modelName}成功')
                return True
            else:
                log.warning(f'已有此模型添加{modelName}失败')
                return False
        except AttributeError as e:
            log.error(f'添加模型失败:{modelName}')
            return False

    def get_response(self, query: str):
        """获取模型输出
        Args:
            query: 给模型的提问
            outDetails: 模型输入内容默认content
        """

        completion = self.client.invoke(query)
        return completion.content

    def get_stream_response(self, query: str):
        """获取模型输出
        Args:
           query: 给模型的提问
           outDetails: 模型输入内容默认content
        """
        return self.client.stream(query)
    # ===== 错误写法（保留对照）=====
    # 错在：
    #   1) system 用了 f-string，{role} 会被 Python 立即求值，但函数里没有 role 变量 -> NameError
    #   2) 传进来的 prompt 字典根本没被用到
    #   3) 返回的是"模板对象"，模板没有 .to_messages()；必须先 .invoke(变量) 填空再转消息
    # def set_propmpt(self, prompt: dict):
    #     return ChatPromptTemplate.from_messages([
    #         ("system", f"你是一位{role},回答要简洁、准确、带例子."),
    #         MessagesPlaceholder("history", optional=True),
    #         ("human", "{question}")
    #     ])
    # ===== 错误写法结束 =====

    def set_propmpt(self):
        """第1步：只负责"造模板"。{role}/{question} 是待填的空，此刻不赋值。
        注意 system 用普通字符串（不是 f-string），否则 {role} 会被立即求值。
        """
        return ChatPromptTemplate.from_messages([
            ("system", "你是一位{role},回答要简洁、准确、带例子."),
            MessagesPlaceholder("history", optional=True),
            ("human", "{question}")
        ])

    def build_messages(self, prompt: dict):
        """第2步+第3步：用变量字典"填空"(invoke)，再转成消息列表(to_messages)。"""
        template = self.set_propmpt()
        return template.invoke(prompt).to_messages()


if __name__ == "__main__":
    modelService = modelService()
    # modelService.add_model('bailian/glm-5.2')
    # print(f'模型列表{get_config("model_config", "model_list")}')
    # print(f'当前模型{get_config("model_config", "current_model")}')
    # modelName = input('请选择模型:')
    # if modelName in get_config("model_config", "model_list"):
    #     modelService.switch_model(modelName)
    # userinput = input("请输入你的问题")
    # print(modelService.get_response(userinput))

    # ===== 错误写法（保留对照）=====
    # 错在：set_propmpt 返回的是模板对象，模板没有 .to_messages()，
    #       且没有把变量字典通过 .invoke(...) 填进去。
    # msg = modelService.set_propmpt(
    #     {
    #         "role": "资深 Python 工程师",
    #         "question": "装饰器是什么？",
    #         "history": [],
    #     }
    # ).to_messages()
    # ===== 错误写法结束 =====

    # 正确流程：造模板 -> 填空(invoke) -> 转消息(to_messages)，一步到位交给 build_messages
    # msg = modelService.build_messages({
    #     "role": "资深 Python 工程师",
    #     "question": "装饰器是什么？",
    #     "history": [],
    # })

    # print(modelService.get_response("你好,请输出一200字以上的段落"))

    # for i in modelService.get_stream_response("你好,请输出一300字以上的段落"):
    #     if i:
    #         print(i.content, end='', flush=True)

    # 组件1：Prompt 模板（定义发给模型的"信纸格式"）
prompt = ChatPromptTemplate.from_template("用 {style} 的风格解释 {topic}")

# 组件2：模型（处理请求的"大脑"）
model = modelService.client

# 组件3：输出解析器（从模型回复中提取纯文本）
parser = StrOutputParser()

# 用 | 把三个组件串成一条链
chain = prompt | model | parser

# 调用这条链——传入模板需要的变量
result = chain.invoke({"style": "像对5岁小孩说话", "topic": "量子力学"})
print(result)
