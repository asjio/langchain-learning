
from pathlib import Path
import toml

from logBase import log

Project_path = Path.cwd()
config_toml = Project_path / 'config.toml'


def get_config(table, name):
    """ 获取配置
    Args:
        table: 查询的配置头
        name: 查询的配置名
    Return: 指定的配置值
    """
    return toml.load(config_toml)[table][name]


def set_config(table, name, value):
    """ 修改配置名称
    Args:
        table: 要修改的配置头
        name: 要修改的配置名
        value: 要修改的配置的值
    Return: modelName
    """
    data = toml.load(config_toml)

    data.setdefault(table, {})[name] = value
    with open(config_toml, 'w', encoding='utf-8') as f:  # 写入也用文本模式 'w'
        toml.dump(data, f)
    return data[table][name]


def add_config(table, name, value):
    """ 修改配置名称
    Args:
        table: 要修改的配置头
        name: 要修改的配置名
        value: 要修改的配置的值
    """
    toml_data = toml.load(config_toml)
    data = toml_data.get(table).get(name)
    if value in data:
        return False
    else:
        data.append(value)

        with open(config_toml, 'w', encoding='utf-8') as f:
            toml.dump(toml_data, f)
    return toml_data


if __name__ == "__main__":

    # print(get_config("model_config", "current_model"))
    # print(set_config('model_config', "current_model", "qwen3.7-max"))
    print(add_config("model_config", "model_list", "bailian/glm-5.3"))
