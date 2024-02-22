import yaml
from attrdict import AttrDict
from pathlib import Path


def parse_yaml_config(config_path):
    def resolve_path(value):
        if isinstance(value, str):
            if Path(value).is_file() or Path(value).is_dir():
                return Path(value).resolve()
        return value

    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)

    # 递归遍历配置数据，将字段中的文件路径转换为绝对路径
    config_data = recursive_resolve_paths(config_data, resolve_path)

    # 当文件夹不存在时不会识别为路径, 手动转换
    path_items = ['save_root']
    for item in path_items:
        config_data[item] = Path(config_data[item])

    # 使用 AttrDict 将配置转换为对象
    config_obj = AttrDict(config_data)

    return config_obj


def recursive_resolve_paths(data, resolve_path_func):
    if isinstance(data, dict):
        resolved_data = {}
        for key, value in data.items():
            resolved_data[key] = recursive_resolve_paths(value, resolve_path_func)
        return resolved_data
    elif isinstance(data, list):
        resolved_data = []
        for item in data:
            resolved_data.append(recursive_resolve_paths(item, resolve_path_func))
        return resolved_data
    else:
        return resolve_path_func(data)


project_root = Path(__file__).resolve().parent.parent
config_file = project_root / 'config.yaml'
config = parse_yaml_config(config_file)
