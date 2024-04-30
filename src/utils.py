# Utils
# BoMeyering, 2024

import yaml

def load_yaml_config(path):
    """_summary_

    Args:
        path (_type_): _description_

    Returns:
        _type_: _description_
    """
    with open(path, 'r') as file:
        return yaml.safe_load(file)
    
def set_args_attr(config, args):
    """_summary_

    Args:
        config (_type_): _description_
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    for k, v in config.items():
        if type(v) is dict:
            set_args_attr(v, args)
        elif getattr(args, k, None) is None:
            setattr(args, k, v)
    return args

class ConfigParser():
    def __init__(self, args, config_path) -> None:

        pass