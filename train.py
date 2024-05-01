# Main training script
# BoMeyering, 2024
import os
import torch
import yaml

from argparse import ArgumentParser

from src.utils import YamlConfigLoader, ArgsAttributeSetter


parser = ArgumentParser()
parser.add_argument('config', nargs='?', default='configs/train_config.yaml')
args = parser.parse_args()

def main(args):
    config_loader = YamlConfigLoader(args.config)
    config = config_loader.load_config()

    arg_setter = ArgsAttributeSetter(args, config)
    args = arg_setter.set_args_attr()

    return args


if __name__ == '__main__':
    configs = main(args)
    print(configs)
