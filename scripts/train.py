# Main training script
# BoMeyering, 2024
import os
import torch
import yaml

from argparse import ArgumentParser

from src.utils import config_parser


parser = ArgumentParser
parser.add_argument('config', default='configs/train_config.yaml')

def main(args):
    return True


if __name__ == '__main__':



    main()