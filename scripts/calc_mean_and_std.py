# Dataset Channel Means and STD Calculations
# BoMeyering 2024

import torch
import os
import cv2
import sys
import argparse
import numpy as np
import albumentations as A
from torch.utils.data import Dataset, DataLoader

script_directory = os.path.dirname(os.path.abspath(__file__))
root_directory = os.path.dirname(script_directory)
sys.path.append(root_directory)

from src.utils.config import YamlConfigLoader, ArgsAttributeSetter
from src.data import StatDataset

parser = argparse.ArgumentParser()

parser.add_argument('config', help="The path to the yaml configuration file which defines the directories containing the images")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args):
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Path to config file {args.config} does not exist. Please specify a different path")
    config = YamlConfigLoader(args.config).load_config()
    print(config)
    args = ArgsAttributeSetter(args, config).set_args_attr(check_run_name=False)

    means = torch.zeros(3)
    std = torch.zeros(3)
    pixel_n = torch.zeros(1)

    for path in args.training_dirs:
        print(path)
        ds = StatDataset(dir_path=path)
        dl = DataLoader(
            ds, 
            shuffle=False, 
            num_workers=2,
            pin_memory=True
        )

        for i, batch in enumerate(dl):

            channel_means = torch.sum(batch, dim=(0, 2, 3))
            channel_std = torch.sum(batch ** 2, dim=(0, 2, 3))
            pixels = batch.shape[2] * batch.shape[3]

            means += channel_means
            std += channel_std
            pixel_n += pixels

            new_batch = batch.squeeze()

        print(means)
        print(channel_std)
        print(pixel_n)


if __name__ == '__main__':
    args = main(args)

    # for i in args.training_dirs:
        # print(i)
