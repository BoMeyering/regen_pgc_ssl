# Dataset Channel Means and STD Calculations
# BoMeyering 2024

import torch
import os
import sys
import json
import argparse
from torch.utils.data import DataLoader
from typing import Tuple

script_directory = os.path.dirname(os.path.abspath(__file__))
root_directory = os.path.dirname(script_directory)
sys.path.append(root_directory)

from src.utils.config import YamlConfigLoader, ArgsAttributeSetter
from src.data import StatDataset
from src.utils.welford import WelfordCalculator

# Grab config and parse
parser = argparse.ArgumentParser()
parser.add_argument('config', help="The path to the yaml configuration file which defines the directories containing the images")
args = parser.parse_args()

# Set image directory paths to args
if not os.path.exists(args.config):
    raise FileNotFoundError(f"Path to config file {args.config} does not exist. Please specify a different path")
config = YamlConfigLoader(args.config).load_config()
args = ArgsAttributeSetter(args, config).set_args_attr(check_run_name=False)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args: argparse.Namespace) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute channel-wise pixel mean and std over a list of datasets in args

    Args:
        args (argparse.Namespace): Args containing key 'training_dirs' with a list value containing all the image directories used for training

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors of channel wise mean and std in RGB format
    """

    welford = WelfordCalculator()

    for path in args.training_dirs:
        ds = StatDataset(dir_path=path)
        dl = DataLoader(
            ds, 
            shuffle=False, 
            num_workers=2,
            pin_memory=True,
            batch_size=1
        )

        for i, batch in enumerate(dl):
            batch = batch.squeeze().to(device)
            welford.update(batch)

    mean, std = welford.compute()
     
    return mean, std

if __name__ == '__main__':
    means, std = main(args)

    norm_dict = {
        'means': [i.item() for i in means],
        'std': [i.item() for i in std]
    }

    with open(args.out_path, 'w') as f:
        json.dump(norm_dict, f)
