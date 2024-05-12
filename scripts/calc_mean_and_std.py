# Dataset Channel Means and STD Calculations
# BoMeyering 2024

import torch
import os
import sys
import json
import argparse
import logging
import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Tuple
from logging import Logger, getLogger

script_directory = os.path.dirname(os.path.abspath(__file__))
root_directory = os.path.dirname(script_directory)
sys.path.append(root_directory)

from src.utils.config import YamlConfigLoader, ArgsAttributes
from src.datasets import StatDataset
from src.utils.welford import WelfordCalculator

logger = Logger('norm_calc_logger', level='DEBUG')
logger = logging.getLogger('norm_calc_logger')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

file_handler = logging.FileHandler('logs/norm_calc_log'+datetime.datetime.now().isoformat(timespec='seconds', sep="_"))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


# Grab config and parse
parser = argparse.ArgumentParser()
parser.add_argument('config', help="The path to the yaml configuration file which defines the directories containing the images")
args = parser.parse_args()

# Set image directory paths to args
if not os.path.exists(args.config):
    raise FileNotFoundError(f"Path to config file {args.config} does not exist. Please specify a different path")
config = YamlConfigLoader(args.config).load_config()
arg_attr = ArgsAttributes(args, config)
arg_attr.set_args_attr(check_run_name=False)
args = arg_attr.args

print(args)

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

    welford = WelfordCalculator(device=device)

    for path in args.training_dirs:
        logger.info(f"Calculating mean and std for path {path}")
        ds = StatDataset(dir_path=path)
        dl = DataLoader(
            ds, 
            shuffle=False, 
            num_workers=2,
            pin_memory=True,
            batch_size=1
        )

        pbar = tqdm(total=len(dl), desc="Overall Progress", unit="image")
        for i, batch in enumerate(dl):
            # unpack the batch
            img, key = batch
            pbar.set_description(f"Processing {key}")
            if len(img.shape) < 3:
                logger.debug(f"Image {key} in dataset {path} is corrupt. Please check the image integrity")
                continue
            
            img = img.squeeze().to(device)
            welford.update(img)
            pbar.update(1)
        logger.info(f"Finished incorporating data from {path} into mean and std calculations")
        pbar.close()

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
