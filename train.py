# Main training script
# BoMeyering, 2024
import os
import torch
import yaml

from argparse import ArgumentParser

from src.utils.config import YamlConfigLoader, ArgsAttributeSetter

from src.utils.parameters import get_params
from src.models import create_smp_model
from src.optim import EMA, get_optimizer


parser = ArgumentParser()
parser.add_argument('config', nargs='?', default='configs/train_config.yaml')
args = parser.parse_args()

def main(args):

    # Load training configuration yaml
    config_loader = YamlConfigLoader(args.config)
    config = config_loader.load_config()

    # Instantiate args namespace with config and set values
    arg_setter = ArgsAttributeSetter(args, config)
    args = arg_setter.set_args_attr()

    # Ensure args has a training run_name
    args = arg_setter.set_nested_key(args, ('general', 'run_name'))

    # Create model specified in configs
    model = create_smp_model(args)

    # Get model parameters and weight decay. Filter out bias and batch norm parameters if necessary
    parameters, weight_decay = get_params(args, model)

    # Get the model optimizer
    optimizer = get_optimizer(args, parameters, weight_decay=weight_decay)

    # Set up EMA if configured
    if args.model.ema:
        ema = EMA(model, args.model.ema_decay)

    # Build Datasets and Dataloaders
    


    return args


if __name__ == '__main__':
    configs = main(args)
