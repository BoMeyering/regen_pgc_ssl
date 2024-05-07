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
from src.data import LabeledDataset, UnlabeledDataset
from src.transforms import get_train_transforms


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
    # args = arg_setter.set_nested_key(args, ('general', 'run_name'))

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
    train_l_ds = LabeledDataset(root_dir=args.directories.train_l_dir)
    train_u_ds = UnlabeledDataset(root_dir=args.directories.train_u_dir)
    val_ds = LabeledDataset(root_dir=args.directories.val_dir)
    test_ds = LabeledDataset(root_dir=args.directories.test_dir)

    


    
    return train_l_ds, train_u_ds, val_ds, test_ds


if __name__ == '__main__':
    train_l_ds, train_u_ds, val_ds, test_ds = main(args)

    print(len(train_l_ds), len(train_u_ds), len(val_ds), len(test_ds))
    img, target = train_l_ds[0]
    print(img)
    print(target)

    weak_img, strong_img = train_u_ds[0]
    print(weak_img)
    print(strong_img)

    img, target = val_ds[0]
    print(img)
    print(target)

    img, target = test_ds[0]
    print(img)
    print(target)