# Main training script
# BoMeyering, 2024
import os
import torch
import yaml

from torch.utils.data import DataLoader
from argparse import ArgumentParser

from src.utils.config import YamlConfigLoader, ArgsAttributes

from src.utils.parameters import get_params
from src.models import create_smp_model
from src.optim import EMA, get_optimizer, ConfigOptim
from src.datasets import LabeledDataset, UnlabeledDataset
from src.transforms import get_train_transforms
from src.dataloaders import DataLoaderBalancer
from src.fixmatch import get_pseudo_labels
from src.trainer import FixMatchTrainer


parser = ArgumentParser()
parser.add_argument('config', nargs='?', default='configs/train_config.yaml')
args = parser.parse_args()

def main(args):

    # Load training configuration yaml
    config_loader = YamlConfigLoader(args.config)
    config = config_loader.load_config()

    # Instantiate args namespace with config and set values
    arg_setter = ArgsAttributes(args, config)
    # Set attributes and validate
    arg_setter.set_args_attr()
    arg_setter.validate()

    # Grab validated args Namespace
    args = arg_setter.args

    # Create model specified in configs
    model = create_smp_model(args)

    # Get model parameters and weight decay. Filter out bias and batch norm parameters if necessary
    parameters = get_params(args, model)

    # Get the model optimizer
    optimizer = get_optimizer(args, parameters)

    opt_stuff = ConfigOptim(args, parameters)
    loss_criterion = opt_stuff.get_loss_criterion()
    optimizer = opt_stuff.get_optimizer()

    print(loss_criterion)


    # # Set up EMA if configured
    if args.optimizer.ema:
        ema = EMA(model, args.optimizer.ema_decay)

    # print(args)

    # Build Datasets and Dataloaders
    train_l_ds = LabeledDataset(root_dir=args.directories.train_l_dir)
    train_u_ds = UnlabeledDataset(root_dir=args.directories.train_u_dir)
    val_ds = LabeledDataset(root_dir=args.directories.val_dir)
    test_ds = LabeledDataset(root_dir=args.directories.test_dir)

    dl_balancer = DataLoaderBalancer(train_l_ds, train_u_ds, batch_sizes=[args.model.lab_bs, args.model.unlab_bs], drop_last=False)
    dataloaders, max_length = dl_balancer.balance_loaders()
    # dataloaders = [iter(dl) for dl in dataloaders]

    val_dataloader = DataLoader(val_ds, batch_size=args.model.lab_bs, shuffle=False, drop_last=False)

    fixmatch_trainer = FixMatchTrainer(
        name='Test_Trainer',
        args=args, 
        model=model, 
        train_loaders = dataloaders, 
        train_length=max_length, 
        val_loader=val_dataloader, 
        optimizer=optimizer, 
        criterion=loss_criterion
    )

    fixmatch_trainer.train()

if __name__ == '__main__':
    main(args)