# Main training script
# BoMeyering, 2024
import os
import torch
import yaml
import sys
import json
import logging.config
import datetime

from torch.utils.data import DataLoader
from argparse import ArgumentParser
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from src.utils.config import YamlConfigLoader, ArgsAttributes, setup_loggers

from src.utils.parameters import get_params
from src.models import create_smp_model
from src.optim import EMA, get_optimizer, ConfigOptim
from src.datasets import LabeledDataset, UnlabeledDataset
from src.transforms import get_train_transforms
from src.dataloaders import DataLoaderBalancer
from src.fixmatch import get_pseudo_labels
from src.trainer import FixMatchTrainer
from src.losses import CELoss, FocalLoss, CBLoss, ACBLoss, RecallLoss

# Parse the command line argument configs
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

    # Set up Tensorboard
    tb_writer = SummaryWriter(comment=args.run_name)

    # Set up the loggers
    setup_loggers(args)
    logger = logging.getLogger()

    # Create model specified in configs
    model = create_smp_model(args)
    model.to(args.device)
    logger.info(f"Instantiated {args.model.model_name} with {args.model.encoder_name} backbone.")

    # Get model parameters and weight decay. Filter out bias and batch norm parameters if necessary
    parameters = get_params(args, model)
    if args.optimizer.filter_bias_and_bn:
        logger.info(f"Applied decay rate to non bias and batch norm parameters.")
    else:
        logger.info(f"Applied decay rate to all parameters.")

    # Get the model optimizer
    # optimizer = get_optimizer(args, parameters)
    logger.info(f"Initialized optimizer {args.optimizer.name}")


    # Optimizer stuff, loss criterion, and sample counts
    opt_stuff = ConfigOptim(args, parameters)
    with open('metadata/class_pixel_counts.json', 'r') as f:
        samples = json.load(f)
    samples = torch.tensor([v for v in samples.values()]).to(args.device)
    # loss_criterion = opt_stuff.get_loss_criterion()
    inv_weights = 1/samples
    # loss_criterion = CELoss(weights=inv_weights)
    # loss_criterion = FocalLoss(alpha = samples)
    # loss_criterion = CBLoss(samples = samples, loss_type='CELoss', reduction='mean')
    loss_criterion = RecallLoss(samples = samples, loss_type='CELoss')


    optimizer = opt_stuff.get_optimizer()
    logger.info(f"Initialized loss criterion {args.loss.name}")
    logger.info(f"Initialized optimizer {args.optimizer.name}")

    # # Set up EMA if configured
    if args.optimizer.ema:
        ema = EMA(model, args.optimizer.ema_decay)
        logger.info(f'Applied exponential moving average of {args.optimizer.ema_decay} to model weights.')

    # Build Datasets and Dataloaders
    logger.info(f"Building datasets from {[v for _, v in vars(args.directories).items() if v.startswith('data')]}")
    train_l_ds = LabeledDataset(root_dir=args.directories.train_l_dir)
    train_u_ds = UnlabeledDataset(root_dir=args.directories.train_u_dir)
    val_ds = LabeledDataset(root_dir=args.directories.val_dir)
    test_ds = LabeledDataset(root_dir=args.directories.test_dir)

    dl_balancer = DataLoaderBalancer(train_l_ds, train_u_ds, batch_sizes=[args.model.lab_bs, args.model.unlab_bs], drop_last=False)
    dataloaders, max_length = dl_balancer.balance_loaders()
    logger.info(f"Training dataloaders balanced. Labeled DL BS: {args.model.lab_bs} Unlabaled DL BS: {args.model.unlab_bs}.")
    logger.info(f"Max loader length for epoch iteration: {max_length}")

    val_dataloader = DataLoader(val_ds, batch_size=args.model.lab_bs, shuffle=False, drop_last=False)
    logger.info(f"Validation dataloader instantiated")

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

    logger.info("Created FixMatchTrainer for semi supervised learning.")
    logger.info("Training Initiated")
    fixmatch_trainer.train()
    logger.info("Training complete")

if __name__ == '__main__':
    main(args)