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
import torch.distributed as dist
from argparse import ArgumentParser
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from src.utils.config import YamlConfigLoader, ArgsAttributes, setup_loggers

from src.utils.parameters import get_params
from src.models import create_smp_model
from src.optim import EMA, ConfigOptim
from src.datasets import LabeledDataset, UnlabeledDataset
from src.transforms import get_train_transforms, get_strong_transforms, get_weak_transforms, get_val_transforms
from src.dataloaders import DataLoaderBalancer, DistributedDataloaderBalancer
from src.fixmatch import get_pseudo_labels
from src.trainer import FixMatchTrainer, SupervisedTrainer, TBLogger
from src.losses import CELoss, FocalLoss, CBLoss, ACBLoss, RecallLoss, get_loss_criterion

from metadata.dataset_maps import mapping

# Parse the command line argument configs
parser = ArgumentParser()
parser.add_argument('config', nargs='?', default='configs/train_config.yaml')
args = parser.parse_args()

def setup(rank, world_size):

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(
        backend="nccl", 
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

def prepare_ddp_dataloader(rank, world_size, dataset, batch_size=16, pin_memory=False, shuffle=True, drop_last=True, num_workers=0):
    sampler =  DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank, 
        drop_last=drop_last, 
        shuffle=shuffle
    )
    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        pin_memory=pin_memory, 
        num_workers=num_workers, 
        drop_last=drop_last, 
        sampler=sampler
    )

    return dataloader, sampler

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, args):

    # Initialize the Torch DDP settings
    setup(rank, world_size)

    # Load training configuration yaml
    config_loader = YamlConfigLoader(args.config)
    config = config_loader.load_config()

    # Instantiate args namespace with config and set values
    arg_setter = ArgsAttributes(args, config)

    # Set attributes and validate
    arg_setter.set_args_attr()
    arg_setter.validate()
    
    # Get class map
    class_map = mapping['new_mapping']

    # Grab validated args Namespace
    args = arg_setter.args

    # Set up Tensorboard
    tb_writer = SummaryWriter(log_dir="/".join(('runs', args.run_name)))
    tb_logger = TBLogger(tb_writer)

    # Set up the loggers
    setup_loggers(args)
    logger = logging.getLogger()

    # Create model specified in configs and move to the device
    torch.cuda.set_device(rank)
    model = create_smp_model(args).to(f"cuda:{rank}")
    model = DDP(model, device_ids=[rank])
    logger.info(f"Instantiated {args.model.model_name} with {args.model.encoder_name} backbone.")

    # Get model parameters and weight decay. Filter out bias and batch norm parameters if necessary
    parameters = get_params(args, model)
    if args.optimizer.filter_bias_and_bn:
        logger.info(f"Applied decay rate to non bias and batch norm parameters.")
    else:
        logger.info(f"Applied decay rate to all parameters.")

    # Get optimizer
    opt_stuff = ConfigOptim(args, parameters)
    optimizer = opt_stuff.get_optimizer()
    scheduler = opt_stuff.get_scheduler()
    logger.info(f"Initialized optimizer {args.optimizer.name}")
    logger.info(f"Initialized scheduler {args.scheduler.name}")

    # Load inverse wights and normalize
    with open('metadata/class_pixel_counts.json', 'r') as f:
        samples = json.load(f)
    samples = torch.tensor([v for v in samples.values()]).to(args.device)
    inv_weights = 1/samples
    inv_weights = inv_weights / inv_weights.sum()
    
    logger.info(f"Loading sample class pixel distribution {samples}")
    logger.info(f"Loading class inverse weights {inv_weights}.")
    
    # Set pixel class samples and inverse weights as args.loss attributes
    setattr(args.loss, 'samples', samples)
    setattr(args.loss, 'weights', inv_weights)

    # Get loss criterion from args
    loss_criterion = get_loss_criterion(args)
    logger.info(f"Initialized loss criterion {args.loss.name}")

    # # Set up EMA if configured
    if args.optimizer.ema:
        ema = EMA(model, args.optimizer.ema_decay)
        logger.info(f'Applied exponential moving average of {args.optimizer.ema_decay} to model weights.')

    # Build Datasets and Dataloaders
    logger.info(f"Building datasets from {[v for _, v in vars(args.directories).items() if v.startswith('data')]}")

    train_l_ds = LabeledDataset(
        root_dir=args.directories.train_l_dir, 
        transforms=get_train_transforms(resize=args.model.resize)
    )
    val_ds = LabeledDataset(
        root_dir=args.directories.val_dir, 
        transforms=get_val_transforms(resize=args.model.resize)
    )
    test_ds = LabeledDataset(
        root_dir=args.directories.test_dir, 
        transforms=get_val_transforms(resize=args.model.resize)
    )

    if 'train_u_dir' in vars(args.directories).keys():
        # Create unlabeled dataset
        train_u_ds = UnlabeledDataset(
            root_dir=args.directories.train_u_dir,
            weak_transforms=get_weak_transforms(resize=args.model.resize),
            strong_transforms=get_strong_transforms(resize=args.model.resize)
        )
        # Create dataloader balancer
        dl_balancer = DistributedDataloaderBalancer(
            train_l_ds, 
            train_u_ds, 
            batch_sizes=[args.model.lab_bs, args.model.unlab_bs], 
            num_replicas=world_size,
            rank=rank
        )

        # Balance loaders
        dataloaders, iter_length = dl_balancer.balance_loaders()

        # Pack up the training samplers in a tuple
        train_samplers = dl_balancer.infinite_sampler, dl_balancer.sampler

        logger.info(f"Training dataloaders balanced. Labeled DL BS: {args.model.lab_bs} Unlabaled DL BS: {args.model.unlab_bs}.")
        logger.info(f"Max loader length for epoch iteration: {iter_length}")

        # Instantiate distributed val dataloader
        val_dataloader, _ = prepare_ddp_dataloader(
            rank=rank,
            world_size=world_size,
            dataset=val_ds,
            batch_size=args.model.lab_bs,
            pin_memory=False,
            num_workers=0,
            drop_last=True,
            shuffle=False,
        )
        logger.info(f"Validation dataloader instantiated")

        fixmatch_trainer = FixMatchTrainer(
            name='Test_Trainer',
            args=args, 
            model=model, 
            train_loaders = dataloaders, 
            train_length=iter_length, 
            train_samplers=train_samplers,
            val_loader=val_dataloader, 
            optimizer=optimizer, 
            criterion=loss_criterion,
            scheduler=scheduler,
            tb_logger=tb_logger,
            class_map=class_map
        )

        logger.info(f"Created FixMatchTrainer {fixmatch_trainer.trainer_id} for semi-supervised learning.")
        logger.info("Training initiated")

        # Train the model
        fixmatch_trainer.train()


        logger.info("Training complete")

        tb_writer.flush()
        tb_writer.close()
    else:
        train_dataloader, train_sampler = prepare_ddp_dataloader(
            rank=rank, 
            world_size=world_size,
            dataset=train_l_ds,
            batch_size=args.model.lab_bs, 
            pin_memory=False,
            num_workers=0, 
            shuffle=True,
            drop_last=True
        )

        val_dataloader, _ = prepare_ddp_dataloader(
            rank=rank,
            world_size=world_size,
            dataset=val_ds,
            batch_size=args.model.lab_bs,
            pin_memory=False,
            num_workers=0,
            drop_last=True,
            shuffle=False,
        )

        test_dataloader, _ = prepare_ddp_dataloader(
            rank=rank,
            world_size=world_size,
            dataset=test_ds,
            batch_size=args.model.lab_bs,
            pin_memory=False,
            num_workers=0,
            drop_last=True,
            shuffle=False
        )

        logger.info(f"All dataloaders instantiated.")

        supervised_trainer = SupervisedTrainer(
            name='Supervised Trainer',
            args=args,
            model=model,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            optimizer=optimizer,
            criterion=loss_criterion,
            scheduler=scheduler,
            tb_logger=tb_logger,
            class_map=class_map
        )

        logger.info(f"Created SupervisedTrainer {supervised_trainer.trainer_id} for fully supervised learning.")
        logger.info("Training initiated")
        supervised_trainer.train()
        logger.info("Training complete")

        tb_writer.flush()
        tb_writer.close()

    cleanup()
    

if __name__ == '__main__':
    world_size =  torch.cuda.device_count()

    mp.spawn(
        main, 
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )
