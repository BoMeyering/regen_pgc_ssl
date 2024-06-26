# Model Parameter Utility Functions
# BoMeyering 2024

import torch
import argparse
import logging
from typing import List, Tuple

logger = logging.getLogger() 

def add_weight_decay(model: torch.nn.Module, weight_decay: float=1e-5, skip_list: List=[]):

    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {'params': no_decay, 'weight_decay': 0},
        {'params': decay, 'weight_decay': weight_decay}
    ]

def get_params(args: argparse.Namespace, model: torch.nn.Module) -> Tuple:
    """
    If args.filter_bias_and_bn is True, applies the weight decay only to non bias and bn parameters
    Else, applies weight decay to all parameters.

    Args:
        args (argparse.Namespace): The args parsed from the config yaml file.
        model (torch.nn.module): A torch.nn.Module model

    Returns:
        Tuple: A tuple containing the model parameters and the udpated weight decay value for the optimizer
    """

    if args.optimizer.weight_decay and args.optimizer.filter_bias_and_bn:
        logger.info("Filtering bias and norm parameters from weight decay parameter.")
        parameters = add_weight_decay(model, args.optimizer.weight_decay)
        weight_decay = 0
        setattr(args.optimizer, 'original_weight_decay', args.optimizer.weight_decay)
        setattr(args.optimizer, 'weight_decay', weight_decay)
    else:
        logger.info("Applying weight decay to all parameters.")
        parameters = model.parameters()
        weight_decay = args.optimizer.weight_decay
    
    return parameters
