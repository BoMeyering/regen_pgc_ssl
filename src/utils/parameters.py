# Model Parameter Utility Functions
# BoMeyering 2024

import torch
import argparse
from typing import List, Tuple

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

    if args.model.weight_decay and args.model.filter_bias_and_bn:
        parameters = add_weight_decay(model, args.model.weight_decay)
        weight_decay = 0
    else:
        parameters = model.parameters()
        weight_decay = args.model.weight_decay

    return parameters, weight_decay
