# Semi-supervised learning functionality
# BoMeyering 2024

import argparse
import torch
import torch.nn.functional as F
from typing import Tuple

@torch.no_grad()
def get_pseudo_labels(args: argparse.Namespace, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """_summary_

    Args:
        args (_type_): _description_
        logits (torch.Tensor): _description_

    Raises:
        ValueError: _description_

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: _description_
    """
    if len(logits.shape) != 4:
        raise ValueError(f"Argument 'logits' should be of shape (N, V, H, W), but has {len(logits.shape)} instead")
    
    probs = F.softmax(logits, dim=1)
    max_p, pseudo_labels = torch.max(probs, dim=1)
    if len(args.fixmatch.tau) == 1:
        tau_mask = max_p.ge(args.fixmatch.tau[0])
    else:
        tau = torch.tensor(args.fixmatch.tau)
        exp_tau = tau[pseudo_labels] 
        tau_mask = max_p.ge(exp_tau)

    return pseudo_labels, tau_mask

@torch.no_grad()
def mask_targets(targets: torch.Tensor, mask: torch.Tensor, ignore_index: int=-1):

    adj_targets = torch.where(mask, targets, torch.full_like(targets, ignore_index))

    return adj_targets