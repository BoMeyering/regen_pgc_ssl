# Semi-supervised learning functionality
# BoMeyering 2024

import torch
import torch.nn.functional as F
from typing import Tuple

@torch.no_grad()
def get_pseudo_labels(args, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    if len(logits.shape) != 4:
        raise ValueError(f"Argument 'logits' should be of shape (N, V, H, W), but has {len(logits.shape)} instead")
    
    probs = F.softmax(logits, dim=1)
    max_p, pseudo_labels = torch.max(probs, dim=1)
    tau_mask = max_p.ge(args.tau).float()

    return pseudo_labels, tau_mask

