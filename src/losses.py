# Loss Classes
# BoMeyering 2024

import torch
import src
import logging
import inspect
import argparse
import torch.nn.functional as F
from typing import Union, Optional

logger = logging.getLogger()

def get_loss_criterion(args: argparse.Namespace) -> torch.nn.Module:
    """
    Get a loss function from the args

    Args:
        args (argparse.Namespace): Arguments namespace from a configuration file.

    Returns:
        torch.nn.Module: An instantiated loss criterion.
    """

    # Set loss name and retrieve the class from src.losses namespace
    loss_name = args.loss.name
    LossClass = getattr(src.losses, loss_name)

    # Get the valid parameters
    loss_params = vars(args.loss).copy()
    valid_params = inspect.signature(LossClass).parameters
    filtered_params = {k: v for k, v in loss_params.items() if k in valid_params}

    # Instnatiate the criterion
    criterion = LossClass(**filtered_params)
    
    return criterion

class CELoss(torch.nn.Module):
    """
    Wrapper class for vanilla cross entropy loss.
    """
    def __init__(self, ignore_index=-1, label_smoothing: float=0.0, weights: Optional[torch.tensor]=None, reduction: str='mean', use_weights: bool=True):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.weights = weights
        self.reduction = reduction
        self.use_weights = use_weights

        if self.use_weights:
            logger.info("Using inverse weights for CELoss.")
        else:
            logger.info("Not using inverse weights for CELoss.")
    
    def _mask_targets(targets: torch.Tensor, mask: torch.BoolTensor, ignore_index: int=-1) -> torch.tensor:
        """
        Helper function to create a new tareget with the ignore_index at the masked locations.

        Args:
            targets (torch.Tensor): a torch.tensor of shape (N, H, W) and dtype int.
            mask (torch.Tensor): a torch.BoolTensor of shape (N, H, W).
            ignore_index (int, optional): Index value to used for the masked values. Defaults to -1.

        Returns:
            torch.tensor: A tensor with the masked target values replaced by the ignore index
        """
        adj_targets = torch.where(mask, targets, torch.full_like(targets, ignore_index)).to(targets.device)

        return adj_targets

    def forward(self, preds: torch.tensor, targets: torch.tensor, mask: Optional[torch.BoolTensor] = None) -> torch.tensor:
        """
        Forward method of CELoss.

        Args:
            preds (torch.tensor): The raw logits from the model.
            targets (torch.tensor): The ground truth targets.
            mask (Optional[torch.BoolTensor], optional): A boolean torch tensor of shape (N, H, W) of pixels to exclude. Defaults to None.

        Returns:
            torch.tensor: A scalar loss value if reduction is 'mean' or 'sum', else a loss tensor of shape (N, H, W).
        """
        if mask is not None:
            targets = self._mask_targets(targets, mask, self.ignore_index)
        if self.use_weights:
            loss = F.cross_entropy(
                input=preds, 
                target=targets, 
                ignore_index=self.ignore_index, 
                label_smoothing=self.label_smoothing, 
                reduction=self.reduction, 
                weight=self.weights
            )
        else:
            loss = F.cross_entropy(
                input=preds, 
                target=targets, 
                ignore_index=self.ignore_index, 
                label_smoothing=self.label_smoothing, 
                reduction=self.reduction
            )

        return loss

class FocalLoss(torch.nn.Module):
    """
    Implementation of Focal Loss
    https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha: Union[float, torch.tensor]=1, gamma: float=2, reduction: str='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, preds: torch.tensor, targets: torch.tensor, mask: Optional[torch.BoolTensor] = None) -> torch.tensor:
        """
        Forward method of FocalLoss.

        Args:
            preds (torch.tensor): The raw logits from the model.
            targets (torch.tensor): The ground truth targets.
            mask (Optional[torch.BoolTensor], optional): A boolean torch tensor of shape (N, H, W) of pixels to exclude. Defaults to None.

        Returns:
            torch.tensor: A scalar loss value if reduction is 'mean' or 'sum', else a loss tensor of shape (N, H, W).
        """
        if isinstance(self.alpha, torch.Tensor) and len(self.alpha) != preds.shape[1]:
            raise ValueError(f"Length of alpha should be 1 or the number of classes in preds, {preds.shape[1]}. Please set a new alpha.")
        
        # Calculate log_pt and pt from the raw logits
        log_pt = F.log_softmax(preds, dim=1)
        log_pt = torch.gather(log_pt, 1, targets.unsqueeze(1)).squeeze(1)
        pt = torch.exp(log_pt)

        # Set alpha weights
        if isinstance(self.alpha, torch.Tensor):
            alpha = self.alpha[targets]
        else:
            alpha = self.alpha
        # Calculate focal loss
        loss = -alpha * ((1 - pt) ** self.gamma) * log_pt

        # Mask the loss if needed
        if mask is not None:
            loss = loss * mask

        # Apply reductions
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else: 
            raise ValueError(f"Invalid reduction mode: {self.reduction}. Must be one of ['mean', 'sum', 'none']")

class CBLoss(torch.nn.Module):
    """
    Implementation of Class Balanced losses
    https://arxiv.org/pdf/1901.05555
    """
    def __init__(self, samples: torch.tensor, loss_type: str, reduction: str='mean', gamma: Optional[float]=2.0):
        super().__init__()
        self.samples = samples.double()
        self.loss_type = loss_type
        self.reduction = reduction
        self.gamma = gamma
        self.N = self.samples.sum()
        self.beta = (self.N - 1) / self.N
        self.C = len(self.samples)
        self.eps = 1e-7

        self._effective_samples()
        if self.loss_type == 'CELoss':
            self.loss_fn = CELoss(weights=self.alpha.float(), reduction=self.reduction)
        elif self.loss_type == 'FocalLoss':
            self.loss_fn = FocalLoss(alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}. Must be one of ['CELoss', 'FocalLoss']")

    def _effective_samples(self):
        """
        Helper function to calculate the effective samples and weights.
        """
        # Calculate effective samples
        E = (1 - torch.pow(self.beta, self.samples)).double() / (1 - self.beta + self.eps).double()
        # E = torch.clamp(E, min=self.eps, max=1e10)

        # Invert to get alpha weights and normalize
        alpha = 1/E * self.C / (1/E).sum()
        # alpha = torch.clamp(alpha, min=self.eps, max=1e10)

        self.E = E
        self.alpha = alpha
     
    def forward(self, preds: torch.tensor, targets: torch.tensor, mask: Optional[torch.BoolTensor] = None) -> torch.tensor:
        """
        Forward method of CBLoss.

        Args:
            preds (torch.tensor): The raw logits from the model.
            targets (torch.tensor): The ground truth targets.
            mask (Optional[torch.BoolTensor], optional): A boolean torch tensor of shape (N, H, W) of pixels to exclude. Defaults to None.

        Returns:
            torch.tensor: A scalar loss value if reduction is 'mean' or 'sum', else a loss tensor of shape (N, H, W).
        """

        loss = self.loss_fn(preds=preds, targets=targets, mask=mask)
        
        return loss
  
class ACBLoss(torch.nn.Module):
    """
    Implement Adaptive Class Balanced Loss from Xu et al 2022.
    https://ieeexplore.ieee.org/document/10137858
    """
    def __init__(self, samples: torch.tensor, loss_type: str, reduction: str='mean', gamma: Optional[float]=2.0):
        super().__init__()
        self.samples = samples.double()
        self.loss_type = loss_type
        self.reduction = reduction
        self.gamma = gamma
        self.N = self.samples.sum()
        self.N_max = torch.max(self.samples)
        self.C = len(self.samples)
        self.eps = 1e-7

        self._effective_samples()
        if self.loss_type == 'CELoss':
            self.loss_fn = CELoss(weights=self.alpha.float(), reduction=self.reduction)
        elif self.loss_type == 'FocalLoss':
            self.loss_fn = FocalLoss(alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}. Must be one of ['CELoss', 'FocalLoss']")

    def _effective_samples(self):
        """
        Helper function to calculate the effective samples and weights based on beta.
        """
        # Sample size, class size, and degree of imbalance calculations
        self.u = torch.log(self.N.double())
        self.v = torch.log(torch.tensor(self.C).double())
        self.b = -torch.log10(self.samples / self.N_max).mean().double()
        self.f_uvb = self.u / (self.v ** torch.sqrt(self.b)).double()
        self.beta = torch.tanh(self.f_uvb).double()

        # Calculate effective samples
        E = (1 - torch.pow(self.beta, self.samples)).double() / (1 - self.beta + self.eps).double()

        # Invert to get alpha weights and normalize
        alpha = 1/E
        alpha = alpha / (alpha).sum() * self.C 

        self.E = E
        self.alpha = alpha

    def forward(self, preds: torch.tensor, targets: torch.tensor, mask: Optional[torch.BoolTensor] = None) -> torch.tensor:
        """
        Forward method of ACBLoss.

        Args:
            preds (torch.tensor): The raw logits from the model.
            targets (torch.tensor): The ground truth targets.
            mask (Optional[torch.BoolTensor], optional): A boolean torch tensor of shape (N, H, W) of pixels to exclude. Defaults to None.

        Returns:
            torch.tensor: A scalar loss value if reduction is 'mean' or 'sum', else a loss tensor of shape (N, H, W).
        """
        
        loss = self.loss_fn(preds=preds, targets=targets, mask=mask)
        
        return loss

class RecallLoss(torch.nn.Module):
    """
    Implementation of Recall Loss with dynamic weighting
    https://arxiv.org/pdf/2106.14917
    """
    def __init__(self, samples: torch.tensor, loss_type: str, reduction: str='mean', gamma: Optional[float]=2.0, eps: float=0.0000001):
        super().__init__()
        self.samples = samples
        self.loss_type = loss_type
        self.reduction = reduction
        self.gamma = gamma
        self.N = self.samples.sum()
        self.C = len(self.samples)
        self.eps = eps

    def _calculate_weights(self, preds: torch.tensor, targets: torch.tensor):
        """
        Helper function to calculate the recall weights.
        """
        # Get probs from logits and calculate one-hot tensors
        probs = F.softmax(preds, dim=1)
        pred_labels = torch.argmax(probs, dim=1)
        pred_oh = F.one_hot(pred_labels, num_classes=self.C)
        target_oh = F.one_hot(targets, num_classes=self.C)

        # Reshape to (-1, num_classes) to sum over one dimension
        pred_oh = pred_oh.view(-1, self.C)
        target_oh = target_oh.view(-1, self.C)

        # print("PRED SUMS: ", pred_oh.sum(dim=0))
        # print("TARGET SUMS: ", target_oh.sum(dim=0))

        # Calculate TP and FN rates
        TP = ((target_oh == 1) * (pred_oh == 1)).sum(dim=0)
        FN = ((target_oh == 1) * (pred_oh == 0)).sum(dim=0)

        # Calculate recall and weights
        R_c = (TP / (FN + TP + self.eps)).clamp(min=self.eps)
        weights = 1 - R_c
        
        if torch.all(weights == 0):
            weights = torch.full_like(weights, 1.)

        weights = weights / weights.sum() * self.C

        self.alpha = weights

    def forward(self, preds: torch.tensor, targets: torch.tensor, mask: Optional[torch.BoolTensor] = None) -> torch.tensor:
        """
        Forward method of RecallLoss.

        Args:
            preds (torch.tensor): The raw logits from the model.
            targets (torch.tensor): The ground truth targets.
            mask (Optional[torch.BoolTensor], optional): A boolean torch tensor of shape (N, H, W) of pixels to exclude. Defaults to None.

        Returns:
            torch.tensor: A scalar loss value if reduction is 'mean' or 'sum', else a loss tensor of shape (N, H, W).
        """
        # Calculate effective samples
        self._calculate_weights(preds=preds, targets=targets)
        
        if self.loss_type == 'CELoss':
            loss_fn = CELoss(weights=self.alpha, reduction=self.reduction)
            loss = loss_fn(preds=preds, targets=targets, mask=mask)
        elif self.loss_type == 'FocalLoss':
            loss_fn = FocalLoss(alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)
            loss = loss_fn(preds=preds, targets=targets, mask=mask)
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}. Must be one of ['CELoss', 'FocalLoss']")
        
        return loss



# class ACWLoss(torch.nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, preds, targets, mask: torch.BoolTensor = None):
#         pass

# class DiceLoss(torch.nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, preds, targets, mask: torch.BoolTensor = None):
#         pass

# class FocalTverskyLoss(torch.nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, preds, targets, mask: torch.BoolTensor = None):
#         pass


# if __name__ == '__main__':
#     logits = torch.randn(3, 4, 5, 5) * 2
#     sm = F.softmax(logits + torch.randn(3, 4, 5, 5), dim=1)
#     max_p, targets = torch.max(sm, dim=1)

#     mask = torch.ge(max_p, 0.8)

#     samples = torch.tensor([2390, 4201, 9011, 6850])
#     inv_weights = 1/samples

#     cross_entropy = CELoss()
#     loss = cross_entropy.forward(preds=logits, targets=targets, mask=mask)
#     print("CE LOSS: \t\t", loss)

#     weighted_ce = CELoss(weights=inv_weights)
#     loss = weighted_ce.forward(preds=logits, targets=targets, mask=mask)
#     print("WEIGHTED CE LOSS: \t", loss)

#     focal_loss = FocalLoss(gamma=.5)
#     loss = focal_loss.forward(preds=logits, targets=targets, mask=mask)
#     print("FOCAL LOSS: \t\t", loss)

#     cb_ce_loss = CBLoss(samples=samples, loss_type='CELoss', gamma=.5)
#     loss = cb_ce_loss.forward(preds=logits, targets=targets, mask=mask)
#     print("CB CE LOSS: \t\t", loss)

#     cb_focal_loss = CBLoss(samples=samples, loss_type='FocalLoss', gamma=.5)
#     loss = cb_focal_loss.forward(preds=logits, targets=targets, mask=mask)
#     print("CB FOCAL LOSS: \t\t", loss)

#     acb_ce_loss = ACBLoss(samples=samples, loss_type='CELoss', gamma=.5)
#     loss = acb_ce_loss.forward(preds=logits, targets=targets, mask=mask)
#     print("ACB CE LOSS: \t\t", loss)

#     acb_focal_loss = ACBLoss(samples=samples, loss_type='FocalLoss', gamma=.5)
#     loss = acb_focal_loss.forward(preds=logits, targets=targets, mask=mask)
#     print("ACB FOCAL LOSS: \t", loss)

#     recall_ce_loss = RecallLoss(samples=samples, loss_type='CELoss', gamma=.5)
#     loss = recall_ce_loss.forward(preds=logits, targets=targets, mask=mask)
#     print("RECALL CE LOSS: \t", loss)

#     recall_focal_loss = RecallLoss(samples=samples, loss_type='FocalLoss', gamma=.5)
#     loss = recall_focal_loss.forward(preds=logits, targets=targets, mask=mask)
#     print("RECALL FOCAL LOSS: \t", loss)