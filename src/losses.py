# Loss Classes
# BoMeyering 2024

import torch
import torch.nn.functional as F
from typing import Union, Optional

@torch.no_grad()
def mask_targets(targets: torch.Tensor, mask: torch.Tensor, ignore_index: int=-1):

    adj_targets = torch.where(mask, targets, torch.full_like(targets, ignore_index))

    return adj_targets
        
class CELoss(torch.nn.Module):
    """
    Wrapper class for vanilla cross entropy loss
    """
    def __init__(self, ignore_index=-1, label_smoothing: float=0.0, weights: Optional[torch.tensor]=None, reduction: str='mean'):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.weights = weights
        self.reduction = reduction

    def forward(self, preds: torch.tensor, targets: torch.tensor, mask: Optional[torch.bool] = None):
        if mask is not None:
            targets = mask_targets(targets, mask, self.ignore_index)

        loss = F.cross_entropy(
            input=preds, 
            target=targets, 
            ignore_index=self.ignore_index, 
            label_smoothing=self.label_smoothing, 
            reduction=self.reduction, 
            weight=self.weights
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
    
    def forward(self, preds: torch.tensor, targets: torch.tensor, mask: Optional[torch.bool] = None):
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
        self.samples = samples
        self.loss_type = loss_type
        self.reduction = reduction
        self.gamma = gamma
        self.N = self.samples.sum()
        self.beta = (self.N - 1) / self.N
        self.C = len(self.samples)

    def _effective_samples(self):

        # Calculate effective samples
        E = (1 - (self.beta ** self.samples))/(1 - self.beta)

        # Invert to get alpha weights and normalize
        alpha = 1/E * self.C / (1/E).sum()

        self.E = E
        self.alpha = alpha
     
    def forward(self, preds: torch.tensor, targets: torch.tensor, mask: Optional[torch.bool] = None):
        # Calculate effective samples
        self._effective_samples()
        
        if self.loss_type == 'CELoss':
            loss_fn = CELoss(weights=self.alpha, reduction=self.reduction)
            loss = loss_fn(preds=preds, targets=targets, mask=mask)
        elif self.loss_type == 'FocalLoss':
            loss_fn = FocalLoss(alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)
            loss = loss_fn(preds=preds, targets=targets, mask=mask)
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}. Must be one of ['CELoss', 'FocalLoss']")
        
        return loss
  
class ACBLoss(torch.nn.Module):
    """
    Implement Adaptive Class Balanced Loss from Xu et al 2022.
    https://ieeexplore.ieee.org/document/10137858
    """
    def __init__(self, samples: torch.tensor, loss_type: str, reduction: str='mean', gamma: Optional[float]=2.0):
        super().__init__()
        self.samples = samples
        self.loss_type = loss_type
        self.reduction = reduction
        self.gamma = gamma
        self.N = self.samples.sum()
        self.N_max = torch.max(self.samples)
        self.C = len(self.samples)

    def _effective_samples(self):

        # Sample size, class size, and degree of imbalance calculations
        self.u = torch.log(self.N.float())
        self.v = torch.log(torch.tensor(self.C).float())
        self.b = -torch.log10(self.samples / self.N_max).mean()
        self.f_uvb = self.u / (self.v ** torch.sqrt(self.b))
        self.beta = torch.tanh(self.f_uvb)

        # Calculate effective samples
        E = (1 - (self.beta ** self.samples))/(1 - self.beta)

        # Invert to get alpha weights and normalize
        alpha = 1/E
        alpha = alpha / (alpha).sum() * self.C 

        self.E = E
        self.alpha = alpha

    def forward(self, preds: torch.tensor, targets: torch.tensor, mask: Optional[torch.bool] = None):
        # Calculate effective samples
        self._effective_samples()
        
        if self.loss_type == 'CELoss':
            loss_fn = CELoss(weights=self.alpha, reduction=self.reduction)
            loss = loss_fn(preds=preds, targets=targets, mask=mask)
        elif self.loss_type == 'FocalLoss':
            loss_fn = FocalLoss(alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)
            loss = loss_fn(preds=preds, targets=targets, mask=mask)
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}. Must be one of ['CELoss', 'FocalLoss']")
        
        return loss

class RecallLoss(torch.nn.Module):
    """
    Implementation of Recall Loss with dynamic weighting
    https://arxiv.org/pdf/2106.14917
    """
    def __init__(self, samples: torch.tensor, loss_type: str, reduction: str='mean', gamma: Optional[float]=2.0):
        super().__init__()
        self.samples = samples
        self.loss_type = loss_type
        self.reduction = reduction
        self.gamma = gamma
        self.N = self.samples.sum()
        self.C = len(self.samples)

    def _calculate_weights(self, preds: torch.tensor, targets: torch.tensor):

        # Get probs from logits and calculate one-hot tensors
        probs = F.softmax(preds, dim=1)
        pred_labels = torch.argmax(probs, dim=1)
        pred_oh = F.one_hot(pred_labels, num_classes=self.C)
        target_oh = F.one_hot(targets, num_classes=self.C)

        # Reshape to (-1, num_classes) to sum over one dimension
        pred_oh = pred_oh.view(-1, self.C)
        target_oh = target_oh.view(-1, self.C)

        # Calculate TP and FN rates
        TP = ((target_oh == 1) * (pred_oh == 1)).sum(dim=0)
        FN = ((target_oh == 1) * (pred_oh == 0)).sum(dim=0)

        # Calculate recall and weights
        R_c = (TP / (FN + TP)).clamp(min=1e-12)
        weights = 1 - R_c
        
        if torch.all(weights == 0):
            weights = torch.full_like(weights, 1.)

        weights = weights / weights.sum() * self.C

        self.alpha = weights

    def forward(self, preds: torch.tensor, targets: torch.tensor, mask: Optional[torch.bool] = None):
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



class ACWLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets, mask: torch.bool = None):
        pass

class DiceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets, mask: torch.bool = None):
        pass

class FocalTverskyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets, mask: torch.bool = None):
        pass


if __name__ == '__main__':
    logits = torch.randn(3, 4, 5, 5) * 2
    sm = F.softmax(logits + torch.randn(3, 4, 5, 5), dim=1)
    max_p, targets = torch.max(sm, dim=1)

    mask = torch.ge(max_p, 0.8)

    samples = torch.tensor([2390, 4201, 9011, 6850])
    inv_weights = 1/samples

    cross_entropy = CELoss()
    loss = cross_entropy.forward(preds=logits, targets=targets, mask=mask)
    print("CE LOSS: \t\t", loss)

    weighted_ce = CELoss(weights=inv_weights)
    loss = weighted_ce.forward(preds=logits, targets=targets, mask=mask)
    print("WEIGHTED CE LOSS: \t", loss)

    focal_loss = FocalLoss(gamma=.5)
    loss = focal_loss.forward(preds=logits, targets=targets, mask=mask)
    print("FOCAL LOSS: \t\t", loss)

    cb_ce_loss = CBLoss(samples=samples, loss_type='CELoss', gamma=.5)
    loss = cb_ce_loss.forward(preds=logits, targets=targets, mask=mask)
    print("CB CE LOSS: \t\t", loss)

    cb_focal_loss = CBLoss(samples=samples, loss_type='FocalLoss', gamma=.5)
    loss = cb_focal_loss.forward(preds=logits, targets=targets, mask=mask)
    print("CB FOCAL LOSS: \t\t", loss)

    acb_ce_loss = ACBLoss(samples=samples, loss_type='CELoss', gamma=.5)
    loss = acb_ce_loss.forward(preds=logits, targets=targets, mask=mask)
    print("ACB CE LOSS: \t\t", loss)

    acb_focal_loss = ACBLoss(samples=samples, loss_type='FocalLoss', gamma=.5)
    loss = acb_focal_loss.forward(preds=logits, targets=targets, mask=mask)
    print("ACB FOCAL LOSS: \t", loss)

    recall_ce_loss = RecallLoss(samples=samples, loss_type='CELoss', gamma=.5)
    loss = recall_ce_loss.forward(preds=logits, targets=targets, mask=mask)
    print("RECALL CE LOSS: \t", loss)

    recall_focal_loss = RecallLoss(samples=samples, loss_type='FocalLoss', gamma=.5)
    loss = recall_focal_loss.forward(preds=logits, targets=targets, mask=mask)
    print("RECALL FOCAL LOSS: \t", loss)