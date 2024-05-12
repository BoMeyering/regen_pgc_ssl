# Trainer functions
# BoMeyering 2024

import torch
import uuid
import argparse
from abc import ABC, abstractmethod
from tqdm import tqdm
from src.eval import AverageMeterSet
from src.fixmatch import get_pseudo_labels, mask_targets
from typing import Tuple

class Trainer(ABC):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.meters = AverageMeterSet()

    @abstractmethod
    def _train_step(self):
        """Implement the train step for one batch"""
        pass

    @abstractmethod
    def _val_step(self):
        """Implement the val step for one batch"""
        pass
    
    @abstractmethod
    def _train_epoch(self):
        """Implement the training method for one epoch"""
        pass
    
    @abstractmethod
    def _val_epoch(self):
        """Implement the validation method for one epoch"""
        pass
    
    @abstractmethod
    def train(self):
        """Implement the whole training loop"""
        pass

class FixMatchTrainer(Trainer):
    def __init__(self, name, args: argparse.Namespace, model: torch.nn.Module, train_loaders, train_length, val_loader, optimizer, criterion, scheduler=None, ema=None):
        super().__init__(name=name)
        self.trainer_id = "_".join(["fixmatch", str(uuid.uuid4())])
        self.args = args
        self.model = model
        self.train_loaders = train_loaders
        self.train_length = train_length
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.ema = ema

    def _train_step(self, batches: Tuple):
        
        # Unpack batches
        l_batch, u_batch = batches
        l_img, l_targets = l_batch
        weak_img, strong_img = u_batch
        
        # Concatenate all inputs and send to device
        inputs = torch.cat((l_img, weak_img, strong_img)).float().to(self.args.device)
        l_targets = l_targets.long().to(self.args.device)

        # Compute logits for labeled and unlabeled images
        logits = self.model(inputs)
        l_logits = logits[:len(l_img)]
        weak_logits, strong_logits = logits[len(l_img): ].chunk(2) # grab the unlabeled logits and split into two

        # labeled loss
        l_loss = self.criterion(l_logits, l_targets)

        # pseudo-labeling data
        u_targets, mask = get_pseudo_labels(self.args, weak_logits)
        u_targets = mask_targets(u_targets, mask, ignore_index=self.args.loss.ignore_index)
        u_loss = self.criterion(strong_logits, u_targets)

        total_loss = l_loss + self.args.fixmatch.lam * u_loss

        self.meters.update("total_loss", total_loss.item(), 1)
        self.meters.update("labeled_loss", l_loss.item(), l_logits.size()[0])
        self.meters.update("unlabeled_loss", u_loss.item(), strong_logits.size()[0])

        return total_loss
    
    def _train_epoch(self, epoch: int):
        
        # Reset meters
        self.model.train()
        self.meters.reset()

        # Set progress bar and unpack batches
        p_bar = tqdm(range(self.train_length))
        train_l_loader, train_u_loader = self.train_loaders

        # Reinstantiate iterator loaders
        train_l_loader = iter(train_l_loader)
        train_u_loader = iter(train_u_loader)

        for batch_idx in range(self.train_length):

            # Zero the optimizer
            self.optimizer.zero_grad()

            # Get batches
            batches = (next(train_l_loader), next(train_u_loader))

            # Train one batch and backpropagate
            loss = self._train_step(batches)
            loss.backward()

            # Update parameters
            self.optimizer.step()
            if self.ema:
                self.ema.update()

            p_bar.set_description(
                "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Loss: {loss:.6f}".format(
                    epoch=epoch + 1,
                    epochs=self.args.model.epochs,
                    batch=batch_idx + 1,
                    iter=len(train_l_loader),
                    # lr=self.scheduler.get_last_lr()[0],
                    lr = self.args.optimizer.lr,
                    loss = loss.item()
                )
            )
            p_bar.update()
        if self.scheduler:
            self.scheduler.step()

        return (
            self.meters["total_loss"].avg,
            self.meters["labeled_loss"].avg,
            self.meters["unlabeled_loss"].avg,
        )
    
    @torch.no_grad()
    def _val_step(self, batch):

        # Unpack batch and send to device
        img, targets = batch
        img.float().to(self.args.device)
        targets.long().to(self.args.device)

        # Forward pass through model
        logits = self.model(img)

        # Calculate validation loss
        loss = self.criterion(logits, targets)

        # Update running meters
        self.meters.update("validation_loss", loss.item(), logits.size()[0])

        return loss

    @torch.no_grad()
    def _val_epoch(self, epoch: int):

        # Reset meters
        self.model.eval()
        self.meters.reset()

        # Set progress bar and unpack batches
        p_bar = tqdm(range(len(self.val_loader)))
        for batch_idx, batch in enumerate(self.val_loader):

            # Validate one batch and 
            loss = self._val_step(batch)

            p_bar.set_description(
                "Val Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Loss: {loss:.6f}".format(
                    epoch=epoch + 1,
                    epochs=self.args.model.epochs,
                    batch=batch_idx + 1,
                    iter=len(self.val_loader),
                    # lr=self.scheduler.get_last_lr()[0],
                    lr = self.args.optimizer.lr,
                    loss = loss.item()
                )
            )
            p_bar.update()
        
        return (
            self.meters["validation_loss"].avg,
        )

    def train(self):

        print(f'training for {self.args.model.epochs}')
        for epoch in range(self.args.model.epochs):
            train_loss = self._train_epoch(epoch)
            print("TrainingLoss", train_loss)
            val_loss = self._val_epoch(epoch)
            print("ValidataionLoss", val_loss)

class SupervisedTrainer(Trainer):
    def __init__(self, name):
        super().__init__(name=name)
        self.trainer_id = "_".join(["supervised", str(uuid.uuid4())])

    def _train_step(self):
        return super()._train_step()
    
    def _train_epoch(self):
        return super()._train_epoch()
    
    def _val_step(self):
        return super()._val_step()
    
    def _val_epoch(self):
        return super()._val_epoch()
    
    def train(self):
        return super().train()

if __name__ == '__main__':
    strainer = SupervisedTrainer(name='supervised_trainer')
    print(strainer.name)
    print(strainer.trainer_id)