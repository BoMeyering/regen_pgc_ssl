# Trainer functions
# BoMeyering 2024

import torch
import uuid
import logging
import argparse
from abc import ABC, abstractmethod
from tqdm import tqdm
from datetime import datetime
from src.eval import AverageMeterSet
from src.fixmatch import get_pseudo_labels, mask_targets
from src.callbacks import ModelCheckpoint
from src.metrics import MetricLogger
from typing import Tuple
from pathlib import Path
from torchmetrics.functional import dice

import torch.nn.functional as F

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
        self.logger = logging.getLogger()

        # setup metrics class
        self.metrics = MetricLogger(num_classes=args.model.num_classes, device=args.device)

        chkpt_path = Path('./model_checkpoints') / "_".join((self.args.run_name, datetime.now().isoformat(timespec='seconds', sep='_')))
        self.checkpoint = ModelCheckpoint(filepath=chkpt_path, metadata=vars(self.args))

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
        # l_loss = F.cross_entropy(l_logits, l_targets, reduction='mean')

        # pseudo-labeling data
        u_targets, mask = get_pseudo_labels(self.args, weak_logits)

        # print(u_targets)
        # print(mask)


        u_targets = mask_targets(u_targets, mask, ignore_index=self.args.loss.ignore_index)
        u_loss = self.criterion(strong_logits, u_targets)


        print(u_targets)
        print(mask)
        
        # u_loss = F.cross_entropy(strong_logits, u_targets, reduction='mean')

        total_loss = l_loss + self.args.fixmatch.lam * u_loss
        
        # Update loss meters
        self.meters.update("total_loss", total_loss.item(), 1)
        self.meters.update("labeled_loss", l_loss.item(), l_logits.size()[0])
        self.meters.update("unlabeled_loss", u_loss.item(), strong_logits.size()[0])

        # Update metrics
        self.metrics.update(preds=l_logits, targets=l_targets)

        # return l_loss
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

            # print metrics
            if batch_idx % 20 == 0:
                self.metrics.print_metrics(type='both')

            # Update parameters
            self.optimizer.step()
            # if self.ema:
            #     self.ema.update()

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

        self.logger.info(f"Epoch {epoch + 1} - Total Loss: {self.meters['total_loss'].avg:.6f} Labeled Loss: {self.meters['labeled_loss'].avg:.6f} Unlabeled Loss: {self.meters['unlabeled_loss'].avg:.6f}")

        return (
            self.meters["total_loss"].avg,
            self.meters["labeled_loss"].avg,
            self.meters["unlabeled_loss"].avg,
        )
    
    @torch.no_grad()
    def _val_step(self, batch):

        # Unpack batch and send to device
        img, targets = batch
        img = img.float().to(self.args.device)
        targets = targets.long().to(self.args.device)

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
        self.logger.info(f"Epoch {epoch + 1} - Validation Loss: {self.meters['validation_loss'].avg:.6f}")

        return (
            self.meters["validation_loss"].avg,
        )

    def train(self):
        self.logger.info(f'Training for {self.args.model.epochs} epochs')
        for epoch in range(self.args.model.epochs):
            train_loss = self._train_epoch(epoch)
            val_loss = self._val_epoch(epoch)
            
            print(train_loss)
            print(val_loss)
            logs = {
                'epoch': epoch,
                'train_loss': torch.tensor(train_loss[0]),
                'val_loss': torch.tensor(val_loss[0]),
                'model_state_dict': self.model.state_dict()
            }

            self.checkpoint(epoch=epoch, logs=logs)

            




class SupervisedTrainer(Trainer):
    def __init__(self, name, args: argparse.Namespace, model: torch.nn.Module, train_loaders, train_length, val_loader, optimizer, criterion, scheduler=None, ema=None):
        super().__init__(name=name)
        self.trainer_id = "_".join(["supervised", str(uuid.uuid4())])
        self.args = args
        self.model = model
        self.train_loaders = train_loaders
        self.train_length = train_length
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.ema = ema
        self.logger = logging.getLogger()

        chkpt_path = Path('./model_checkpoints') / "_".join((self.args.run_name, datetime.now().isoformat(timespec='seconds', sep='_')))
        self.checkpoint = ModelCheckpoint(filepath=chkpt_path, metadata=vars(self.args))

    def _train_step(self, batch: Tuple):
        
        # Unpack batch
        img, targets = batch
        
        # Send inputs to device
        inputs = img.to(self.args.device)
        targets = targets.to(self.args.device)

        # Compute logits for labeled and unlabeled images
        logits = self.model(inputs)

        # Loss
        loss = self.criterion(logits, targets)

        self.meters.update("train_loss", loss.item(), )

        return total_loss
    
    def _train_epoch(self):
        return super()._train_epoch()
    
    def _val_step(self):
        return super()._val_step()
    
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
        self.logger.info(f"Epoch {epoch + 1} - Validation Loss: {self.meters['validation_loss'].avg:.6f}")

        return (
            self.meters["validation_loss"].avg,
        )
    
    def train(self):
        for epoch in range(self.args.model.epochs):
            train_loss = self._train_epoch(epoch)
            self.logger.info(f"Epoch {epoch} Training Loss: {train_loss}")

            val_loss = self._val_epoch(epoch)
            self.logger.info("Epoch {epoch} Validation Loss: {val_loss}")
            
            logs = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'model_state_dict': self.model.state_dict()
            }

if __name__ == '__main__':
    strainer = SupervisedTrainer(name='supervised_trainer')
    print(strainer.name)
    print(strainer.trainer_id)