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
    def __init__(
            self, 
            name, 
            args: argparse.Namespace, 
            model: torch.nn.Module, 
            train_loaders, 
            train_length, 
            val_loader, 
            optimizer, 
            criterion, 
            scheduler=None, 
            ema=None, 
            tb_logger: torch.utils.tensorboard.writer.SummaryWriter=None,
            class_map: dict=None):
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
        self.tb_logger = tb_logger
        self.class_map = class_map

        # setup metrics class
        self.train_metrics = MetricLogger(num_classes=args.model.num_classes, device=args.device)
        self.val_metrics = MetricLogger(num_classes=args.model.num_classes, device=args.device)

        chkpt_path = Path('./model_checkpoints') / self.args.run_name
        self.checkpoint = ModelCheckpoint(filepath=chkpt_path, metadata=vars(self.args))

    def _train_step(self, batches: Tuple):
        "Train on one batch of labeled and unlabeled images."
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

        # Calculate labeled loss
        l_loss = self.criterion(l_logits, l_targets)

        # Pseudo-label the unlabled images
        u_targets, mask = get_pseudo_labels(self.args, weak_logits)

        # Calculate unlabeled loss
        u_loss = self.criterion(strong_logits, u_targets, mask)
        
        # Check for invalid loss
        if torch.isnan(u_loss):
            raise ValueError("Unlabeled loss is 'Nan'. Stopping model training.")
        
        # Compute total loss
        total_loss = l_loss + self.args.fixmatch.lam * u_loss

        # Update loss meters
        self.meters.update("total_loss", total_loss.item(), 1)
        self.meters.update("labeled_loss", l_loss.item(), l_logits.size()[0])
        self.meters.update("unlabeled_loss", u_loss.item(), strong_logits.size()[0])
    
        # Update metrics
        self.train_metrics.update(preds=l_logits, targets=l_targets)

        return total_loss, l_loss, u_loss
    
    def _train_epoch(self, epoch: int):
        
        # Reset meters
        self.model.train()
        self.meters.reset()
        self.train_metrics.reset()
        self.val_metrics.reset()

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
            loss, l_loss, u_loss = self._train_step(batches)
            # loss.backward()

            # Step optimizer and update parameters for EMA
            self.optimizer.step()
            if self.ema:
                self.ema.update()

            # Update progress bar
            p_bar.set_description(
                "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Loss: {loss:.6f}".format(
                    epoch=epoch + 1,
                    epochs=self.args.model.epochs,
                    batch=batch_idx + 1,
                    iter=len(train_l_loader),
                    lr=self.scheduler.get_last_lr()[0],
                    # lr = self.args.optimizer.lr,
                    loss = loss.item()
                )
            )
            p_bar.update()

            # Tensorboard batch writing
            loss_dict = {"train_loss": loss, "train_labeled_loss": l_loss, "train_unlabeled_loss": u_loss}
            batch_step = (epoch*self.train_length) + batch_idx
            self.tb_logger.log_scalar_dict(main_tag='step_loss', scalar_dict=loss_dict, step=batch_step)

            # Step logging
            # self.logger.info(f'{self.args.run_name} Epoch: {epoch + 1} - Step: {batch_idx} - Total Loss: {loss:.6f} Labeled Loss: {l_loss:.6f} Unlabeled Loss: {u_loss:.6f}')

        # Step LR scheduler
        if self.scheduler:
            self.scheduler.step()

        # Compute epoch metrics and loss
        avg_metrics, mc_metrics = self.train_metrics.compute()
        loss = self.meters['total_loss'].avg
        l_loss = self.meters['labeled_loss'].avg
        u_loss = self.meters['unlabeled_loss'].avg

        # Set the epoch step
        epoch_step = epoch + 1

        # Epoch Loss Logging
        loss_dict = tag_scalar_dict={"train_loss": loss, "train_labeled_loss": l_loss, "train_unlabeled_loss": u_loss}
        self.tb_logger.log_scalar_dict(main_tag='epoch_loss', scalar_dict=loss_dict, step=epoch_step)

        # Epoch Average Metric Logging
        self.tb_logger.log_scalar_dict(main_tag='epoch_train_metrics', scalar_dict=avg_metrics, step=epoch_step)

        # Epoch Multiclass Metric Logging
        self.tb_logger.log_tensor_dict(main_tag='epoch_train_metrics', tensor_dict=mc_metrics, step=epoch_step, class_map=self.class_map)

        # Logger Logging
        self.logger.info(f"Epoch {epoch + 1} - Total Loss: {loss:.6f} Labeled Loss: {l_loss:.6f} Unlabeled Loss: {u_loss:.6f}")
        self.logger.info(f"Epoch {epoch + 1} - Avg Metrics {avg_metrics}")
        self.logger.info(f"Epoch {epoch + 1} - Multiclass Metrics {mc_metrics}")

        

        return loss, l_loss, u_loss
    
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

        # Update metrics
        self.val_metrics.update(preds=logits, targets=targets)

        return loss

    @torch.no_grad()
    def _val_epoch(self, epoch: int):

        # Reset meters
        self.model.eval()
        self.meters.reset()

        # Set progress bar and unpack batches
        p_bar = tqdm(range(len(self.val_loader)))
        for batch_idx, batch in enumerate(self.val_loader):

            # Validate one batch
            loss = self._val_step(batch)

            # Update the progress bar
            p_bar.set_description(
                "Val Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Loss: {loss:.6f}".format(
                    epoch=epoch + 1,
                    epochs=self.args.model.epochs,
                    batch=batch_idx + 1,
                    iter=len(self.val_loader),
                    lr=self.scheduler.get_last_lr()[0],
                    # lr = self.args.optimizer.lr,
                    loss = loss.item()
                )
            )
            p_bar.update()

        # Compute epoch metrics
        avg_metrics, mc_metrics = self.val_metrics.compute()
        loss = self.meters['validation_loss'].avg

        self.logger.info(f'V')

        # Epoch step
        epoch_step = epoch + 1

        # Epoch Loss Logging
        loss_dict = {'validation_loss': loss}
        self.tb_logger.log_scalar_dict(main_tag='epoch_loss', scalar_dict=loss_dict, step=epoch_step)

        # Epoch Average Validation Metric Logging
        self.tb_logger.log_scalar_dict(main_tag='epoch_val_metrics', scalar_dict=avg_metrics, step=epoch_step)

        # Epoch Multiclass Validation Metric Logging
        self.tb_logger.log_tensor_dict(main_tag='epoch_val_metrics', tensor_dict=mc_metrics, step=epoch_step, class_map=self.class_map)

        # Logger Logging
        self.logger.info(f"Epoch {epoch + 1} - Validation Loss: {loss:.6f}")
        self.logger.info(f"Epoch {epoch + 1} - Avg Metrics {avg_metrics}")
        self.logger.info(f"Epoch {epoch + 1} - Multiclass Metrics {mc_metrics}")
        

        return loss

    def train(self):
        self.logger.info(f'Training for {self.args.model.epochs} epochs')
        for epoch in range(self.args.model.epochs):
            train_loss = self._train_epoch(epoch)
            val_loss = self._val_epoch(epoch)
            
            logs = {
                'epoch': epoch,
                'train_loss': torch.tensor(train_loss[0]),
                'val_loss': torch.tensor(val_loss),
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

class TBLogger:
    def __init__(self, writer: torch.utils.tensorboard.writer.SummaryWriter):
        self.writer = writer

    def log_scalar_dict(self, main_tag, scalar_dict, step):
        # Epoch Average Metric Logging
        for k, v in scalar_dict.items():
            self.writer.add_scalar(
                tag=f'{main_tag}/{k}',
                scalar_value=v,
                global_step=step,
                new_style=True
            )

    def log_tensor_dict(self, main_tag, tensor_dict, step, class_map):
        # Log each tensor value for each key in the tensor_dict
        for k, v in tensor_dict.items():
            print(k, v)
            for i, tensor_value in enumerate(v):
                class_name = class_map[i]
                self.writer.add_scalar(
                    tag=f'{main_tag}/{k}/{class_name}',
                    scalar_value=tensor_value,
                    global_step=step,
                    new_style=True
                )

