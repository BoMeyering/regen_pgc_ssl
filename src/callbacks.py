# Torch model callbacks
# BoMeyering 2024

import torch
import logging

class ModelCheckpoint:
    def __init__(self, filepath, monitor='val_loss', metadata=None):
        self.filepath = filepath
        self.monitor = monitor
        self.best = float('inf')
        self.monitor_op = torch.lt
        self.logger = logging.getLogger()
        self.metadata = metadata if metadata is not None else {}

    def __call__(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            self.logger.warning(f"Warning: Metric {self.monitor} is not available. Skipping checkpoint")
            return None

        if self.monitor_op(current, self.best):
            self.logger.info(f"Epoch {epoch + 1}: '{self.monitor}' improved from {self.best:.6f} to {current:.6f}. Saving model to {self.filepath}")
            self.best = current
            chkpt = {
                'model_state_dict': logs['model_state_dict'],
                'epoch': epoch,
                'monitor': self.monitor,
                self.monitor: current,
                **self.metadata
            }
            torch.save(chkpt, self.filepath)
        else:
            self.logger.info(f"Epoch {epoch}: '{self.monitor}' did not improve from {self.best:.6f}.Skipping checkpoint.")

