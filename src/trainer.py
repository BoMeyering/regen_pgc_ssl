# Trainer functions
# BoMeyering 2024

import torch
import uuid
from abc import ABC, abstractmethod

class Trainer(ABC):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.trainer_id = uuid.uuid4()

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

# class FixMatchTrainer(Trainer):
    # def __init__(self):
    #     pass

    # def _train_step(self):
    #     pass

    # def _val_step(self):
    #     pass

    # def _train_epoch(self):
    #     pass

    # def _val_epoch(self):
    #     pass

    # def train(self):
    #     pass

class SupervisedTrainer(Trainer):
    def __init__(self, name):
        super().__init__(name=name)

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