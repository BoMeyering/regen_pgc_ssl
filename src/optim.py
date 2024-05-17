# Optimization Functions
# BoMeyering 2024

import torch
import inspect
from typing import List, Generator
import segmentation_models_pytorch as smp



class ConfigOptim:
    def __init__(self, args, model_parameters):
        self.args = args
        self.optim_params = vars(self.args.optimizer).copy()
        self.loss_params = vars(self.args.loss).copy()
        self.model_params = model_parameters

    def get_optimizer(self):
        
        try:
            OptimClass = getattr(torch.optim, self.optim_params['name'])
        except AttributeError:
            print(f"The loss function {self.optim_params['name']} is not in torch.nn. Defaulting to torch.optim.SGD.")
            OptimClass = torch.optim.SGD
        
        valid_params = inspect.signature(OptimClass).parameters
        filtered_params = {k: v for k, v in self.optim_params.items() if k in valid_params}
        optim_params = {'params': self.model_params}
        optim_params.update(filtered_params)

        optimizer = OptimClass(**optim_params)

        return optimizer

    def get_loss_criterion(self) -> torch.nn.Module:
        """
        Return an instantiated loss criterion from the config specs.

        Returns:
            torch.nn.Module: A loss criterion
        """
        loss_name = self.loss_params.get('name')
        if loss_name:
            try:
                LossClass = getattr(smp.losses, loss_name)
            except AttributeError:
                print(f"The loss function {loss_name} is not in smp.losses. Defaulting to torch.nn.CrossEntropyLoss.")
                LossClass = torch.nn.CrossEntropyLoss
        
        if loss_name == 'CBLoss':
            LossClass = CBLoss
        elif loss_name == 'RecallLoss':
            LossClass = RecallLoss
        elif loss_name == 'ACWLoss':
            LossClass = ACWLoss
        
        valid_params = inspect.signature(LossClass).parameters
        filtered_params = {k: v for k, v in self.loss_params.items() if k in valid_params}

        criterion = LossClass(**filtered_params)

        return criterion


def get_optimizer(args, parameters: Generator):

    if args.optimizer.name == 'sgd':
        optimizer = torch.optim.SGD(
            params=parameters, 
            lr=args.optimizer.lr,
            momentum=args.optimizer.momentum,
            # weight_decay=args.optimizer.weight_decay, 
            weight_decay=args.optimizer.weight_decay,
            nesterov=args.optimizer.nesterov
        )
        return optimizer
    
    # REPEAT FOR OTHER TYPES
    # elif args.optim_name == 'adam':
    #     pass


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float, verbose: bool=True):
        """
        Initialize exponential moving average of named model parameters
        Smooths the model parameters based on an exponential moving average equation:

        s_(t) = decay * s_(t-1) + (1 - decay) * x_(t)

        Where:
        s_(t) = the shadow parameter value at time t
        decay = the EMA decay rate
        s_(t-1) = the shadow parameter value at time t-1
        x_(t) = the model parameter value at time t

        The higher the decay rate, the smoother the updates to the parameters are since they take more of the previous parameters values into account.

        Args:
            model (torch.nn.Module): The model to update
            decay (float): Exponential moving average decay rate
        """
        
        self.model = model
        self.decay = decay
        self.shadow_params = {}
        self.original_params = {}
        self.update_params()
        self.verbose = verbose

    def  update_params(self):
        """
        Assigns the current parameters to the shadow params if they don't exist.
        Or updates the shadow_params by the decay rate and the current param values.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name not in self.shadow_params:
                    self.shadow_params[name] = param.data.clone()
                else:
                    self.shadow_params[name] = self.decay * self.shadow_params[name] + (1 - self.decay) * param.data
    
    def assign_params(self):
        """
        Assign shadow parameters to the model's parameters
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Update the original parameters in the original parameters dictionary
                self.original_params[name] = param.clone()
                # Copy the data from the shadow params to the model param
                param.data.copy_(self.shadow_params[name].data)

    def update(self):
        """
        Update shadow parameters and apply them to the model
        """
        self.update_params()
        self.assign_params()
        if self.verbose:
            print("Model parameters updated with shadow params")

    def restore_params(self):
        """
        Restores the original model parameters to the model
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.original_params[name].data)

