# Optimization Functions
# BoMeyering 2024

import torch
from typing import List, Generator

def get_optimizer(args, parameters: Generator, weight_decay: float=1e-5):

    if args.optimizer.optim_name == 'sgd':
        optimizer = torch.optim.SGD(
            params=parameters, 
            lr=args.optimizer.lr,
            momentum=args.optimizer.momentum,
            weight_decay=weight_decay, 
            nesterov=args.optimizer.nesterov
            #...
        )
        return optimizer
    
    # REPEAT FOR OTHER TYPES
    # elif args.optim_name == 'adam':
    #     pass

class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
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
        self.update_parms()
        self.apply_shadow()

    def restore_params(self):
        """
        Restores the original model parameters to the model
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.original_params[name].data)
