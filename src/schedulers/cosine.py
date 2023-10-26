

import torch
import math

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR


class CosineAnnealingLRWrapper:
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        """
        Parameters:
        - optimizer (Optimizer): Wrapped optimizer.
        - T_max (int): Maximum number of iterations/epochs.
        - eta_min (float, optional): Minimum learning rate. Default: 0.
        - last_epoch (int, optional): The index of last epoch. Default: -1.
        """
        self.scheduler = CosineAnnealingLR(optimizer, T_max, eta_min, last_epoch)
        
    def step(self):
        self.scheduler.step()
        
    def get_lr(self):
        return self.scheduler.get_lr()



class CustomCosineAnnealingWarmUpLR:
    def __init__(self, optimizer, lr_warmup_epochs, lr_warmup_decay, T_max, last_epoch=-1):
        self.optimizer = optimizer
        self.lr_warmup_epochs = lr_warmup_epochs
        self.lr_warmup_decay = lr_warmup_decay
        self.T_max = T_max
        self.last_epoch = last_epoch
        self.cosine_annealing_scheduler = CosineAnnealingLR(self.optimizer, self.T_max - self.lr_warmup_epochs, last_epoch=self.last_epoch - self.lr_warmup_epochs if self.last_epoch != -1 else -1)
        self.epoch = self.last_epoch

    def step(self):
        self.epoch += 1
        if self.epoch <= self.lr_warmup_epochs:
            warmup_lr = self.lr_warmup_decay + (0.5 - self.lr_warmup_decay) * (self.epoch / self.lr_warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_lr
        else:
            self.cosine_annealing_scheduler.step()
