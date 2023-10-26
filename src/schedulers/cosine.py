

import torch
import math

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



class CosineAnnealingLRWrapperWithWarmup:
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, warmup_epochs=0, warmup_start_lr=None, warmup_decay="linear"):
        self.scheduler = CosineAnnealingLR(optimizer, T_max, eta_min, last_epoch)
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr is not None else [group['lr'] for group in optimizer.param_groups]
        self.warmup_decay = warmup_decay
        self.current_epoch = last_epoch + 1
        self.initial_lr = [group['lr'] for group in optimizer.param_groups]

    def step(self):
        if self.current_epoch < self.warmup_epochs:
            fraction = self.current_epoch / self.warmup_epochs
            for i, param_group in enumerate(self.optimizer.param_groups):
                if self.warmup_decay == "linear":
                    new_lr = self.warmup_start_lr[i] + fraction * (self.initial_lr[i] - self.warmup_start_lr[i])
                elif self.warmup_decay == "exponential":
                    new_lr = self.warmup_start_lr[i] * (self.initial_lr[i] / self.warmup_start_lr[i]) ** fraction
                elif self.warmup_decay == "cosine":
                    new_lr = self.warmup_start_lr[i] + (1 + math.cos(math.pi * fraction)) / 2 * (self.initial_lr[i] - self.warmup_start_lr[i])
                else:
                    raise ValueError(f"Invalid warmup_decay: {self.warmup_decay}")

                param_group['lr'] = new_lr
        else:
            self.scheduler.step()

        self.current_epoch += 1

    def get_lr(self):
        if self.current_epoch < self.warmup_epochs:
            return [group['lr'] for group in self.optimizer.param_groups]
        else:
            return self.scheduler.get_lr()
