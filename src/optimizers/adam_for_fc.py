import torch.optim as optim
from torch.optim import Adam

class AdamOptimizerFC:
    def __init__(self, model, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5, amsgrad=False):
        self.model = model
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad

    def get_optimizer(self):
        # Create a list of all parameter groups with a unified learning rate
        param_groups = [{'params': param_group, 'lr': self.lr} for param_group in self.model.parameters()]
        
        # Create the Adam optimizer with the unified learning rate
        optimizer = Adam(param_groups, lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay, amsgrad=self.amsgrad)
        return optimizer
