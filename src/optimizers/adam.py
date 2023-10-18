import torch.optim as optim
from torch.optim import Adam


class AdamOptimizer:
    def __init__(self, model_parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False):
        """
            Initializes the Adam optimizer with a specified learning rate, betas, eps, weight decay, and amsgrad.

            Args:
            lr (float): The learning rate for the Adam optimizer.
            betas (tuple): The betas for the Adam optimizer.
            eps (float): The eps for the Adam optimizer.
            weight_decay (float): The weight decay for the Adam optimizer.
            amsgrad (bool): The amsgrad for the Adam optimizer.
        """
        self.model_parameters = model_parameters
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad

    def get_optimizer(self):
        """
            Creates and returns an Adam optimizer with the specified hyperparameters.

            Returns:
            Adam: An Adam optimizer with the specified hyperparameters.
        """
        # optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay, amsgrad=self.amsgrad)
        optimizer = Adam(self.model_parameters, lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay, amsgrad=self.amsgrad)

        return optimizer