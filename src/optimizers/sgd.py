import torch
import torch.nn as nn
from torch.optim import SGD



class SGDOptimizerVariableLR:
    def __init__(self, model, lr=0.001, momentum=0.9, weight_decay=0):
        """
            Initializes the SGD optimizer with a specified learning rate and momentum.

            Args:
            lr (float): The learning rate for the SGD optimizer.
            momentum (float): The momentum for the SGD optimizer.
        """
        self.model = model
        self.param_groups = [
            {'params': self.model.features[0].parameters(), 'lr': lr * 0.001},  # conv1
            {'params': self.model.features[1].parameters(), 'lr': lr * 0.004},  # bn1
            {'params': self.model.features[4].parameters(), 'lr': lr * 0.012},  # layer1
            {'params': self.model.features[5].parameters(), 'lr': lr * 0.036},  # layer2
            {'params': self.model.features[6].parameters(), 'lr': lr},       # layer3
            {'params': self.model.features[7].parameters(), 'lr': lr},       # layer4
            {'params': self.model.classifier.parameters(), 'lr': lr}         # fc (now called classifier)
        ]

        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum

    def get_optimizer(self):
        """
            Creates and returns an SGD optimizer with the specified hyperparameters.

            Returns:
            SGD: An SGD optimizer with the specified hyperparameters.
        """
        optimizer = SGD(self.param_groups, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

        return optimizer



class SGDOptimizer:
    def __init__(self, model, lr=0.001, momentum=0.9, weight_decay=0):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.weight_decay = weight_decay
        self.momentum = momentum

    def get_optimizer(self):
        optimizer = SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        return optimizer
