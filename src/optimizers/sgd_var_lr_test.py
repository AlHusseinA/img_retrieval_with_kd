
import torch.optim as optim
from torch.optim import SGD

class SGDOptimizerVariableLR:
    def __init__(self, model, lr=0.001, momentum=0.9, weight_decay=0, dampening=0, nesterov=False):
        self.model = model
        # Define the param_groups with different learning rates
        self.param_groups = [
            {'params': self.model.features[0].parameters(), 'lr': lr * 0.003},  # conv1
            {'params': self.model.features[1].parameters(), 'lr': lr * 0.009},  # bn1
            {'params': self.model.features[4].parameters(), 'lr': lr * 0.016},  # layer1
            {'params': self.model.features[5].parameters(), 'lr': lr * 0.036},  # layer2
            {'params': self.model.features[6].parameters(), 'lr': lr},          # layer3
            {'params': self.model.features[7].parameters(), 'lr': lr},          # layer4
            {'params': self.model.fc.parameters(), 'lr': lr}                     # fc
        ]
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov

    def get_optimizer(self):
        optimizer = SGD(self.param_groups, lr=self.lr, momentum=self.momentum, dampening=self.dampening, weight_decay=self.weight_decay, nesterov=self.nesterov)
        return optimizer
