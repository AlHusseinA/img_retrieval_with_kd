import torch
import torch.nn as nn


# write modular class for cross entropy loss 
class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        # self.loss = nn.CrossEntropyLoss()
    def forward(self, y_pred, y_true):
        return self.loss(y_pred, y_true)


