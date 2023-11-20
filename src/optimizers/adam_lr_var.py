import torch.optim as optim
from torch.optim import Adam

class AdamOptimizerVar:
    def __init__(self, model, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5, amsgrad=False):
        # Validate that the model has the expected attributes
        # for attr in ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
        #     if not any(hasattr(layer, attr) for layer in self.model.features):

        #     # if not hasattr(self.model, attr):
        #         raise AttributeError(f"The model does not have '{attr}' attribute.")
                
        self.model = model
        self.param_groups = [
            {'params': self.model.features[0].parameters(), 'lr': lr * 0.001},  # conv1
            {'params': self.model.features[1].parameters(), 'lr': lr * 0.004},  # bn1
            {'params': self.model.features[4].parameters(), 'lr': lr * 0.012},  # layer1
            {'params': self.model.features[5].parameters(), 'lr': lr * 0.036},  # layer2
            {'params': self.model.features[6].parameters(), 'lr': lr},       # layer3
            {'params': self.model.features[7].parameters(), 'lr': lr},       # layer4
            # {'params': self.model.classifier.parameters(), 'lr': lr}         # fc (now called classifier)
            {'params': self.model.fc.parameters(), 'lr': lr}         # fc (now called classifier)
        ]
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad

    def get_optimizer(self):
        optimizer = Adam(self.param_groups, lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay, amsgrad=self.amsgrad)
        return optimizer
