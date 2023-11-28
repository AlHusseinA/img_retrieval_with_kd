import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50_vanilla(nn.Module):
    def __init__(self, num_classes=200, set_eval_mode=False, weights=ResNet50_Weights.DEFAULT, pretrained_weights=None, **kwargs):

        super(ResNet50_vanilla, self).__init__()
        
        self.resnet50 = resnet50(weights=weights)  # Adjusted to load pretrained weights correctly
        self.num_classes = num_classes
        # Set eval mode flag
        self.set_eval_mode = set_eval_mode

        # Remove the last fully connected layer
        self.features = nn.Sequential(*list(self.resnet50.children())[:-2])
        
        # Adaptive Average Pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # In features for the fully connected layer
        in_features = self.resnet50.fc.in_features
        
        # Fully connected layer for CUB-200
        self.fc = nn.Linear(in_features, num_classes)
        
    def fine_tune_mode(self):
        """Activate fine-tuning mode: classification head active and all weights trainable."""
        for param in self.parameters():
            param.requires_grad = True    

    def forward(self, x):
        # Set to eval mode if flag is True
        if self.set_eval_mode:
            self.features.eval()
        
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten tensor
        # Forward pass through fully connected layer
        x = self.fc(x)
        return x
