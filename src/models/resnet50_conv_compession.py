import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50_conv(nn.Module):
    def __init__(self, features_size, num_classes, weights=ResNet50_Weights.DEFAULT, pretrained_weights=None, **kwargs):
        super().__init__()

        model = resnet50(weights=weights)  # Adjusted to load pretrained weights correctly
        self.num_classes = num_classes
        self.features_size = features_size

        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool,
            nn.Conv2d(2048, features_size, 1, 1, 0),
            nn.Flatten()
        )

        # self.features_out = nn.Linear(in_features=2048, out_features=features_size)
        self.classifier = nn.Linear(in_features=features_size, out_features=num_classes)

        if pretrained_weights:
            self.load_state_dict(torch.load(pretrained_weights))

    def forward(self, x):
        x = self.features(x)
        # Forward pass through the extra linear layer
        # x = self.features_out(x)
        # Assert the feature size
        assert x.shape[1] == self.features_size, f"Expected feature size {self.features_size}, got {x.shape[1]}"   
        x = self.classifier(x)
        return x

    def fine_tune_mode(self):
        """Activate fine-tuning mode: classification head active and all weights trainable."""
        for param in self.parameters():
            param.requires_grad = True

    def feature_extractor_mode(self):
        """Activate feature extractor mode: remove classification head and freeze all weights."""
        for param in self.parameters():
            param.requires_grad = False
            
        self.classifier = nn.Identity()  # Remove classification head






