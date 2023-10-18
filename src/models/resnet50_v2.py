import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class DynamicResNet50(nn.Module):
    def __init__(self, feature_size, num_classes, weights=ResNet50_Weights.DEFAULT):
        super(DynamicResNet50, self).__init__()

        # Load the pre-trained ResNet-50 model
        model = resnet50(weights=weights)
        
        # Create the feature extractor from the pre-trained model
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3
        )
        
        # # Modify the last layer to change the output channels
        # in_channels = model.layer4[0].conv1.in_channels  # Get input channels of the last layer
        # self.features.add_module('layer4_mod', nn.Sequential(
        #     Bottleneck(in_channels, feature_size // 4, feature_size, downsample=nn.Sequential(
        #         nn.Conv2d(in_channels, feature_size, kernel_size=1, stride=2, bias=False),
        #         nn.BatchNorm2d(feature_size)
        #     )),
        #     model.layer4[1],
        #     model.layer4[2]
        # ))
        
        # Modify the last layer to change the output channels
        in_channels = model.layer4[0].conv1.in_channels  # Get input channels of the last layer
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, feature_size, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(feature_size)
        )
        bottleneck = Bottleneck(in_channels, feature_size // 4, feature_size, downsample=downsample)
        self.features.add_module('layer4_mod', nn.Sequential(
            bottleneck,
            model.layer4[1],
            model.layer4[2]
        ))

        # Add the average pooling and flatten layers
        self.features.add_module('avgpool', model.avgpool)
        self.features.add_module('flatten', nn.Flatten())
        
        # Create the classifier
        self.classifier = nn.Linear(feature_size, num_classes)

    def fine_tune_mode(self):
        """Activate fine-tuning mode: all weights trainable."""
        for param in self.parameters():
            param.requires_grad = True
        print("Fine-tuning mode activated: all layers are now trainable.")

    def feature_extractor_mode(self):
        """Activate feature extractor mode: freeze all weights and remove classification head."""
        for param in self.parameters():
            param.requires_grad = False
        self.classifier = nn.Identity()
        print("Feature extractor mode activated: all layers are frozen and the classification head is removed.")

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class Bottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

