import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from compression.pca import PCAWrapper

class ResNet50_vanilla_with_PCA(nn.Module):
    def __init__(self, num_classes=200, pca_components=2048, num_components_to_keep=8, set_eval_mode=False, weights=ResNet50_Weights.DEFAULT, pretrained_weights=None, **kwargs):
        super().__init__()
        
        self.resnet50 = resnet50(weights=weights)
        self.num_classes = num_classes
        self.set_eval_mode = set_eval_mode
        self.features = nn.Sequential(*list(self.resnet50.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        in_features = self.resnet50.fc.in_features
        # write assert that in_features == 2048
        # Initialize PCA Wrapper
        self.pca_wrapper = PCAWrapper(pca_components, num_components_to_keep)

        assert pca_components == in_features, f"PCA components must be equal to the number of features (2048) from the resnet50 model. Expected {in_features}, got {pca_components}"

        self.fc = nn.Linear(pca_components, num_classes)  # Adjusted for PCA output size

    def fine_tune_mode(self):
        for param in self.parameters():
            param.requires_grad = True

# add a feature extractor mode  
    # def feature_extractor_mode(self):
    #     for param in self.parameters():
    #         param.requires_grad = False
            
    #     self.fc = nn.Identity()  # Remove classification head

    def fit_pca(self, feature_batch):
        """Fit PCA on a batch of features."""
        self.pca_wrapper.compress(feature_batch)

    def forward(self, x):
        if self.set_eval_mode:
            self.features.eval()
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Apply PCA to the features
        x = self.pca_wrapper.compress(x)

        x = self.fc(x)
        return x


