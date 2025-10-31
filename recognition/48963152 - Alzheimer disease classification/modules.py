import torch.nn as nn
from torchvision import models
from torchvision.models import ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights, ConvNeXt_Base_Weights

class ConvNeXtClassifier(nn.Module):

    def __init__ (self, variant='tiny', number_of_classes=2, dropout = 0.2):
        super().__init__()
        variant = variant.lower()
        if variant == 'tiny':
            self.backbone = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
            feature_channels = 768
        elif variant == 'small':
            self.backbone = models.convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)
            feature_channels = 768
        elif variant == 'base':
            self.backbone = models.convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
            feature_channels = 1024
        else:
            print("variant must be 'tiny', 'small' or 'base'. Proceeding with 'tiny'")
            self.backbone = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
            feature_channels = 768

        #Replace original classifier to classify the number of classes we want. It helps to use backbone only for feature extraction
        if hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_channels, feature_channels // 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(feature_channels // 2, number_of_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        if features.ndim == 2:
            result = self.classifier(features)
        else:
            pooled = self.pool(features)
            result = self.classifier(pooled)
        return result
    

