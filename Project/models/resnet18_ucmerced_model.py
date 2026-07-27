"""
ResNet18 Model for UC Merced Dataset (21 Classes)
-----------------------------------------------
Pretrained ResNet18 backbone with a 21-output classification linear layer.
"""

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18UCMercedModel(nn.Module):
    """
    ResNet18 architecture adapted for 21-class UC Merced Land Use classification.
    """

    def __init__(self, num_classes: int = 21):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = self.model.fc.in_features  # 512
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
