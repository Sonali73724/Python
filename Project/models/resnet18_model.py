import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Model(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        self.model = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        # Freeze backbone
        self.freeze_backbone()

        # Replace classifier
        in_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def freeze_backbone(self):
        """Freeze all backbone layers."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_last_blocks(self):
        """Unfreeze layer3, layer4 and classifier."""

        for param in self.model.layer3.parameters():
            param.requires_grad = True

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)