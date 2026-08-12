import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Conv2d(3, 64, 3, padding=1)
        self.norm = nn.BatchNorm2d(64)
        self.block1 = nn.Conv2d(64, 64, 3, padding=1)
        self.block2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.norm(self.stem(x)))
        skip = x
        x = F.relu(self.block1(x))
        x = self.block2(x)
        x = x + skip
        x = self.pool(x)
        x = torch.flatten(x, 1)
        logits = self.head(x)
        return logits
