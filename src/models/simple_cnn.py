# File: src/models/simple_cnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    Lightweight CNN for CIFAR-10: two conv+pool stages to go from 32→16→8.
    """
    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.25)
        # After two pools: 32→16→8 spatial size
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)                # 32×32 → 16×16
        x = F.relu(self.conv2(x))
        x = self.pool(x)                # 16×16 → 8×8
        x = self.dropout(x)
        x = x.view(x.size(0), -1)       # flatten batch×64×8×8
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
