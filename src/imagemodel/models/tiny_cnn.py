import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyCNN(nn.Module):
    """
    Simple CNN for CIFAR-10 (32x32 RGB).

    Input:  (B, 3, 32, 32)
    Output: (B, num_classes) logits
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)     # -> (B, 64, 32, 32)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)   # -> (B, 128, 16, 16)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # -> (B, 256, 8, 8)

        self.pool = nn.MaxPool2d(2)  # halves H/W each time

        self.dropout = nn.Dropout(p=0.2)

        # After 3 pools: 32 -> 16 -> 8 -> 4, so feature map is (B, 256, 4, 4)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))  # (B, 64, 16, 16)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 128, 8, 8)
        x = self.pool(F.relu(self.conv3(x)))  # (B, 256, 4, 4)

        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # logits
        return x
