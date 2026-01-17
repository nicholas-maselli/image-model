import torch
import torch.nn as nn
import torch.nn.functional as F


class MilliCNN(nn.Module):
    """
    Small CNN for CIFAR-10.

    Input:  (B, 3, 32, 32)
    Output: (B, num_classes) logits

    Design goals:
    - small parameter count
    - fast inference
    - simple/robust (no fancy blocks)

    Parameter count (learnable parameters):
    - conv1: (c1 * (3 * 3 * 3)) + c1
    - conv2: (c2 * (c1 * 3 * 3)) + c2
    - conv3: (c3 * (c2 * 3 * 3)) + c3
    - fc:    (num_classes * c3) + num_classes
      Total: conv1 + conv2 + conv3 + fc

    With the default args (c1=16, c2=32, c3=64, num_classes=10):
    - conv1: 16*(3*3*3) + 16 = 448
    - conv2: 32*(16*3*3) + 32 = 4,640
    - conv3: 64*(32*3*3) + 64 = 18,496
    - fc:    10*64 + 10 = 650
      Total = 448 + 4,640 + 18,496 + 650 = 24,234 parameters
    """

    def __init__(self, num_classes: int = 10, c1: int = 16, c2: int = 32, c3: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, c1, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2)  # 32->16->8
        self.fc = nn.Linear(c3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.mean(dim=(2, 3))  # global average pool -> (B, c3)
        return self.fc(x)
