import torch
import torch.nn as nn
import torch.nn.functional as F


class MicroCNN(nn.Module):
    """
    Very small CNN for CIFAR-10.

    Input:  (B, 3, 32, 32)
    Output: (B, num_classes) logits

    Design goals:
    - tiny parameter count
    - fast inference
    - simple/robust (no fancy blocks)

    Parameter count (learnable parameters):
    - conv1: (c1 * (3 * 3 * 3)) + c1
    - conv2: (c2 * (c1 * 3 * 3)) + c2
    - fc:    (num_classes * c2) + num_classes
      Total: conv1 + conv2 + fc

    With the default args (c1=16, c2=32, num_classes=10):
    - conv1: 16*(3*3*3) + 16 = 448
    - conv2: 32*(16*3*3) + 32 = 4,640
    - fc:    10*32 + 10 = 330
      Total = 448 + 4,640 + 330 = 5,418 parameters
    """

    def __init__(self, num_classes: int = 10, c1: int = 16, c2: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(3, c1, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2)  # 32->16->8
        self.fc = nn.Linear(c2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.mean(dim=(2, 3))  # global average pool -> (B, c2)
        return self.fc(x)
