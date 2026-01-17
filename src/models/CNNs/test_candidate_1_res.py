import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class TestCandidate(nn.Module):
    """
    Standard-sized CNN for CIFAR-10.

    Layout (example):
      [64]  conv-bn-relu -> conv-bn-relu -> pool   (32 -> 16)
      [128] conv-bn-relu -> conv-bn-relu -> pool   (16 -> 8)
      [256] conv-bn-relu -> conv-bn-relu           (8 -> 8)
      global avg pool -> fc

      Parameter count (learnable parameters):
    For each conv_bn_relu(in_ch -> out_ch) block:
    - conv (bias=False): out_ch * in_ch * 3 * 3
    - batchnorm: 2 * out_ch   (gamma + beta; running mean/var are buffers, not params)
      block_total = (out_ch * in_ch * 9) + (2 * out_ch)

    Stages:
    - stage1: block(3->c1) + block(c1->c1)
    - stage2: block(c1->c2) + block(c2->c2)
    - stage3: block(c2->c3) + block(c3->c3)
    - fc (Linear): (num_classes * c3) + num_classes

    With the default args (c1=64, c2=128, c3=256, num_classes=10):
    - block(3->64):   64*3*9   + 2*64   = 1,856
    - block(64->64):  64*64*9  + 2*64   = 36,992
      stage1 total = 38,848

    - block(64->128): 128*64*9 + 2*128  = 73,984
    - block(128->128):128*128*9+ 2*128  = 147,712
      stage2 total = 221,696

    - block(128->256):256*128*9+ 2*256  = 295,424
    - block(256->256):256*256*9+ 2*256  = 590,336
      stage3 total = 885,760

    - fc: 10*256 + 10 = 2,570

    Total = 38,848 + 221,696 + 885,760 + 2,570 = 1,148,874 parameters
    """

    def __init__(self, num_classes: int = 10, c1: int = 64, c2: int = 128, c3: int = 256):
        super().__init__()
        self.stage1 = nn.Sequential(conv_bn_relu(3, c1), conv_bn_relu(c1, c1))
        self.stage2 = nn.Sequential(conv_bn_relu(c1, c2), conv_bn_relu(c2, c2))
        self.stage3 = nn.Sequential(conv_bn_relu(c2, c3), conv_bn_relu(c3, c3))

        # Projection shortcuts if shape changes (1x1 convs)
        self.proj1 = nn.Conv2d(3, c1, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj2 = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj3 = nn.Conv2d(c2, c3, kernel_size=1, stride=1, padding=0, bias=False)

        self.pool = nn.MaxPool2d(2)  # downsample by 2
        self.fc = nn.Linear(c3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_proj1 = self.proj1(x) # 3 -> 32
        x = self.stage1(x) + x_proj1 # 32 -> 32
        x = self.pool(x)  # 32->16

        x_proj2 = self.proj2(x) # 32 -> 16
        x = self.stage2(x) + x_proj2 # 16 -> 16
        x = self.pool(x)  # 16->8

        x_proj3 = self.proj3(x) # 16 -> 8
        x = self.stage3(x) + x_proj3 # 8 -> 8
        x = x.mean(dim=(2, 3))         # GAP -> (B, c3)
        return self.fc(x)
