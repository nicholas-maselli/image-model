import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
  """
    ResNet-style BasicBlock (CIFAR-friendly).

    Shape behavior:
      - If stride=1 and in_ch==out_ch: shortcut is identity.
      - Else: shortcut is a 1x1 conv (+ BN) with the same stride to match shape.

    Forward:
      out = ReLU( BN(conv3x3(stride)) (x) )
      out =      BN(conv3x3) (out)
      out = ReLU( out + shortcut(x) )
  """
  def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
    super().__init__()
    self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride = stride, padding = 1, bias = False)
    self.batchnorm1 = nn.BatchNorm2d(out_ch)
    self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride = 1, padding = 1, bias = False)
    self.batchnorm2 = nn.BatchNorm2d(out_ch)
    
    if stride != 1 or in_ch != out_ch:
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size =1, stride = stride, padding = 0, bias = False),
        nn.BatchNorm2d(out_ch),
      )
    else :
      self.shortcut = nn.Identity()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = self.batchnorm1(x)
    out = F.relu(out, inplace=True)
    out = self.conv1(out)

    out = self.batchnorm2(out)
    out = F.relu(out, inplace=True)
    out = self.conv2(out)

    out = out + self.shortcut(x)
    return out

class TestCandidate0(nn.Module):
    def __init__(self, num_classes: int = 10, c1: int = 64, c2: int = 128, c3: int = 256):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
          BasicBlock(c1,c1,1), 
          BasicBlock(c1,c1,1), 
          BasicBlock(c1,c1,1)
        )

        self.stage2 = nn.Sequential(
          BasicBlock(c1,c2,2), 
          BasicBlock(c2,c2,1), 
          BasicBlock(c2,c2,1)
        )

        self.stage3 = nn.Sequential(
          BasicBlock(c2,c3,2), 
          BasicBlock(c3,c3,1), 
          BasicBlock(c3,c3,1)
        )

        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(c3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = x.mean(dim=(2, 3))
        x = self.dropout(x)
        return self.fc(x)