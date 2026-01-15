import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_ch: int, out_ch: int, *, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_ch: int, out_ch: int, *, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=False)


class BasicBlock(nn.Module):
    """
    ResNet-style basic block:
      conv3x3 -> bn -> relu -> conv3x3 -> bn -> add(skip) -> relu
    """
    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = conv3x3(out_ch, out_ch, stride=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        # Projection shortcut if shape changes (channels or spatial via stride)
        self.proj = None
        if stride != 1 or in_ch != out_ch:
            self.proj = nn.Sequential(
                conv1x1(in_ch, out_ch, stride=stride),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.proj is not None:
            identity = self.proj(identity)

        out = out + identity
        return F.relu(out)


class KiloCNN(nn.Module):
    """
    Kilo-sized CNN for CIFAR-10 (small ResNet-like).

    Input:  (B, 3, 32, 32)
    Output: (B, num_classes) logits

    Key improvements vs StandardCNN:
    - residual (skip) connections: makes deeper nets train much better
    - downsampling via stride=2 blocks instead of MaxPool
    - BatchNorm everywhere (conv bias disabled)

    Layout (defaults):
      stem: conv(3->64) + bn + relu                 (32 -> 32)
      stage1: 2x BasicBlock(64 -> 64, stride=1)     (32 -> 32)
      stage2: 2x BasicBlock(64 -> 128, first s=2)   (32 -> 16)
      stage3: 2x BasicBlock(128 -> 256, first s=2)  (16 -> 8)
      global avg pool -> fc

    Parameter count notes:
    - Conv2d (bias=False): out_ch * in_ch * k * k
    - BatchNorm2d(C): 2 * C learnable params (gamma, beta)
    - 1x1 projection conv: out_ch * in_ch
    - fc: (num_classes * C) + num_classes

    Default parameter calculation (c1=64, c2=128, c3=256, blocks_per_stage=2, num_classes=10):

    Helper formulas:
    - conv3x3(in->out): out * in * 9
    - conv1x1(in->out): out * in
    - BN(out): 2*out
    - BasicBlock(in->out, stride=1, in==out):
        conv1 + bn1 + conv2 + bn2
      = (out*in*9) + (2*out) + (out*out*9) + (2*out)
    - BasicBlock(in->out, stride=2 or in!=out) adds projection:
        + proj_conv1x1(in->out) + proj_bn(out)
      = + (out*in) + (2*out)

    Stem (3 -> 64):
    - conv3x3: 64*3*9 = 1,728
    - bn: 2*64 = 128
      stem total = 1,856

    Stage1: 2 blocks of (64 -> 64), stride=1, no projection
    - block(64->64): 64*64*9 + 2*64 + 64*64*9 + 2*64
                  = 36,864 + 128 + 36,864 + 128
                  = 73,984
      stage1 total = 2 * 73,984 = 147,968

    Stage2: first block (64 -> 128) stride=2 with projection, then (128 -> 128)
    - block(64->128, proj):
        conv1: 128*64*9  = 73,728
        bn1:  2*128      = 256
        conv2:128*128*9  = 147,456
        bn2:  2*128      = 256
        proj: 128*64     = 8,192
        proj_bn:2*128    = 256
        total = 73,728 + 256 + 147,456 + 256 + 8,192 + 256 = 230,144
    - block(128->128, no proj):
        128*128*9 + 256 + 128*128*9 + 256
      = 147,456 + 256 + 147,456 + 256 = 295,424
      stage2 total = 230,144 + 295,424 = 525,568

    Stage3: first block (128 -> 256) stride=2 with projection, then (256 -> 256)
    - block(128->256, proj):
        conv1: 256*128*9 = 294,912
        bn1:  2*256      = 512
        conv2:256*256*9  = 589,824
        bn2:  2*256      = 512
        proj: 256*128    = 32,768
        proj_bn:2*256    = 512
        total = 294,912 + 512 + 589,824 + 512 + 32,768 + 512 = 919,040
    - block(256->256, no proj):
        256*256*9 + 512 + 256*256*9 + 512
      = 589,824 + 512 + 589,824 + 512 = 1,180,672
      stage3 total = 919,040 + 1,180,672 = 2,099,712

    FC (256 -> 10):
    - weights: 10*256 = 2,560
    - bias: 10
      fc total = 2,570

    Grand total = stem(1,856) + stage1(147,968) + stage2(525,568) + stage3(2,099,712) + fc(2,570)
                = 2,777,674 parameters
    """
    def __init__(
        self,
        num_classes: int = 10,
        c1: int = 64,
        c2: int = 128,
        c3: int = 256,
        blocks_per_stage: int = 2,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            conv3x3(3, c1, stride=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        self.stage1 = self._make_stage(in_ch=c1, out_ch=c1, blocks=blocks_per_stage, first_stride=1)
        self.stage2 = self._make_stage(in_ch=c1, out_ch=c2, blocks=blocks_per_stage, first_stride=2)
        self.stage3 = self._make_stage(in_ch=c2, out_ch=c3, blocks=blocks_per_stage, first_stride=2)

        self.fc = nn.Linear(c3, num_classes)

    @staticmethod
    def _make_stage(*, in_ch: int, out_ch: int, blocks: int, first_stride: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        layers.append(BasicBlock(in_ch, out_ch, stride=first_stride))
        for _ in range(blocks - 1):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = x.mean(dim=(2, 3))  # global average pool -> (B, c3)
        return self.fc(x)
