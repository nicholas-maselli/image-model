import torch
import torch.nn as nn

from .vit_blocks import EncoderBlock

class NanoViT(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        image_size: int = 32,
        patch_size: int = 4,
        dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        pool: str = "cls",  # "cls" | "mean"
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.image_size = image_size
        self.patch_size = patch_size
        grid = image_size // patch_size
        num_patches = grid * grid
        self.pool = pool

        # Patchify + linear embed in one step (conv with stride=patch_size)
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True
        )
        self.patch_norm = nn.LayerNorm(dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, dim))
        self.pos_drop = nn.Dropout(drop)

        dpr = torch.linspace(0, float(drop_path_rate), steps=depth).tolist()
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    dim=dim, 
                    num_heads=num_heads, 
                    mlp_ratio=mlp_ratio, 
                    attn_drop=attn_drop, 
                    drop=drop,
                    drop_path=float(dpr[i]),
                ) 
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch embed: (B, 3, H, W) -> (B, D, H/P, W/P)
        x = self.patch_embed(x)

        # Flatten to tokens: (B, D, H/P, W/P) -> (B, T, D)
        x = x.flatten(2).transpose(1, 2)
        x = self.patch_norm(x)

        B, T, D = x.shape
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        if self.pool == "mean":
            feat = x[:, 1:].mean(dim=1)
        else:
            feat = x[:, 0]  # CLS token
        return self.head(feat)