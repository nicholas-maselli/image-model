import torch
import torch.nn as nn

from .vit_blocks import _make_gn, EncoderBlockLS

class StandardViT(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        image_size: int = 32,
        patch_size: int = 4,
        # Bigger defaults than mini_vit, but still reasonable for ~50k steps on CIFAR-10
        dim: int = 384,
        depth: int = 10,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop: float = 0.1,
        attn_drop: float = 0.1,
        drop_path_rate: float = 0.15,
        pool: str = "mean",  # "cls" | "mean"
        # Robust across small batches (BN can get noisy if you run batch_size=8 etc.)
        stem_norm: str = "group",  # "batch" | "group"
        # LayerScale often helps when you push depth up
        layer_scale_init: float = 1e-5,  # set 0.0 to disable
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.image_size = image_size
        self.patch_size = patch_size
        grid = image_size // patch_size
        num_patches = grid * grid
        self.pool = pool
        self.use_cls_token = (pool == "cls")

        def N(ch: int) -> nn.Module:
            if stem_norm == "batch":
                return nn.BatchNorm2d(ch)
            if stem_norm == "group":
                return _make_gn(ch)
            raise ValueError("stem_norm must be 'batch' or 'group'")

        # Conv stem: (B,3,32,32) -> (B,dim,8,8)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            N(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
            N(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, kernel_size=3, stride=2, padding=1, bias=False),
            N(dim),
            nn.GELU(),
        )
        self.patch_norm = nn.LayerNorm(dim)

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, dim))
        else:
            self.cls_token = None
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))

        self.pos_drop = nn.Dropout(drop)

        dpr = torch.linspace(0, float(drop_path_rate), steps=depth).tolist()
        self.blocks = nn.ModuleList(
            [
                EncoderBlockLS(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=float(dpr[i]),
                    layer_scale_init=float(layer_scale_init),
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
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
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # (B, T, D)
        x = self.patch_norm(x)

        if self.use_cls_token:
            B = x.size(0)
            cls = self.cls_token.expand(B, -1, -1)  # type: ignore[union-attr]
            x = torch.cat([cls, x], dim=1)

        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        if self.pool == "mean":
            feat = x[:, 1:].mean(dim=1) if self.use_cls_token else x.mean(dim=1)
        else:
            feat = x[:, 0]
        return self.head(feat)
