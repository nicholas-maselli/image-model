import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.1):
        super().__init__()
        self.activation = nn.GELU()
        
        self.mlp_expand = nn.Linear(dim, hidden_dim)
        self.mlp_contract = nn.Linear(hidden_dim, dim)

        self.dropout1 = nn.Dropout(drop)
        self.dropout2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp_expand(x)
        x = self.activation(x)
        x = self.dropout1(x)

        x = self.mlp_contract(x)
        x = self.dropout2(x)

        return x

class EncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        self.mlp = MLP(dim=dim, hidden_dim=int(dim * mlp_ratio), drop=proj_drop)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.drop_path = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x_norm1 = self.norm1(x)
        attn_out, _ = self.attn(x_norm1, x_norm1, x_norm1, need_weights=False)
        x = x + self.drop_path(attn_out)


        x_norm2 = self.norm2(x)
        mlp_out = self.mlp(x_norm2)
        x = x + self.drop_path(mlp_out)
        return x

class MicroViT(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        image_size: int = 32,
        patch_size: int = 2,
        dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.image_size = image_size
        self.patch_size = patch_size
        grid = image_size // patch_size
        num_patches = grid * grid

        # Patchify + linear embed in one step (conv with stride=patch_size)
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, dim))
        self.pos_drop = nn.Dropout(drop)

        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    dim=dim, 
                    num_heads=num_heads, 
                    mlp_ratio=mlp_ratio, 
                    attn_drop=attn_drop, 
                    proj_drop=drop
                ) 
                for _ in range(depth)
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

        B, T, D = x.shape
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_out = x[:, 0] # CLS token
        return self.head(cls_out)