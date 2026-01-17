import math
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, dim: int, dim_hidden: int, drop: float = 0.1):
        self.activation = nn.GELU()
        
        self.mlp_expand = nn.Linear(dim, dim_hidden)
        self.mlp_contract = nn.Linear(dim_hidden, dim)

        self.dropout1 = nn.Dropout(drop)
        self.dropout2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.mlp_expand(x)
        x = self.activation(x)
        x = self.dropout1(x)

        x = self.mlp_contract(x)
        x = self.dropout2(x)

        return x