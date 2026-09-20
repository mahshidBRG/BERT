import torch
import torch.nn as nn


class AddNorm(nn.Module):

    def __init__(
        self,
        hidden_size,
        dropout = 0.1
    ):
        super().__init__()

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
      

    def forward(
        self,
        x,
        sublayer_out
    ):
        
        return self.layer_norm(x + self.dropout(sublayer_out))