import torch
import torch.nn as nn

import EncoderLayer


class BertEncoder(nn.Module):

    def __init__(
        self,
        hidden_size=768,
        num_heads=12,
        num_hidden_layers=12,
        inner_size=3072,
        dropout=0.1
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                inner_size=inner_size,
                dropout=dropout
            )
            for _ in range(num_hidden_layers)
        ])

    def forward(
        self,
        x,
        attention_mask=None
    ):

        for layer in self.layers:

            x = layer(x, attention_mask)   

        return x     