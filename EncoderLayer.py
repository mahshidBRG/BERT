import torch
import torch.nn as nn

import MultiHeadSelfAttention
import AddNorm
import FeedForward


class EncoderLayer(nn.Module):

    def __init__(
        self,
        hidden_size=768,
        num_heads=12,
        inner_size=3072,
        dropout=0.1
    ):
        super().__init__()

        self.attention = MultiHeadSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        )

        self.attention_dropout = nn.Dropout(dropout)

        self.add_norm1 = AddNorm(hidden_size, dropout)

        self.feed_forward = FeedForward(
            hidden_size=hidden_size,
            inner_size=inner_size,
            dropout=dropout
        )

        self.add_norm2 = AddNorm(hidden_size, dropout)

    
    def forward(
        self,
        x,
        attention_mask
    ):
        
        attention_output = self.attention(
            x,
            attention_mask=attention_mask
        )

        attention_output = self.attention_dropout(attention_output)

        x = self.add_norm1(x, attention_output)

        feed_forward_output = self.feed_forward(x)

        x = self.add_norm2(x, feed_forward_output)

        return x


