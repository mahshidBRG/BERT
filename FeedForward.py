import torch
import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(
        self,
        hidden_size=768,
        inner_size=3072,
        dropout = 0.1
    ):
        super().__init__()

        self.fc1 = nn.Linear(hidden_size, inner_size)

        self.activation = nn.GELU()

        self.fc2 = nn.Linear(inner_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)  
        x = self.dropout(x)

        return x 