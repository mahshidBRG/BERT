import torch
import torch.nn as nn


class MLMHead(nn.Module):

    def __init__(
        self,
        hidden_size,
        vocab_size,
        embedding_weights
    ):
        super().__init__()

        self.dense = nn.Linear(
            hidden_size,
            hidden_size
        )

        self.activation = nn.GELU()

        self.layer_norm = nn.LayerNorm(hidden_size)

        self.decoder = nn.Linear(
            hidden_size,
            vocab_size,
            bias=False
        )

        self.bias = nn.Parameter(torch.zeros(vocab_size))

        self.decoder.weight = embedding_weights


    def forward(self, hidden_states):

        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.layer_norm(x)
        x = self.decoder(x) + self.bias 

        return x  