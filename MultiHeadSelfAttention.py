import torch
import torch.nn as nn
import math


class MultiHeadSelfAttention(nn.Module):

    def __init__(
            self,
            hidden_size, # d model(embeddding dim)
            num_heads,
            dropout=0.1
    ):
        super().__init__()

        assert hidden_size % num_heads == 0, \
            "hidden-size must be divisble by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size / num_heads

        # Query, Key, Value projections
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        # Output projection(W_o)
        self.output = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)


    def split_heads(self, x):

        batch_size, seq_length, _ = x.shape

        x = x.view(
            batch_size,
            seq_length,
            self.num_heads,
            self.head_dim
        )

        # (B, S, H, D) → (B, H, S, D)
        x = x.transpose(1, 2)

        return x


    def forward(
        self,
        x,
        attention_mask=None
    ):
        
        # 1. Create Q, K, V
        Q =  self.query(x)
        K = self.keys(x)
        V = self.value(x)

        # 2. Split into heads
        Q = self.split(Q)
        K = self.split(K)
        V = self.split(V)

        # 3. Attention Scores
        scores = torch.matmul(Q , K.transpose(-2,-1))

        # 4. Scale
        scores = scores / math.sqrt(self.head_dim)

        # 5. Mask (padding)
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]

            scores = scores.masked_fill(   # Wherever the attention mask is 0 (a padding token),
                attention_mask == 0,            # replace its attention score with an extremely negative number.
                torch.finfo(scores.dtype).min                              
            )

        # 6. Softmax
        attention_weights = torch.softmax(
            scores,
            dim=-1
        )  

        # 7. Weightrd sum of V
        context = torch.matmul(attention_weights, V)

        # 8. Combine heads
        context = context.transpose(1,2) 
        context = context.contiguous().view(
            context.size(0),
            context.size(1),
            self.hidden_size
        )

        # 9. Output projection
        output = self.output(context)

        return output 




