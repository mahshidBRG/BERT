import torch
import torch.nn as nn 

import BertEmbeddings
import BertEncoder 


class BertModeel(nn.Module):

    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_heads=12,
        inner_size=3072,
        max_position_embeddings=512,
        type_vocab_size=2,
        dropout=0.1,
    ):
        
        super().__init__()

        self.embeddings = BertEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            dropout=dropout
        )

        self.encoder = BertEncoder(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_hidden_layers=num_hidden_layers,
            inner_size=inner_size,
            dropout=dropout
        )


    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None
    ):  

        # 1. Embeddings

        x =  self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids
        )


        # 2. Transformer Encoder
        x = self.encode(
            x,
            attention_mask=attention_mask
        )

        return x 

          