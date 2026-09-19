import torch 
import torch.nn as nn


#                    ┌─ Token Embedding ─────┐
#input_ids ──────────┤                       │
#                    │                       ↓
#position_ids ───────┤                  Addition
#                    │                       ↓
#token_type_ids ─────┘                  LayerNorm
#                                            ↓
#                                         Dropout
#                                            ↓
#                                      BERT Embeddings


class BertEmbeddings(nn.Module):

    def __init__(
            self,
            vocab_size,
            hidden_size,
            max_position_embeddings=512,
            type_vocab_size=2,
            dropout=0.1):
        
        super().__init__()

        # 1. Token embeddings
        self.word_embeddings = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=hidden_size
        )

        # 2. Position embeddings
        self.position_embeddings = nn.Embedding(
            max_position_embeddings,
            hidden_size
        )

        # 3. Segment / token type embeddings
        self.token_type_embeddings = nn.Embedding(
            type_vocab_size,
            hidden_size
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(dropout)

        def forward(
                self,
                input_ids,
                token_type_ids=None
        ):
            
            batch_size, seq_length = input_ids.shape


            # If token_type_ids are not provided,
            # assume that all tokens belong to segment 0.
            if token_type_ids is None:
                token_type_ids = torch.zeros_like(input_ids)
            
            # Create position IDs : [0, 1, 2, ..., seq_lengh-1]
            position_ids  = torch.arange(
                seq_length,
                device=input_ids.device
            )

            # Add batch dimension
            position_ids = position_ids.unsqueeze(0)

            # Token embeddings
            word_embeddings = self.word_embeddings(input_ids)

            # Posotion embeddings
            position_embeddings = self.position_embeddings(input_ids)

            # Segment embeddingd 
            token_type_embeddings = self.token_type_embedding(token_type_ids)

            # Combine the embeddings
            embedding = (
                word_embeddings 
                + position_embeddings
                + token_type_embeddings
            )

            # LayerNorm
            embedding = self.layer_norm(embedding)

            # Dropout
            embedding = self.dropout(embedding)

            return embedding