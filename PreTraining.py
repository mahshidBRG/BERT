import torch
import torch.nn as nn

import BertModel
import MLMHead
import NSPHead


class PreTraining(nn.Module):

    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_heads=12,
        inner_size=3072,
        max_position_embeddings=512,
        type_vocab_size=2,
        dropout=0.1
    ):
        super().__init__()

        # BERT encoder
        self.bert = BertModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_heads=num_heads,
            inner_size=inner_size,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            dropout=dropout,
        )

        # MLM head
        self.mlm_head = MLMHead(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            embedding_weights=
                self.bert.embeddings.word_embeddings.weight
        )

        # NSP head
        self.nsp_head = NSPHead(hidden_size=hidden_size)


        def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None
        ):
            
            # BERT encoder 
            hidden_states = self.bert(
               input_ids=input_ids,
               attention_mask=attention_mask,
               token_type_ids=token_type_ids
            )

            # MLM predictions
            mlm_logits = self.mlm_head(hidden_states)

            # NSP predictions
            nsp_logits = self.nsp_head(hidden_states)

            return {
                "hidden_states": hidden_states,
                "mlm_logits" : mlm_logits,
                "nsp_logits": nsp_logits
            }
        