import torch
import torch.nn as nn

# A part of the data pipeline 
class MLMDataCollator:

    def __init__(
        self,
        tokenizer,
        mlm_probability
    ):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability


    def __call__(self, input_ids):

        labels = input_ids.clone()

        # 1. Decide which tokens will be prediction targets  
        probability_matrix = torch.full(
            labels.shape,
            self.mlm_probability
        )

        # Identify special tokens such as: [CLS], [SEP], [PAD], etc.
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(
                seq.tolist(),
                already_has_special_tokens=True
            )
            for seq in labels
        ]

        special_tokens_mask = torch.tensor(
            special_tokens_mask,
            dtype=torch.bool
        )

        # Special tokens should never be selected for MLM
        probability_matrix.masked_fill_(
            special_tokens_mask,
            0.0
        )

        # Randomly select tokens according to the probability
        masked_indices = torch.bernoulli(
            probability_matrix
        ).bool()

        # 2. Create labels
        labels[~masked_indices] = -100

        # 3. 80% of selected tokens => [MASK]
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool()
            & masked_indices
        )

        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 4. 10% of slected tokens => random token
        indices_random = torch.bernoulli(
            torch.full(labels.shape, 0.1).bool()
            & masked_indices    
            & ~indices_replaced
        )

        random_words = torch.randint(
            low=0,
            high=self.tokenizer.vocab_size,
            size=labels.shape,
            dtype=torch.long
        )

        input_ids[indices_random] = random_words[indices_random]

        # and remaining slected tokens (10%) => unchanged

        return input_ids, labels


