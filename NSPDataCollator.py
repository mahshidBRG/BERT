import torch
import random


class NSPDataCollator:

    def __init__(
        self,
        tokenizer,
        max_length=32
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length


    def create_nsp_example(self, sentences):

        i = random.randint(0, len(sentences)-2)

        sentence_a = sentences[i]

        # 50% IsNext
        if random.random() < 0.5:

            sentence_b = sentences[i+1]
            label = 1

        # 50% NotNet
        else:

            possible_indices = list(range(len(sentences)))
            possible_indices.remove(i)
            possible_indices.remove(i+1)

            j = random.choice(possible_indices)
            sentence_b =sentences[j]
            label = 0

        return sentence_a, sentence_b, label    


    def __call__(self, sentences):

        sentence_a, sentence_b, nsp_label = self.create_nsp_example(sentences)

        encoded = self.tokenizer(
            sentence_a,
            sentence_b,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "token_type_ids": encoded["token_type_ids"],
            "nsp_labels": torch.tensor([nsp_label])
        }     

