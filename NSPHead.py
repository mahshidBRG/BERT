import torch
import torch.nn as nn


class NSPHead(nn.Module):

    def __init__(self, hidden_size=768):
        super().__init__()

        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, hidden_states):

        # Take the [CLS] representation
        cls_output = hidden_states[:, 0, :]
        logits = self.classifier(cls_output)

        return logits   
