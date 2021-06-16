import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(hidden_size, hidden_size)

    def forward(self, feats):
        attention = self.fc1(feats) # B, T, H
        attention_score = torch.bmm(attention, feats.permute(0, 2, 1).contiguous()) # B, T, T
        attention_weight = nn.functional.softmax(attention_score, dim=2)
        attention_value = torch.bmm(attention_weight, feats) # B, T, H
        attention_value = torch.sum(attention_value, 1) # B, H

        return attention_value
