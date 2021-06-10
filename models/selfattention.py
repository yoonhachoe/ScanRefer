import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, feats):
        attention_score = self.fc1(feats) # B, T, H
        attention_weight = nn.functional.softmax(attention_score, dim=1)
        attention_value = torch.bmm(attention_weight.permute(0, 2, 1).contiguous(), feats) # B, H, H
        attention_value = self.fc2(attention_value, 1) # B, H

        return attention_value
