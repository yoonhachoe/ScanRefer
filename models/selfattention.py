import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, feats, lengths):
        attention = self.fc1(feats) # B, T, H
        # mask attention scores
        B, S, _ = attention.size()  # S = len longest hiddenstate
        idx = lengths.new_tensor(torch.arange(0, S).unsqueeze(0).repeat(B, 1)).long()  # clone lengths tensor
        lengths = lengths.unsqueeze(1).repeat(1, S)

        mask = (idx >= lengths)
        mask = mask.unsqueeze(2).repeat(1, 1, self.hidden_size)
        attention.masked_fill_(mask, float('-1e30'))  # attn mask

        # softmax
        weights = nn.functional.softmax(attention, dim=1)  # B, T, H
        weights = weights.transpose(1, 2)  # B, H, T
        value = torch.bmm(weights, feats) # B, H, H
        value = value.permute(0, 2, 1).contiguous()
        value = self.fc2(value)  # B, H, 1
        value = torch.squeeze(value)
        return value
