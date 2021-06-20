import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, feats):
        score = self.fc(feats) # B, T, H
        score = torch.bmm(score, feats.permute(0, 2, 1).contiguous())  # B, T, T
        _, _, T = score.size()
        mask = feats[:,:,0].repeat(T, 1, 1).permute(1, 2, 0).contiguous() # B, T, T
        score = score.masked_fill(mask==0, -1e9)
        weight = nn.functional.softmax(score, dim=-1) # B, T, T
        value = torch.bmm(weight, feats) # B, T, H
        value = torch.sum(value, 1) # B, H, 1

        return value
