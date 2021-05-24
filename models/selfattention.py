import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.scale = 1.0/math.sqrt(hidden_size)

    def forward(self, hidden):
        query = hidden # B, T, H
        key = hidden.permute(1, 0, 2).contiguous() # B, H, T
        value = hidden # B, T, H
        score = torch.bmm(query, key) # B, T, T
        weight = F.softmax(score.mul_(self.scale), dim=2) # scale, normalization
        attention_value = torch.bmm(weight, value) # B, T, H
        attention_value = torch.sum(attention_value, 1) # B, H

        return attention_value
