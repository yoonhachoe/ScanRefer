import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        #self.attention_size = attention_size
        self.fc1 = nn.Linear(hidden_size, hidden_size)

    def forward(self, feats):
        attention = self.fc1(feats) # B, T, H
        attention = torch.bmm(attention, feats.permute(0, 2, 1).contiguous()) # B, T, T
        attention = torch.softmax(attention, dim=1)
        attention_value = torch.bmm(feats, attention) # B, T, H
        attention_value = torch.sum(attention_value, 1) # B, T

        return attention_value
