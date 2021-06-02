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
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, feats):
        attention = self.fc1(feats) # B, T, attention_size
        attention = torch.bmm(feats.permute(0, 2, 1).contiguous(), attention) # B, H, H
        attention = torch.softmax(attention, dim=1)
        #attention = torch.bmm(feats, attention) # B, T, H
        attention_value = self.fc2(attention) #B, H

        return attention_value
