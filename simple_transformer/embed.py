import math
import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(Embedding, self).__init__()
        self.embed_dim = embed_dim
        self.embed_layer = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        output = self.embed_layer(x) * math.sqrt(self.embed_dim)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=1000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)

        positional_encoding = torch.zeros(max_seq_len, self.embed_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim)
        )
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        pe = positional_encoding.unsqueeze(0)

        self.register_buffer('pe', pe)

    def pe_sin(self, position, i):
        return math.sin(position / (10000 ** (2 * i) / self.embed_dim))

    def pe_cos(self, position, i):
        return math.cos(position / (10000 ** (2 * i) / self.embed_dim))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
