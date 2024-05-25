import math
import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, heads=8):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.head = int(self.embed_dim / self.heads)

        self.query = nn.Linear(self.head, self.head, bias=False)
        self.value = nn.Linear(self.head, self.head, bias=False)
        self.key = nn.Linear(self.head, self.head, bias=False)

        self.fc_out = nn.Linear(self.head * self.heads, embed_dim)

    def forward(self, key, query, value, mask=None):
        batch_size = key.size(0)
        k_len, q_len, v_len = key.size(1), query.size(1), value.size(1)

        key = key.reshape(batch_size, k_len, self.heads, self.head)
        query = query.reshape(batch_size, q_len, self.heads, self.head)
        value = value.reshape(batch_size, v_len, self.heads, self.head)

        key = self.key(key)
        query = self.query(query)
        value = self.value(value)

        product = torch.einsum("bqhd, bkhd->bhqk", [query, key])

        if mask is not None:
            product = product.masked_fill(mask == 0, float("-1e20"))

        product = product / math.sqrt(self.head)
        scores = F.softmax(product, dim=-1)

        output = torch.einsum("nhql, nlhd->nqhd",
                              [scores, value]).reshape(batch_size, q_len, self.heads * self.head)

        output = self.fc_out(output)

        return output
