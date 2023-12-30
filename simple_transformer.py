import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class SimpleTransformer(nn.Module):
    def __init__(self, n_token, d_model, n_head, d_hid, n_layers, dropout=0.5):
        super(SimpleTransformer, self).__init__()

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_head, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.embedding = nn.Embedding(n_token, d_model)
        self.d_model = d_model
        self.fc = nn.Linear(d_model, n_token)

    def forward(self, src, src_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src))
        output = self.transformer_encoder(src, src_mask)
        output = self.fc(output)
        return output
