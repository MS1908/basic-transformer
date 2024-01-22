import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, vocab_size=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class SimpleTransformer(nn.Module):
    def __init__(self, n_classes, vocab_size, d_model, n_head, d_hid, n_layers, dropout=0.5):
        super(SimpleTransformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            vocab_size=vocab_size,
        )
        
        enc_layers = nn.TransformerEncoderLayer(d_model, n_head, d_hid, dropout)
        self.encoder = nn.TransformerEncoder(enc_layers, n_layers)
        self.fc = nn.Linear(d_model, n_classes)

        self.d_model = d_model

    def forward(self, inputs):
        x = self.embedding(inputs) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        output = self.fc(x.mean(dim=1))
        return output
