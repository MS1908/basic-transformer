import torch
from torch import nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder


class TransformerForGeneration(nn.Module):
    def __init__(self,
                 embed_dim,
                 src_vocab_size,
                 tgt_vocab_size,
                 seq_len,
                 num_blocks=6,
                 expansion_factor=4,
                 heads=8,
                 dropout=0.2):
        super(TransformerForGeneration, self).__init__()

        self.tgt_vocab_size = tgt_vocab_size

        self.encoder = Encoder(seq_len=seq_len,
                               vocab_size=src_vocab_size,
                               embed_dim=embed_dim,
                               num_blocks=num_blocks,
                               expansion_factor=expansion_factor,
                               heads=heads,
                               dropout=dropout)

        self.decoder = Decoder(seq_len=seq_len,
                               target_vocab_size=tgt_vocab_size,
                               embed_dim=embed_dim,
                               num_blocks=num_blocks,
                               expansion_factor=expansion_factor,
                               heads=heads,
                               dropout=dropout)

        self.fc_out = nn.Linear(embed_dim, tgt_vocab_size)

    @staticmethod
    def make_tgt_mask(tgt):
        batch_size, tgt_len = tgt.shape
        tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len)).expand(batch_size, 1, tgt_len, tgt_len)
        return tgt_mask

    def forward(self, source, target):
        tgt_mask = self.make_tgt_mask(target)
        enc_out = self.encoder(source)
        outputs = self.decoder(target, enc_out, tgt_mask)
        outputs = F.softmax(self.fc_out(outputs), dim=-1)
        return outputs


class TransformerForClassification(nn.Module):
    def __init__(self,
                 embed_dim,
                 src_vocab_size,
                 num_classes,
                 seq_len,
                 num_blocks=6,
                 expansion_factor=4,
                 heads=8,
                 dropout=0.2):
        super(TransformerForClassification, self).__init__()

        self.encoder = Encoder(seq_len=seq_len,
                               vocab_size=src_vocab_size,
                               embed_dim=embed_dim,
                               num_blocks=num_blocks,
                               expansion_factor=expansion_factor,
                               heads=heads,
                               dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_dim * seq_len, num_classes)

    def forward(self, inputs):
        enc_out = self.encoder(inputs)
        enc_out = torch.flatten(enc_out, start_dim=1)
        outputs = self.dropout(self.fc_out(enc_out))
        return outputs
