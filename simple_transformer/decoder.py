import copy
from torch import nn

from .embed import PositionalEncoding
from .block import DecoderBlock


class Decoder(nn.Module):
    def __init__(self,
                 seq_len,
                 target_vocab_size,
                 embed_dim=512,
                 num_blocks=6,
                 expansion_factor=4,
                 heads=8,
                 dropout=0.2):
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(target_vocab_size, embed_dim)
        self.positional_encoder = PositionalEncoding(embed_dim, seq_len)

        self.blocks = nn.ModuleList(
            [copy.deepcopy(DecoderBlock(embed_dim, heads, expansion_factor, dropout))
             for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, mask):
        x = self.dropout(self.positional_encoder(self.embedding(x)))

        for block in self.blocks:
            x = block(encoder_output, x, encoder_output, mask)

        return x
