import copy
from torch import nn

from .embed import Embedding, PositionalEncoding
from .block import EncoderBlock


class Encoder(nn.Module):
    def __init__(self,
                 seq_len,
                 vocab_size,
                 embed_dim=512,
                 num_blocks=6,
                 expansion_factor=4,
                 heads=8,
                 dropout=0.2):
        super(Encoder, self).__init__()

        self.embedding = Embedding(vocab_size, embed_dim)
        self.positional_encoder = PositionalEncoding(embed_dim, seq_len)

        self.blocks = nn.ModuleList(
            [copy.deepcopy(EncoderBlock(embed_dim, heads, expansion_factor, dropout))
             for _ in range(num_blocks)]
        )

    def forward(self, x):
        out = self.positional_encoder(self.embedding(x))
        for block in self.blocks:
            out = block(out, out, out)
        return out
