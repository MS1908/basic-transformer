from torch import nn

from .attention import MultiHeadAttention


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim=512, heads=8, expansion_factor=4, dropout=0.2):
        super(EncoderBlock, self).__init__()

        self.attention = MultiHeadAttention(embed_dim=embed_dim, heads=heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor * embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * expansion_factor, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, key, query, value, mask=None):
        attn_out = self.attention(key, query, value, mask)
        attn_out = attn_out + value

        attn_norm = self.dropout(self.norm(attn_out))

        fc_out = self.feedforward(attn_norm)
        fc_out = fc_out + attn_norm

        outputs = self.dropout(self.norm(fc_out))

        return outputs


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim=512, heads=8, expansion_factor=4, dropout=0.2):
        super(DecoderBlock, self).__init__()

        self.attention = MultiHeadAttention(embed_dim, heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.encoder_block = EncoderBlock(embed_dim, heads, expansion_factor, dropout)

    def forward(self, key, query, x, mask):
        attention = self.attention(x, x, x, mask)
        value = self.dropout(self.norm(attention + x))
        outputs = self.encoder_block(key, query, value)
        return outputs
