import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .attention import MultiHeadAttention
from .FFN import FeedForward
from .position import PositionalEncoding
class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, X, mask=None):
        # self-attention
        attn_output, attn_weight = self.self_attention(X, mask=mask)
        # residual connection, normalization, dropout
        X = self.norm1(X+ self.dropout1(attn_output))
        # feed forward network
        ffn_output = self.ffn(X)
        # residual connection, normalization, dropout
        X = self.norm2(X + self.dropout2(ffn_output))
        return X, attn_weight

class Encoder(nn.Module):
    def __init__(self, vocab_size,
        d_model,
        num_heads,
        d_ff,
        num_layers,
        max_len=5000,
        dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, d_ff, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, mask=None):
        X = self.embedding(X) * math.sqrt(self.d_model)
        X = self.positional_encoding(X)
        X = self.dropout(X)
        for layer in self.layers:
            X = layer(X, mask)[0]
        return X
