import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .attention import MultiHeadAttention
from .FFN import FeedForward
from .position import PositionalEncoding

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, X, encoder_output, self_mask=None, cross_mask=None):
        # self-attention
        # 解码器自注意力必须始终带因果掩码：位置 i 只能看到 ≤ i 的 key，
        # 否则训练时模型会“偷看”未来的目标 token，推理时也无法自回归生成。
        L = X.size(1)
        causal_mask = torch.tril(
            torch.ones(L, L, device=X.device, dtype=torch.bool)
        )
        if self_mask is not None:
            self_mask = self_mask & causal_mask
        else:
            self_mask = causal_mask
        self_attn_output, self_attn_weight = self.self_attention(X, mask=self_mask)
        # residual connection, normalization, dropout
        X = self.norm1(X + self.dropout1(self_attn_output))
        # cross-attention
        cross_attn_output, cross_attn_weight = self.cross_attention(X, context=encoder_output, mask=cross_mask)
        # residual connection, normalization, dropout
        X = self.norm2(X + self.dropout2(cross_attn_output))
        # feed forward network
        ffn_output = self.ffn(X)
        # residual connection, normalization, dropout
        X = self.norm3(X + self.dropout3(ffn_output))
        return X

class Decoder(nn.Module):
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
            DecoderLayer(d_model, d_ff, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, encoder_output, self_mask=None, cross_mask=None):
        X = self.embedding(X) * math.sqrt(self.d_model)
        X = self.positional_encoding(X)
        X = self.dropout(X)
        for layer in self.layers:
            X = layer(X, encoder_output, self_mask, cross_mask) #encoder_output在层内保持不变，不会迭代
        return X
