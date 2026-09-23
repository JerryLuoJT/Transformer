import torch
from .attention import MultiHeadAttention, create_causal_mask
torch.manual_seed(42)
torch.manual_seed(42)

X = torch.randn(2, 4, 8)

attention = MultiHeadAttention(
    d_model=8,
    num_heads=2
)
mask = create_causal_mask(
    seq_len=4,
    device="cpu"
)

output, weights = attention.forward(
    X,
    mask=mask
)

print(output.shape)
print(weights.shape)
print(weights[0, 0])
