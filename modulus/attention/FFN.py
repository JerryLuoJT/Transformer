import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
    # X -> linear1 -> gelu -> linear2
    def forward(self,X):
        X = self.linear1(X)
        X = F.gelu(X)
        X = self.linear2(X)
        return X
