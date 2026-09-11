import torch 
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        # [max_len, 1]
        # 生成[0,1,2,...,max_len-1]的列向量再变成二维，相当于[[0],[1],[2],...,[max_len-1]]
        # position就是每个token的索引pos
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 10000^(-(2i/d_model)) = exp(-log(10000)/d_model * 2i)
        # arange(0,d_model,2) -> [0,2,4,...,d_model-2] 相当于2i
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        #偶数维, sin(pos*div)
        pe[:,0::2] = torch.sin(position*div_term)
        #奇数维
        pe[:,1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0) 
        #固定参数，不参与训练
        self.register_buffer('pe', pe)

    def forward(self, X):

        L = X.size(1)
        # 把位置编码加到输入的词嵌入矩阵上
        X = X + self.pe[:, :L]

        return X




