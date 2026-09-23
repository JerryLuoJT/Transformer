import torch
import torch.nn as nn
import torch.nn.functional as F
import math
d_model = 8

def create_causal_mask(seq_len, device):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask
class QKVProjection(nn.Module):
    # basic self-attention
    def __init__(self,d_model):
        super().__init__()
        #构建全连接层
        self.W_q = nn.Linear(d_model,d_model,bias=False) 
        self.W_k = nn.Linear(d_model,d_model,bias=False)
        self.W_v = nn.Linear(d_model,d_model,bias=False)

    def forward(self,X):
        Q = self.W_q(X) # y = XW_q
        K = self.W_k(X)
        V = self.W_v(X)

        """
        Q: [B, Lq, dk]
        K: [B, Lk, dk]
        V: [B, Lk, dv]
        计算attention=QK/sqrt(dk)
        注意QKV是三维向量（未展开）
        """
        dk = Q.size(-1)
        K = K.transpose(-2, -1)
        # QK^T
        score = torch.matmul(Q, K)
        # /sqrt(dk)
        score = score / math.sqrt(dk)
        L = score.size(-1)
        #构建掩码矩阵：先构建一个全1矩阵，然后变成下三角矩阵，最后将其放在X的设备上
        mask = torch.tril(torch.ones(L,L,device= X.device))
        #将上三角矩阵的值置为-inf
        score = score.masked_fill(mask==0, float('-inf')) 
        #softmax(score)
        attention_weights = F.softmax(score, dim = -1) #沿着score的最后一维度dk进行softmax
        #AV
        output = torch.matmul(attention_weights,V)
        return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.W_k = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.W_v = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.W_o = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

    def forward(self, X, context= None, mask=None):
        B, L, _ = X.size()
        #1. 线性变换得到Q,K,V
        Q = self.W_q(X)
        # context代表cross attention输入时候来自encoder的那部分。在encoder模块中默认为空。
        if context is None:
            K = self.W_k(X)
            V = self.W_v(X)
        else:
            K = self.W_k(context)
            V = self.W_v(context)
        #2. 拆分多头
        # [B, L, d_model]
        # ->
        # [B, L, h, d_k]
        # 交叉注意力时 K/V 来自 encoder，长度是 Lk，不能复用查询长度 L
        Lk = K.size(1)
        Q = Q.view(B,L,self.num_heads,self.d_k)
        K = K.view(B,Lk,self.num_heads,self.d_k)
        V = V.view(B,Lk,self.num_heads,self.d_k)
        #3. 转置
        # [B, L, h, d_k]
        # ->
        # [B, h, L, d_k]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        #4. 计算注意力
        scores = torch.matmul(Q, K.transpose(-2,-1))/math.sqrt(self.d_k)
        #传入mask
        if mask is not None:
            scores = scores.masked_fill(mask==0, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        #5. 计算输出
        output = torch.matmul(attention_weights, V)
        # [B, h, L, d_k]
        # ->
        # [B, L, h, d_k]
        output = output.transpose(1,2)
        # ->
        # [B, L, d_model]
        output = output.contiguous().view(B,L,self.d_model) #view函数就是按照给定的参数格式来排列这些数据
        #6. W_o线性变换得到最终输出
        output = self.W_o(output)
        return output, attention_weights



        




   
  
