''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformer.Modules import ScaledDotProductAttention

__author__ = "Yu-Hsiang Huang"


# q k v 先经过不同的线性层 再用ScaledDotProductAttention 最后再经过一个线性层
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        # 这里的n_head, d_model, d_k, d_v分别默认为8, 512, 64, 64
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        # len_q, len_k, len_v 为输入的序列长度
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # 用作残差连接
        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        # q k v 分别经过一个线性层再改变维度
        # 由(sz_b, len_q, n_head*d_k) => (sz_b, len_q, n_head, d_k) (sz_b, len_q, 8*64) => (sz_b, len_q, 8, 64)
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        # 交换维度做attention
        # 由(sz_b, len_q, n_head, d_k) => (sz_b, n_head, len_q, d_k) (sz_b, len_q, 8, 64) => (sz_b, 8, len_q, 64)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            # 为head增加一个维度
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        # 做attention
        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        # (sz_b, 8, len_k, 64) => (sz_b, len_k, 8, 64) => (sz_b, len_k, 512)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        # 经过fc和dropout
        q = self.dropout(self.fc(q))
        # 残差连接 论文中的Add & Norm中的Add
        q += residual
        # 论文中的Add & Norm中的Norm
        q = self.layer_norm(q)
        # q的shape为(sz_b, len_q, 512)
        # attn的shape为(sz_b, n_head, len_q, len_k)
        return q, attn


# 其实就是一个MLP而已
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        # d_in默认为512 d_hid默认为2048
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        # 下面两句对应论文中的Add & Norm中
        x += residual

        x = self.layer_norm(x)

        return x
