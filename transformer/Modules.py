import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = "Yu-Hsiang Huang"

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        # 其实就是论文中的根号d_k
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # sz_b: batch_size 批量大小
        # len_q,len_k,len_v: 序列长度 在这里他们都相等
        # n_head: 多头注意力 默认为8
        # d_k,d_v: k v 的dim(维度) 默认都是64
        # 此时q的shape为(sz_b, n_head, len_q, d_k) (sz_b, 8, len_q, 64)
        # 此时k的shape为(sz_b, n_head, len_k, d_k) (sz_b, 8, len_k, 64)
        # 此时v的shape为(sz_b, n_head, len_k, d_v) (sz_b, 8, len_k, 64)
        # q先除以self.temperature(论文中的根号d_k) k交换最后两个维度(这样才可以进行矩阵相乘) 最后两个张量进行矩阵相乘
        # attn的shape为(sz_b, n_head, len_q, len_k)
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            # 用-1e9代替0 -1e9是一个很大的负数 经过softmax之后接近与0
            # 其一：去除掉各种padding在训练过程中的影响
            # 其二，将输入进行遮盖，避免decoder看到后面要预测的东西。（只用在decoder中）
            attn = attn.masked_fill(mask == 0, -1e9)

        # 先在attn的最后一个维度做softmax 再dropout 得到注意力分数
        attn = self.dropout(F.softmax(attn, dim=-1))
        # 最后attn与v进行矩阵相乘
        # output的shape为(sz_b, 8, len_q, 64)
        output = torch.matmul(attn, v)
        # 返回 output和注意力分数
        return output, attn
