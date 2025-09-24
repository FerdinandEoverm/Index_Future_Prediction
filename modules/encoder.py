import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import modules.attention as attention
import modules.addnorm as addnorm
import modules.ffn as ffn

class EncoderBlock(nn.Module):
    """
    Transformer Encoder Block: Multi-Head Attention + AddNorm + FFN + AddNorm
    """
    def __init__(
            self,
            batch_size,
            dim_feature,
            dim_sequence,

            num_head,
            num_ffn_hidden,

            dropout,
            ):
        super(EncoderBlock, self).__init__()
        self.attn = attention.MultiHeadAttention(dim_feature, num_head)
        self.addnorm1 = addnorm.AddNorm(normalized_shape=(dim_sequence, dim_feature) ,dropout=dropout)
        self.ffn = ffn.PositionWiseFFN(dim_feature, num_ffn_hidden, dim_feature)
        self.addnorm2 = addnorm.AddNorm(normalized_shape=(dim_sequence, dim_feature), dropout=dropout)
    def forward(self, x, mask=None):
        # Multi-head attention + AddNorm
        attn_out = self.attn(x,x,x, mask)
        x = self.addnorm1(x, attn_out)
        # Position-wise FFN + AddNorm
        ffn_out = self.ffn(x)
        x = self.addnorm2(x, ffn_out)
        return x



class MultiLayerEncoder(nn.Module):
    """
    多层Transformer Encoder，由多个EncoderBlock堆叠而成
    """
    def __init__(
            self,
            batch_size,
            dim_feature,
            dim_sequence,

            num_enc_layer,
            num_head,
            num_ffn_hidden,

            dropout,

            ):
        super(MultiLayerEncoder, self).__init__()
        self.layers = nn.ModuleList([EncoderBlock(
            batch_size = batch_size,
            dim_feature = dim_feature,
            dim_sequence = dim_sequence,

            num_head = num_head,
            num_ffn_hidden = num_ffn_hidden,

            dropout = dropout,

            )for _ in range(num_enc_layer)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
