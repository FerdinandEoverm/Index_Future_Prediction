import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import modules.attention as attention
import modules.addnorm as addnorm
import modules.ffn as ffn

class DecoderBlock(nn.Module):
    """
    Transformer Decoder Block: Cross Attention + AddNorm + Self Attention + AddNorm + FFN + AddNorm
    """
    def __init__(
            self,
            batch_size,
            dim_feature,
            dim_sequence,

            len_hist,
            len_pred,

            num_head,
            num_ffn_hidden,

            dropout,
            ):
        super(DecoderBlock, self).__init__()

        self.self_attn = attention.MultiHeadAttention(dim_feature, num_head)
        self.addnorm1 = addnorm.AddNorm(normalized_shape=(len_hist + len_pred, dim_feature), dropout=dropout)

        self.cross_attn = attention.MultiHeadAttention(dim_feature, num_head)
        self.addnorm2 = addnorm.AddNorm(normalized_shape=(len_hist + len_pred, dim_feature), dropout=dropout)

        self.ffn = ffn.PositionWiseFFN(dim_feature, num_ffn_hidden, dim_feature)
        self.addnorm3 = addnorm.AddNorm(normalized_shape=(len_hist + len_pred, dim_feature), dropout=dropout)

    def forward(self, x, enc_out, causal_mask):

        # self attention + AddNorm
        self_attn_out = self.self_attn(x, x, x, mask = causal_mask)
        x = self.addnorm1(x, self_attn_out)

        # cross attition + AddNorm
        cross_attn_out = self.cross_attn(x, enc_out, enc_out)
        x = self.addnorm2(x, cross_attn_out)

        # Position-wise FFN + AddNorm
        ffn_out = self.ffn(x)
        x = self.addnorm3(x, ffn_out)

        return x



class MultiLayerDecoder(nn.Module):
    """
    多层Transformer Decoder，由多个DecoderBlock堆叠而成
    """
    def __init__(
            self,
            batch_size,
            dim_feature,
            dim_sequence,

            len_hist,
            len_pred,

            num_dec_layer,
            num_head,
            num_ffn_hidden,

            dropout,
            ):
        super(MultiLayerDecoder, self).__init__()
        self.layers = nn.ModuleList([DecoderBlock(
            batch_size,
            dim_feature,
            dim_sequence,

            len_hist,
            len_pred,

            num_head,
            num_ffn_hidden,

            dropout,
            )for _ in range(num_dec_layer)])

    def forward(self, x, enc_out, mask):
        for layer in self.layers:
            x = layer(x, enc_out, mask)
        return x