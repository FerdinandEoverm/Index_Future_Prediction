import torch
import torch.nn as nn
from modules.attention import MultiHeadAttention
from modules.addnorm import AddNorm
from modules.ffn import PositionWiseFFN

class PanelEncoderBlock(nn.Module):
    """
    Panel data transformer
    """
    def __init__(self, d_model, num_head, num_ffn_hidden, dropout):
        super().__init__()
        # 纵向时间序列注意力；
        self.time_series_attention = MultiHeadAttention(d_model, num_head)
        # 横向截面注意力；
        self.cross_section_attention = MultiHeadAttention(d_model, num_head)
        # addnorm 层
        self.addnorm = AddNorm(normalized_shape=(d_model,), dropout=dropout)
        # 通过ffn 整合信息
        self.ffn = PositionWiseFFN(d_model, num_ffn_hidden, d_model)

    def forward(self, x, mask=None):
        """
        imput and output size: (batch_size, num_assets, seq_len or num_patch, d_model)
        """
        # 注意力机制会自动展平前面的层，把倒数第二层作为注意力的范围。对于时序注意力，倒数第二维度应该是时间步长度
        time_series_attention_out = self.time_series_attention(x,x,x, mask)
        x = self.addnorm(x, time_series_attention_out)
        # 这里交换num_assets 和 seq_len 来把资产数交换到倒数第二个维度上，让注意力关注截面
        x = x.permute(0,2,1,3)
        cross_section_attention = self.cross_section_attention(x,x,x, mask)
        x = self.addnorm(x, cross_section_attention)
        # 记得交换回来
        x = x.permute(0,2,1,3)
        # 最后通过ffn 整理当前时间步内部的信息
        ffn_out = self.ffn(x)
        x = self.addnorm(x, ffn_out)
        return x


class MultiLayerPanelEncoder(nn.Module):
    """
    多层PanelEncoder，由多个PanelEncoderBlock堆叠而成
    """
    def __init__(self, num_layer, d_model, num_head, num_ffn_hidden, dropout):
        super().__init__()
        self.layers = nn.ModuleList([PanelEncoderBlock(d_model = d_model, num_head = num_head,num_ffn_hidden = num_ffn_hidden,dropout = dropout,)for _ in range(num_layer)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x