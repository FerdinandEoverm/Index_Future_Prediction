import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

class FeatureEmbedding(nn.Module):
    '''
    可学习的特征映射
    根据 mlp_structure 构建多层感知机（MLP）支持多种激活函数（GELU、ReLU、Tanh）在每个隐藏层后添加 Dropout 防止过拟合
    验证输入特征的维度与 MLP 输入维度匹配，处理任意维度的输入（保持除最后一维外的所有维度），通过 MLP 进行特征变换，恢复原始形状（仅改变特征维度）
    灵活支持不同的 MLP 结构，保持输入张量的批量维度和其他维度不变，使用 GELU 作为默认激活函数（效果通常优于 ReLU）
    '''

    def __init__(self, dim_raw_feature, dim_extension, dropout=0.5):
        super(FeatureEmbedding, self).__init__()

        self.dim_raw_feature = dim_raw_feature

        self.mlp = nn.Sequential(
            nn.Linear(dim_raw_feature, dim_extension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_extension, dim_extension),
            nn.GELU(),
            nn.Dropout(dropout),
            )
        

    def forward(self, raw_feature: torch.FloatTensor):

        assert raw_feature.size()[-1] == self.dim_raw_feature, f"Input feature size {raw_feature.size()[-1]} does not match expected size {self.dim_raw_feature} in feature embedding"
        embedded = self.mlp(raw_feature)
        return embedded



class ProductEmbedding(nn.Module):
    '''
    品种嵌入
    '''
    def __init__(self, dim_embedding):
        '''
        parameter：
            num_product: 品种个数，数据集内共87个不同品种
            dim_embedding: 用于品种嵌入的增加维度
        '''
        super(ProductEmbedding, self).__init__()
        self.product_embed = nn.Embedding(100, dim_embedding)

    def forward(self, product_id: torch.IntTensor):
        return self.product_embed(product_id).to(torch.float32)



class TemporalEmbedding(nn.Module):
    """
    Time2Vec时序编码，以concat形式扩展位置编码。
    原始输入维度: (*, seq_len, d_model)
    输出维度: (*, seq_len, d_model + dim_embedding)
    """
    def __init__(self, dim_embedding):
        super().__init__()
        self.dim_embedding = dim_embedding
        
        # Time2Vec 的可学习参数
        self.w = nn.Parameter(torch.empty(1, self.dim_embedding), requires_grad=True)
        self.b = nn.Parameter(torch.empty(1, self.dim_embedding), requires_grad=True)
        # 初始化参数
        nn.init.uniform_(self.w, -0.1, 0.1)
        nn.init.uniform_(self.b, -0.1, 0.1)

    def forward(self, x):
        """
        输入形状为 (*, seq_len, d_model)
        输出形状为 (*, seq_len, d_model + dim_embedding)
        """
        # 保存初始形状
        original_shape = x.shape # (*, seq_len, feature_dim)
        seq_len = original_shape[-2]
        batch_dims = original_shape[:-2]
        
        # 相对时间序号： [0, 1, 2, ..., seq_len-1]
        tau = torch.arange(seq_len, dtype=torch.float, device=x.device).unsqueeze(-1)

        # 计算时间嵌入
        time_embedding = tau @ self.w + self.b
        
        linear_part = time_embedding[:, :1] # 线性部分
        periodic_part = torch.sin(time_embedding[:, 1:]) # 周期性部分

        time_embedding = torch.cat([linear_part, periodic_part], dim=-1)

        # 把编码广播到所有维度
        target_shape = batch_dims + (seq_len, self.dim_embedding)
        time_embedding = time_embedding.expand(target_shape)

        # 拼接
        output = torch.cat([x, time_embedding], dim=-1)
        
        return output


class CalendarEmbedding(nn.Module):
    '''
    绝对时间信息嵌入，补全相对时间信息嵌入中丢失的日历信息，例如周一效应、月末效应等。
    该模块为月、周、日分别学习一个嵌入向量，然后将它们拼接起来。
    
    输入:
    - month (torch.Tensor): 月份张量，整数类型，值应在 [1, 12] 范围内。
    - weekday (torch.Tensor): 星期几张量，整数类型，值应在 [0, 6] 范围内。
    - day (torch.Tensor): 日期张量，整数类型，值应在 [1, 31] 范围内 (代表1号到31号)。
    输入张量的形状可以是 (batch_size, sequence_size) 或 (sequence_size,)。
    
    返回:
    - torch.Tensor: 拼接后的嵌入向量，形状为 (batch_size, sequence_size, 24)。
    '''
    def __init__(self):
        """
        初始化日历嵌入模块。
        
        参数:
        embed_dim (int): 每个日历特征（月、周、日）的嵌入维度。
                         最终输出的维度将是 3 * embed_dim。
        """
        super(CalendarEmbedding, self).__init__()
        self.month_embed = nn.Embedding(num_embeddings=13, embedding_dim=3)
        self.weekday_embed = nn.Embedding(num_embeddings=7, embedding_dim=3)
        self.day_embed = nn.Embedding(num_embeddings=32, embedding_dim=6)

    def forward(self, date: torch.LongTensor):
        """
        前向传播
        
        参数 (必须是LongTensor或IntTensor):
        - month: 月份张量（1-12）
        - weekday: 星期几张量（0-6）
        - day: 日期张量（1-31）
        """
        # 分别获取每个特征的嵌入向量
        # nn.Embedding 会自动处理输入张量的形状
        # (B, S) -> (B, S, embed_dim)
        month_vec = self.month_embed(date[:,:,0])
        weekday_vec = self.weekday_embed(date[:,:,1])
        day_vec = self.day_embed(date[:,:,2])
        
        # 沿最后一个维度（特征维度）将三个嵌入向量拼接起来
        combined_embeds = torch.cat([month_vec, weekday_vec, day_vec], dim=-1)
        
        return combined_embeds



class Embedding(nn.Module):
    '''
    封装上述所有的embedding模块
    '''

    def __init__(self, dim_raw_feature, dim_extension, dim_product_embedding, dim_temporal_embedding, dropout = 0.5):
        super(Embedding, self).__init__()
        self.feature_embed = FeatureEmbedding(dim_raw_feature = dim_raw_feature, dim_extension = dim_extension, dropout = dropout)
        self.product_embed = ProductEmbedding(dim_embedding = dim_product_embedding)
        self.temporal_embed = TemporalEmbedding(embed_size = dim_temporal_embedding)
        self.calender_embed = CalendarEmbedding()

    def forward(self, market_feature, product_id, relative_date, calender_feature):
        return torch.cat((
            market_feature,
            self.product_embed(product_id),
            self.temporal_embed(relative_date),
            self.calender_embed(calender_feature),
            self.feature_embed(market_feature),
        ),
        dim = -1
        )


# 测试
if __name__ == '__main__':
    # 超参数
    batch_size = 16
    sequence_length = 1024

    dim_raw_feature = 11
    dim_extension = 12
    num_product = 13
    dim_product_embedding = 14
    dim_temporal_embedding = 15

    dim_feature = dim_raw_feature + dim_extension + dim_product_embedding + dim_temporal_embedding + 24

    raw_input = torch.randn(size=(batch_size, sequence_length, dim_raw_feature))

    product_id = torch.randint(size=(batch_size, sequence_length), low = 0, high = num_product)

    relative_date = torch.arange(start = - sequence_length*batch_size/2, end = sequence_length*batch_size/2).reshape(batch_size,-1)

    months = torch.randint(1, 13, (batch_size, sequence_length))
    weekdays = torch.randint(0, 7, (batch_size, sequence_length))
    days = torch.randint(1, 32, (batch_size, sequence_length))
    calender_feature = torch.stack((months, weekdays, days), dim = -1)

    print('========================')
    print('Feature Embedding Test:')
    embedding = FeatureEmbedding(dim_raw_feature = dim_raw_feature, dim_extension = dim_extension)
    output = embedding(raw_input)
    print('Input shape:', raw_input.size())
    print('Output shape:', output.size())
    print(f"Expected output shape: ({batch_size}, {sequence_length}, {dim_extension})")
    assert output.shape[0] == batch_size, 'error: unmatched dim in 0'
    assert output.shape[1] == sequence_length, 'error: unmatched dim in 1'
    assert output.shape[2] == dim_extension, 'error: unmatched dim in 2'
    print('========================')
    

    print('Product Embedding Test:')
    embedding = ProductEmbedding(dim_embedding = dim_product_embedding)
    output = embedding(product_id)
    print('Input shape:', product_id.size())
    print('Output shape:', output.size())
    print(f"Expected output shape: ({batch_size}, {sequence_length}, {dim_product_embedding})")
    assert output.shape[0] == batch_size, 'error: unmatched dim in 0'
    assert output.shape[1] == sequence_length, 'error: unmatched dim in 1'
    assert output.shape[2] == dim_product_embedding, 'error: unmatched dim in 2'
    print('========================')

    print('Temporal Embedding Test:')
    embedding = TemporalEmbedding(embed_size=dim_temporal_embedding)
    output = embedding(relative_date)
    print('Input shape:', relative_date.size())
    print('Output shape:', output.size())
    print(f"Expected output shape: ({batch_size}, {sequence_length}, {dim_temporal_embedding})")
    assert output.shape[0] == batch_size, 'error: unmatched dim in 0'
    assert output.shape[1] == sequence_length, 'error: unmatched dim in 1'
    assert output.shape[2] == dim_temporal_embedding, 'error: unmatched dim in 2'
    print('========================')

    print('Calendar Embedding Test:')
    embedding = CalendarEmbedding()
    print(f"Month shape:    {months.shape}")
    print(f"Weekday shape:  {weekdays.shape}")
    print(f"Day shape:      {days.shape}")
    print(f"Calender shape: {calender_feature.shape}")
    output = embedding(calender_feature)
    print('Input shape:', calender_feature.size())
    print('Output shape:', output.size())
    print(f"Expected output shape: ({batch_size}, {sequence_length}, {24})")
    assert output.shape[0] == batch_size, 'error: unmatched dim in 0'
    assert output.shape[1] == sequence_length, 'error: unmatched dim in 1'
    assert output.shape[2] == 24, 'error: unmatched dim in 2'
    print('========================')

    print('Joint Embedding Test:')
    embed = Embedding(dim_raw_feature = dim_raw_feature, dim_extension = dim_extension, dim_product_embedding = dim_product_embedding,  dim_temporal_embedding = dim_temporal_embedding, dropout = 0.5)
    output = embed(raw_input, product_id, relative_date, calender_feature)
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {sequence_length}, {dim_feature})")
    assert output.shape[0] == batch_size, 'error: unmatched dim in 0'
    assert output.shape[1] == sequence_length, 'error: unmatched dim in 1'
    assert output.shape[2] == dim_feature, 'error: unmatched dim in 2'
    print('========================')

