import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_feature, num_head):
        super(MultiHeadAttention, self).__init__()
        assert dim_feature % num_head == 0, "d_model must be divisible by num_heads"
        self.dim_feature = dim_feature
        self.num_head = num_head
        self.dim_head = dim_feature // num_head

        # 线性变换层，用于生成Q, K, V
        self.wq = nn.Linear(dim_feature, dim_feature)
        self.wk = nn.Linear(dim_feature, dim_feature)
        self.wv = nn.Linear(dim_feature, dim_feature)
        self.fc_out = nn.Linear(dim_feature, dim_feature) # 最终的线性输出层

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1. 线性变换并分头
        Q = self.wq(query).view(batch_size, -1, self.num_head, self.dim_head).transpose(1, 2)
        K = self.wk(key).view(batch_size, -1, self.num_head, self.dim_head).transpose(1, 2)
        V = self.wv(value).view(batch_size, -1, self.num_head, self.dim_head).transpose(1, 2)

        Q = Q.to(torch.float32)
        K = K.to(torch.float32)
        V = V.to(torch.float32)

        # 2. 计算缩放点积注意力
        energy = torch.matmul(Q, K.transpose(-2, -1)) / (self.dim_head ** 0.5)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20')) # 将mask为0的位置置为负无穷
        attention = torch.softmax(energy, dim=-1)
        attention = attention.to(torch.float32)
        scaled_attention = torch.matmul(attention, V)

        # 3. 合并头
        scaled_attention = scaled_attention.transpose(1, 2).contiguous().view(batch_size, -1, self.dim_feature)

        # 4. 最终线性输出
        output = self.fc_out(scaled_attention)
        return output