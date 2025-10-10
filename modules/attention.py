import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_feature, num_head):
        super().__init__()
        assert dim_feature % num_head == 0, "dim_feature 必须能被 num_head 整除"
        self.dim_feature = dim_feature
        self.num_head = num_head
        self.dim_head = dim_feature // num_head

        # 线性变换层，用于生成Q, K, V
        self.wq = nn.Linear(dim_feature, dim_feature)
        self.wk = nn.Linear(dim_feature, dim_feature)
        self.wv = nn.Linear(dim_feature, dim_feature)
        self.fc_out = nn.Linear(dim_feature, dim_feature) # 最终的线性输出层

    def forward(self, query, key, value, mask=None):
        # 假设 query, key, value 的前置维度形状都相同
        original_shape = query.shape
        seq_len = original_shape[-2]
        
        # 将所有前置维度展平成一个新的 batch_size
        # 例如: (B, T, H, W) -> (B*T, H, W) 如果 H 是 seq_len, W 是 dim_feature
        # 使用 -1 可以自动计算展平后的维度大小
        effective_batch_size = math.prod(original_shape[:-2])
        
        query = query.reshape(effective_batch_size, seq_len, self.dim_feature)
        key = key.reshape(effective_batch_size, seq_len, self.dim_feature)
        value = value.reshape(effective_batch_size, seq_len, self.dim_feature)
        # ----------------------------------------
        
        # batch_size 现在是展平后的 effective_batch_size
        batch_size = query.size(0)

        # 线性变换并分头
        # view中的-1会自动推断为 seq_len
        Q = self.wq(query).view(batch_size, -1, self.num_head, self.dim_head).transpose(1, 2)
        K = self.wk(key).view(batch_size, -1, self.num_head, self.dim_head).transpose(1, 2)
        V = self.wv(value).view(batch_size, -1, self.num_head, self.dim_head).transpose(1, 2)

        # 计算缩放点积注意力
        energy = torch.matmul(Q, K.transpose(-2, -1)) / (self.dim_head ** 0.5)
        
        if mask is not None:
            # 注意: 如果有mask, mask也需要被正确地广播到 (effective_batch_size, num_head, ...)
            energy = energy.masked_fill(mask == 0, -1e9) # 使用-1e9代替负无穷，更稳定
            
        attention = torch.softmax(energy, dim=-1)
        scaled_attention = torch.matmul(attention, V)

        # contiguous() 保证内存连续，view操作前通常需要
        scaled_attention = scaled_attention.transpose(1, 2).contiguous().view(batch_size, -1, self.dim_feature)
        # 4. 最终线性输出
        output = self.fc_out(scaled_attention)
        
        # 将展平的 batch_size 恢复为原始的多个前置维度
        output = output.view(*original_shape)
        return output