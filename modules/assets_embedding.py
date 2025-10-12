import torch
import torch.nn as nn
import math

class AssetsEmbedding(nn.Module):
    def __init__(self, num_base_assets, embedding_dim, target_ratio = 0.2, freeze = False):
        super().__init__()
        self.num_base_assets = num_base_assets
        self.embedding_dim  = embedding_dim
        # 设置目标资产的比例
        self.num_target = math.ceil((num_base_assets - embedding_dim) * target_ratio)
        # 定义嵌入层
        self.embedding = nn.Embedding(num_embeddings = num_base_assets, embedding_dim = embedding_dim, _freeze = freeze)
    
    def pre_train(self, x):
        """
        预训练，现在每次输入的x是一个批次的收益率数据
        x 的形状: (batch_size, num_base_assets)
        返回: x_pred, x_real, 形状均为 (batch_size, 5)
        """
        # 对于整个批次，我们选择相同的目标和基准资产
        # 索引目标资产和基准资产
        target_indices = torch.randperm(self.num_base_assets, device = x.device)[:self.num_target]
        mask = torch.ones(self.num_base_assets, dtype=torch.bool, device = x.device)
        mask[target_indices] = False

        # 分离预测目标和基向量的嵌入
        A_target = self.embedding.weight[target_indices]
        A_base = self.embedding.weight[mask]

        # 求解方程组，这个 solution 矩阵对于整个批次是通用的
        solution = torch.linalg.lstsq(A_base.T, A_target.T).solution
        
        # 映射线性关系到批次的收益率上
        # 从批次数据中选取所有样本的目标资产真实收益率
        # x 形状 (batch_size, num_assets) -> x_real 形状 (batch_size, self.num_target)
        x_real = x[:, target_indices] 

        # 从批次数据中选取所有样本的基准资产真实收益率
        # x 形状 (batch_size, num_assets) -> x_base 形状 (batch_size, num_assets - self.num_target)
        x_base = x[:, mask]
        x_pred = x_base @ solution 
        
        return x_pred, x_real
    
    def forward(self, x, portfolio_weights = None):
        """
        根据资产权重计算资产组合的嵌入。
        """
        # portfolio_weights 的输入形状要求 (batch_size, num_assets, num_base_assets)
        # 如果没有输入weight, 默认倒数第三个维度是num_base_assets 按照顺序默认为基资产
        batch_size = x.shape[0]
        seq_len = x.shape[-2]
        if portfolio_weights == None:
            assert x.shape[-3] == self.num_base_assets, 'please provide portfolio weights for portfolio assets'
            portfolio_weights = torch.eye(n = self.num_base_assets, device = x.device).unsqueeze(0).repeat(batch_size, 1, 1)

        portfolio_embedding = torch.matmul(portfolio_weights, self.embedding.weight).unsqueeze(-2).repeat(1,1, seq_len, 1)
        # print(portfolio_embedding.shape)
        return torch.concat((x, portfolio_embedding), dim = -1)