import torch
import torch.nn as nn

class WeightedRankLoss(nn.Module):
    """
    加权排序损失函数，对于排序错配施加不同的权重损失
    """
    def __init__(self, alpha=0.5, p=1, q=2):
        """
        p 控制幅度权重的指数，衡量本次错配的距离，1为绝对差值 2为平方差之
        q 控制位置权重的指数，衡量本次错配发生的位置的重要性，1为绝对差值 2为平方差值
        alpha控制两个权重的比例，alpha越接近1，幅度权重越大
        """
        super().__init__()
        self.alpha = alpha
        self.p = p
        self.q = q
        
        self.bceloss = nn.BCEWithLogitsLoss(reduction='none') # 用这个版本的BCE可以直接输入logits，但现在需要设置reduction='none' 来拿到原始的loss来加权

    def forward(self, pred, real):
        # 通过双重索引强制将real转换为rank
        sorted_indices = torch.argsort(real, dim=1)
        rank = torch.argsort(sorted_indices, dim=1)

        # 列出所有的ij组合
        num_assets = pred.shape[1]
        indices = torch.combinations(torch.arange(num_assets), r=2)
        indices = indices.to(pred.device)
        i_indices = indices[:, 0]
        j_indices = indices[:, 1]
        
        # 提取成对的预测分数和真实排名
        pred_i = pred[:, i_indices]
        pred_j = pred[:, j_indices]
        rank_i = rank[:, i_indices].float()
        rank_j = rank[:, j_indices].float()

        # 计算预测得分差和真实目标
        # 我们期望 pred_i > pred_j 当 rank_i > rank_j 时
        pred_diff = pred_i - pred_j
        # target=1.0 表示 i 应该排在 j 前面
        real_diff = (rank_i > rank_j).float()

        # 权重p: 错配的幅度 (Magnitude Weight)
        w_mag = torch.pow(torch.abs(rank_i - rank_j), self.p)
        
        # 权重q: 错配的位置 (Location Weight)
        mean_rank = (num_assets - 1) / 2.0
        mid_point = (rank_i + rank_j) / 2.0
        w_loc = torch.pow(torch.abs(mid_point - mean_rank), self.q)
        
        # 为了避免不同尺度问题，在混合前先对每个样本的权重进行归一化
        w_mag_normalized = w_mag / (w_mag.max(dim=1, keepdim=True)[0] + 1e-9)
        w_loc_normalized = w_loc / (w_loc.max(dim=1, keepdim=True)[0] + 1e-9)
        
        # 组合权重
        raw_weights = self.alpha * w_mag_normalized + (1 - self.alpha) * w_loc_normalized
        
        # 对最终权重进行归一化并缩放，使其平均值为1，以保持损失的量级稳定
        num_combinations = i_indices.shape[0]
        final_weights = (raw_weights / (raw_weights.sum(dim=1, keepdim=True) + 1e-9)) * num_combinations

        # 计算原始loss应用自定义权重
        unreduced_loss = self.bceloss(pred_diff, real_diff)
        weighted_loss = unreduced_loss * final_weights

        return weighted_loss.mean()