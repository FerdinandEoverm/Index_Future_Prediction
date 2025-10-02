import torch
import torch.nn as nn

class HybridLoss(nn.Module):
    """
    应用于金融资产收益率预测的损失函数，对接HybridDecoder
    避免离散化为标签时丢失信息，同时利用成交量等波动率参数强化预测效果。
    将线性预测输出输入到Huber损失函数，将概率预测输出输入到KL散度损失函数，再通过超参数控制两个损失的和作为最终损失
    """
    def __init__(self, alpha = 0.5, delta = 1, show_loss = False):
        """
        alpha: 损失函数的混合比例，alpha越接近1, Huber 损失占比越大
        delta：Huber损失的内置参数
        """
        super().__init__()
        self.alpha = alpha
        self.huber_loss = nn.HuberLoss(delta = delta, reduction='mean')
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.show_loss = show_loss

    def forward(self, pred: torch.Tensor, real: torch.Tensor):
        huber_loss = self.huber_loss(pred[:,:1], real[:,:1])
        kl_loss = self.kl_loss(pred[:,1:], real[:,1:])
        if self.show_loss:
            print('huber_loss',huber_loss, 'kl_loss', kl_loss)

        return self.alpha * huber_loss + (1-self.alpha)*kl_loss