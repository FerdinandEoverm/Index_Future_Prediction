import torch
import torch.nn as nn
class SequenceTruncate(nn.Module):
    """
    随机从序列中裁去前端的一部分时间步，在保持序列整体顺序结构不变的情况下丰富输入的信息，进一步降低过拟合的影响，类似于时间序列上的dropout
    """
    def __init__(self, dropout):
        """
        警告：dropout 过大，可能导致序列过短无法完成其他基础任务
        """
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        if self.training:
            front_size = tuple(x.shape[:-2])
            seq_len = x.shape[-2]
            feature_size = x.shape[-1]
            x_rebatch = x.reshape(-1, seq_len, feature_size)

            random_drop = torch.randint(0, int(self.dropout*seq_len) + 1, (1,), device=x.device).item() # dropout 为最大舍弃比例，会随即从0到dropout舍弃一定比例

            x_valid = x_rebatch[:,random_drop:,:].clone() # 切片操作破坏了内存连续性，需要复制一份
            x_recover = x_valid.reshape(*front_size, seq_len - random_drop, feature_size)
            return x_recover

        else:
            return x
        
if __name__ == '__main__':
    x = torch.randn(size = (5,7,20,9))
    st = SequenceTruncate(0.5)
    print(st(x).shape)