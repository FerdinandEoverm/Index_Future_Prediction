import torch
import torch.nn as nn

class HybridDecoder(nn.Module):
    """
    混合输出解码器架构，对接HybridLoss
    从单一的信息向量中，通过两种方式分别解码出线性预测输出和概率预测输出
    """
    def __init__(self, dim_state, **kwargs):
        super(HybridDecoder, self).__init__(**kwargs)
        self.device = 'cuda:0'
        self.log_prob = nn.Sequential(nn.Linear(dim_state, 3),nn.LogSoftmax(dim = 1))
        self.regress = nn.Linear(dim_state, 1)
    def forward(self, x):
        return torch.concat((self.regress(x), self.log_prob(x)),dim = 1)
    
if __name__ == '__main__':
    state = torch.ones(size = (10, 64))
    hd = HybridDecoder(dim_state = 64)
    print(hd(state).shape)