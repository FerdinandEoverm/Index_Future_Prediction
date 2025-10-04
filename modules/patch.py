import torch
import torch.nn as nn

class SimplePatch(nn.Module):
    """"
    Simple Patch for RNN
    服务于RNN的时间序列分块，因为RNN不会忽略位置信息，因此不需要嵌入RoPE
    """
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        """"
        倒数第二个维度为需要patch的时间步
        """
        # 保存形状并展平前面的层
        front_size = tuple(x.shape[:-2])
        seq_len = x.shape[-2]
        feature_size = x.shape[-1]
        x_rebatch = x.reshape(-1, seq_len, feature_size)


        # 舍去前面无法被整分为patch的的部分
        max_patch = seq_len//self.patch_size
        vaild_seq_len = self.patch_size * max_patch
        x_valid = x_rebatch[:,-vaild_seq_len:,:].clone() # 切片操作破坏了内存连续性，需要复制一份
        x_recover = x_valid.reshape(*front_size, max_patch, self.patch_size, feature_size)

        return x_recover
    
class TimeSeriesPatcher(nn.Module):
    """
    将形状为 (*, seq_len, feature) 的tensor重塑为 (*, num_patch, patch_size, feature)，允许patch重叠
    """

    def __init__(self, patch_size: int, stride: int):
        super().__init__()
        if not isinstance(patch_size, int) or patch_size <= 0:
            raise ValueError("patch_size 必须是一个正整数。")
        if not isinstance(stride, int) or stride <= 0:
            raise ValueError("step 必须是一个正整数。")

        self.patch_size = patch_size
        self.stride = stride

    def forward(self, x) :
        """
        num_patch = floor((seq_len - patch_size) / stride) + 1
        """
        seq_len = x.shape[-2]
        assert seq_len >= self.patch_size, 'patch_size 超过了序列长度'
        patches = x.unfold(dimension=-2, size=self.patch_size, step=self.stride)
        patches = patches.swapaxes(-1, -2)
        patches = torch.flatten(patches, start_dim = -2)
        return patches
