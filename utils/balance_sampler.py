import torch
import numpy as np
from torch.utils.data import Sampler
from typing import Iterator, List

class BalancedSampler(Sampler[List[int]]):
    """
    自定义采样器，确保每个批次中正负样本的数量均衡。
    """
    def __init__(self, labels, batch_size):

        assert batch_size % 2 == 0 , "batch_size必须为偶数"

        self.batch_size = batch_size
        self.one_side_size = batch_size // 2
        
        # 确保标签是 Torch Tensor
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.float32)

        # 假设标签是一个 [n_samples, n_features] 的张量，我们根据第一列来判断正负
        # 正样本定义为 value >= 0, 负样本定义为 value < 0
        values = labels[:,0]
        self.positive_indices = torch.where(values >= 0)[0].tolist()
        self.negative_indices = torch.where(values < 0)[0].tolist()

        # 检查是否有足够的样本来创建至少一个批次
        if len(self.positive_indices) < self.one_side_size  or len(self.negative_indices) < self.one_side_size:
            raise ValueError("数据集中没有足够的正样本或负样本来创建至少一个批次。")
            
        # 计算一个 epoch 内可以生成的批次数
        # 这取决于样本量较少的那个类别
        num_pos_batches = len(self.positive_indices) // self.one_side_size
        num_neg_batches = len(self.negative_indices) // self.one_side_size
        self.num_batches = min(num_pos_batches, num_neg_batches)

    def __iter__(self) -> Iterator[List[int]]:
        # 在每个 epoch 开始时，打乱索引
        np.random.shuffle(self.positive_indices)
        np.random.shuffle(self.negative_indices)

        pos_ptr = 0
        neg_ptr = 0

        for _ in range(self.num_batches):
            # 取出 n_positive 个正样本索引
            batch_indices = self.positive_indices[pos_ptr : pos_ptr + self.one_side_size]
            pos_ptr += self.one_side_size

            # 取出 n_negative 个负样本索引并合并
            batch_indices.extend(self.negative_indices[neg_ptr : neg_ptr + self.one_side_size])
            neg_ptr += self.one_side_size

            # 打乱批次内的顺序，避免模型学到“前半部分总是正样本”的模式
            np.random.shuffle(batch_indices)
            
            yield batch_indices

    def __len__(self) -> int:
        return self.num_batches
