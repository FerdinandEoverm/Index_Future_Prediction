import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Sampler
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


class RandomLoader:
    def __init__(self, feature, label):
        """
        将输入的时间序列打包为随机位置但连续的切片用于时间序列训练
        feature label
        label的最后一个维度必须是4，且第一个分量表示预测目标，以第一个分量的符号进行均衡分配sample size
        """
        assert len(feature) == len(label), 'unmatched length of feature and label'
        
        self.feature = feature
        self.label = label
        self.length = len(feature)

    def __call__(self, batch_size, slice_size = [0.5, 0.2, 0.1], balance = [True, False, False]):
        """
        slice_size: list of float 数量表示切片数量
        balance： list of bool
        """
        assert len(slice_size) == len(balance), 'unmatched length of slice_size and balance'

        
        total_ratio = np.sum(slice_size)

        start = np.random.randint(0, int(self.length * (1 - total_ratio)))
        print(start)
        dataloaders = []

        for i in range(len(slice_size)):
            size = int(self.length * slice_size[i])

            if i == 0:
                current_feature = torch.flatten(self.feature[:start+size], start_dim = 0, end_dim = -3)
                current_label = torch.flatten(self.label[:start+size], start_dim = 0, end_dim = -2)
                dataset = TensorDataset(current_feature, current_label)
            else:
                current_feature = torch.flatten(self.feature[start:start+size], start_dim = 0, end_dim = -3)
                current_label = torch.flatten(self.label[start:start+size], start_dim = 0, end_dim = -2)
                dataset = TensorDataset(current_feature, current_label)

            if balance[i]:
                balance_sampler = BalancedSampler(current_label, batch_size)
                loader = DataLoader(dataset, batch_sampler=balance_sampler)
            else:
                loader = DataLoader(dataset, batch_size=batch_size, drop_last = True)

            dataloaders.append(loader)
            start = start + size

            
        return tuple(dataloaders)


class CallableDataset(TensorDataset):
    """
    封装数据接口
    """
    def __init__(self, *args):
        super(CallableDataset, self).__init__(*args)
    
    def __call__(self, batch_size):
        """
        根据ModelTrain 类的要求，我们这里继承自原生的Dataset类，实现一个callable的Dataset
        """
        total_size = len(self)
        indices = torch.randint(0, total_size, (batch_size,))
        batch = [tensor[indices] for tensor in self.tensors]
        
        return tuple(batch)
    
    def __add__(self, dataset_b):
        """
        合并两个CallableDataset，返回新的CallableDataset
        """
        # 检查类型
        if not isinstance(dataset_b, CallableDataset):
            raise TypeError("只能合并CallableDataset类型")
        
        # 检查tensors数量是否一致
        if len(self.tensors) != len(dataset_b.tensors):
            raise ValueError("两个数据集的tensors数量必须一致")
        
        # 检查每个tensor的维度（除了batch维度）是否匹配
        for i, (tensor_a, tensor_b) in enumerate(zip(self.tensors, dataset_b.tensors)):
            if tensor_a.shape[1:] != tensor_b.shape[1:]:
                raise ValueError(f"第{i}个tensor的维度不匹配: {tensor_a.shape[1:]} vs {tensor_b.shape[1:]}")
        
        # 合并所有tensors
        merged_tensors = []
        for tensor_a, tensor_b in zip(self.tensors, dataset_b.tensors):
            merged_tensor = torch.cat([tensor_a, tensor_b], dim=0)
            merged_tensors.append(merged_tensor)
        
        # 创建新的CallableDataset
        return CallableDataset(*merged_tensors)
    
    def all(self):
        return tuple(self.tensors)
    


class RandomSplit():
    def __init__(self, data, device):
        self.data = data.dropna()
        self.device = device

        self.column_names = [] # 记录需要输出的列名
        self.dtypes = [] # 记录需要输出的列名
        self.unfold = [] # 记录是否需要滑动窗口

    def set_output(self, column_name, dtype, unfold = False):
        self.column_names.append(column_name)
        self.dtypes.append(dtype)
        self.unfold.append(unfold)

    def get_split_data(self, train_size, validation_size, test_size, window_size):

        if len(self.unfold) == 0:
            raise ValueError('you have not set output yet')
        
        tensors = []
        for i in range(len(self.unfold)):
            tensor = torch.tensor(self.data[self.column_names[i]].values, dtype = self.dtypes[i], device = self.device)
            if self.unfold[i]:
                tensor = tensor.unfold(dimension = 0, size = window_size, step = 1).transpose(1,2)
            else:
                tensor = tensor[window_size-1:] # 如果该tensor未进行滑动窗口，则需要裁去前面部分以对齐滑动窗口
            tensors.append(tensor)


        split = np.random.randint(train_size, len(self.data) - validation_size - test_size)
        train_tensors = []
        validation_tensors = []
        test_tensors = []

        for tensor in tensors:
            train_tensors.append(tensor[:split])
            validation_tensors.append(tensor[split:split+validation_size])
            test_tensors.append(tensor[split+validation_size:split+validation_size+test_size])
        

        return CallableDataset(*tuple(train_tensors)), CallableDataset(*tuple(validation_tensors)), CallableDataset(*tuple(test_tensors))
