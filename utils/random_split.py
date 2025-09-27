import numpy as np
import torch
from torch.utils.data import TensorDataset

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
    