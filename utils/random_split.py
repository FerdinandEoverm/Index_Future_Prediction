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
    