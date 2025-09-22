import matplotlib.pyplot as plt
from IPython import display
import numpy as np

class TrainAnimator:
    """
    在动画中绘制数据，用于在模型训练中动态监控损失、预测概率、logits的变化。
    """

    def __init__(self, figsize=(12, 6)):
        self.num_subplots = 6
        self.fig, self.axes = plt.subplots(2, 3, figsize=figsize)
        self.axes = self.axes.flatten()
        titles = ['train loss', 'train classes prob', 'train classes logits', 'test loss', 'test classes prob', 'test classes logits']
        for i, ax in enumerate(self.axes):
            ax.set_title(titles[i])
            ax.grid()
        self.fig.tight_layout()

    def add(self, x, y, subplot_idx=0):
        """
        向指定的子图添加数据点。
        参数:
            x : 当前epoch
            y : 记录的值，对于prob和logits，传入元组
            subplot_idx (int): 子图的编号
        """
        if subplot_idx < 0 or subplot_idx >= self.num_subplots:
            raise ValueError(f"subplot_idx must be between 0 and {self.num_subplots - 1}.")
            
        target_plot = self.data[subplot_idx]
        
        # 确保y是列表
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        # 确保x是列表
        if not hasattr(x, "__len__"):
            x = [x] * n
            
        # 第一次添加数据时需要初始化
        if not target_plot['X']:
            target_plot['X'] = [[] for _ in range(n)]
            target_plot['Y'] = [[] for _ in range(n)]

        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                target_plot['X'][i].append(a)
                target_plot['Y'][i].append(b)

        self.draw()

    def draw(self):
        """绘制子图"""
        display.clear_output(wait=True)
        for i, ax in enumerate(self.axes):
            ax.cla()
            plot_data = self.data[i]
            if plot_data['X']:
                fmts = ('-', 'm--', 'g-.', 'r:')
                for j in range(len(plot_data['X'])):
                    ax.plot(plot_data['X'][j], plot_data['Y'][j], fmts[j % len(fmts)])
            ax.grid()
            # ax.legend()
        self.fig.tight_layout()
        display.display(self.fig)

    def reset(self):
        """清空数据"""
        self.data = [{'X': [], 'Y': []} for _ in range(self.num_subplots)]
        print("Animator data has been reset.")