import matplotlib.pyplot as plt
from IPython import display
import numpy as np

class TrainAnimator:
    """
    在动画中绘制数据，用于在模型训练中动态监控损失、预测概率、logits的变化。
    """

    def __init__(self, figsize=(12, 6), legend_labels=None):
        self.num_subplots = 6
        
        # 默认图例标签
        default_legends = [
            ['loss'],  # train loss
            ['Down', 'Abstain', 'Up'],
            ['Precision', 'Severe'],
            ['loss'],  # test loss
            ['Down', 'Abstain', 'Up'],
            ['Precision', 'Severe']
        ]
        
        # 使用提供的图例标签或默认值
        self.legend_labels = legend_labels if legend_labels is not None else default_legends
        
        # 确保图例标签数量与子图数量一致
        if len(self.legend_labels) != self.num_subplots:
            raise ValueError(f"legend_labels must have {self.num_subplots} elements")
            
        # 不立即显示图形
        self.fig, self.axes = plt.subplots(2, 3, figsize=figsize)
        plt.close(self.fig)  # 关闭图形，避免立即显示
        self.axes = self.axes.flatten()
        
        # 初始化数据存储
        self.reset()
        
        # 设置标题和网格（但不绘制）
        titles = ['train loss', 'train classes ratio', 'Precision and Severe', 'test loss', 'test classes ratio', 'Precision and Severe']
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


    def draw(self):
        """绘制子图"""
        display.clear_output(wait=True)
        
        # 重新创建图形（避免关闭后无法绘制）
        if not plt.fignum_exists(self.fig.number):
            self.fig, self.axes = plt.subplots(2, 3, figsize=self.fig.get_size_inches())
            self.axes = self.axes.flatten()
            titles = ['train loss', 'train classes prob', 'Precision and Severe', 'test loss', 'test classes prob', 'Precision and Severe']
            for i, ax in enumerate(self.axes):
                ax.set_title(titles[i])
                ax.grid()
            self.fig.tight_layout()
        
        for i, ax in enumerate(self.axes):
            ax.clear()  # 使用clear而不是cla，保留标题等设置
            plot_data = self.data[i]
            
            # 重新设置标题和网格（因为clear会清除它们）
            titles = ['train loss', 'train classes prob', 'Precision and Severe', 'test loss', 'test classes prob', 'Precision and Severe']
            ax.set_title(titles[i])
            ax.grid()
            
            if plot_data['X']:
                fmts = ('-', 'm--', 'g-.', 'r:')
                lines = []  # 存储线条对象用于图例
                labels = []  # 存储标签
                
                for j in range(len(plot_data['X'])):
                    if plot_data['X'][j] and plot_data['Y'][j]:  # 确保有数据
                        # 只绘制实际有数据的线条
                        line, = ax.plot(plot_data['X'][j], plot_data['Y'][j], fmts[j % len(fmts)])
                        lines.append(line)
                        
                        # 使用预设的图例标签，但不超过实际数据线条数量
                        if j < len(self.legend_labels[i]):
                            labels.append(self.legend_labels[i][j])
                
                # 如果有线条，添加图例
                if lines and labels:
                    ax.legend(lines, labels, loc='best')
        
        self.fig.tight_layout()
        display.display(self.fig)

    def reset(self):
        """清空数据"""
        self.data = [{'X': [], 'Y': []} for _ in range(self.num_subplots)]
        print("Animator data has been reset.")
        
    def set_legend_labels(self, legend_labels):
        """设置图例标签"""
        if len(legend_labels) != self.num_subplots:
            raise ValueError(f"legend_labels must have {self.num_subplots} elements")
        self.legend_labels = legend_labels