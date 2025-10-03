import numpy as np
import pandas as pd
import torch
from sklearn.metrics import  confusion_matrix
import matplotlib.pyplot as plt
from IPython import display

class PredictionRecorder:
    """
    记录和分析预测结果的类。
    """

    def __init__(self):
        self.records = pd.DataFrame(columns=[
            'pred_value', 'pred_neg', 'pred_abstain', 'pred_pos', 'pred_class', 'pred_direction',
            'real_value', 'real_neg', 'real_abstain', 'real_pos', 'real_class', 'real_direction',
        ])

    def add(self, pred: torch.Tensor, real: torch.Tensor):
        pred = pred.reshape(-1,4)
        real = real.reshape(-1,4)
        if pred.shape[0] != real.shape[0]:
            raise ValueError("预测张量和真实值张量的batch_size必须相同。")
        if pred.dim() != 2 or pred.shape[1] != 4:
            raise ValueError("预测张量的形状必须是 (batch_size, 4)。")
        if real.dim() != 2 or real.shape[1] != 4:
            raise ValueError("真实值张量的形状必须是 (batch_size, 4)。")

        pred_numpy = pred.cpu().detach().numpy()
        real_numpy = real.cpu().detach().numpy()

        pred_class = torch.argmax(pred[:,1:], dim=1).cpu().detach().numpy()
        real_class = torch.argmax(real[:,1:], dim=1).cpu().detach().numpy()

        pred_direction  = pred_numpy[:, 0] > 0
        real_direction  = real_numpy[:, 0] > 0

        new_records = pd.DataFrame({
            'pred_value'    : pred_numpy[:, 0],
            'pred_neg'      : pred_numpy[:, 1],
            'pred_abstain'  : pred_numpy[:, 2],
            'pred_pos'      : pred_numpy[:, 3],
            'pred_class'    : pred_class,
            'pred_direction': pred_direction,

            'real_value'    : real_numpy[:, 0],
            'real_neg'      : real_numpy[:, 1],
            'real_abstain'  : real_numpy[:, 2],
            'real_pos'      : real_numpy[:, 3],
            'real_class'    : real_class,
            'real_direction': real_direction,
        })
        self.records = pd.concat([self.records, new_records], ignore_index=True)

    def clear(self):
        self.__init__()

    def summary(self) -> pd.DataFrame:
        """
        Generates and prints a detailed classification performance summary DataFrame.
        """
        if self.records.empty:
            print("记录为空，无法生成摘要。")
            return pd.DataFrame()

        cm = confusion_matrix(self.records['real_class'].astype(int), self.records['pred_class'].astype(int), labels = [0,1,2])

        results = []
        for i in range(3):
            tp = cm[i, i]
            predicted_count = cm[:, i].sum()
            true_count = cm[i, :].sum()

            predicted_ratio = predicted_count / cm.sum()
            true_ratio = true_count / cm.sum()

            precision = tp / predicted_count if predicted_count > 0 else 0
            recall = tp / true_count if true_count > 0 else 0

            severe_error = 0
            if i == 0:
                severe_error = cm[2, 0] / predicted_count if predicted_count > 0 else 0
            elif i == 2:
                severe_error = cm[0, 2] / predicted_count if predicted_count > 0 else 0
            
            results.append({
                'Prediction Ratio': predicted_ratio,
                'Precision: Right/Pred': precision,
                'Severe: Wrong/Pred': severe_error,
                'Real Ratio': true_ratio,
                'Accuracy: Right/Real': recall,
            })

        total_samples = cm.sum()

        radical_prediction = cm[:, 0].sum() + cm[:,2].sum()
        radical_real = cm[0, :].sum() + cm[2,:].sum()

        radical_correct = cm[0, 0] + cm[2, 2]
        radical_wrong = cm[2, 0] + cm[0, 2]


        results.append({
            'Prediction Ratio': radical_prediction / total_samples if total_samples > 0 else 0.0,
            'Precision: Right/Pred': radical_correct / radical_prediction if radical_prediction > 0 else 0.0,
            'Severe: Wrong/Pred': radical_wrong / radical_prediction if radical_prediction > 0 else 0.0,
            
            'Real Ratio': radical_real / total_samples if total_samples > 0 else 0.0,
            'Accuracy: Right/Real': radical_correct / radical_real if radical_real > 0 else 0.0,
        })
        summary_df = pd.DataFrame(results, index=['negative', 'neutral', 'positive', 'neg + pos'])
        return summary_df
    



class TrainMonitor:
    """
    在动画中绘制数据，用于在模型训练中动态监控损失、预测概率、logits的变化。
    """

    def __init__(self, figsize=(12, 6)):

        self.data = [{'X': [], 'Y': []} for _ in range(6)]

        self.legend_labels = [
            ['loss'],  # train loss
            ['Down', 'Abstain', 'Up'],
            ['Precision', 'Severe'],
            ['loss'],  # test loss
            ['Down', 'Abstain', 'Up'],
            ['Precision', 'Severe']
        ]
        
        self.fig, self.axes = plt.subplots(2, 3, figsize=figsize)
        plt.close(self.fig)
        self.axes = self.axes.flatten()
        titles = ['Train Loss', 'Class Ratio', 'Precision and Severe', 'Test Loss', 'Class Ratio', 'Precision and Severe']
        for i, ax in enumerate(self.axes):
            ax.set_title(titles[i])
            ax.grid()
        self.fig.tight_layout()


    def add(self, x, y, subplot_idx=0):
        """
        向指定的子图添加数据点。
        参数:
            x : 当前epoch
            y : 记录的值，同组的多个记录传入列表或元组
            subplot_idx: 子图的编号
        """

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
            titles = ['Train Loss', 'Class Ratio', 'Precision and Severe', 'Test Loss', 'Class Ratio', 'Precision and Severe']
            for i, ax in enumerate(self.axes):
                ax.set_title(titles[i])
                ax.grid()
            self.fig.tight_layout()
        
        for i, ax in enumerate(self.axes):
            ax.clear()  # 使用clear而不是cla，保留标题等设置
            plot_data = self.data[i]
            
            titles = ['Train Loss', 'Class Ratio', 'Precision and Severe', 'Test Loss', 'Class Ratio', 'Precision and Severe']
            ax.set_title(titles[i])
            ax.grid()
            
            if plot_data['X']:
                if i%3==0:
                    fmts = ('-',)
                elif i%3==1:
                    fmts = ('g-', 'm--', 'r-',)
                else :
                    fmts = ('-', 'r-',)
                
                lines = []
                labels = []
                
                for j in range(len(plot_data['X'])):
                    if plot_data['X'][j] and plot_data['Y'][j]:  
                        line, = ax.plot(plot_data['X'][j], plot_data['Y'][j], fmts[j % len(fmts)])
                        lines.append(line)
                        if j < len(self.legend_labels[i]):
                            labels.append(self.legend_labels[i][j])
                
                if lines and labels:
                    ax.legend(lines, labels, loc='best')
        
        self.fig.tight_layout()
        display.display(self.fig)

    def reset(self):
        self.data = [{'X': [], 'Y': []} for i in range(6)]
        print("record has been reset.")
