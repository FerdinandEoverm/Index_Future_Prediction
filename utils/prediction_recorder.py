import numpy as np
import pandas as pd
import torch
from sklearn.metrics import  confusion_matrix

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