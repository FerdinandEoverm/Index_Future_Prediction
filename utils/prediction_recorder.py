import numpy as np
import pandas as pd
import torch
from sklearn.metrics import  confusion_matrix

class PredictionRecorder:
    """
    记录和分析预测结果的类。
    """

    def __init__(self, is_logits = True):
        self.records = pd.DataFrame(columns=[
            'pred_neg', 'pred_abstain', 'pred_pos',
            'logit_neg', 'logit_abstain', 'logit_pos',
            'real', 'predicted_class'
        ])
        self.is_logits = is_logits

    def add(self, predict: torch.Tensor, real: torch.Tensor):
        if predict.shape[0] != real.shape[0]:
            raise ValueError("预测张量和真实值张量的batch_size必须相同。")
        if predict.dim() != 2 or predict.shape[1] != 3:
            raise ValueError("预测张量的形状必须是 (batch_size, 3)。")
        if real.dim() != 2 or real.shape[1] != 1:
            raise ValueError("真实值张量的形状必须是 (batch_size, 1)。")

        if self.is_logits :
            prob = torch.softmax(predict, dim = 1).cpu().detach().numpy()
            logits = predict.cpu().detach().numpy()
        else:
            prob = predict.cpu().detach().numpy()
            logits = torch.log(predict + 1e-9).cpu().detach().numpy()
        
        predicted_class = torch.argmax(predict, dim=1).cpu().detach().numpy()

        new_records_df = pd.DataFrame({
            'pred_neg': prob[:, 0],
            'pred_abstain': prob[:, 1],
            'pred_pos': prob[:, 2],
            'logit_neg': logits[:, 0],
            'logit_abstain': logits[:, 1],
            'logit_pos': logits[:, 2],
            'real': real.squeeze().cpu().detach().numpy(),
            'predicted_class': predicted_class,
        })
        self.records = pd.concat([self.records, new_records_df], ignore_index=True)

    def clear(self):
        self.__init__()

    def summary(self, threshold: float = 0.0) -> pd.DataFrame:
        """
        Generates and prints a detailed classification performance summary DataFrame.
        """
        if self.records.empty:
            print("记录为空，无法生成摘要。")
            return pd.DataFrame()

        # 1. Classify 'real' values
        def classify_real(value):
            if value < -abs(threshold): return 0
            elif value > abs(threshold): return 2
            else: return 1

        y_true = self.records['real'].apply(classify_real)
        y_pred = self.records['predicted_class']


        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
    
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        

        results = []
        for i in range(3):
            tp = cm[i, i]
            predicted_count = cm[:, i].sum()
            true_count = cm[i, :].sum()
            precision = tp / predicted_count if predicted_count > 0 else 0
            recall = tp / true_count if true_count > 0 else 0
            severe_error = 0
            if i == 0:
                severe_error = cm[2, 0] / predicted_count if predicted_count > 0 else 0
            elif i == 2:
                severe_error = cm[0, 2] / predicted_count if predicted_count > 0 else 0
            results.append({
                '预测为该分类的个数': predicted_count,
                'Precision (精确率)': precision,
                '真实为该分类的个数': true_count,
                'Accuracy (召回率)': recall,
                'Severe (严重错误率)': severe_error
            })

        total_samples = cm.sum()
        total_correct = np.trace(cm)
        total_severe_errors = cm[2, 0] + cm[0, 2]
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
        overall_severe_rate = total_severe_errors / total_samples if total_samples > 0 else 0
        results.append({
            '预测为该分类的个数': total_samples,
            'Precision (精确率)': overall_accuracy,
            '真实为该分类的个数': total_samples,
            'Accuracy (召回率)': overall_accuracy,
            'Severe (严重错误率)': overall_severe_rate
        })
        summary_df = pd.DataFrame(results, index=['分类 0 (负)', '分类 1 (放弃)', '分类 2 (正)', '总计'])
        # print(f"--- 基于阈值 {threshold} 的分类性能摘要 ---")
        # print(summary_df.to_string(float_format="%.4f"))
        return summary_df

    def distribution(self) -> tuple[float, float, float]:
        if self.records.empty:
            return (0.0, 0.0, 0.0)
        props = self.records['predicted_class'].value_counts(normalize=True).reindex([0, 1, 2]).fillna(0)
        return (props[0], props[1], props[2])

    def average_score(self) -> tuple[float, float, float]:
        """
        计算三个分类的 logits 的全局平均值。
        tuple[float, float, float]: 分别代表 logit_neg, logit_abstain, logit_pos 的平均值。
        """
        if self.records.empty:
            return (0.0, 0.0, 0.0)

        # 选取 logits 相关的列
        logit_columns = ['logit_neg', 'logit_abstain', 'logit_pos']
        
        # 使用 .mean() 计算每列的平均值
        avg_logits = self.records[logit_columns].mean()

        return (avg_logits['logit_neg'], avg_logits['logit_abstain'], avg_logits['logit_pos'])