import numpy as np
import pandas as pd
from scipy.stats import norm
import tushare as ts
pro = ts.pro_api('700c1d6015ad030ff20bf310c088243da030e6b79a2a1098d58d2614')

class GetOHLCV():
    def __init__(self):
        pass


    def get_data(self, assets_code, pred_len, threshold_ratio, std_type = 'ma'):
        data_1 = pro.fut_daily(ts_code = assets_code, start_date = '20110101', end_date = '20180101')
        data_2 = pro.fut_daily(ts_code = assets_code, start_date = '20180101')

        data = pd.concat([data_1, data_2], ignore_index = True)

        data['oi_chg'] = 1
        data.dropna(inplace=True)
        data.sort_values(by = 'trade_date', inplace = True)
        data['log_open'] = np.log(data['open'] / data['pre_close']) * 100 # 标准化为对数百分比（不含百分号）
        data['log_high'] = np.log(data['high'] / data['pre_close']) * 100 
        data['log_low'] = np.log(data['low'] / data['pre_close']) * 100 
        data['log_close'] = np.log(data['close'] / data['pre_close']) * 100
        data['log_amount'] = np.log(data['amount'] / data['amount'].shift(1)) * 10

        data['label_return'] = data['log_close'].rolling(window = pred_len).sum().shift(-pred_len) # 标准化为对数百分比（不含百分号），可以直接相加

        data['ma_amount'] = data['amount'].rolling(window = 250).mean() # 过去一年的成交量均值
        data['ma_return_std'] = data['label_return'].rolling(window = 250).std()# 过去一年的收益标准差

        data['label_pred_high'] = data['high'].rolling(window = pred_len).max().shift(-pred_len)
        data['label_pred_low'] = data['low'].rolling(window = pred_len).min().shift(-pred_len)

        data['label_amplitude'] = data['label_pred_high'] - data['label_pred_low']
        data['label_amplitude_ma'] = data['label_amplitude'].rolling(window = 250).mean()

        if std_type == 'ma':
            # 固定标准差，稳定训练
            data['label_std'] =  data['ma_return_std'] 
        elif std_type == 'amount':
            # 根据当前成交量和历史成交量，估计当前隐含的标准差 由于用1年滚动，避免数据泄露
            data['label_std'] = data['amount'].rolling(window = pred_len).mean().shift(-pred_len)/ data['ma_amount'] * data['ma_return_std'] 
        elif std_type == 'amplitude':
            data['label_std'] = data['label_amplitude'] / data['label_amplitude_ma'] * data['ma_return_std']

        else:
            raise ValueError('unrecognized std type')


        data['upper_bond'] = data['label_return'].rolling(window = 250).quantile(1 - threshold_ratio) # 过去一年的收益下分位数
        data['lower_bond'] = data['label_return'].rolling(window = 250).quantile(threshold_ratio) # 过去一年的收益上分位数
        data['threshold'] = (abs(data['upper_bond']) + abs(data['lower_bond']))/2 # 过去一年的收益的分割阈值

        def down_probability(row):
            return norm.cdf(-row['threshold'], loc = row['label_return'], scale=row['label_std'])

        def middle_probability(row):
            return norm.cdf(row['threshold'], loc = row['label_return'], scale=row['label_std']) - norm.cdf(-row['threshold'], loc = row['label_return'], scale=row['label_std'])

        def up_probability(row):
            return 1 - norm.cdf(row['threshold'], loc = row['label_return'], scale=row['label_std'])
        
        data['down_prob'] = data.apply(down_probability, axis = 1)
        data['middle_prob'] = data.apply(middle_probability, axis = 1)
        data['up_prob'] = data.apply(up_probability, axis = 1)
        
        data.dropna(inplace=True)

        return data
