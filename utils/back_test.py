import numpy as np

class BackTest:
    def __init__(self, model, x_test, y_test, sliding_window, init_value):

        logits = model(x_test).detach().cpu().numpy()[:,1:]
        trading_directions = np.argmax(logits, axis=1) - 1
        sliding_profit = y_test.detach().cpu().numpy()[:,0]
        strategy_sliding_profit = sliding_profit * trading_directions
        daily_return = strategy_sliding_profit / sliding_window
        for i in range(1,sliding_window):
            daily_return += np.concatenate((np.zeros(shape = (i,)),(strategy_sliding_profit[:-i]/ sliding_window)))
        daily_return = daily_return / sliding_window / 100

        
        print(f'yearly return :{np.mean(daily_return)*250:.2%}')
        print(f'std           :{np.std(daily_return)*15.87:.2%}')
        print(f'sharpe ratio  :{(np.mean(daily_return)*250 - 0.03)/(np.std(daily_return)*15.87):.2f}')
