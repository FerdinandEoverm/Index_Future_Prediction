import numpy as np

class BackTest:
    def __init__(self, model, x_test, y_test):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
        self.test_size = len(x_test) // 2

    def test(self, rounds):
        profits_per_round = []

        for r in range(rounds):
            start_index = np.random.randint(0, self.test_size)
            end_index = start_index + self.test_size
            x_round = self.x_test[start_index:end_index]
            y_round = self.y_test[start_index:end_index]
            round_profit = self.calculate_round_profit(x_round, y_round)
            profits_per_round.append(round_profit)
        
        profits_per_round = np.array(profits_per_round)
        yearly_return = np.mean(profits_per_round)/self.test_size*250
        yearly_std = np.std(profits_per_round)/self.test_size*250
        sharpe_ratio = (yearly_return - 0.03)/yearly_std

        print(f'yearly_return:{yearly_return:.2%}')
        print(f'yearly_std:{yearly_std:.2%}') 
        print(f'sharpe_ratio:{sharpe_ratio:.2f}') 


    def calculate_round_profit(self, x_round, y_round):

        total_profit = 0
        logits = self.model(x_round)
        trading_directions = np.argmax(logits, axis=1)
        for i in range(len(trading_directions) - 1):
            direction = trading_directions[i]
            current_price = y_round[i]
            next_price = y_round[i+1]
            
            daily_return = (next_price - current_price) / current_price if current_price != 0 else 0

            if direction == 0:
                profit = -daily_return
            elif direction == 1:
                profit = 0
            elif direction == 2:
                profit = daily_return
            else:
                profit = 0
            
            total_profit += profit

        return total_profit