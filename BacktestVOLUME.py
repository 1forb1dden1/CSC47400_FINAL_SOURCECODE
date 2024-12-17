import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        try:
            data = pd.read_csv(self.file_path)

            for col in ['Close/Last', 'Open', 'High', 'Low', 'Percentage Change', 'Normalized Volume(scaled down)']:
                if col in data.columns and data[col].dtype == object:
                    data[col] = data[col].str.replace('$', '').str.replace(',', '').str.replace('%', '').astype(float)

            if 'Volume' in data.columns:
                data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce').fillna(0).astype(int)

            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
                data.set_index('Date', inplace=True)

            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None



class Strategy:
    def __init__(self):
        self.signals_df = pd.DataFrame(columns=['timestamp', 'percentage change'])

    def generate_percentage_change(self, data):
        if 'Close/Last' not in data.columns:
            raise ValueError("Data must contain a 'Close/Last' column.")
        self.signals_df = self.signals_df._append({'timestamp': data.index[0], 'percentage change': 0}, ignore_index=True)

        for i in range(1, len(data['Close/Last'])):
            previous_price = data['Close/Last'].iloc[i-1]
            current_price = data['Close/Last'].iloc[i]

            percentage_change = ((current_price - previous_price) / previous_price) * 100 if previous_price != 0 else 0

            self.signals_df = self.signals_df._append({
                'timestamp': data.index[i],
                'percentage change': percentage_change
            }, ignore_index=True)

        return self.signals_df

    def calculate_rsi(self, window=14):
        if len(self.signals_df) < window:
            raise ValueError("Not enough data points to calculate RSI.")
        
        self.signals_df['gain'] = self.signals_df['percentage change'].where(self.signals_df['percentage change'] > 0, 0)
        self.signals_df['loss'] = -self.signals_df['percentage change'].where(self.signals_df['percentage change'] < 0, 0)

        avg_gain = self.signals_df['gain'].rolling(window=window, min_periods=1).mean()
        avg_loss = self.signals_df['loss'].rolling(window=window, min_periods=1).mean()

        avg_loss = avg_loss.replace(0, 1e-10)
        
        rs = avg_gain / avg_loss

        self.signals_df['RSI'] = 100 - (100 / (1 + rs))
        return self.signals_df



class Backtest:
    def __init__(self, data, strategy, initial_balance):
        self.data = data
        self.strategy = strategy
        self.balance = initial_balance
        self.results = []
        self.position = None  # Tracks whether a position is open ('buy') or None
        self.shares = 0  # Tracks the number of shares owned
        self.net_worth_history = []
        self.buy_price = 0

    def handle_buy_order(self, volume, current_price, timestamp, net_worth):
        # Buying logic
        self.position = 'buy'
        self.shares = float(self.balance / current_price)
        self.balance -= self.shares * current_price
        self.balance = round(self.balance, 2)
        self.buy_price = current_price
        self.results.append((timestamp, 'Buy', volume, current_price, self.shares, self.balance, round(net_worth, 2)))

    def handle_sell_order(self, volume, current_price, timestamp, net_worth):
        # Selling logic
        self.position = None
        self.balance += self.shares * current_price
        self.balance = round(self.balance, 2)
        self.shares = 0
        self.buy_price = 0
        self.results.append((timestamp, 'Sell', volume, current_price, self.shares, self.balance, round(net_worth, 2)))

    def run(self, strategy):
        
        if strategy == "Volume":
            threshold = 91.80  # Volume threshold

            for i in range(len(self.data['Volume'])):
                volume = self.data['Volume'].iloc[i]/1000000
                current_price = self.data['Close/Last'].iloc[i]
                timestamp = self.data.index[i]

                # Calculate the current net worth based on shares held
                net_worth = self.balance + (self.shares * current_price)
                self.net_worth_history.append(net_worth)

                if volume < threshold and self.position is None and self.balance >= current_price:
                    # Buy when volume is below the threshold and there's enough balance
                    self.handle_buy_order(volume, current_price, timestamp, net_worth)

                elif volume > threshold and self.position == 'buy':
                    # Sell when volume is above the threshold and a position is open
                    self.handle_sell_order(volume, current_price, timestamp, net_worth)

            return self.results

    def plot_portfolio(self):
        self.net_worth_history.reverse()
        if len(self.net_worth_history) > len(self.data.index):
            self.net_worth_history = self.net_worth_history

        plot_item = self.net_worth_history


        buy_points = []
        buy_timestamps = []
        sell_points = []
        sell_timestamps = []

        for result in self.results:
            timestamp, action, volume, current_price, shares, balance, net_worth = result
            if action == 'Buy':
                buy_points.append(net_worth)
                buy_timestamps.append(timestamp)
            elif action == 'Sell':
                sell_points.append(net_worth)
                sell_timestamps.append(timestamp)
        
        plt.figure(figsize=(14, 8))
        plt.plot(self.data.index, plot_item, label='Portfolio Balance', color='Black')
        plt.xlabel('Date')
        plt.ylabel('Balance (USD)')
        plt.title('Portfolio Balance Over Time(Buy Volume < 91.80)')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


data_loader = DataLoader('SPY.csv')
data = data_loader.load_data()


if data is not None:
    strategy = Strategy()
    strategy.generate_percentage_change(data)
    strategy.calculate_rsi(window=14)

    backtest = Backtest(data, strategy, 10000)
    results = backtest.run("Volume")
    backtest.plot_portfolio()

    # Print results
    for result in results:
        print(result)