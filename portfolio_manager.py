import pandas as pd
import numpy as np

class PortfolioManager:
    def __init__(self, initial_capital=10000):
        self.cash = initial_capital
        self.holdings = {} # e.g. { 'NVDA': 10 }
        self.history = []
        self.equity_curve = []

    def get_total_value(self, current_prices):
        stock_val = 0
        for ticker, shares in self.holdings.items():
            price = current_prices.get(ticker, 0)
            stock_val += shares * price
        return self.cash + stock_val

    def execute(self, action, ticker, current_price):
        if action == 'BUY':
            if self.cash >= current_price:
                # Buy 1 unit (Conservative sizing for Monte Carlo stability)
                self.cash -= current_price
                self.holdings[ticker] = self.holdings.get(ticker, 0) + 1
                self.history.append(f"BUY {ticker} @ ${current_price:.2f}")

        elif action == 'SELL':
            if self.holdings.get(ticker, 0) > 0:
                self.holdings[ticker] -= 1
                self.cash += current_price
                self.history.append(f"SELL {ticker} @ ${current_price:.2f}")
        
        self.equity_curve.append(self.get_total_value({ticker: current_price}))

    def calculate_metrics(self):
        if not self.equity_curve or len(self.equity_curve) < 2:
            return {"Sharpe": 0.0, "Max Drawdown": "0.00%", "Win Rate": "0.00%"}
        
        curve = pd.Series(self.equity_curve)
        returns = curve.pct_change().dropna()
        
        excess_returns = returns - (0.04 / 252)
        if returns.std() == 0: sharpe = 0.0
        else: sharpe = np.sqrt(252) * excess_returns.mean() / returns.std()
        
        cumulative_max = curve.cummax()
        drawdown = (curve - cumulative_max) / cumulative_max
        max_dd = drawdown.min()
        
        wins = len(returns[returns > 0])
        total = len(returns)
        win_rate = (wins / total) * 100 if total > 0 else 0
            
        return {
            "Sharpe": float(sharpe),
            "Max Drawdown": f"{max_dd*100:.2f}%",
            "Win Rate": f"{win_rate:.2f}%"
        }