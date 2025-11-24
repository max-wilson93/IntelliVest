class PortfolioManager:
    def __init__(self, initial_capital=10000):
        self.cash = initial_capital
        self.holdings = {} # e.g. { 'NVDA': 10 }
        self.history = []

    def execute(self, action, ticker, current_price):
        if action == 'BUY':
            # Buy 1 share if we have cash
            if self.cash >= current_price:
                self.cash -= current_price
                self.holdings[ticker] = self.holdings.get(ticker, 0) + 1
                self.history.append(f"BUY {ticker} @ ${current_price:.2f}")
                print(f" -> EXECUTE: Bought 1 {ticker} at ${current_price:.2f}")
            else:
                print(f" -> EXECUTE: Insufficient funds to buy {ticker}")
                
        elif action == 'SELL':
            # Sell 1 share if we have it
            if self.holdings.get(ticker, 0) > 0:
                self.holdings[ticker] -= 1
                self.cash += current_price
                self.history.append(f"SELL {ticker} @ ${current_price:.2f}")
                print(f" -> EXECUTE: Sold 1 {ticker} at ${current_price:.2f}")
            else:
                print(f" -> EXECUTE: No holdings to sell for {ticker}")
        
        else:
            print(" -> EXECUTE: Hold")

    def get_total_value(self, current_prices):
        """
        Calculate total Net Worth (Cash + Stock Value)
        """
        stock_val = sum(amt * current_prices.get(t, 0) for t, amt in self.holdings.items())
        return self.cash + stock_val