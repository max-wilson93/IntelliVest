import yfinance as yf
import numpy as np
from scipy import signal
from skimage.transform import resize
import pandas as pd

class MarketLoader:
    def __init__(self, tickers, lookback_window=60, label_threshold=0.01):
        self.tickers = tickers
        self.lookback = lookback_window
        self.threshold = label_threshold

    def fetch_data(self):
        """
        Downloads data and returns:
        1. images (Spectrograms for the AI)
        2. labels (Truth for training)
        3. prices (Real closing prices for the simulation execution)
        """
        print(f"Fetching data for: {self.tickers}...")
        
        # Download data
        data = yf.download(self.tickers, period="2y", interval="1d", progress=False, auto_adjust=True)
        
        if data is None or data.empty:
            raise ValueError("yfinance returned no data.")

        # Handle Data Structures safely
        try:
            close_data = data['Close']
        except KeyError:
            close_data = data

        if isinstance(close_data, pd.DataFrame) and len(self.tickers) > 1:
            prices_series = close_data[self.tickers[0]]
        elif isinstance(close_data, pd.DataFrame) and self.tickers[0] in close_data.columns:
             prices_series = close_data[self.tickers[0]]
        else:
            prices_series = close_data

        if hasattr(prices_series, 'to_numpy'):
            prices = prices_series.to_numpy()
        elif hasattr(prices_series, 'values'):
            prices = prices_series.values
        else:
            prices = np.array(prices_series)

        prices = prices.flatten()
        prices = prices[~np.isnan(prices)]
        
        # Calculate Returns
        prices_clean = np.where(prices <= 0, 1e-8, prices) 
        returns = np.diff(np.log(prices_clean)) 
        
        images = []
        labels = []
        sim_prices = [] # The price at the moment of decision
        
        # Sliding Window
        for i in range(len(returns) - self.lookback - 5):
            window = returns[i : i + self.lookback]
            future_return = np.sum(returns[i + self.lookback : i + self.lookback + 5])
            
            # Capture the price at the END of this lookback window (Decision Time)
            current_sim_price = prices[i + self.lookback]
            
            # --- SPECTROGRAM ---
            f, t, Sxx = signal.spectrogram(window, fs=1.0, nperseg=15, noverlap=10)
            Sxx = np.log(Sxx + 1e-10)
            s_min, s_max = np.min(Sxx), np.max(Sxx)
            if s_max - s_min == 0: Sxx = np.zeros_like(Sxx)
            else: Sxx = (Sxx - s_min) / (s_max - s_min)
            
            img_2d = resize(Sxx, (28, 28), mode='reflect', anti_aliasing=True)
            img_flat = np.asarray(img_2d).flatten()
            
            # --- LABELS (Dynamic Threshold) ---
            if future_return > self.threshold:   label = 2 # Bull
            elif future_return < -self.threshold: label = 0 # Bear
            else:                                 label = 1 # Neutral
                
            images.append(img_flat)
            labels.append(label)
            sim_prices.append(current_sim_price)
            
        return np.array(images), np.array(labels), np.array(sim_prices)