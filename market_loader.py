import yfinance as yf
import numpy as np
from scipy import signal
from skimage.transform import resize
import pandas as pd

class MarketLoader:
    def __init__(self, tickers, lookback_window=60):
        self.tickers = tickers
        self.lookback = lookback_window

    def fetch_data(self):
        """
        Downloads data and generates 'Images' (Spectrograms) and Labels.
        """
        print(f"Fetching data for: {self.tickers}...")
        
        # Download data
        # auto_adjust=True fixes the Future Warning
        data = yf.download(self.tickers, period="2y", interval="1d", progress=False, auto_adjust=True)
        
        # --- FIX 1: Explicitly handle None or Empty data ---
        if data is None or data.empty:
            raise ValueError("yfinance returned no data. Check your internet connection or ticker symbols.")

        # --- FIX 2: Handle 1D vs 2D Data Structures safely ---
        # Access the 'Close' column
        try:
            close_data = data['Close']
        except KeyError:
            # Fallback for some yfinance versions that return simple DataFrames
            close_data = data

        # Check if we need to select a specific ticker column
        if isinstance(close_data, pd.DataFrame) and len(self.tickers) > 1:
            prices_series = close_data[self.tickers[0]]
        elif isinstance(close_data, pd.DataFrame) and self.tickers[0] in close_data.columns:
             prices_series = close_data[self.tickers[0]]
        else:
            # It's likely already a Series or a single-column DataFrame
            prices_series = close_data

        # --- FIX 3: Convert Pandas Object -> NumPy Array explicitly ---
        # This fixes the "Attribute 'flatten' is unknown" error
        if hasattr(prices_series, 'to_numpy'):
            prices = prices_series.to_numpy()
        elif hasattr(prices_series, 'values'):
            prices = prices_series.values
        else:
            prices = np.array(prices_series)

        # Now .flatten() is guaranteed to work
        prices = prices.flatten()
        
        # Clean NaNs (often present at start of downloads)
        prices = prices[~np.isnan(prices)]
        
        if len(prices) < self.lookback + 10:
            raise ValueError(f"Not enough valid data fetched for {self.tickers}. Got {len(prices)} points.")

        # Calculate Returns
        # Add small epsilon to avoid log(0)
        prices = np.where(prices <= 0, 1e-8, prices) 
        returns = np.diff(np.log(prices)) 
        
        images = []
        labels = []
        
        # Sliding Window
        for i in range(len(returns) - self.lookback - 5):
            window = returns[i : i + self.lookback]
            future_return = np.sum(returns[i + self.lookback : i + self.lookback + 5])
            
            # --- CREATE SPECTROGRAM ---
            # Now 'window' is strictly 1D numpy array, so this works
            f, t, Sxx = signal.spectrogram(window, fs=1.0, nperseg=15, noverlap=10)
            
            # Normalize
            Sxx = np.log(Sxx + 1e-10)
            
            # Min-Max Scale (Safe check for constant input)
            s_min, s_max = np.min(Sxx), np.max(Sxx)
            if s_max - s_min == 0:
                Sxx = np.zeros_like(Sxx)
            else:
                Sxx = (Sxx - s_min) / (s_max - s_min)
            
            # Resize to 28x28 for the Quantum Cortex
            img_2d = resize(Sxx, (28, 28), mode='reflect', anti_aliasing=True)
            
            # --- FIX 4: Explicit cast for appending to list ---
            # This helps if Pylance is confused about skimage return types
            img_flat = np.asarray(img_2d).flatten()
            
            # --- CREATE LABELS ---
            if future_return > 0.01:   label = 2 # Bull
            elif future_return < -0.01: label = 0 # Bear
            else:                       label = 1 # Neutral
                
            images.append(img_flat)
            labels.append(label)
            
        return np.array(images), np.array(labels)