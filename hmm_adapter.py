import numpy as np
from hmmlearn import hmm
import warnings

# Suppress specific hmmlearn warnings about convergence and initialization
warnings.filterwarnings("ignore")

class MarketHMM:
    def __init__(self, n_states=3, train_window=100):
        """
        n_states=3 corresponds to Bear (0), Neutral (1), Bull (2)
        train_window: How many past days the HMM sees to define the current regime.
        """
        self.n_states = n_states
        self.train_window = train_window
        
    def get_signal(self, recent_returns):
        """
        1. Trains HMM on recent history (rolling window).
        2. Identifies the current hidden state.
        3. Maps that state to BUY/SELL based on its historical mean return.
        """
        if len(recent_returns) < 50:
            return 'HOLD' # Not enough data
            
        # 1. Prepare Data
        X = recent_returns.reshape(-1, 1)
        
        # 2. Initialize FRESH Model (Prevents "overwrite" warnings)
        # We create a new instance for every window to ensure no memory leaks
        model = hmm.GaussianHMM(n_components=self.n_states, 
                                     covariance_type="full", 
                                     n_iter=100,
                                     random_state=42,
                                     init_params='stmc') # Explicitly init everything
        
        # 3. Fit Model
        try:
            model.fit(X)
        except:
            return 'HOLD' # Convergence failure fallback
            
        # 4. Decode States
        # We need to know which state is "Bull" and which is "Bear".
        # HMM labels (0,1,2) are random. We sort them by their Mean Return.
        means = model.means_.flatten()
        sorted_indices = np.argsort(means)
        
        # Map: Lowest Mean = SELL, Highest Mean = BUY, Middle = HOLD
        state_map = {
            sorted_indices[0]: 'SELL', # Bear Regime
            sorted_indices[1]: 'HOLD', # Neutral/Choppy
            sorted_indices[2]: 'BUY'   # Bull Regime
        }
        
        # 5. Predict Current Regime
        try:
            current_state = model.predict(X)[-1]
            return state_map[current_state]
        except:
            return 'HOLD'