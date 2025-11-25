import numpy as np
import yfinance as yf
import pandas as pd
from market_loader import MarketLoader
from fourier_optics import FourierOptics
from quantum_cortex import QuantumCortex
from adversarial_search import AdversarialGame
from portfolio_manager import PortfolioManager
from hmm_adapter import MarketHMM

# === THE ARENA ===
TEST_TICKERS = ['NVDA', 'JPM', 'XOM', 'JNJ', 'COST', 'TSLA', 'GOOGL', 'GLD']

# === SPECIALIST DNA (Derived from your Monte Carlo Optimization) ===
# Each stock gets its own unique Hyperparameters
SPECIALIST_DNA = {
    'NVDA': {'learning_rate': 0.08, 'lookback_window': 30, 'buy_threshold': 20, 'sell_threshold': -20},
    'JPM':  {'learning_rate': 0.06, 'lookback_window': 30, 'buy_threshold': 15, 'sell_threshold': -15},
    'XOM':  {'learning_rate': 0.10, 'lookback_window': 45, 'buy_threshold': 5,  'sell_threshold': -5},  # Low thresh for boring stock
    'JNJ':  {'learning_rate': 0.11, 'lookback_window': 45, 'buy_threshold': 8,  'sell_threshold': -8},
    'COST': {'learning_rate': 0.14, 'lookback_window': 30, 'buy_threshold': 12, 'sell_threshold': -12},
    'TSLA': {'learning_rate': 0.15, 'lookback_window': 45, 'buy_threshold': 25, 'sell_threshold': -25},
    'GOOGL':{'learning_rate': 0.13, 'lookback_window': 30, 'buy_threshold': 18, 'sell_threshold': -18},
    'GLD':  {'learning_rate': 0.11, 'lookback_window': 30, 'buy_threshold': 10, 'sell_threshold': -10}
}

# Default Physics (Shared across all specialists unless overridden)
DEFAULT_PHYSICS = {
    'initial_capital': 10000,
    'label_threshold': 0.015,
    'phase_flexibility': 0.10,
    'lateral_strength': 0.20,
    'input_threshold': 0.75,
    'kerr_constant': 0.20,
    'system_energy': 50.0,
    'search_depth': 3
}

def run_head_to_head():
    print(f"{'='*70}")
    print(f"{'QUANTUM CORTEX (SPECIALIST)  vs.  HIDDEN MARKOV MODEL (HMM)':^70}")
    print(f"{'='*70}\n")
    
    results = []
    
    for ticker in TEST_TICKERS:
        print(f"matchup: {ticker}...", end=" ", flush=True)
        
        # 1. CONSTRUCT DNA FOR THIS TICKER
        specialist_dna = DEFAULT_PHYSICS.copy()
        # Update with specific tuning
        specialist_dna.update(SPECIALIST_DNA.get(ticker, {}))
        
        # 2. FETCH DATA
        loader = MarketLoader([ticker], 
                              lookback_window=specialist_dna['lookback_window'], 
                              label_threshold=specialist_dna['label_threshold'])
        try:
            X_data, y_data, prices = loader.fetch_data()
            prices_clean = np.where(prices <= 0, 1e-8, prices)
            raw_returns = np.diff(np.log(prices_clean))
        except Exception as e:
            print(f"Data Error: {e}")
            continue

        # === PLAYER 1: QUANTUM CORTEX (SPECIALIST) ===
        q_pm = PortfolioManager(10000)
        scout = QuantumCortex(3136, 3, 15, config=specialist_dna)
        optics = FourierOptics()
        game = AdversarialGame(scout)
        
        for i, (market_img, label) in enumerate(zip(X_data, y_data)):
            img_2d = market_img.reshape(28, 28)
            features = optics.apply(img_2d)
            signal = game.get_best_move(features, 
                                        buy_thresh=specialist_dna['buy_threshold'],
                                        sell_thresh=specialist_dna['sell_threshold'])
            
            q_pm.execute(signal, ticker, prices[i])
            scout.process_image(features, label, train=True)
            
        q_final = q_pm.get_total_value({ticker: prices[-1]})
        q_ret = ((q_final - 10000) / 10000) * 100

        # === PLAYER 2: HIDDEN MARKOV MODEL (BASELINE) ===
        h_pm = PortfolioManager(10000)
        hmm_brain = MarketHMM(train_window=100)
        
        # HMM Loop
        # We must align start times based on the specific lookback window of this specialist
        start_day_idx = specialist_dna['lookback_window']
        
        for i in range(len(X_data)):
            current_day_idx = i + start_day_idx
            
            if current_day_idx < 100: continue
            
            window = raw_returns[current_day_idx-100 : current_day_idx]
            signal = hmm_brain.get_signal(window)
            h_pm.execute(signal, ticker, prices[i])
            
        h_final = h_pm.get_total_value({ticker: prices[-1]})
        h_ret = ((h_final - 10000) / 10000) * 100
        
        # === RECORD RESULT ===
        winner = "QUANTUM" if q_ret > h_ret else "HMM"
        print(f"Winner: {winner}")
        
        results.append({
            'ticker': ticker,
            'quantum': q_ret,
            'hmm': h_ret,
            'diff': q_ret - h_ret
        })

    # === FINAL SCORECARD ===
    print(f"\n{'='*70}")
    print(f"{'FINAL SCORECARD':^70}")
    print(f"{'='*70}")
    print(f"{'TICKER':<10} | {'QUANTUM %':<12} | {'HMM %':<12} | {'EDGE':<10}")
    print("-" * 70)
    
    q_avg = 0
    h_avg = 0
    
    for res in results:
        print(f"{res['ticker']:<10} | {res['quantum']:>9.2f}%   | {res['hmm']:>9.2f}%   | {res['diff']:>+8.2f}%")
        q_avg += res['quantum']
        h_avg += res['hmm']
        
    print("-" * 70)
    q_final_avg = q_avg/len(results)
    h_final_avg = h_avg/len(results)
    print(f"{'AVERAGE':<10} | {q_final_avg:>9.2f}%   | {h_final_avg:>9.2f}%   | {(q_final_avg-h_final_avg):>+8.2f}%")
    print(f"{'='*70}")

if __name__ == "__main__":
    run_head_to_head()