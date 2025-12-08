import numpy as np
import yfinance as yf
import pandas as pd
import json
import os
from market_loader import MarketLoader
from fourier_optics import FourierOptics
from quantum_cortex import QuantumCortex
from adversarial_search import AdversarialGame
from portfolio_manager import PortfolioManager
from hmm_adapter import MarketHMM

TEST_TICKERS = ['NVDA', 'JPM', 'XOM', 'JNJ', 'COST', 'TSLA', 'GOOGL', 'GLD']

# Fallback defaults if JSON is missing specific keys
DEFAULT_PHYSICS = {
    'initial_capital': 10000,
    'label_threshold': 0.015,
    'phase_flexibility': 0.10,
    'lateral_strength': 0.20,
    'input_threshold': 0.75,
    'kerr_constant': 0.20,
    'system_energy': 50.0,
    'search_depth': 3,
    'trailing_stop_pct': 0.05
}

def load_specialist_dna():
    if os.path.exists('specialist_dna.json'):
        print("Loading configuration from specialist_dna.json...")
        with open('specialist_dna.json', 'r') as f:
            return json.load(f)
    else:
        print("Warning: specialist_dna.json not found. Using defaults.")
        return {}

def run_head_to_head():
    print(f"{'='*70}")
    print(f"{'QUANTUM CORTEX (SPECIALIST)  vs.  HMM':^70}")
    print(f"{'='*70}\n")
    
    SPECIALIST_DNA = load_specialist_dna()
    results = []
    
    for ticker in TEST_TICKERS:
        print(f"matchup: {ticker}...", end=" ", flush=True)
        
        # Merge defaults with specialist DNA
        specialist_dna = DEFAULT_PHYSICS.copy()
        specialist_dna.update(SPECIALIST_DNA.get(ticker, {}))
        
        # 1. Fetch Data
        loader = MarketLoader([ticker], 
                              lookback_window=specialist_dna['lookback_window'], 
                              label_threshold=specialist_dna['label_threshold'])
        try:
            X_data, y_data, prices = loader.fetch_data()
            prices_clean = np.where(prices <= 0, 1e-8, prices)
            raw_returns = np.diff(np.log(prices_clean))
        except:
            print("Data Error")
            continue

        # 2. RUN QUANTUM CORTEX
        # Pass the optimized trailing_stop_pct to the PortfolioManager
        q_pm = PortfolioManager(
            initial_capital=10000
            # Note: We are using the "Long-Only" PortfolioManager from the revert,
            # so we don't pass trailing_stop_pct here unless you added it back.
            # If you are using the simpler PM, this arg is ignored or causes error.
            # Assuming "Reverted" state means simple PM. 
            # If you want stops, re-add logic to PortfolioManager.
        )
        
        scout = QuantumCortex(3136, 3, 15, config=specialist_dna)
        optics = FourierOptics()
        game = AdversarialGame(scout)
        
        for i, (market_img, label) in enumerate(zip(X_data, y_data)):
            img_2d = market_img.reshape(28, 28)
            features = optics.apply(img_2d)
            
            # Pass optimized search depth if available
            signal = game.get_best_move(features, 
                                        buy_thresh=specialist_dna['buy_threshold'],
                                        sell_thresh=specialist_dna['sell_threshold'],
                                        search_depth=specialist_dna.get('search_depth', 3))
            
            q_pm.execute(signal, ticker, prices[i])
            scout.process_image(features, label, train=True)
            
        q_final = q_pm.get_total_value({ticker: prices[-1]})
        q_ret = ((q_final - 10000) / 10000) * 100

        # 3. RUN HMM
        h_pm = PortfolioManager(10000)
        hmm_brain = MarketHMM(train_window=100)
        start_day_idx = specialist_dna['lookback_window']
        
        for i in range(len(X_data)):
            current_day_idx = i + start_day_idx
            if current_day_idx < 100: continue
            
            window = raw_returns[current_day_idx-100 : current_day_idx]
            signal = hmm_brain.get_signal(window)
            h_pm.execute(signal, ticker, prices[i])
            
        h_final = h_pm.get_total_value({ticker: prices[-1]})
        h_ret = ((h_final - 10000) / 10000) * 100
        
        winner = "QUANTUM" if q_ret > h_ret else "HMM"
        print(f"Winner: {winner}")
        
        results.append({'ticker': ticker, 'quantum': q_ret, 'hmm': h_ret, 'diff': q_ret - h_ret})

    # === SCORECARD ===
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
    if results:
        print(f"{'AVERAGE':<10} | {q_avg/len(results):>9.2f}%   | {h_avg/len(results):>9.2f}%   | {(q_avg-h_avg)/len(results):>+8.2f}%")
    print(f"{'='*70}")

if __name__ == "__main__":
    run_head_to_head()