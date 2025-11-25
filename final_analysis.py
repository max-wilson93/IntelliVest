import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import json
import os
from main_intellivest import run_simulation

# Standard Physics Defaults
DEFAULT_PHYSICS = {
    'initial_capital': 10000,
    'phase_flexibility': 0.10,
    'lateral_strength': 0.20,
    'input_threshold': 0.75,
    'kerr_constant': 0.20,
    'system_energy': 50.0,
    'search_depth': 3
}

def load_specialist_dna():
    """Loads optimized parameters from JSON or falls back to hardcoded defaults."""
    if os.path.exists('specialist_dna.json'):
        print("Loading optimized DNA from 'specialist_dna.json'...")
        with open('specialist_dna.json', 'r') as f:
            return json.load(f)
    else:
        print("WARNING: 'specialist_dna.json' not found. Using hardcoded defaults.")
        # Fallback (Old hardcoded values)
        return {
            'NVDA': {'learning_rate': 0.08, 'lookback_window': 30, 'buy_threshold': 20, 'sell_threshold': -20, 'label_threshold': 0.015},
            'JPM':  {'learning_rate': 0.06, 'lookback_window': 30, 'buy_threshold': 15, 'sell_threshold': -15, 'label_threshold': 0.015},
            # ... (rest of your old list if needed as backup)
        }

def run_final_analysis():
    print(f"{'='*80}")
    print(f"{'INTELLIVEST FINAL ANALYSIS DASHBOARD':^80}")
    print(f"{'='*80}\n")
    
    SPECIALIST_DNA = load_specialist_dna()
    
    summary_data = []
    equity_curves = {} 
    
    # Sort tickers to ensure consistent order
    sorted_tickers = sorted(SPECIALIST_DNA.keys())
    
    for ticker in sorted_tickers:
        dna = SPECIALIST_DNA[ticker]
        print(f"Analyzing {ticker}...", end=" ", flush=True)
        
        # 1. Build Config
        config = DEFAULT_PHYSICS.copy()
        config.update(dna)
        config['ticker'] = ticker
        
        # 2. Run Simulation
        res = run_simulation(config)
        
        # 3. Get Benchmark Data (Buy & Hold)
        try:
            data = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
            close = data['Close'] if 'Close' in data else data
            if hasattr(close, 'values'): vals = close.values.flatten()
            else: vals = np.array(close).flatten()
            vals = vals[~np.isnan(vals)]
            
            sim_len = len(res['equity_curve'])
            bench_vals = vals[-sim_len:] 
            bench_ret = ((bench_vals[-1] - bench_vals[0]) / bench_vals[0]) * 100
            bench_curve = (bench_vals / bench_vals[0]) * 10000
            
        except:
            bench_ret = 0.0
            bench_curve = []

        print("Done.")
        
        # 4. Record Stats
        summary_data.append({
            'Ticker': ticker,
            'Model Return': f"{res['return_pct']:.2f}%",
            'Market Return': f"{bench_ret:.2f}%",
            'Alpha': f"{res['return_pct'] - bench_ret:.2f}%",
            'Sharpe': f"{res['sharpe']:.2f}",
            'Max DD': res['max_drawdown'],
            'Win Rate': res['win_rate']
        })
        
        equity_curves[ticker] = {
            'Model': res['equity_curve'],
            'BuyHold': bench_curve
        }

    # === PRINT TABLE ===
    df = pd.DataFrame(summary_data)
    print("\n=== PERFORMANCE SUMMARY ===")
    print(df.to_string(index=False))
    
    # === GENERATE PLOTS ===
    print("\nGenerating 'comparison_chart.png'...")
    
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    axes = axes.flatten()
    
    for i, ticker in enumerate(sorted_tickers):
        if i >= len(axes): break
        ax = axes[i]
        curves = equity_curves[ticker]
        
        ax.plot(curves['Model'], label='IntelliVest', color='blue', linewidth=2)
        if len(curves['BuyHold']) > 0:
            ax.plot(curves['BuyHold'], label='Buy & Hold', color='gray', linestyle='--', alpha=0.7)
            
        ax.set_title(f"{ticker} Performance")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Portfolio Value ($)")
    
    plt.tight_layout()
    plt.savefig('comparison_chart.png')
    print("Chart saved successfully.")

if __name__ == "__main__":
    run_final_analysis()