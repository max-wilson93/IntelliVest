import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import json
import os
from main_intellivest import run_simulation
from statistical_analysis import StatisticalAnalyzer
from analytics_engine import AnalyticsEngine

# Default Physics Defaults
DEFAULT_PHYSICS = {
    'initial_capital': 10000,
    'phase_flexibility': 0.10,
    'lateral_strength': 0.20,
    'input_threshold': 0.75,
    'kerr_constant': 0.20,
    'system_energy': 50.0,
    'search_depth': 3,
    'trailing_stop_pct': 0.05
}

def load_specialist_dna():
    """Loads optimized parameters from JSON or falls back to hardcoded defaults."""
    if os.path.exists('specialist_dna.json'):
        print("Loading optimized DNA from 'specialist_dna.json'...")
        with open('specialist_dna.json', 'r') as f:
            return json.load(f)
    else:
        print("WARNING: 'specialist_dna.json' not found. Using defaults.")
        # Fallback to empty dicts so defaults are used
        return {
            'NVDA': {}, 'JPM': {}, 'XOM': {}, 'JNJ': {}, 
            'COST': {}, 'TSLA': {}, 'GOOGL': {}, 'GLD': {}
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
    if not sorted_tickers:
        sorted_tickers = ['NVDA'] # Default if JSON empty

    # Store aggregated daily equity for portfolio-level stats
    portfolio_equity = None
    benchmark_equity = None
    
    for ticker in sorted_tickers:
        dna = SPECIALIST_DNA.get(ticker, {})
        print(f"Analyzing {ticker}...", end=" ", flush=True)
        
        config = DEFAULT_PHYSICS.copy()
        config.update(dna)
        config['ticker'] = ticker
        
        # 1. Run Simulation
        res = run_simulation(config)
        # Note: In the reverted version, 'equity_curve' might not be in res if you used the simple main_intellivest.
        # But PortfolioManager has it. We need to ensure main_intellivest returns it.
        # I will assume main_intellivest.py returns 'equity_curve' as per the robust version request, 
        # or we might need to access it from the pm object if exposed.
        # Based on previous robust main_intellivest, it returns 'equity_curve'.
        
        # If your main_intellivest doesn't return equity_curve, this will fail.
        # Let's handle that gracefully or assume the user has the robust main_intellivest.
        if 'equity_curve' not in res:
             print("Error: 'equity_curve' missing from simulation result.")
             continue
             
        model_curve = res['equity_curve']
        
        # 2. Get Benchmark Data (Buy & Hold)
        try:
            data = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
            close = data['Close'] if 'Close' in data else data
            if hasattr(close, 'values'): vals = close.values.flatten()
            else: vals = np.array(close).flatten()
            vals = vals[~np.isnan(vals)]
            
            # Align lengths
            sim_len = len(model_curve)
            if len(vals) > sim_len:
                bench_vals = vals[-sim_len:]
            else:
                bench_vals = vals
            
            # Normalize benchmark to start at $10k
            if len(bench_vals) > 0:
                bench_curve = (bench_vals / bench_vals[0]) * 10000
                bench_ret = ((bench_vals[-1] - bench_vals[0]) / bench_vals[0]) * 100
            else:
                bench_curve = [10000] * sim_len
                bench_ret = 0.0
            
        except Exception as e:
            # print(f"Bench error: {e}")
            bench_ret = 0.0
            bench_curve = [10000] * len(model_curve)
        # Initialize the engine for THIS ticker
        engine = AnalyticsEngine(model_curve, bench_curve)

        # Generate the 2x2 dashboard and print the fancy stats
        engine.generate_full_report(ticker)
        print("Done.")
        
        summary_data.append({
            'Ticker': ticker,
            'Model Return': f"{res['return_pct']:.2f}%",
            'Market Return': f"{bench_ret:.2f}%",
            'Alpha': f"{res['return_pct'] - bench_ret:.2f}%",
            'Sharpe': f"{res.get('sharpe', 0):.2f}",
            'Max DD': res.get('max_drawdown', '0%'),
            'Win Rate': res.get('win_rate', '0%')
        })
        
        equity_curves[ticker] = {
            'Model': model_curve,
            'BuyHold': bench_curve
        }
        
        # Aggregate Portfolio Logic
        if portfolio_equity is None:
            portfolio_equity = np.array(model_curve)
            benchmark_equity = np.array(bench_curve)
        else:
            # Handle potential length mismatches by trimming to min
            min_len = min(len(portfolio_equity), len(model_curve))
            portfolio_equity = portfolio_equity[:min_len] + np.array(model_curve)[:min_len]
            benchmark_equity = benchmark_equity[:min_len] + np.array(bench_curve)[:min_len]

    # === PRINT TABLE ===
    if summary_data:
        df = pd.DataFrame(summary_data)
        print("\n=== PERFORMANCE SUMMARY ===")
        print(df.to_string(index=False))
    
    # === GENERATE PLOTS ===
    if equity_curves:
        print("\nGenerating 'comparison_chart.png'...")
        # Adjust subplot size based on number of tickers
        rows = (len(sorted_tickers) + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(15, 5 * rows))
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
        
        plt.tight_layout()
        plt.savefig('comparison_chart.png')
        print("Chart saved.")
        # 2. Run Advanced Analytics

    
    # === RUN STATISTICAL ANALYSIS ===
    # We analyze the aggregate portfolio vs aggregate benchmark
    if portfolio_equity is not None:
        analyzer = StatisticalAnalyzer(portfolio_equity, benchmark_equity)
        analyzer.run_full_analysis()

if __name__ == "__main__":
    run_final_analysis()