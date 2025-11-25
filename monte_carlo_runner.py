import random
import time
import numpy as np
import yfinance as yf
from main_intellivest import run_simulation
from experiment_logger import log_experiment_result

# === THE SPECIALIST SQUAD ===
TEST_TICKERS = [
    'NVDA', 'JPM', 'XOM', 'JNJ', 
    'COST', 'TSLA', 'GOOGL', 'GLD'
]
BENCHMARK_TICKER = 'VTI'

def get_benchmark_return(period="2y"):
    """Get VTI return for comparison"""
    try:
        data = yf.download(BENCHMARK_TICKER, period=period, interval="1d", progress=False, auto_adjust=True)
        close = data['Close'] if 'Close' in data else data
        if hasattr(close, 'values'): vals = close.values.flatten()
        else: vals = np.array(close).flatten()
        
        vals = vals[~np.isnan(vals)]
        if len(vals) < 2: return 0.0
        return ((vals[-1] - vals[0]) / vals[0]) * 100
    except:
        return 0.0

def generate_random_dna():
    """Generates random Hyperparameters (DNA)"""
    return {
        'initial_capital': 10000,
        # DATA DNA
        'lookback_window': random.choice([30, 45, 60]),
        'label_threshold': random.uniform(0.01, 0.035), 
        
        # PHYSICS DNA
        'learning_rate':    random.uniform(0.002, 0.15),
        'phase_flexibility': random.uniform(0.05, 0.25),
        'lateral_strength':  random.uniform(0.1, 0.3),
        'input_threshold':   random.uniform(0.6, 0.8),
        'kerr_constant':     random.uniform(0.1, 0.4),
        'system_energy':     random.uniform(40.0, 60.0),
        
        # STRATEGY DNA
        'buy_threshold':  random.uniform(15, 40),
        'sell_threshold': random.uniform(-40, -15),
        'search_depth':   5
    }

def run_monte_carlo(iterations=1000):
    print(f"=== STARTING MONTE CARLO ({iterations} Runs) ===")
    print("Optimization Target: Maximize Return % on Real Market Data")
    print(f"Benchmark: {BENCHMARK_TICKER}")
    
    benchmark_return = get_benchmark_return()
    print(f"Benchmark Target (VTI 2yr): {benchmark_return:.2f}%")
    
    best_avg_return = -100.0
    best_config = None
    
    for i in range(iterations):
        print(f"\n--- Run {i+1}/{iterations} ---")
        
        # 1. Generate One Config to rule them all (Generalist Approach)
        # Or modify to per-ticker if you prefer specialist
        base_config = generate_random_dna()
        print(f"Testing DNA: LR={base_config['learning_rate']:.3f}")
        
        ticker_results = {}
        total_return = 0.0
        
        # 2. Test Generalization
        for ticker in TEST_TICKERS:
            run_config = base_config.copy()
            run_config['ticker'] = ticker
            
            res = run_simulation(run_config)
            
            ticker_results[ticker] = res['return_pct']
            total_return += res['return_pct']
            print(f" > {ticker}: {res['return_pct']:.2f}%")
            
        # 3. Calculate Stats
        avg_return = total_return / len(TEST_TICKERS)
        
        results_package = {
            'avg_return': avg_return,
            'breakdown': ticker_results # Note: Logger needs update if you want breakdown columns
        }
        
        # 4. Log
        # Simplified logging for this runner version
        log_experiment_result(base_config, {'return_pct': avg_return})
        
        if avg_return > best_avg_return:
            best_avg_return = avg_return
            best_config = base_config
            print(f">>> NEW RECORD! (Avg: {avg_return:.2f}%) <<<")

    print("\n=== OPTIMIZATION COMPLETE ===")
    print(f"Best Average Return: {best_avg_return:.2f}%")
    print(f"Best Config DNA: {best_config}")

if __name__ == "__main__":
    run_monte_carlo(iterations=1000)