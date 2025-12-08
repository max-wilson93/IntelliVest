import random
import time
import numpy as np
import yfinance as yf
from main_intellivest import run_simulation
from experiment_logger import log_experiment_result
import json

# === THE SPECIALIST SQUAD ===
TEST_TICKERS = ['NVDA', 'JPM', 'XOM', 'JNJ', 'COST', 'TSLA', 'GOOGL', 'GLD']
BENCHMARK_TICKER = 'VTI'

def get_benchmark_return(period="2y"):
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
    """Long-Only Optimized Search Space"""
    return {
        'initial_capital': 10000,
        'lookback_window': random.choice([25, 30, 35]),
        'label_threshold': random.uniform(0.001, 0.035), 
        'learning_rate':    random.uniform(0.02, 0.15),
        'phase_flexibility': random.uniform(0.05, 0.20),
        'lateral_strength':  random.uniform(0.1, 0.3),
        'input_threshold':   random.uniform(0.65, 0.85),
        'kerr_constant':     random.uniform(0.1, 0.4),
        'system_energy':     random.uniform(40.0, 60.0),
        'buy_threshold':  random.uniform(10, 35),
        'sell_threshold': random.uniform(-35, -10),
        'search_depth':   3,
        'trailing_stop_pct': random.uniform(0.04, 0.08)
    }

def mutate(dna):
    new_dna = dna.copy()
    if random.random() < 0.3:
        new_dna['learning_rate'] = np.clip(new_dna['learning_rate'] + random.normalvariate(0, 0.01), 0.01, 0.3)
    if random.random() < 0.3:
        new_dna['buy_threshold'] = np.clip(new_dna['buy_threshold'] + random.normalvariate(0, 2), 1, 50)
    return new_dna

def crossover(parent1, parent2):
    child = {}
    for key in parent1:
        if key == 'ticker': continue
        child[key] = parent1[key] if random.random() > 0.5 else parent2[key]
    return child

def run_genetic_optimization(generations=3, population_size=8):
    print(f"=== INITIALIZING GENETIC OPTIMIZATION ===")
    benchmark_ret = get_benchmark_return()
    print(f"Benchmark (VTI): {benchmark_ret:.2f}%")
    final_portfolio = {}
    
    for ticker in TEST_TICKERS:
        print(f"\n>>> EVOLVING: {ticker} <<<")
        population = [generate_random_dna() for _ in range(population_size)]
        for p in population: p['ticker'] = ticker
        
        best_fitness = -999.0
        best_dna = None
        
        for gen in range(generations):
            scored_pop = []
            for dna in population:
                res = run_simulation(dna)
                fitness = res['return_pct'] 
                scored_pop.append((fitness, dna))
            
            scored_pop.sort(key=lambda x: x[0], reverse=True)
            if scored_pop[0][0] > best_fitness:
                best_fitness = scored_pop[0][0]
                best_dna = scored_pop[0][1]
                print(f"   Gen {gen}: Record -> {best_fitness:.2f}%")
            
            survivors = [x[1] for x in scored_pop[:population_size//2]]
            next_gen = survivors[:] 
            while len(next_gen) < population_size:
                p1 = random.choice(survivors)
                p2 = random.choice(survivors)
                child = crossover(p1, p2)
                child = mutate(child)
                child['ticker'] = ticker
                next_gen.append(child)
            population = next_gen
            
        print(f"   WINNER {ticker}: {best_fitness:.2f}%")
        final_portfolio[ticker] = {'return': best_fitness, 'dna': best_dna}

    with open('specialist_dna.json', 'w') as f:
        save_data = {k: v['dna'] for k, v in final_portfolio.items()}
        json.dump(save_data, f, indent=4)

if __name__ == "__main__":
    run_genetic_optimization()