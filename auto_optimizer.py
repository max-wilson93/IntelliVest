import json
import time
from monte_carlo_runner import run_random_search, run_genetic_optimization, TEST_TICKERS

def main():
    print("========================================================")
    print("   INTELLIVEST: AUTOMATED HYPERPARAMETER TUNING SYSTEM   ")
    print("========================================================")
    
    final_portfolio = {}
    total_start_time = time.time()
    
    for ticker in TEST_TICKERS:
        print(f"\n>>> OPTIMIZING ASSET: {ticker} <<<")
        ticker_start = time.time()
        
        # STEP 1: WIDE EXPLORATION (Random Search)
        # Runs 100 quick random guesses to find a good neighborhood
        print("1. Phase: Wide Exploration")
        seed_dna = run_random_search(ticker, iterations=100)
        
        # STEP 2: DEEP EXPLOITATION (Genetic Evolution)
        # Takes the best random guess and refines it for 5 generations
        print("2. Phase: Genetic Evolution")
        final_dna, final_return = run_genetic_optimization(
            ticker, 
            seed_dna=seed_dna, 
            generations=5, 
            population_size=12
        )
        
        duration = time.time() - ticker_start
        print(f"   [COMPLETE] Best Return: {final_return:.2f}% (Took {duration:.1f}s)")
        
        final_portfolio[ticker] = final_dna

    print("\n========================================================")
    print("   OPTIMIZATION COMPLETE. SAVING DNA.   ")
    print("========================================================")
    
    # Save to JSON for final_analysis.py to use
    with open('specialist_dna.json', 'w') as f:
        json.dump(final_portfolio, f, indent=4)
        
    print(f"Saved optimized parameters to 'specialist_dna.json'.")
    print(f"Total Time: {(time.time() - total_start_time)/60:.1f} minutes.")

if __name__ == "__main__":
    main()