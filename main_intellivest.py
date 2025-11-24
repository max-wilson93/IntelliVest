import numpy as np
import time

# Import our custom modules
from market_loader import MarketLoader
from fourier_optics import FourierOptics
from quantum_cortex import QuantumCortex
from csp_solver import ConstraintSolver
from adversarial_search import AdversarialGame
from portfolio_manager import PortfolioManager

def run_simulation():
    print("=== INTELLIVEST: INITIALIZING SYSTEM ===")
    
    # 1. DEFINE USER CONSTRAINTS (The "Rulebook")
    user_rules = {
        'max_price_per_share': 500,
        'max_risk_volatility': 0.05,
        'excluded_sectors': ['Fossil Fuels', 'Gambling']
    }
    csp = ConstraintSolver(user_rules)
    
    # 2. DEFINE MARKET UNIVERSE (Mock Metadata for filtering)
    # In a real app, this would come from an API. 
    # Here we define NVDA as valid and XOM (Exxon) as invalid to test CSP.
    raw_universe = [
        {'ticker': 'NVDA', 'price': 130, 'sector': 'Tech', 'volatility': 0.02},
        {'ticker': 'XOM',  'price': 110, 'sector': 'Fossil Fuels', 'volatility': 0.01}
    ]
    
    # 3. RUN CONSTRAINT SOLVER
    valid_tickers = csp.filter_universe(raw_universe)
    
    if not valid_tickers:
        print("No stocks satisfied your constraints. Exiting.")
        return

    # 4. FETCH REAL DATA FOR VALID TICKERS
    # This uses your corrected market_loader.py
    loader = MarketLoader(valid_tickers)
    print("\n[Data] Fetching market spectrograms...")
    X_data, y_data = loader.fetch_data()
    
    # 5. INITIALIZE AI MODULES
    # Scout (The Brain) - Expects 3136 inputs (from Fourier Optics)
    scout = QuantumCortex(num_inputs=3136, num_classes=3, neurons_per_class=15)
    optics = FourierOptics()
    
    # Strategist (The Search)
    game = AdversarialGame(scout)
    
    # Manager (The Ledger)
    pm = PortfolioManager(initial_capital=10000)
    
    # 6. RUN THE SIMULATION LOOP
    print(f"\n=== STARTING SIMULATION ({len(X_data)} Days) ===")
    
    for day_idx, (market_img, label) in enumerate(zip(X_data, y_data)):
        
        # A. PRE-PROCESSING (Fourier Optics)
        # Reshape flat vector back to 28x28
        img_2d = market_img.reshape(28, 28)
        
        # KEY STEP: Extract 3136 features from the 784 pixel image
        features = optics.apply(img_2d)
        
        # B. STRATEGY (Adversarial Search)
        # FIX: Pass 'features' (3136 size), NOT 'img_2d' (784 size)
        best_move = game.get_best_move(features)
        
        # C. EXECUTION (Portfolio Manager)
        # Mocking the price movement for the prototype simulation
        # In production, you would align this with the 'Close' price of that day.
        current_mock_price = 100 + (day_idx * 0.2) + (np.random.randn() * 2)
        
        # Print status every 50 days to keep log clean
        if day_idx % 50 == 0:
            print(f"\n[Day {day_idx}] AI Signal: {best_move} | Price: ${current_mock_price:.2f}")
            
        pm.execute(best_move, valid_tickers[0], current_mock_price)
        
        # D. LEARNING (Online Hebbian Update)
        # The Scout learns from the "Truth" (label) of what actually happened
        # We pass 'features' here too
        scout.process_image(features, label, train=True)

    # 7. FINAL REPORT
    final_val = pm.get_total_value({valid_tickers[0]: 150}) # Assuming final price $150
    print("\n=== SIMULATION COMPLETE ===")
    print(f"Final Portfolio Value: ${final_val:.2f}")
    print(f"Total Trades: {len(pm.history)}")

if __name__ == "__main__":
    run_simulation()