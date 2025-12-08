import numpy as np
import time
from market_loader import MarketLoader
from fourier_optics import FourierOptics
from quantum_cortex import QuantumCortex
from csp_solver import ConstraintSolver
from adversarial_search import AdversarialGame
from portfolio_manager import PortfolioManager

def run_simulation(config=None):
    if config is None:
        config = {
            'ticker': 'NVDA',
            'initial_capital': 10000,
            'lookback_window': 30,
            'label_threshold': 0.015,
            'buy_threshold': 20,
            'sell_threshold': -20,
            'search_depth': 3,
        }

    pm = PortfolioManager(initial_capital=config['initial_capital'])
    
    loader = MarketLoader([config['ticker']], 
                          lookback_window=config['lookback_window'],
                          label_threshold=config['label_threshold'])
    try:
        X_data, y_data, real_prices = loader.fetch_data()
    except Exception as e:
        return {'final_value': 0, 'trades': 0, 'return_pct': 0.0, 'accuracy': 0.0}

    scout = QuantumCortex(3136, 3, 15, config=config)
    optics = FourierOptics()
    game = AdversarialGame(scout)
    
    correct_predictions = 0
    total_predictions = 0

    for i, (market_img, label) in enumerate(zip(X_data, y_data)):
        img_2d = market_img.reshape(28, 28)
        features = optics.apply(img_2d)
        
        is_correct, _, _ = scout.process_image(features, label, train=False)
        if is_correct: correct_predictions += 1
        total_predictions += 1

        best_move = game.get_best_move(features, 
                                       buy_thresh=config['buy_threshold'],
                                       sell_thresh=config['sell_threshold'],
                                       search_depth=config['search_depth'])
        
        current_price = real_prices[i]
        pm.execute(best_move, config['ticker'], current_price)
        scout.process_image(features, label, train=True)

    final_price = real_prices[-1]
    final_val = pm.get_total_value({config['ticker']: final_price})
    return_pct = ((final_val - config['initial_capital']) / config['initial_capital']) * 100
    
    metrics = pm.calculate_metrics()
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0.0
    
    return {
        'final_value': final_val,
        'trades': len(pm.history),
        'return_pct': return_pct,
        'sharpe': metrics['Sharpe'],
        'max_drawdown': metrics['Max Drawdown'],
        'win_rate': metrics['Win Rate'],
        'equity_curve': pm.equity_curve,
        'accuracy': accuracy
    }

if __name__ == "__main__":
    res = run_simulation()
    print(f"Result: {res['return_pct']:.2f}%")