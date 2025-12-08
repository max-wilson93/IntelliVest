import numpy as np

class AdversarialGame:
    def __init__(self, market_scout_model):
        self.scout = market_scout_model
        # Actions: Buy, Hold, Sell (Long Only)
        self.actions = ['BUY', 'HOLD', 'SELL']

    def evaluate_state(self, market_state_image):
        _, prediction, energy = self.scout.process_image(market_state_image, label=0, train=False)
        
        if prediction == 2: # Bull
            return energy * 10  
        elif prediction == 0: # Bear
            return -energy * 10
        else: 
            return 0

    def minimax(self, market_image, depth, is_maximizing, alpha, beta):
        if depth == 0:
            return self.evaluate_state(market_image)

        if is_maximizing:
            best_val = -np.inf
            for _ in self.actions:
                val = self.minimax(market_image, depth - 1, False, alpha, beta)
                best_val = max(best_val, val)
                alpha = max(alpha, best_val)
                if beta <= alpha: break
            return best_val
            
        else:
            best_val = np.inf
            for _ in self.actions:
                val = self.minimax(market_image, depth - 1, True, alpha, beta)
                best_val = min(best_val, val)
                beta = min(beta, best_val)
                if beta <= alpha: break
            return best_val

    def get_best_move(self, current_spectrogram, buy_thresh=20, sell_thresh=-20, search_depth=3):
        score = self.minimax(current_spectrogram, depth=search_depth, is_maximizing=True, alpha=-np.inf, beta=np.inf)
        
        if score > buy_thresh: return 'BUY'
        elif score < sell_thresh: return 'SELL'
        else: return 'HOLD'