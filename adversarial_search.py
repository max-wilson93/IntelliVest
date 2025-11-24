import numpy as np

class AdversarialGame:
    def __init__(self, market_scout_model):
        self.scout = market_scout_model
        self.actions = ['BUY', 'HOLD', 'SELL']

    def evaluate_state(self, market_state_image):
        """
        Uses the Quantum Cortex to get the 'Energy' of the market state.
        High Energy + Bull Class = Positive Score (Good for Buyer)
        High Energy + Bear Class = Negative Score (Bad for Buyer)
        """
        # Run the Scout (Prediction only, no training)
        # We pass label=0 because we don't know the truth yet, we just want the energy/pred
        _, prediction, energy = self.scout.process_image(market_state_image, label=0, train=False)
        
        # Heuristic Logic
        # Class 2 = Bull, Class 0 = Bear, Class 1 = Neutral
        if prediction == 2: 
            return energy * 10  
        elif prediction == 0: 
            return -energy * 10
        else: 
            return 0

    def minimax(self, market_image, depth, is_maximizing, alpha, beta):
        """
        Standard Minimax with Alpha-Beta Pruning.
        """
        # Base Case
        if depth == 0:
            return self.evaluate_state(market_image)

        if is_maximizing: # The Investor (You)
            best_val = -np.inf
            for _ in self.actions:
                # Simulate the move (simplified recursion)
                val = self.minimax(market_image, depth - 1, False, alpha, beta)
                best_val = max(best_val, val)
                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break
            return best_val
            
        else: # The Adversary (The Market)
            best_val = np.inf
            # FIX: Added loop here. The market "moves" to minimize your score.
            for _ in self.actions:
                val = self.minimax(market_image, depth - 1, True, alpha, beta)
                best_val = min(best_val, val)
                beta = min(beta, best_val)
                # Now 'break' is inside a loop, so it works
                if beta <= alpha:
                    break
            return best_val

    def get_best_move(self, current_spectrogram):
        """
        Wrapper to start the search.
        """
        # Run Minimax Depth 3
        score = self.minimax(current_spectrogram, depth=3, is_maximizing=True, alpha=-np.inf, beta=np.inf)
        
        # Map Score to Policy
        # High positive score -> Strong Bull Signal -> BUY
        # Low negative score -> Strong Bear Signal -> SELL
        if score > 20: return 'BUY'
        elif score < -20: return 'SELL'
        else: return 'HOLD'