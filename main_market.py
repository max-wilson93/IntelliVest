import time
import numpy as np
from market_loader import MarketLoader
from quantum_cortex import QuantumCortex
from fourier_optics import FourierOptics

def run_intellivest():
    # 1. SETUP DATA
    # We use a volatile stock (like NVDA) to get good signal patterns
    loader = MarketLoader(tickers=["NVDA"], lookback_window=60)
    
    print("-> Converting Market Cycles to Holographic Spectrograms...")
    X_data, y_data = loader.fetch_data()
    
    # Split Train/Test (80/20)
    split = int(len(X_data) * 0.8)
    train_images = X_data[:split]
    train_labels = y_data[:split]
    test_images = X_data[split:]
    test_labels = y_data[split:]
    
    print(f"-> Generated {len(X_data)} Market Holograms.")
    print(f"-> Training Samples: {len(train_images)} | Test Samples: {len(test_images)}")

    # 2. CONFIGURATION
    # 3 Classes: Bear (0), Neutral (1), Bull (2)
    NUM_CLASSES = 3  
    NEURONS_PER_CLASS = 15 # Increased slightly for robustness
    
    physics_config = {
        'learning_rate':    0.12, # Slightly higher for noisy market data
        'phase_flexibility': 0.15,
        'lateral_strength':  0.2,
        'input_threshold':   0.6,
        'kerr_constant':     0.25,
        'system_energy':     50.0 
    }

    # 3. INITIALIZE SYSTEM
    optics = FourierOptics(shape=(28, 28)) # Expects the resized spectrograms
    
    # Input size calculation: 4 masks * 28 * 28 = 3136 features
    cortex_A = QuantumCortex(3136, NUM_CLASSES, NEURONS_PER_CLASS, config=physics_config)
    cortex_B = QuantumCortex(3136, NUM_CLASSES, NEURONS_PER_CLASS, config=physics_config)
    cortex_C = QuantumCortex(3136, NUM_CLASSES, NEURONS_PER_CLASS, config=physics_config)

    # --- PHASE 1: ONLINE LEARNING ---
    print(f"\n=== PHASE 1: ONLINE LEARNING (Market Regimes) ===")
    correct_count = 0
    
    for i in range(len(train_images)):
        # Reshape flat vector back to 28x28 for Optics
        img_2d = train_images[i].reshape(28, 28)
        
        # Apply Fourier Filters (Extract Phase Topology)
        features = optics.apply(img_2d)
        label = train_labels[i]
        
        # Train Trinity
        _, pred_a, _ = cortex_A.process_image(features, label, train=True)
        _, pred_b, _ = cortex_B.process_image(features, label, train=True)
        _, pred_c, _ = cortex_C.process_image(features, label, train=True)
        
        # Vote
        votes = np.zeros(NUM_CLASSES)
        votes[pred_a] += 1; votes[pred_b] += 1; votes[pred_c] += 1
        ensemble_pred = np.argmax(votes)
        
        if ensemble_pred == label: correct_count += 1
        
        if (i+1) % 100 == 0:
            acc = (correct_count / (i+1)) * 100
            print(f"Day {i+1} | Acc: {acc:.2f}% | Pred: {ensemble_pred} vs Truth: {label}")

    print(f"Training Complete. Train Acc: {(correct_count / len(train_images)) * 100:.2f}%")

    # --- PHASE 2: VALIDATION ---
    print(f"\n=== PHASE 2: VALIDATION (Unseen Market Data) ===")
    test_correct = 0
    
    for i in range(len(test_images)):
        img_2d = test_images[i].reshape(28, 28)
        features = optics.apply(img_2d)
        label = test_labels[i]
        
        # Test (Plasticity OFF)
        _, pred_a, _ = cortex_A.process_image(features, label, train=False)
        _, pred_b, _ = cortex_B.process_image(features, label, train=False)
        _, pred_c, _ = cortex_C.process_image(features, label, train=False)
        
        votes = np.zeros(NUM_CLASSES)
        votes[pred_a] += 1; votes[pred_b] += 1; votes[pred_c] += 1
        ensemble_pred = np.argmax(votes)
        
        if ensemble_pred == label: test_correct += 1

    final_acc = (test_correct / len(test_images)) * 100
    print(f"\n=== FINAL INTELIVEST RESULTS ===")
    print(f"Test Accuracy: {final_acc:.2f}%")
    
    # NOTE FOR ADVERSARIAL SEARCH:
    # This 'ensemble_pred' is what you will feed into your Minimax algorithm 
    # as the 'Market State'.

if __name__ == "__main__":
    run_intellivest()