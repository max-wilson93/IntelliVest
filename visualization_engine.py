import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from scipy import signal
import networkx as nx

class VisualizationEngine:
    def __init__(self):
        plt.style.use('bmh')

    def plot_all(self, price_data, specialist_dna):
        print("Generating Model Visualizations...")
        self.plot_architecture_concept()
        self.plot_quantum_neuron()
        self.plot_spectrogram_vs_price(price_data)
        self.plot_game_tree()
        self.plot_dna_heatmap(specialist_dna)
        print("All visualization charts saved.")

    def plot_architecture_concept(self):
        """1. High-Level Architecture Flowchart"""
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 4)
        ax.axis('off')
        
        # Nodes
        blocks = [
            (1, "Market Data\n(Time Series)", 'gray'),
            (3, "Fourier Optics\n(Spectrogram)", 'blue'),
            (5, "Quantum Cortex\n(Interference)", 'purple'),
            (7, "Adversarial Agent\n(Minimax)", 'red'),
            (9, "Portfolio Manager\n(Execution)", 'green')
        ]
        
        for x, label, color in blocks:
            # Box
            rect = patches.FancyBboxPatch((x-0.8, 1.5), 1.6, 1, boxstyle="round,pad=0.1", 
                                          linewidth=2, edgecolor=color, facecolor='white')
            ax.add_patch(rect)
            # Text
            ax.text(x, 2, label, ha='center', va='center', fontsize=10, fontweight='bold')
            
            # Arrows
            if x < 9:
                ax.arrow(x+0.9, 2, 0.2, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
                
        ax.set_title("IntelliVest System Architecture", fontsize=14)
        plt.tight_layout()
        plt.savefig("viz_1_architecture.png")

    def plot_quantum_neuron(self):
        """2. Quantum Neuron Physics (Phase Interference)"""
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # Standard Neuron (Scalar)
        ax[0].set_title("Standard Neuron (Scalar Sum)")
        ax[0].arrow(0, 0, 0.5, 0.5, head_width=0.05, color='gray', label='Input 1')
        ax[0].arrow(0.5, 0.5, 0.5, -0.2, head_width=0.05, color='gray', label='Input 2')
        ax[0].text(0.5, 0.1, "Simple Addition\n1 + 1 = 2", ha='center')
        ax[0].set_xlim(-0.5, 1.5); ax[0].set_ylim(-0.5, 1.5)
        ax[0].axis('off')

        # Quantum Neuron (Vector Interference)
        ax[1].set_title("Quantum Neuron (Phase Interference)")
        
        # Constructive (Signal)
        ax[1].arrow(0, 0.8, 0.4, 0.1, head_width=0.03, color='blue', label='Signal A')
        ax[1].arrow(0.4, 0.9, 0.4, 0.1, head_width=0.03, color='blue', label='Signal B')
        ax[1].text(0.4, 1.1, "Constructive (Trend)", color='blue', ha='center')
        
        # Destructive (Noise)
        ax[1].arrow(0, 0.2, 0.4, 0.1, head_width=0.03, color='red', label='Noise A')
        ax[1].arrow(0.4, 0.3, -0.4, -0.1, head_width=0.03, color='red', label='Noise B')
        ax[1].text(0.4, 0.0, "Destructive (Noise Cancel)", color='red', ha='center')
        
        ax[1].set_xlim(-0.2, 1.0); ax[1].set_ylim(-0.2, 1.5)
        ax[1].axis('off')
        
        plt.tight_layout()
        plt.savefig("viz_2_quantum_physics.png")

    def plot_spectrogram_vs_price(self, prices):
        """3. The 'Eye' of the Model"""
        # Generate dummy spectrogram from real price data snippet
        # Use last 100 days of provided price data
        snippet = prices[-200:]
        returns = np.diff(np.log(snippet + 1e-8))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Price Chart
        ax1.plot(snippet, color='black')
        ax1.set_title("Market View: Raw Price (Time Domain)")
        ax1.set_ylabel("Price ($)")
        ax1.grid(True, alpha=0.3)
        
        # Spectrogram
        f, t, Sxx = signal.spectrogram(returns, fs=1.0, nperseg=15, noverlap=10)
        ax2.pcolormesh(t, f, np.log(Sxx + 1e-10), shading='gouraud', cmap='inferno')
        ax2.set_title("AI View: Spectrogram (Frequency Domain)")
        ax2.set_ylabel("Volatility Frequency")
        ax2.set_xlabel("Time (Days)")
        
        plt.tight_layout()
        plt.savefig("viz_3_spectrogram.png")

    def plot_game_tree(self):
        """4. Adversarial Game Tree"""
        G = nx.DiGraph()
        
        # Root
        G.add_node("Market State", pos=(0, 4))
        
        # Actions (Max)
        G.add_node("Buy", pos=(-2, 2))
        G.add_node("Hold", pos=(0, 2))
        G.add_node("Sell", pos=(2, 2))
        
        G.add_edge("Market State", "Buy")
        G.add_edge("Market State", "Hold")
        G.add_edge("Market State", "Sell")
        
        # Outcomes (Min) - Simplified for visual
        outcomes = ["Crash", "Rally", "Flat"]
        x_offsets = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
        
        # Just drawing a few representative branches
        G.add_node("Loss (-10)", pos=(-3, 0)) # Buy -> Crash
        G.add_node("Profit (+10)", pos=(-1, 0)) # Buy -> Rally
        
        G.add_edge("Buy", "Loss (-10)")
        G.add_edge("Buy", "Profit (+10)")
        
        pos = nx.get_node_attributes(G, 'pos')
        
        plt.figure(figsize=(10, 6))
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', 
                font_size=10, font_weight='bold', edge_color='gray', arrowsize=20)
        plt.title("Adversarial Search Tree (Minimax Concept)")
        plt.savefig("viz_4_gametree.png")

    def plot_dna_heatmap(self, specialist_dna):
        """5. Specialist DNA Heatmap"""
        # Convert JSON dict to DataFrame
        # Rows: Tickers, Cols: Parameters
        data = []
        tickers = []
        
        # Select key params to visualize
        keys = ['lookback_window', 'learning_rate', 'buy_threshold', 'input_threshold']
        
        for ticker, dna in specialist_dna.items():
            if not dna: continue
            row = [dna.get(k, 0) for k in keys]
            data.append(row)
            tickers.append(ticker)
            
        df = pd.DataFrame(data, columns=keys, index=tickers)
        
        # Normalize columns for heatmap visualization (0 to 1 scaling)
        df_norm = (df - df.min()) / (df.max() - df.min())
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_norm, annot=df, fmt=".2f", cmap="viridis", linewidths=.5)
        plt.title("Specialist DNA: No Universal Algorithm")
        plt.savefig("viz_5_dna_heatmap.png")

if __name__ == "__main__":
    # Test run with dummy data
    viz = VisualizationEngine()
    dummy_prices = np.cumsum(np.random.randn(200)) + 100
    dummy_dna = {
        'NVDA': {'lookback_window': 30, 'learning_rate': 0.08, 'buy_threshold': 20, 'input_threshold': 0.7},
        'JNJ': {'lookback_window': 60, 'learning_rate': 0.02, 'buy_threshold': 10, 'input_threshold': 0.6}
    }
    viz.plot_all(dummy_prices, dummy_dna)