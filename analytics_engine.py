import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Set style for professional charts
plt.style.use('bmh')

class AnalyticsEngine:
    def __init__(self, model_equity, benchmark_equity, risk_free_rate=0.04):
        self.model_equity = np.array(model_equity)
        self.benchmark_equity = np.array(benchmark_equity)
        self.rf_daily = risk_free_rate / 252
        
        # Calculate Returns
        self.model_returns = pd.Series(self.model_equity).pct_change().fillna(0)
        self.bench_returns = pd.Series(self.benchmark_equity).pct_change().fillna(0)
        
    def _calculate_drawdown(self, equity_curve):
        """Calculates the underwater curve (percentage drop from peak)."""
        series = pd.Series(equity_curve)
        running_max = series.cummax()
        drawdown = (series - running_max) / running_max
        return drawdown

    def _calculate_rolling_sharpe(self, returns, window=60):
        """Calculates Sharpe Ratio over a rolling window."""
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        # Annualized
        return (rolling_mean - self.rf_daily) / rolling_std * np.sqrt(252)

    def generate_full_report(self, ticker="ASSET"):
        """
        Generates a 4-panel dashboard and prints statistical stats.
        """
        self.print_statistics(ticker)
        self.plot_dashboard(ticker)

    def print_statistics(self, ticker):
        print(f"\n{'='*60}")
        print(f"   ADVANCED ANALYTICS REPORT: {ticker}")
        print(f"{'='*60}")
        
        m_ret = self.model_returns
        b_ret = self.bench_returns
        
        # 1. CAGR (Compound Annual Growth Rate)
        # Approx days
        days = len(self.model_equity)
        years = days / 252
        cagr_m = (self.model_equity[-1] / self.model_equity[0]) ** (1/years) - 1
        cagr_b = (self.benchmark_equity[-1] / self.benchmark_equity[0]) ** (1/years) - 1
        
        # 2. Max Drawdown
        dd_m = self._calculate_drawdown(self.model_equity).min()
        dd_b = self._calculate_drawdown(self.benchmark_equity).min()
        
        # 3. Sortino Ratio (Penalizes only downside volatility)
        downside_returns = m_ret[m_ret < 0]
        downside_std = downside_returns.std()
        sortino = (m_ret.mean() - self.rf_daily) / downside_std * np.sqrt(252) if downside_std > 0 else 0
        
        # 4. Beta (Correlation to Market)
        cov_matrix = np.cov(m_ret, b_ret)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
        
        # 5. Alpha (Jensen's Alpha)
        # Alpha = R_p - (R_f + Beta * (R_m - R_f))
        annual_m_ret = m_ret.mean() * 252
        annual_b_ret = b_ret.mean() * 252
        alpha = annual_m_ret - (0.04 + beta * (annual_b_ret - 0.04))
        
        print(f"{'METRIC':<20} | {'MODEL':<15} | {'MARKET':<15}")
        print("-" * 60)
        print(f"{'CAGR':<20} | {cagr_m:>14.2%} | {cagr_b:>14.2%}")
        print(f"{'Max Drawdown':<20} | {dd_m:>14.2%} | {dd_b:>14.2%}")
        print(f"{'Sharpe Ratio':<20} | {self._get_sharpe(m_ret):>14.2f} | {self._get_sharpe(b_ret):>14.2f}")
        print(f"{'Sortino Ratio':<20} | {sortino:>14.2f} | {'--':>14}")
        print("-" * 60)
        print(f"{'Beta':<20} | {beta:>14.2f}")
        print(f"{'Alpha (Annualized)':<20} | {alpha:>14.2%}")
        print(f"{'='*60}\n")

    def _get_sharpe(self, returns):
        if returns.std() == 0: return 0.0
        return (returns.mean() - self.rf_daily) / returns.std() * np.sqrt(252)

    def plot_dashboard(self, ticker):
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f"IntelliVest Performance Analysis: {ticker}", fontsize=16)
        
        # Plot 1: Equity Curve (The Growth)
        ax1 = axes[0, 0]
        ax1.plot(self.model_equity, label='Quantum Cortex', color='#1f77b4', linewidth=2)
        ax1.plot(self.benchmark_equity, label='Buy & Hold', color='gray', linestyle='--', alpha=0.7)
        ax1.set_title("Equity Growth ($10k Start)")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Drawdown (The Risk)
        # Shows "Underwater" periods. 
        # Crucial for proving safety.
        ax2 = axes[0, 1]
        dd_model = self._calculate_drawdown(self.model_equity) * 100
        dd_bench = self._calculate_drawdown(self.benchmark_equity) * 100
        
        ax2.fill_between(range(len(dd_model)), dd_model, 0, color='#1f77b4', alpha=0.3, label='Quantum DD')
        ax2.plot(dd_bench, color='red', alpha=0.6, linewidth=1, label='Market DD')
        ax2.set_title("Drawdown Profile (Risk)")
        ax2.set_ylabel("Drawdown %")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Return Distribution (The Topology)
        # Shows non-gaussian nature.
        ax3 = axes[1, 0]
        sns.kdeplot(self.model_returns, ax=ax3, fill=True, color='#1f77b4', label='Quantum', clip=(-0.05, 0.05))
        sns.kdeplot(self.bench_returns, ax=ax3, fill=True, color='gray', label='Market', clip=(-0.05, 0.05))
        ax3.set_title("Daily Return Distribution (Topology)")
        ax3.set_xlabel("Daily Return")
        ax3.legend()
        
        # Plot 4: Rolling Sharpe (The Consistency)
        # Shows if performance is steady or lucky.
        ax4 = axes[1, 1]
        roll_sharpe_m = self._calculate_rolling_sharpe(self.model_returns)
        roll_sharpe_b = self._calculate_rolling_sharpe(self.bench_returns)
        
        ax4.plot(roll_sharpe_m, label='Quantum (60d Rolling)', color='#1f77b4')
        ax4.plot(roll_sharpe_b, label='Market (60d Rolling)', color='gray', alpha=0.5)
        ax4.axhline(0, color='black', linewidth=1, linestyle='-')
        ax4.set_title("Rolling Risk-Adjusted Return (Sharpe)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{ticker}_tearsheet.png")
        print(f"[Chart] Saved dashboard to '{ticker}_tearsheet.png'")