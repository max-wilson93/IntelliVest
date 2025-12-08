import numpy as np
import pandas as pd
from scipy import stats

class StatisticalAnalyzer:
    def __init__(self, model_equity, benchmark_equity, risk_free_rate=0.04):
        # Explicit conversion to float numpy arrays
        self.model_equity = np.array(model_equity, dtype=float)
        self.benchmark_equity = np.array(benchmark_equity, dtype=float)
        
        # Calculate Returns
        self.model_returns = np.diff(self.model_equity) / self.model_equity[:-1]
        self.benchmark_returns = np.diff(self.benchmark_equity) / self.benchmark_equity[:-1]
        
        # Clean NaNs
        self.model_returns = self.model_returns[~np.isnan(self.model_returns)]
        self.benchmark_returns = self.benchmark_returns[~np.isnan(self.benchmark_returns)]
        
        min_len = min(len(self.model_returns), len(self.benchmark_returns))
        self.model_returns = self.model_returns[-min_len:]
        self.benchmark_returns = self.benchmark_returns[-min_len:]
        
        self.rf_daily = risk_free_rate / 252
        
    def run_full_analysis(self):
        print(f"{'='*60}")
        print(f"{'STATISTICAL SIGNIFICANCE ANALYSIS':^60}")
        print(f"{'='*60}\n")
        
        results = {}
        
        # 1. Normality
        if len(self.model_returns) >= 3:
            stat_m, p_m = stats.shapiro(self.model_returns)
            stat_b, p_b = stats.shapiro(self.benchmark_returns)
            
            results['Normality_Model'] = 'Gaussian' if p_m > 0.05 else 'Non-Gaussian'
            results['Normality_Benchmark'] = 'Gaussian' if p_b > 0.05 else 'Non-Gaussian'
            
            print(f"--- 1. NORMALITY TEST (Shapiro-Wilk) ---")
            print(f"Quantum Model: p={p_m:.5f} -> {results['Normality_Model']}")
            print(f"Benchmark:     p={p_b:.5f} -> {results['Normality_Benchmark']}")
        
        # 2. T-Test
        if len(self.model_returns) >= 2:
            t_stat, p_val = stats.ttest_ind(self.model_returns, self.benchmark_returns, equal_var=False)
            
            print(f"\n--- 2. HYPOTHESIS TEST (Welch's T-Test) ---")
            print(f"T-Statistic: {t_stat:.4f}")
            print(f"P-Value:     {p_val:.4f}")
            if p_val < 0.05:
                if t_stat > 0:
                    print("   -> RESULT: Quantum Model is SIGNIFICANTLY BETTER.")
                else:
                    print("   -> RESULT: Benchmark is SIGNIFICANTLY BETTER.")
            else:
                print("   -> RESULT: No statistically significant difference.")
            
        # 3. Risk Metrics
        std_m = float(np.std(self.model_returns))
        std_b = float(np.std(self.benchmark_returns))
        
        model_sharpe = 0.0
        if std_m > 0:
            model_sharpe = float(np.mean(self.model_returns - self.rf_daily) / std_m * np.sqrt(252))
            
        bench_sharpe = 0.0
        if std_b > 0:
            bench_sharpe = float(np.mean(self.benchmark_returns - self.rf_daily) / std_b * np.sqrt(252))
        
        print(f"\n--- 3. RISK METRICS ---")
        print(f"Model Sharpe: {model_sharpe:.2f}")
        print(f"Bench Sharpe: {bench_sharpe:.2f}")

        return results