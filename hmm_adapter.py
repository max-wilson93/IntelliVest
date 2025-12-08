import numpy as np
import math

class GaussianHMM_Scratch:
    def __init__(self, n_states=3, n_iter=50, tol=1e-4):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.start_prob = None
        self.trans_mat = None
        self.means = None
        self.covars = None

    def _init_params(self, X):
        n_samples = len(X)
        self.start_prob = np.ones(self.n_states) / self.n_states
        self.trans_mat = np.random.rand(self.n_states, self.n_states)
        self.trans_mat /= self.trans_mat.sum(axis=1, keepdims=True)
        
        if n_samples < self.n_states:
             indices = np.random.choice(n_samples, self.n_states, replace=True)
        else:
             indices = np.random.choice(n_samples, self.n_states, replace=False)
             
        self.means = X[indices].flatten()
        global_var = np.var(X)
        if global_var == 0: global_var = 1e-6
        self.covars = np.ones(self.n_states) * global_var

    def _gaussian_pdf(self, x, mean, var):
        eps = 1e-9
        var = max(var, eps)
        coeff = 1.0 / np.sqrt(2.0 * np.pi * var)
        exponent = -((x - mean) ** 2) / (2.0 * var)
        exponent = np.clip(exponent, -700, 700) 
        return coeff * np.exp(exponent)

    def _forward_backward(self, X):
        n_samples = len(X)
        B = np.zeros((n_samples, self.n_states))
        for j in range(self.n_states):
            B[:, j] = self._gaussian_pdf(X.flatten(), self.means[j], self.covars[j])
        B = np.maximum(B, 1e-300)

        alpha = np.zeros((n_samples, self.n_states))
        scale = np.zeros(n_samples) 
        
        alpha[0] = self.start_prob * B[0]
        scale[0] = 1.0 / (np.sum(alpha[0]) + 1e-300)
        alpha[0] *= scale[0]
        
        for t in range(1, n_samples):
            for j in range(self.n_states):
                alpha[t, j] = np.dot(alpha[t-1], self.trans_mat[:, j]) * B[t, j]
            scale[t] = 1.0 / (np.sum(alpha[t]) + 1e-300)
            alpha[t] *= scale[t]
            
        beta = np.zeros((n_samples, self.n_states))
        beta[-1] = 1.0 * scale[-1]
        
        for t in range(n_samples - 2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.trans_mat[i, :] * B[t+1, :] * beta[t+1, :])
            beta[t] *= scale[t]
            
        gamma = np.zeros((n_samples, self.n_states))
        xi = np.zeros((n_samples - 1, self.n_states, self.n_states))
        
        for t in range(n_samples):
            denom = np.sum(alpha[t] * beta[t]) + 1e-300
            gamma[t] = (alpha[t] * beta[t]) / denom
            
        for t in range(n_samples - 1):
            denom = np.sum(alpha[t] * beta[t]) + 1e-300
            for i in range(self.n_states):
                for j in range(self.n_states):
                    val = (alpha[t, i] * self.trans_mat[i, j] * B[t+1, j] * beta[t+1, j])
                    xi[t, i, j] = val / denom
        return gamma, xi

    def fit(self, X):
        if len(X) < 2: return
        self._init_params(X)
        for i in range(self.n_iter):
            try:
                gamma, xi = self._forward_backward(X)
                self.start_prob = gamma[0]
                
                xi_sum = np.sum(xi, axis=0)
                gamma_sum = np.sum(gamma[:-1], axis=0).reshape(-1, 1)
                self.trans_mat = xi_sum / (gamma_sum + 1e-300)
                
                gamma_sum_total = np.sum(gamma, axis=0)
                for j in range(self.n_states):
                    denom = gamma_sum_total[j] + 1e-300
                    self.means[j] = np.sum(gamma[:, j] * X.flatten()) / denom
                    diff = X.flatten() - self.means[j]
                    self.covars[j] = np.sum(gamma[:, j] * (diff ** 2)) / denom
                self.covars = np.maximum(self.covars, 1e-6)
            except Exception:
                break

    def predict(self, X):
        n_samples = len(X)
        B = np.zeros((n_samples, self.n_states))
        for j in range(self.n_states):
            var = self.covars[j] + 1e-6
            mean = self.means[j]
            log_coeff = -0.5 * np.log(2 * np.pi * var)
            log_exp = -((X.flatten() - mean) ** 2) / (2 * var)
            B[:, j] = log_coeff + log_exp

        V = np.zeros((n_samples, self.n_states))
        path = np.zeros((n_samples, self.n_states), dtype=int)
        log_trans = np.log(self.trans_mat + 1e-300)
        log_start = np.log(self.start_prob + 1e-300)
        V[0] = log_start + B[0]
        
        for t in range(1, n_samples):
            for j in range(self.n_states):
                probs = V[t-1] + log_trans[:, j] 
                best_prev = np.argmax(probs)
                path[t, j] = best_prev
                V[t, j] = probs[best_prev] + B[t, j]
                
        best_path = np.zeros(n_samples, dtype=int)
        best_path[-1] = np.argmax(V[-1])
        for t in range(n_samples - 2, -1, -1):
            best_path[t] = path[t+1, best_path[t+1]]
        return best_path

class MarketHMM:
    def __init__(self, n_states=3, train_window=100):
        self.n_states = n_states
        self.train_window = train_window
        
    def get_signal(self, recent_returns):
        if recent_returns is None or len(recent_returns) < 50:
            return 'HOLD'
            
        X = np.array(recent_returns)
        try:
            model = GaussianHMM_Scratch(n_states=self.n_states, n_iter=50)
            model.fit(X)
            
            means = model.means
            vars_ = model.covars
            if means is None or vars_ is None: return 'HOLD'

            score = means / (np.sqrt(vars_) + 1e-6)
            sorted_indices = np.argsort(score)
            
            state_map = {
                sorted_indices[0]: 'SELL',
                sorted_indices[1]: 'HOLD',
                sorted_indices[2]: 'BUY'
            }
            states = model.predict(X)
            return state_map[states[-1]]
        except Exception:
            return 'HOLD'