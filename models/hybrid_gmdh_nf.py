import numpy as np
from sklearn.linear_model import LinearRegression

class GMDHLayer:
    """Одна шар-«оболонка» GMDH із квадратичними нейронами."""
    def __init__(self, k_best: int = 8):
        self.k_best = k_best
        self.neurons = []  

    def fit(self, X, y):
        n = X.shape[1]
        candidates = []
        for i in range(n):
            for j in range(i, n):
                Z = self._poly_features(X[:, i], X[:, j])
                w = np.linalg.lstsq(Z, y, rcond=None)[0]
                y_hat = Z @ w
                mse = np.mean((y - y_hat) ** 2)
                candidates.append((mse, i, j, w))
        # залишаємо k_best
        self.neurons = sorted(candidates)[:self.k_best]

    def transform(self, X):
        out = []
        for _, i, j, w in self.neurons:
            Z = self._poly_features(X[:, i], X[:, j])
            out.append(Z @ w)
        return np.column_stack(out)

    @staticmethod
    def _poly_features(xi, xj):
        return np.column_stack([
            np.ones_like(xi), xi, xj,
            xi**2, xi * xj, xj**2
        ])

class NeoFuzzyLayer:
    """Три трикутні MF на вхід; лінійна агрегація."""
    def fit(self, X, y):
        # MF центри у [min, mean, max]
        mins, means, maxs = X.min(0), X.mean(0), X.max(0)
        self.C = np.vstack([mins, means, maxs])  # 3 × d
        Phi = self.membership(X)                 # n × (3d)
        self.theta = np.linalg.lstsq(Phi, y, rcond=None)[0]

    def predict(self, X):
        return self.membership(X) @ self.theta

    def membership(self, X):
        Phi = []
        for j in range(X.shape[1]):
            a, b, c = self.C[:, j]
            x = X[:, j]
            mu_low  = np.maximum((b - x) / (b - a), 0)
            mu_mid  = np.maximum(np.minimum((x - a) / (b - a),
                                            (c - x) / (c - b)), 0)
            mu_high = np.maximum((x - b) / (c - b), 0)
            Phi.extend([mu_low, mu_mid, mu_high])
        return np.column_stack(Phi)

class HybridGMDHNeoFuzzy:
    def __init__(self, gmdh_layers: int = 2, k_best: int = 8):
        self.g_layers = [GMDHLayer(k_best) for _ in range(gmdh_layers)]
        self.fuzzy = NeoFuzzyLayer()

    def fit(self, X, y):
        for g in self.g_layers:
            g.fit(X, y)
            X = g.transform(X)
        self.fuzzy.fit(X, y)

    def predict(self, X):
        for g in self.g_layers:
            X = g.transform(X)
        return self.fuzzy.predict(X)
