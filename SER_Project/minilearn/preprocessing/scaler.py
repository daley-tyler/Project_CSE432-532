import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = np.array(X, dtype=float)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1
        return self
    
    def transform(self, X):
        X = np.array(X, dtype=float)
        return (X-self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)