import numpy as np

class LinearSVM:
    def __init__(self, learning_rate=0.001, n_iter=500, C=1.0):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.C_= C

    def add_bias(self, X):
        X = np.array(X, dtype=float)
        ones = np.ones((X.shape[0], 1))
        return np.column_stack((ones,X))
    