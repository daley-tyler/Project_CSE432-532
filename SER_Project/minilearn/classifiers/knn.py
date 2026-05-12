import numpy as np

class KNNClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors_ = n_neighbors
        self.X_train_ = None
        self.Y_train_ = None

    def fit(self, X, y):
        self.X_train_ = np.array(X, dtype=float)
        self.y_train_ = np.array(y)
        return self