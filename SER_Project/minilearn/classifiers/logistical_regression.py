import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.1, n_iter=500):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.class_ = None
        self.weight_ = None

    
    def sigmoid(self, z):
        return np.exp(z) / (1 + np.exp(z))

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)
        
        
    def predict():
        pass
    
    def score():
        pass