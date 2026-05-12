import numpy as np

class ANNClassifier:
    def __init__(self, learning_rate=0.05, n_iter=500):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.class_ = None
        self.weights_ = None
        self.bias_ = None
        self.loss_ = []