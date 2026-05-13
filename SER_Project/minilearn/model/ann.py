import numpy as np

class ANNClassifier:
    def __init__(self, hidden_size=16, learning_rate=0.05, n_iter=500, random_state=None):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        self.class_ = None
        self.classes_ = None
        self.W1_ = None
        self.b1_ = None
        self.W2_ = None
        self.b2_ = None
        self.loss_ = []