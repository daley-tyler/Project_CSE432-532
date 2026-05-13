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

    def relu(self, z):
        return np.maximum(0,z)
    
    def relu_grad(self,z):
        return (z>0).astype(float)
    
    def softmax(self,z):
        z = z - np.max(z, axis=1, keepdims=True)
        e = np.exp(z)
        return e / np.sum(e, axis=1, keepdims=True)