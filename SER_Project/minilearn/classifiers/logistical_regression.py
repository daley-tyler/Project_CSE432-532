import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.1, n_iter=500):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.class_ = None
        self.weight_ = None

    def add_intercept_col(self, X):
        ones = np.ones((X.shape[0],1))
        return np.column_stack((ones, X))

    def sigmoid(self, z):
        return np.exp(z) / (1 + np.exp(z))

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)
        X_bias = self.add_intercept_col(X)

        self.class_ = np.unique(y)
        self.weight_ = np.zeros((len(self.class_), X_bias.shape[1]))

        for c_idx in range(len(self.class_)):
            c_label = self.class_[c_idx]
            y_binc = (y == c_label).astype(float)
            weight = np.zeros(X_bias.shape[1])

            for n in range(self.n_iter):
                scores = X_bias.dot(weight)
                probs = self.sigmoid(scores)
                errors = probs - y_binc
                weight_change = X_bias.T.dot(errors) / len(X_bias)
                weight = weight - self.learning_rate * weight_change

            self.weight_[c_idx] = weight

        return self
        
    def predict_prob(self, X):
        X = np.array(X, dtype=float)
        X_bias = self.add_intercept_col(X)
        score = X_bias.dot(self.weight_.T)
        prob = self.sigmoid(score)
        row_sum = np.sum(prob, axis=1)
        row_sum[row_sum==0] = 1

        return prob / row_sum.reshape(-1,1)

    def predict():
        pass
    
    def score():
        pass