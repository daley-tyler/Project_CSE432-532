import numpy as np

class KNNClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors_ = n_neighbors
        self.X_train_ = None
        self.Y_train_ = None

    def fit(self, X, y):
        self.X_train_ = np.array(X, dtype=float)
        self.Y_train_ = np.array(y)
        return self
    
    def predict_r(self, row):
        diff = self.X_train_ - row
        squared_distances = np.sum(diff ** 2, axis=1)
        distance = np.sqrt(squared_distances)
        near_inds = np.argsort(distance)[:self.n_neighbors_]
        near_labs = self.Y_train_[near_inds]

        labels, counts = np.unique(near_labs, return_counts=True)
        b_idx = np.argmax(counts)
        return labels[b_idx]
    
    def predict(self,X):
        X = np.array(X, dtype=float)
        predicts = []

        for row in X:
            predicts.append(self.predict_r(row))
        
        return np.array(predicts)
    
    def score(self, X, y):
        predicts = self.predict(X)
        return np.mean(predicts == np.array(y))
    
KNN = KNNClassifier