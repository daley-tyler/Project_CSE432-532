import numpy as np 

class GaussianNaiveBayes:
    def __init__(self):
        self.class_ = None
        self.means_ = None
        self.vars_ = None
        self.prior_ = None

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)
        self.class_ = np.unique(y)
        self.means_ = []
        self.vars_ = []
        self.prior_ = []

        for c_label in self.class_:
            X_class = X[y == c_label]
            self.means_.append(np.mean(X_class, axis=0))
            self.vars_.append(np.var(X_class, axis=0))
            self.prior_.append(len(X_class)/len(X))

        self.means_ = np.array(self.means_)
        self.vars_ = np.array(self.vars_)
        self.prior_ = np.array(self.prior_)

        return self

    def log_prob(self, row, c_idx):
        mean = self.means_[c_idx]
        var = self.vars_[c_idx]
        prior = self.prior_[c_idx]
        log_prior = np.log(prior)
        log_feats = -.5 * np.log(2 * np.pi * var)
        log_feats = log_feats - ((row - mean) ** 2) / (2 * var)
        return log_prior + np.sum(log_feats)

    def predict(self, X):
        X = np.array(X, dtype=float)
        predicts = []

        for r in X:
            c_score = []
            for c_idx in range(len(self.class_)):
                c_score.append(self.log_prob(r, c_idx))
            
            top_idx = np.argmax(c_score)
            predicts.append(self.class_[top_idx])
                            
        return np.array(predicts)

    def score(self, X, y):
        predicts = self.predict(X)
        return np.mean(predicts == np.array(y))
