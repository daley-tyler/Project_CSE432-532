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

    def log_prob():

    def predict():

    def score():