import numpy as np

class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_r_ = None

    def fit(self, X):
        X = np.array(X, dtype=float)

        self.mean_ = np.mean(X, axis=0)

        X_center = X - self.mean_

        covar = np.cov(X_center, rowvar=False)

        #eigenvalues
        eig_val, eig_vec = np.linalg.eigh(covar)

        #Sort but from large to small
        idx = np.argsort()[::-1]

        eig_val = eig_val[idx]
        eig_vec = eig_vec[:,idx]
        # rows and n_components cols then transposed
        self.components_ = eig_vec[:, :self.n_components].T
        self.explained_variance_ = eig_val[:self.n_components]

        total_var = np.sum(eig_val)
        self.explained_variance_r_ = self.explained_variance_ / total_var

        return self

