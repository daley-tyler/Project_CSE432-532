import numpy as np

class PCA:
    def __init__(self, n_components=2):
        self.n_comonents = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_r_ = None