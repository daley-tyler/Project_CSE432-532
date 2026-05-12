import numpy as np

class DecisionTreeClassifier:
    def __init__(self, max_depth=3, min_samples_split=2, max_thresholds=20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_thresholds = max_thresholds
        self.tree_ = None
        self.class_ = None

    def gini(self, y):