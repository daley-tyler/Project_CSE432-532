import numpy as np

class DecisionTreeClassifier:
    def __init__(self, max_depth=3, min_samples_split=2, max_thresholds=20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_thresholds = max_thresholds
        self.tree_ = None
        self.class_ = None

    def gini(self, y):
        y = np.array(y)

        g = 1

        for c in np.unique(y):
            p = np.mean(y==c)
            g = g-p ** 2
        
        return g
    
    def m_class(self,y):
        labels, counts = np.unique(y, return_counts=True)
        b_idx = np.argmax(counts)
        return labels[b_idx]
    
    