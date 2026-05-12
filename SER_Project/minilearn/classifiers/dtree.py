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
    
    def get_thresholds(self, col):
        vals = np.unique(col)
        thresholds = []

        for i in range(len(vals) - 1):
            mid = (vals[i] + vals[i + 1]) / 2
            thresholds.append(mid)

        if len(thresholds) > self.max_thresholds:
            idxs = np.linspace(0, len(thresholds) - 1, self.max_thresholds)
            idxs = idxs.astype(int)

            small_thresholds = []

            for i in range(len(idxs)):
                small_thresholds.append(thresholds[idxs[i]])

            thresholds = small_thresholds

        return thresholds
    
    def best_split(self, X, y):
        best_feat = None
        best_thresh = None
        best_gini = self.gini(y)

        n_rows = X.shape[0]

        for feat in range(X.shape[1]):
            thresholds = self.get_thresholds(X[:, feat])
            for thresh in thresholds:
                left = X[:, feat] <= thresh
                right = X[:, feat] > thresh

                if np.sum(left) == 0 or np.sum(right) == 0:
                    continue

                left_gini = self.gini(y[left])
                right_gini = self.gini(y[right])
                split_gini = ((np.sum(left) / n_rows) * left_gini + (np.sum(right) / n_rows) * right_gini)

                if split_gini < best_gini:
                    best_gini = split_gini
                    best_feat = feat
                    best_thresh = thresh

        return best_feat, best_thresh
    
    def build_tree(self,X,y,depth):
        if len(np.unique(y)) == 1

            