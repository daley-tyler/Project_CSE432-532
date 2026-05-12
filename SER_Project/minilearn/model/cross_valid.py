import numpy as np

def k_fold_split(y, n_splits=5, shuffle=True):
    y = np.array(y)
    folds = []

    for i in range(n_splits):
        folds.append([])

    for c in np.unique(y):
        c_idx = np.where(y==c)[0]

        if shuffle == True:
            np.random.shuffle(c_idx)
        
        for i in range(len(c_idx)):
            f_num = i % n_splits
            folds[f_num].append(c_idx[i])