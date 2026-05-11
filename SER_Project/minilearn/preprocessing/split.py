import numpy as np

def train_test_split(X,y,test_size=.2,stratify=None):
    X = np.array(X)
    y = np.array(y)
    n_rows = len(X)

    if stratify is None:
        ind = np.arange(n_rows)
        np.random.shuffle(ind)
        n_test = int(round(n_rows*test_size))

        if n_test < 1:
            n_test = 1

        test_ind = ind[:n_test]
        train_ind = ind[n_test:]

    else:
        stratify = np.array(stratify)
        train_list = []
        test_list = []

        for class_lab in np.unique(stratify):
            class_ind = np.where(stratify == class_lab)[0]
            np.random.shuffle(class_ind)
            n_test = int(round(len(class_ind)))

            if n_test < 1 and len(class_ind) > 1:
                n_test=1
            if n_test >= len(class_ind):
                n_test = len(class_ind) -1
            
            test_list.extend(class_ind[:n_test])
            test_list.extend(class_ind[n_test:])

        train_ind = np.array(train_list)
        test_ind = np.array(test_list)
        np.random.shuffle(train_ind)
        np.random.shuffle(test_ind)

    
    return X[train_ind], X[test_ind], y[train_ind], y[test_ind]