import numpy as np

def accuracy_score(y_t, y_p):
    y_t = np.array(y_t)
    y_p = np.array(y_p)
    return np.mean(y_t == y_p)