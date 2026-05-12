import numpy as np

def accuracy_score(y_t, y_p):
    y_t = np.array(y_t)
    y_p = np.array(y_p)
    return np.mean(y_t == y_p)

def class_labels(y_t, y_p, labels=None):
    if labels is not None:
        return np.array(labels)
    
    b_labs = list(y_t) + list(y_p)
    return np.unique(np.array(b_labs))

def confusion_matrix(y_t, y_p, labels=None):
    y_t = np.array(y_t)
    y_p = np.array(y_p)

    labels = class_labels(y_t, y_p, labels)

    ematrix = np.zeros((len(labels), len(labels)), dtype=int)

    for i in range(len(y_t)):
        true_v = y_t[i]
        pred_v = y_p[i]
        true_idx = np.where(labels == true_v)[0][0]
        pred_idx = np.where(labels == pred_v)[0][0]

        ematrix[true_idx, pred_idx] = ematrix[true_idx, pred_idx] + 1

    return ematrix

def precision_score

def recall

def F1_score

