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

def avg_score(scores, supports, avg):
    scores = np.array(scores)
    supports = np.array(supports)

    if avg == "macro":
        return np.mean(scores)
    
    if avg == "weighted":
        total = np.sum(supports)

        if total == 0:
            return 0
        
        return np.sum(scores*supports) / total
    
    return scores

def precision_score(y_t, y_p, avg="macro", labels=None):
    labels = class_labels(y_t, y_p, labels)
    cmatrix = confusion_matrix(y_t, y_p, labels=labels)

    scores=[]
    supports=[]

    for i in range(len(labels)):
        tposi = cmatrix[i,i]
        fposi = np.sum(cmatrix[:,i]) - tposi

        support = np.sum(cmatrix[i, :])

        if tposi + fposi == 0:
            precis = 0
        else:
            precis = tposi / (tposi+fposi)
        
        scores.append(precis)
        supports.append(support)
    return avg_score(scores, supports, avg)

def recall_score(y_t, y_p, avg="macro", labels=None):
    labels = class_labels(y_t, y_p, labels)
    cmatrix = confusion_matrix(y_t, y_p, labels=labels)

    scores = []
    supports = []

    for i in range(len(labels)):
        tposi = cmatrix[i,i]
        fneg = np.sum(cmatrix[i,:]) - tposi
        support = np.sum(cmatrix[i,:])

        if tposi + fneg == 0:
            re = 0
        else:
            re = tposi / (tposi+fneg)
        scores.append(re)
        supports.append(support)

    return avg_score(scores, supports, avg)

def f1_score(y_t, y_p, avg="macro", labels=None):
    labels = class_labels(y_t, y_p, labels)
    cmatrix = confusion_matrix(y_t, y_p, labels=labels)

    scores = []
    supports = []

    for i in range(len(labels)):
        tposi = cmatrix[i,i]
        fposi = np.sum(cmatrix[:,i]) - tposi
        fneg = np.sum(cmatrix[i,:]) - tposi
        support = np.sum(cmatrix[i,:])

        if tposi + fposi == 0:
            precis = 0
        else:
            precis = tposi / (tposi+fposi)
        
        if tposi + fneg ==0:
            re = 0
        else:
            re = tposi / (tposi+fneg)
        
        if precis + re == 0:
            f1 = 0
        else:
            f1 = 2*precis*re/(precis+re)

        scores.append(f1)
        supports.append(support)
    return avg_score(scores, supports, avg)
