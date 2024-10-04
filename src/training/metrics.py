import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


def compute_roc_auc(target, preds):
    score = roc_auc_score(target, preds)
    return score

def compute_f1_score(target, preds):
    preds = (preds > 0.5).astype(int)
    target = (target > 0.5).astype(int)
    if len(preds.shape) > 1:
        score = f1_score(target, preds, average='macro')
    else:
        score = f1_score(target, preds)
    return score
