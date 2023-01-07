# The evaluation metric but probably not the best loss...
# Source: https://aclanthology.org/2020.eval4nlp-1.9.pdf
import torch
import numpy as np

def pfbeta(labels, predictions, beta):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if (labels[idx]):
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0

#https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/369267  
def pfbeta_torch(preds, labels, beta=1):
    # if preds.dim() != 2 or (preds.dim() == 2 and preds.shape[1] !=2): raise ValueError('Houston, we got a problem')
    # preds = preds[:, 1]
    preds = preds.clip(0, 1)
    y_true_count = labels.sum()
    ctp = preds[labels==1].sum()
    cfp = preds[labels==0].sum()
    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return torch.tensor(result)
    else:
        return torch.tensor(0.0)

# https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/369886    
def pfbeta_torch_thresh(preds, labels):
    optimized_preds = optimize_preds(preds, labels)
    return pfbeta_torch(optimized_preds, labels)

def optimize_preds(preds, labels=None, thresh=None, return_thresh=False, print_results=False):
    preds = preds.clone()
    if labels is not None: without_thresh = pfbeta_torch(preds, labels)
    
    if not thresh and labels is not None:
        threshs = np.linspace(0, 1, 101)
        f1s = [pfbeta_torch((preds > thr).float(), labels) for thr in threshs]
        idx = np.argmax(f1s)
        thresh, best_pfbeta = threshs[idx], f1s[idx]

    preds = (preds > thresh).float()

    if print_results:
        print(f'without optimization: {without_thresh}')
        pfbeta = pfbeta_torch(preds, labels)
        print(f'with optimization: {pfbeta}')
        print(f'best_thresh = {thresh}')
    if return_thresh:
        return thresh
    return preds
