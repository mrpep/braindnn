import sklearn.metrics as skm
import numpy as np
from scipy import stats

def accuracy_score(y, ypred):
    ypred = np.argmax(ypred, axis=-1)
    if y.ndim == 2:
        y = np.argmax(y, axis=-1)
    return skm.accuracy_score(y, ypred)

def mAP(y, ypred):
    if y.ndim == 1:
        yoh = y[:,None] == np.arange(ypred.shape[-1])[None,:]
    else:
        yoh = y
    return skm.average_precision_score(yoh, ypred, average='macro')

def aucroc(y, ypred):
    if y.ndim == 1:
        yoh = y[:,None] == np.arange(ypred.shape[-1])[None,:]
    else:
        yoh = y
    return skm.roc_auc_score(yoh, ypred, average='macro')

def d_prime(y, ypred):
    if y.ndim == 1:
        yoh = y[:,None] == np.arange(ypred.shape[-1])[None,:]
    else:
        yoh = y
    auc = skm.roc_auc_score(yoh, ypred, average=None)
    d_prime = stats.norm().ppf(auc) * np.sqrt(2.0)
    d_prime_macro = np.mean(d_prime)
    
    return d_prime_macro

metric_map = {'top1_acc': accuracy_score,
              'mAP': mAP,
              'd_prime': d_prime,
              'aucroc': aucroc,
              'pitch_acc': accuracy_score}