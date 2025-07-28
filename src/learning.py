import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import pickle
from hyper import search_hyperparameters_cv

import pandas as pd

class RidgeWithNorm:
    def __init__(self, alpha, layer, demean_x=True, demean_y=True):
        self.model = Ridge(alpha)
        self.demean_x, self.demean_y = demean_x, demean_y
        self.layer = layer
        
    def fit(self, x, y):
        x = x[self.layer]
        if self.demean_x:
            self.mean_x = x.mean(axis=0)
        else:
            self.mean_x = 0
        if self.demean_y:
            self.mean_y = y.mean()
        else:
            self.mean_y = 0
        self.model.fit(x - self.mean_x, y - self.mean_y)
        
    def predict(self, x):
        x = x[self.layer]
        return self.model.predict(x - self.mean_x) + self.mean_y

def nested_xval(x, y, folds, hyper_fn, model_cls, metric_fns, save_search_results=True, save_preds=False):
    all_preds = np.zeros_like(y)
    fold_results = []
    for fold in np.unique(folds):
        train_mask = (folds != fold)
        test_mask = (folds == fold)
        
        y_train, y_test = y[train_mask], y[test_mask]
        if isinstance(x, dict):
            x_train = {k: v[train_mask] for k,v in x.items()}
            x_test = {k: v[test_mask] for k,v in x.items()}
        else:
            x_train, x_test = x[train_mask], x[test_mask]

        best_model, search_results = search_hyperparameters_cv(x_train, y_train, folds[train_mask], metric_fns[0], hyper_fn, model_cls)
        best_model.fit(x_train, y_train)
        all_preds[test_mask] = best_model.predict(x_test)
        if save_search_results:
            fold_results.append(search_results)
    metrics = [m(y, all_preds) for m in metric_fns]
    results = {'metrics': metrics}
    if save_search_results:
        results['search_results'] = fold_results
    if save_preds:
        results['preds'] = all_preds
        results['y'] = y
    return results