import numpy as np
import pandas as pd

class GridSearch:
    def __init__(self, grid, extend_edges=True):
        self.grid = grid
        self.extend_edges = extend_edges
        if self.extend_edges:
            vals_by_arg = {}
            for g in self.grid:
                for k,v in g.items():
                    if k in vals_by_arg:
                        vals_by_arg[k].append(v)
                    else:
                        vals_by_arg[k] = [v]
                    
            self.bounds = {k: [np.min(v), np.max(v)] for k,v in vals_by_arg.items()}
        else:
            self.bounds = None

    def find(self, fn):
        results = np.array([fn(xi) for xi in self.grid])

        idx_best = np.argmax(results)
        if self.extend_edges is not None:
            while True:
                new_hyp = {}
                extend = False
                for k,v in self.grid[idx_best].items():
                    if (v in self.bounds[k]) and (k in self.extend_edges):
                        extend = True
                        if v == self.bounds[k][0]:
                            new_hyp[k] = v / 2
                            self.bounds[k][0] = v/2
                        else:
                            new_hyp[k] = v*2
                            self.bounds[k][1] = v*2
                    else:
                        new_hyp[k] = v
                if extend:
                    self.grid.append(new_hyp)
                    results = np.append(results,fn(new_hyp))
                    idx_best = np.argmax(results)
                else:
                    break
        results_df = pd.DataFrame(self.grid)
        results_df['score'] = results
        best_model = self.grid[idx_best]

        return best_model, results_df

def search_hyperparameters_cv(x, y, folds, metric_fn, hyper_fn, model_cls):
    def train_one(hyp):
        all_preds = np.zeros_like(y)
        for fold in np.unique(folds):
            test_mask = (folds == fold)
            train_mask = (folds != fold)
            y_train, y_test = y[train_mask], y[test_mask]
            if isinstance(x, dict):
                x_train = {k: v[train_mask] for k,v in x.items()}
                x_test = {k: v[test_mask] for k,v in x.items()}
            else:
                x_train, x_test = x[train_mask], x[test_mask]
            model = model_cls(**hyp)
            model.fit(x_train, y_train)
            all_preds[test_mask] = model.predict(x_test)
        return metric_fn(y, all_preds)
    
    hyperopt = hyper_fn()
    best_model, results = hyperopt.find(train_one)
    best_model = model_cls(**best_model)
    return best_model, results

