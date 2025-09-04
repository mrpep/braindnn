import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import pickle
from hyper import search_hyperparameters_cv
from scipy import stats

import pandas as pd

import torch
import pytorch_lightning as pl

from metrics import metric_map

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

class RSA:
    def __init__(self, 
                 pairwise_distance='pearson', 
                 rdm_distance='spearman',
                 zscore=True):
        self.pairwise_distance, self.rdm_distance = pairwise_distance, rdm_distance
        self.zscore = zscore

    def calculate_rdm(self, x, eps=1e-10):
        if self.zscore:
            x = (x - np.mean(x, axis=0))/(np.std(x, axis=0)+eps)
        if self.pairwise_distance == 'pearson':
            return 1 - np.corrcoef(x)
        elif self.pairwise_distance == 'spearman':
            rdm, _ = stats.spearmanr(x.T)
            return 1 - rdm

    def calculate_matrix_distance(self, x, y, mask=None):
        upper_tri = np.triu_indices_from(x, k=1)
        if mask is not None:
            combined_mask = mask[upper_tri]
            upper_tri = (upper_tri[0][combined_mask], upper_tri[1][combined_mask])
        if self.rdm_distance == 'pearson':
            r, _ = stats.pearsonr(x[upper_tri], y[upper_tri])
        elif self.rdm_distance == 'spearman':
            r, _ = stats.spearmanr(x[upper_tri], y[upper_tri])
        return r

    def fit(self, x, y):
        self.rdm_x = self.calculate_rdm(x)
        self.rdm_y = self.calculate_rdm(y)

        self.r = self.calculate_matrix_distance(self.rdm_x, self.rdm_y)      

class MLP(pl.LightningModule):
    def __init__(self, num_embeddings=1,
                 input_dim=768,
                 hidden_dims=[1024,1024],
                 norm_after_activation=False,
                 hidden_norm=torch.nn.BatchNorm1d,
                 dropout=0.1,
                 num_classes=4,
                 lr=1e-4,
                 metrics=None,
                 initialization=torch.nn.init.xavier_uniform_,
                 prediction_type='multiclass'):
        super().__init__()
        self.lw = torch.nn.Parameter(torch.ones(num_embeddings), requires_grad=True)
        layers = []
        hi = [input_dim] + hidden_dims[:-1]
        ho = hidden_dims

        for hii, hoi in zip(hi,ho):
            #ToDo: Agregar inits
            linear = torch.nn.Linear(hii,hoi)
            initialization(linear.weight,
                                gain=torch.nn.init.calculate_gain('linear'))
            layers.append(linear)
            if not norm_after_activation:
                layers.append(hidden_norm(hoi))
            layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.ReLU())
            if norm_after_activation:
                layers.append(hidden_norm(hoi))
        self.hidden = torch.nn.Sequential(*layers)
        self.out_layer = torch.nn.Linear(hidden_dims[-1], num_classes)
        initialization(self.out_layer.weight,
                            gain=torch.nn.init.calculate_gain('relu'))

        if prediction_type == 'multiclass':
            self.loss = torch.nn.CrossEntropyLoss()
        else:
            self.loss = torch.nn.BCEWithLogitsLoss()
        self.lr = lr

        self.metrics = metrics
        self.val_preds = []
        self.test_preds = []

    def forward(self, batch):
        x = batch['embeddings']
        if self.lw is not None:
            lw = torch.abs(self.lw)
            lw = lw/torch.sum(lw)
            x = torch.sum(x * lw[None,:,None], dim=1)
        x = self.hidden(x)
        x = self.out_layer(x)
        batch['y_hat'] = x

    def validation_step(self, batch):
        self(batch)
        self.val_preds.append({'yhat': batch['y_hat'], 
                               'y': batch['y']})
        loss = self.loss(batch['y_hat'], batch['y'])
        self.log('val_loss', loss)

    def training_step(self, batch):
        self(batch)
        loss = self.loss(batch['y_hat'], batch['y'])
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch):
        self(batch)
        self.test_preds.append({'yhat': batch['y_hat'], 
                               'y': batch['y']})
        loss = self.loss(batch['y_hat'], batch['y'])
        self.log('test_loss', loss)

    def on_validation_epoch_end(self):
        y = torch.cat([yi['y'] for yi in self.val_preds]).detach().cpu().numpy()
        ypred = torch.cat([yi['yhat'] for yi in self.val_preds]).detach().cpu().numpy()
        self.val_preds = []
        for m in self.metrics:
            if m in metric_map:
                self.log(f'val_{m}', metric_map[m](y, ypred))

    def on_test_epoch_end(self):
        y = torch.cat([yi['y'] for yi in self.test_preds]).detach().cpu().numpy()
        ypred = torch.cat([yi['yhat'] for yi in self.test_preds]).detach().cpu().numpy()
        self.test_preds = []
        for m in self.metrics:
            if m in metric_map:
                self.log(f'test_{m}', metric_map[m](y, ypred))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


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

def rsa_layer_select(x,y,selection_method='cv', folds=None):
    unique_folds = np.unique(folds)
    test_rdm = np.zeros((x.shape[0], x.shape[0]))
    selected_layer = np.zeros((x.shape[0], x.shape[0]))
    layer_names = list(y.keys())
    layer_rdms = []
    rsa_model = RSA()
    fmri_rdm = rsa_model.calculate_rdm(x)
    for l in layer_names:
        rdm = rsa_model.calculate_rdm(y[l])
        layer_rdms.append(rdm)
    
    for i in range(len(unique_folds)):
        for j in range(i, len(unique_folds)):
            mask_stimuli_1 = folds == unique_folds[i]
            mask_stimuli_2 = folds == unique_folds[j]
            rdm_train_mask = ~mask_stimuli_1[:,None] * ~mask_stimuli_2[None,:]
            train_r = [rsa_model.calculate_matrix_distance(fmri_rdm, rdm_y, mask=rdm_train_mask) for rdm_y in layer_rdms]
            test_rdm[~rdm_train_mask] = layer_rdms[np.argmax(train_r)][~rdm_train_mask]
            selected_layer[~rdm_train_mask] = np.argmax(train_r)
    final_r = rsa_model.calculate_matrix_distance(fmri_rdm, test_rdm)
    return final_r, test_rdm, selected_layer
