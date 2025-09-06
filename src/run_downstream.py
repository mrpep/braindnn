from pathlib import Path
from invariant.representations import AudioFeature
import fire
from tqdm import tqdm
import json
import librosa
import joblib
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from learning import MLP, DynamicPCA
from data import AudioDataset

from sklearn.model_selection import ParameterGrid
import shutil

hyp_confs = {
    'hidden_dims': [[1024], [1024,1024]],
    'lr': [3.2e-3, 1e-3, 3.2e-4, 1e-4],
    'initialization': [torch.nn.init.xavier_uniform_, torch.nn.init.xavier_normal_]
}

def extract_task_embeddings(upstream_model, task_dir, output_dir):
    data_filenames = []
    for split in Path(task_dir, str(upstream_model.sr)).glob('*'):
        embeddings = {}
        labels = {}
        outdir = Path(output_dir, split.stem + '.pkl')
        data_filenames.append(outdir)
        outdir.parent.mkdir(parents=True, exist_ok=True)

        if not outdir.exists():
            with open(Path(task_dir, split.stem+'.json'), 'r') as f:
                label_data = json.load(f)
            for fname in tqdm(split.rglob('*.wav')):
                x, fs = librosa.core.load(fname, sr=upstream_model.sr)
                feats = upstream_model(x)
                embeddings[fname.stem] = feats
                labels[fname.stem] = label_data[fname.name]

            joblib.dump({'embeddings': embeddings,
                        'labels': labels}, outdir)
    
    return data_filenames

def multilabel_to_ohv(data, num_labels):
    oh = np.zeros((len(data), num_labels), dtype=np.float32)
    for i,xi in enumerate(data):
        oh[i,xi]=1.0
    return oh

def squeeze(x):
    if x.ndim == 2:
        return x[0]
    else:
        return x

def train_test_model(train_data_files,
                val_data_files,
                test_data_files,
                output_dir, label_type='single', 
                batch_size=128, 
                num_workers=4, 
                patience=20,
                max_epochs=200,
                metrics=None,
                fold=None,
                hyp_conf=None):
    #Load Data
    train_pkls = [joblib.load(f) for f in train_data_files]
    val_pkls = [joblib.load(f) for f in val_data_files]
    test_pkls = [joblib.load(f) for f in test_data_files]

    all_labels = set()

    train_labels = []
    test_labels = []
    val_labels = []

    train_embeddings = []
    test_embeddings = []
    val_embeddings = []

    for p in train_pkls:
        for k,v in p['labels'].items():
            all_labels = all_labels.union(set(v))
    label_map = {k:i for i,k in enumerate(all_labels)}
    layer_key_order = list(next(iter(train_pkls[0]['embeddings'].values())).keys())

    for p in train_pkls:
        for k,v in p['labels'].items():
            if label_type == 'multiclass':
                train_labels.append(label_map[v[0]])
            elif label_type == 'multilabel':
                train_labels.append([label_map[vi] for vi in v])
            else:
                from IPython import embed; embed()
            train_embeddings.append([squeeze(p['embeddings'][k][l]) for l in layer_key_order])
    for p in test_pkls:
        for k,v in p['labels'].items():
            if label_type == 'multiclass':
                test_labels.append(label_map[v[0]])
            else:
                test_labels.append([label_map[vi] for vi in v])
            test_embeddings.append([squeeze(p['embeddings'][k][l]) for l in layer_key_order])
    for p in val_pkls:
        for k,v in p['labels'].items():
            if label_type == 'multiclass':
                val_labels.append(label_map[v[0]])
            else:
                val_labels.append([label_map[vi] for vi in v])
            val_embeddings.append([squeeze(p['embeddings'][k][l]) for l in layer_key_order])    
    #Make array / PCA
    model_dims = [x.shape[0] for x in train_embeddings[0]]
    if len(set(model_dims)) == 1:
        train_embeddings = np.array(train_embeddings)
        test_embeddings = np.array(test_embeddings)
        val_embeddings = np.array(val_embeddings)
        model_dim = model_dims[0]
    else:
        #PCA thing
        pca_model = DynamicPCA(model_dims, variance_threshold=1.0)
        train_embeddings = pca_model.fit_transform(train_embeddings)
        test_embeddings = pca_model.transform(test_embeddings)
        val_embeddings = pca_model.transform(val_embeddings)
        model_dim = pca_model.num_components
    if label_type == 'multiclass':
        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)
        val_labels = np.array(val_labels)
    else:
        train_labels = multilabel_to_ohv(train_labels, len(label_map))
        test_labels = multilabel_to_ohv(test_labels, len(label_map))
        val_labels = multilabel_to_ohv(val_labels, len(label_map))

    #Instantiate model
    model = MLP(num_embeddings = len(model_dims),
                input_dim=model_dim,
                num_classes=len(label_map),
                metrics=metrics,
                prediction_type=label_type,
                **hyp_conf)

    #Instantiate datasets
    train_dataset = AudioDataset(train_embeddings, train_labels)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_dataset = AudioDataset(test_embeddings, test_labels)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    val_dataset = AudioDataset(val_embeddings, val_labels)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    #Instantiate trainer
    downstream_folder = list(Path(output_dir).parts)
    downstream_folder[-2] = 'downstream'
    downstream_folder = Path(*downstream_folder)
    log_folder = Path(downstream_folder, 'logs')
    ckpt_folder = Path(downstream_folder, 'ckpt')
    if fold is not None:
        log_folder = Path(log_folder, fold)
        ckpt_folder = Path(ckpt_folder, fold)
    log_folder.mkdir(parents=True, exist_ok=True)
    ckpt_folder.mkdir(parents=True, exist_ok=True)

    logger = CSVLogger(log_folder)
    model_ckpt = ModelCheckpoint(dirpath=ckpt_folder, monitor='val_{}'.format(metrics[0]), mode='max')
    early_stop_callback = EarlyStopping(
        monitor='val_{}'.format(metrics[0]),
        min_delta=0.00,
        patience=patience,
        check_on_train_epoch_end=False,
        verbose=False,
        mode='max',
    )
    trainer = pl.Trainer(devices=1, 
                         accelerator='gpu',
                         num_sanity_val_steps=0,
                         callbacks=[model_ckpt, early_stop_callback],
                         logger=logger,
                         max_epochs=max_epochs,
                         )

    #Fit model
    trainer.fit(model, train_dl, val_dl)
    best_ckpt = model_ckpt.best_model_path
    best_sd = torch.load(best_ckpt, map_location='cpu')['state_dict']
    model.load_state_dict(best_sd)
    val_score = trainer.validate(model, val_dl)[0]
    test_score = trainer.test(model, test_dl)[0]
    scores = {}
    scores.update(val_score)
    scores.update(test_score)
    scores['hyperparameters'] = hyp_conf
    scores['input_dim'] = model_dim
    return scores

def find_best_hyp(results, score_key='val_top1_acc'):
    scores = [r[score_key] for r in results]
    best_score = np.argmax(scores)
    best_hyp = results[best_score]['hyperparameters']

    return best_hyp, results[best_score]

def run_downstream(upstream_model, tasks_dir = '/mnt/data/hear-selected', output_dir='results',
                   remove_activations_after=True, remove_ckpts_after=True):
    model = AudioFeature(upstream_model, device='cuda:0')
    for task_dir in Path(tasks_dir).glob('*'):
        if not Path(output_dir, upstream_model,'downstream',task_dir.parts[-1],'results.pkl').exists():
            #Extract activations / labels
            outdir = Path(output_dir, upstream_model, 'activations', task_dir.parts[-1])
            outdir.mkdir(parents=True, exist_ok=True)
            data_files = extract_task_embeddings(model, task_dir, outdir)
            data_files = sorted(data_files)

            with open(Path(task_dir, 'task_metadata.json'), 'r') as f:
                task_metadata = json.load(f)
            metrics = task_metadata['evaluation']
            #Train models
            #ToDo: Implementar logica folds vs train/test splits
            all_results = []
            if 'fold' in data_files[0].stem:
                for i in range(len(data_files)):
                    test_fold = [data_files[i]]
                    val_fold = [data_files[(i+1)%len(data_files)]]
                    train_fold = [x for x in data_files if (x not in test_fold) and (x not in val_fold)]
                    if i==0: #Search hyperparams
                        param_grid = ParameterGrid(hyp_confs)
                        split_results = []
                        for k,hyp_conf in enumerate(param_grid):
                            results = train_test_model(train_fold, val_fold, test_fold, outdir, metrics=metrics, fold=f'fold-{i}_hyp-{k}', hyp_conf=hyp_conf, label_type=task_metadata['prediction_type'])
                            split_results.append(results)
                        best_conf, best_results = find_best_hyp(split_results, score_key='val_{}'.format(metrics[0]))
                        all_results.append(split_results)
                        all_results.append(best_results)
                    else:
                        results = train_test_model(train_fold, val_fold, test_fold, outdir, metrics=metrics, fold=f'fold-{i}_hyp-{k}', hyp_conf=best_conf, label_type=task_metadata['prediction_type'])
                        results['fold'] = i
                        all_results.append(results)
            else:
                train_fold = [d for d in data_files if d.stem == 'train']
                test_fold = [d for d in data_files if d.stem == 'test']
                val_fold = [d for d in data_files if d.stem == 'valid']

                param_grid = ParameterGrid(hyp_confs)
                split_results = []
                
                for k,hyp_conf in enumerate(param_grid):
                    results = train_test_model(train_fold, val_fold, test_fold, outdir, metrics=metrics, fold=f'hyp-{k}', hyp_conf=hyp_conf, label_type=task_metadata['prediction_type'])
                    split_results.append(results)
                best_conf, best_results = find_best_hyp(split_results, score_key='val_{}'.format(metrics[0]))
                all_results.append(split_results)
                all_results.append(best_results)
                
            results_path_parts = list(outdir.parts)
            results_path_parts[-2] = 'downstream'
            results_path = Path(*results_path_parts, 'results.pkl')
            joblib.dump(all_results, results_path)
            if remove_activations_after:
                shutil.rmtree(outdir)
            if remove_ckpts_after:
                shutil.rmtree(Path(output_dir, upstream_model, 'downstream', task_dir.parts[-1], 'ckpt'))

if __name__ == '__main__':
    fire.Fire(run_downstream)