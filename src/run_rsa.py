import warnings
import numpy as np

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from data import load_fmri, load_activations

from learning import rsa_layer_select, RSA

from tqdm import tqdm
from functools import partial
import joblib
from sklearn.metrics import r2_score

from joblib import Parallel, delayed
from contextlib import contextmanager
from joblib.parallel import BatchCompletionCallBack
import joblib.parallel
from pathlib import Path

from specs import layer_map, ALL_MODELS
from learning import RSA

import fire

def run_rsa(model='all', dataset='NH2015', output_dir='results', method='rdm_xval', roi='all'):
    fold_file=Path(__file__, '../../lists/stratified-fold-assignment.pkl').resolve()
    fmri_dir=Path(__file__, '../../data/neural').resolve()

    fmri_data = load_fmri(fmri_dir, dataset)
    folds = joblib.load(fold_file)

    if model == 'all':
        models = ALL_MODELS + ['topline']
    else:
        models = [model]
    folds = joblib.load(fold_file)
    voxel_feats = np.mean(fmri_data['voxel_features'], axis=-1)
    subj_ids = fmri_data['voxel_metadata']['subj_idx'].values
    roi_metadata = fmri_data['roi_metadata'].set_index('voxel_id')
    rois = [roi_metadata.loc[vid]['roi_label_general'] if vid in roi_metadata.index else None for vid in fmri_data['voxel_metadata']['voxel_id'].values]
    
    if roi == 'all':
        roi_suffix = ''
    else:
        roi_suffix = roi
    
    
    for m in models:
        if not Path(output_dir,m,f'RSA_{dataset}{roi_suffix}.pkl').exists():
            if m!='topline':
                activation_dir = Path(output_dir, m, 'activations', 'natural_sounds')
                layer_filter = layer_map.get(m)
                activations = load_activations(activation_dir, fmri_data['stimuli_metadata'], layer_filter)
                if method == 'rdm_xval':
                    subjects_r = []
                    test_rdms = []
                    layer_selections = []
                    for s in np.unique(subj_ids):
                        if roi != 'all':
                            mask = np.logical_and(subj_ids == s,np.array(rois)==roi)
                        else:
                            mask = subj_ids==s
                        voxel_subset = voxel_feats[:, mask]
                        is_nan = np.isnan(voxel_subset).sum(axis=0) > 0
                        voxel_subset = voxel_subset[:, ~is_nan]
                        subject_r, test_rdm, layer_idxs = rsa_layer_select(voxel_subset, activations, folds=folds)
                        subjects_r.append(subject_r)
                        test_rdms.append(test_rdm)
                        layer_selections.append(layer_idxs)
                elif method == 'rdm_layerwise':
                    subjects_r = []
                    for k,v in activations.items():
                        for s in np.unique(subj_ids):
                            if roi != 'all':
                                mask = np.logical_and(subj_ids == s,np.array(rois)==roi)
                            else:
                                mask = subj_ids==s
                            voxel_subset = voxel_feats[:, mask]
                            is_nan = np.isnan(voxel_subset).sum(axis=0) > 0
                            rsa_model = RSA()
                            rsa_model.fit(voxel_subset, v)
                            subjects_r.append({'subj_id': s, 'subj_r': rsa_model.r, 
                                                'layer': k})
                    layer_selections = None
                    test_rdms = None
            else:
                voxel_metadata = fmri_data['voxel_metadata'].reset_index()
                subject_rdms = []
                subjects_r = []
                for s in np.unique(subj_ids):
                    voxels_s = voxel_metadata.loc[voxel_metadata['subj_idx']==s]
                    voxel_idxs_s = voxels_s['index']
                    s_features = voxel_feats[:,voxel_idxs_s]
                    rsa = RSA()
                    s_rdm = rsa.calculate_rdm(s_features)
                    if np.isnan(s_rdm).sum()==0:
                        subject_rdms.append(s_rdm)
                subject_rdms = np.array(subject_rdms)
                for i in range(subject_rdms.shape[0]):
                    mask = np.zeros(subject_rdms.shape[0], dtype=bool)
                    mask[i] = 1
                    rdm_y = subject_rdms[mask].mean(axis=0)
                    rdm_x = subject_rdms[~mask].mean(axis=0)
                    rsa = RSA()
                    subjects_r.append(rsa.calculate_matrix_distance(rdm_x, rdm_y))
                test_rdms = None
                layer_selections = None
                
            outdir = Path(output_dir, m)
            Path(outdir).mkdir(parents=True, exist_ok=True)
            rsa_results = {'subject_test_rdms': test_rdms,
                            'subjects_r': subjects_r,
                            'subjects_layer_selection': layer_selections}
            joblib.dump(rsa_results, Path(outdir,f'RSA_{dataset}{roi_suffix}.pkl'))

if __name__ == '__main__':
    fire.Fire(run_rsa)