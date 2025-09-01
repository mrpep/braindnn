import warnings
import numpy as np

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)  # Just in case
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

from specs import layer_map, ALL_MODELS

#MODELS = ALL_MODELS
MODELS = []
FOLD_FILE = '/home/lpepino/braindnn/braindnn-enhanced/lists/stratified-fold-assignment.pkl'
FMRI_DATA = f'/home/lpepino/braindnn/tp-picml/auditory_brain_dnn/data/neural'
DATASET='B2021'

fmri_data = load_fmri(FMRI_DATA, DATASET)
folds = joblib.load(FOLD_FILE)

for MODEL in MODELS:
    ACTIVATION_DATA = f'/home/lpepino/braindnn/tp-picml/auditory_brain_dnn/model_actv/{MODEL}'
    layer_filter = layer_map.get(MODEL)

    activations = load_activations(ACTIVATION_DATA, fmri_data['stimuli_metadata'], layer_filter)
    folds = joblib.load(FOLD_FILE)
    voxel_feats = np.mean(fmri_data['voxel_features'], axis=-1)
    subj_ids = fmri_data['voxel_metadata']['subj_idx'].values

    subjects_r = []
    test_rdms = []
    layer_selections = []
    for s in np.unique(subj_ids):
        voxel_subset = voxel_feats[:, subj_ids==s]
        is_nan = np.isnan(voxel_subset).sum(axis=0) > 0
        voxel_subset = voxel_subset[:, ~is_nan]
        subject_r, test_rdm, layer_idxs = rsa_layer_select(voxel_subset, activations, folds=folds)
        subjects_r.append(subject_r)
        test_rdms.append(test_rdm)
        layer_selections.append(layer_idxs)
    
    rsa_results = {'subject_test_rdms': test_rdms,
                   'subjects_r': subjects_r,
                   'subjects_layer_selection': layer_selections}
    joblib.dump(rsa_results, f'RSA_{DATASET}_{MODEL}.pkl')

#Topline
voxel_feats = np.mean(fmri_data['voxel_features'], axis=-1)
subj_ids = fmri_data['voxel_metadata']['subj_idx'].values
voxel_metadata = fmri_data['voxel_metadata'].reset_index()
#if 'coord_id' not in voxel_metadata.columns:
#    voxel_metadata['coord_id'] = voxel_metadata.apply(lambda x: '{}_{}'.format(x['x_ras'],x['y_ras']), axis=1)

subject_rdms = []
subject_r = []
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
    subject_r.append(rsa.calculate_matrix_distance(rdm_x, rdm_y))
    
rsa_results = {'subject_test_rdms': None,
                'subjects_r': subject_r,
                'subjects_layer_selection': None}
joblib.dump(rsa_results, f'RSA_{DATASET}_topline.pkl')