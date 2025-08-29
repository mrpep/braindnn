import warnings
import numpy as np

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)  # Just in case
warnings.filterwarnings("ignore", category=UserWarning)

from data import load_fmri, load_activations

from learning import rsa_layer_select

from tqdm import tqdm
from functools import partial
import joblib
from sklearn.metrics import r2_score

from joblib import Parallel, delayed
from contextlib import contextmanager
from joblib.parallel import BatchCompletionCallBack
import joblib.parallel

from specs import layer_map

MODELS = ['DS2', 'mel256-ec-large']
FOLD_FILE = '/home/lpepino/braindnn/braindnn-enhanced/lists/stratified-fold-assignment.pkl'

folds = joblib.load(FOLD_FILE)
for MODEL in MODELS:
    DATASET='NH2015'
    #MODEL='mel256-ec-base'
    FMRI_DATA = f'/home/lpepino/braindnn/tp-picml/auditory_brain_dnn/data/neural/{DATASET}'
    ACTIVATION_DATA = f'/home/lpepino/braindnn/tp-picml/auditory_brain_dnn/model_actv/{MODEL}'
    layer_filter = layer_map.get(MODEL)

    fmri_data = load_fmri(FMRI_DATA)
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


    