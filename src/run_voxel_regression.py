import warnings
import numpy as np

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)  # Just in case
warnings.filterwarnings("ignore", category=UserWarning)

from data import load_fmri, load_activations

from hyper import GridSearch
from learning import RidgeWithNorm, nested_xval
from tqdm import tqdm
from functools import partial
import joblib
from sklearn.metrics import r2_score

from joblib import Parallel, delayed
from contextlib import contextmanager
from joblib.parallel import BatchCompletionCallBack
import joblib.parallel

from specs import layer_map, ALL_MODELS

@contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()

def r2_score(x,y):
    if np.std(x) == 0:
        return 0
    else:
        r = np.corrcoef(x.ravel(), y.ravel())[1,0]
        if r<0:
            r = 0
        return r**2

#from sklearn.metrics import r2_score
#MODELS = ['mel256-ec-base-step-500000']
#Falta 'VGGish'
MODELS = ALL_MODELS
DATASET='NH2015comp'
if DATASET == 'NH2015comp':
    NUM_JOBS=1
else:
    NUM_JOBS=30
for MODEL in MODELS:
    #MODEL='mel256-ec-base'
    FMRI_DATA = f'/home/lpepino/braindnn/tp-picml/auditory_brain_dnn/data/neural'
    ACTIVATION_DATA = f'/home/lpepino/braindnn/tp-picml/auditory_brain_dnn/model_actv/{MODEL}'
    FOLD_FILE = '/home/lpepino/braindnn/braindnn-enhanced/lists/stratified-fold-assignment.pkl'
    layer_filter = layer_map.get(MODEL)

    fmri_data = load_fmri(FMRI_DATA, DATASET)
    activations = load_activations(ACTIVATION_DATA, fmri_data['stimuli_metadata'], layer_filter)
    #folds = np.random.permutation(np.repeat(np.arange(0,5),33))
    folds = joblib.load(FOLD_FILE)

    def create_grid():
        return GridSearch(grid = [{'alpha': alpha_i,
                                'layer': layer_i} for alpha_i in [0.01,0.05,0.1,0.5,1.0,5.0,10.0,50.0] for layer_i in activations.keys()],
                          extend_edges=['alpha'],
                          edge_limits={'alpha': [1e-49, 1e50]})

    def regress_voxel(i):
        import warnings
        warnings.filterwarnings("ignore")
        if fmri_data['voxel_features'].ndim == 3:
            y = fmri_data['voxel_features'][:,i].mean(axis=-1)
        else:
            y = fmri_data['voxel_features'][:,i]
        if np.any(np.isnan(y)):
            print(f'Ignoring voxel {i} as it contains NANs')
            return None
        else:
            result_i = nested_xval(activations,y,folds,hyp_fn,RidgeWithNorm,[r2_score])
            result_i['voxel_id'] = i
            return result_i

    hyp_fn = create_grid
    NUM_VOXELS = fmri_data['voxel_features'].shape[1]

    with tqdm_joblib(tqdm(desc="Training models", total=NUM_VOXELS)) as progress_bar:
        results = Parallel(n_jobs=NUM_JOBS)(
            delayed(regress_voxel)(i) for i in range(NUM_VOXELS)
        )
    voxel_results = [r for r in results if r is not None]
        
    joblib.dump(voxel_results, f'{DATASET}_{MODEL}.pkl')
