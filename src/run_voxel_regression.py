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
MODELS = ['wav2vec2',
          'ZeroSpeech2020']
for MODEL in MODELS:
    DATASET='NH2015'
    #MODEL='mel256-ec-base'
    FMRI_DATA = f'/home/lpepino/braindnn/tp-picml/auditory_brain_dnn/data/neural/{DATASET}'
    ACTIVATION_DATA = f'/home/lpepino/braindnn/tp-picml/auditory_brain_dnn/model_actv/{MODEL}'
    FOLD_FILE = '/home/lpepino/braindnn/braindnn-enhanced/lists/stratified-fold-assignment.pkl'

    fmri_data = load_fmri(FMRI_DATA)
    activations = load_activations(ACTIVATION_DATA, fmri_data['stimuli_metadata'])

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
        y = fmri_data['voxel_features'][:,i].mean(axis=-1)
        if np.any(np.isnan(y)):
            print(f'Ignoring voxel {i} as it contains NANs')
            return None
        else:
            result_i = nested_xval(activations,y,folds,hyp_fn,RidgeWithNorm,[r2_score])
            result_i['voxel_id'] = i
            return result_i

    hyp_fn = create_grid
    NUM_VOXELS = fmri_data['voxel_features'].shape[1]
    #voxel_results = []

    with tqdm_joblib(tqdm(desc="Training models", total=NUM_VOXELS)) as progress_bar:
        results = Parallel(n_jobs=30)(
            delayed(regress_voxel)(i) for i in range(NUM_VOXELS)
        )
    voxel_results = [r for r in results if r is not None]
        
    joblib.dump(voxel_results, f'{DATASET}_{MODEL}.pkl')