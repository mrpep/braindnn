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
import fire

from pathlib import Path

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

def run_regression(model='all', dataset='NH2015', output_dir='results'):
    if model == 'all':
        models = ALL_MODELS + ['topline']
    else:
        models = [model]

    if dataset == 'NH2015comp':
        num_jobs=1
    else:
        num_jobs=30

    fold_file=Path(__file__, '../../lists/stratified-fold-assignment.pkl').resolve()
    fmri_dir=Path(__file__, '../../data/neural').resolve()
    fmri_data = load_fmri(fmri_dir, dataset)
    folds = joblib.load(fold_file)

    for m in models:
        activation_dir = f'/home/lpepino/braindnn/tp-picml/auditory_brain_dnn/model_actv/{m}'
        layer_filter = layer_map.get(m)
        activations = load_activations(activation_dir, fmri_data['stimuli_metadata'], layer_filter)

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
        num_voxels = fmri_data['voxel_features'].shape[1]

        with tqdm_joblib(tqdm(desc="Training models", total=num_voxels)) as progress_bar:
            results = Parallel(n_jobs=num_jobs)(
                delayed(regress_voxel)(i) for i in range(num_voxels)
            )
        voxel_results = [r for r in results if r is not None]

        outdir = Path(output_dir, m)
        Path(outdir).mkdir(parents=True, exist_ok=True)

        joblib.dump(voxel_results, Path(outdir,f'REG_{dataset}.pkl'))

if __name__ == '__main__':
    fire.Fire(run_regression)