from data import load_fmri, load_activations
import numpy as np
from hyper import GridSearch
from learning import RidgeWithNorm, nested_xval
from sklearn.metrics import r2_score
from tqdm import tqdm
from functools import partial

FMRI_DATA = '/home/lpepino/braindnn/tp-picml/auditory_brain_dnn/data/neural/NH2015'
ACTIVATION_DATA = '/home/lpepino/braindnn/tp-picml/auditory_brain_dnn/model_actv/mel256-ec-base'
LAYER = 8

fmri_data = load_fmri(FMRI_DATA)
activations = load_activations(ACTIVATION_DATA, fmri_data['stimuli_metadata'])

folds = np.random.permutation(np.repeat(np.arange(0,5),33))

def create_grid():
    return GridSearch(grid = [{'alpha': xi} for xi in [0.01,0.1,1.0,5.0,10.0,100.0]],
                      extend_edges=True)

hyp_fn = create_grid
x = activations[LAYER]

voxel_results = []
NUM_VOXELS = fmri_data['voxel_features'].shape[1]
for i in tqdm(range(NUM_VOXELS)):
    y = fmri_data['voxel_features'][:,i].mean(axis=-1)
    result_i = nested_xval(x,y,folds,hyp_fn,RidgeWithNorm,[r2_score])
    voxel_results.append(result_i)
    
from IPython import embed; embed()

