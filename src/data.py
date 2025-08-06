import numpy as np
from pathlib import Path
import pandas as pd
import pickle
import h5py

def load_fmri(path):
    voxel_data = np.load(Path(path, 'voxel_features_array.npy'))
    voxel_metadata = np.load(Path(path, 'voxel_features_meta.npy'))
    stimuli_metadata = np.load(Path(path, 'neural_stim_meta.npy'))
    
    return {'voxel_features': voxel_data,
            'voxel_metadata': pd.DataFrame(voxel_metadata),
            'stimuli_metadata': pd.DataFrame(stimuli_metadata).set_index('id')}
    
def load_activations(path, stimuli_metadata, layer_filter=None):
    h5_path = Path(path, 'natsound_activations.h5')
    activations_ = {}
    if h5_path.exists():
        with h5py.File(h5_path, 'r') as f:
            if layer_filter is not None:
                keys = layer_filter
            else:
                keys = f.keys()
            for k in keys:
                try:
                    activations_[k] = f[k][:]
                except:
                    from IPython import embed; embed()
    else:
        activations = []
        for idx, row in stimuli_metadata.iterrows():
            filename = row['filename'].decode('utf-8').split('.')[0] + '_activations.pkl'
            with open(Path(path, filename), 'rb') as f:
                activations.append(pickle.load(f))
        for k in activations[0].keys():
            activations_[k] = np.array([a[k] for a in activations])

    #if layer_filter is not None:
    #
    # activations_ = {k:v for k,v in activations_.items() if k in layer_filter}
    return activations_
