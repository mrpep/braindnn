import numpy as np
from pathlib import Path
import pandas as pd
import pickle

def load_fmri(path):
    voxel_data = np.load(Path(path, 'voxel_features_array.npy'))
    voxel_metadata = np.load(Path(path, 'voxel_features_meta.npy'))
    stimuli_metadata = np.load(Path(path, 'neural_stim_meta.npy'))
    
    return {'voxel_features': voxel_data,
            'voxel_metadata': pd.DataFrame(voxel_metadata),
            'stimuli_metadata': pd.DataFrame(stimuli_metadata).set_index('id')}
    
def load_activations(path, stimuli_metadata):
    activations = []
    activations_ = {}
    for idx, row in stimuli_metadata.iterrows():
        filename = row['filename'].decode('utf-8').split('.')[0] + '_activations.pkl'
        with open(Path(path, filename), 'rb') as f:
            activations.append(pickle.load(f))
    for k in activations[0].keys():
        activations_[k] = np.array([a[k] for a in activations])

    return activations_