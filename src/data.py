import numpy as np
from pathlib import Path
import pandas as pd
import pickle
import h5py
from scipy.io import loadmat
import torch
import joblib


NUM_STIMULI = 165
def load_fmri(path, dataset='B2021'):
    fmri_path = Path(path, dataset)
    stim_data = np.load(Path(path,'NH2015', 'neural_stim_meta.npy'))
    if dataset == 'B2021':
        voxel_data = np.load(Path(fmri_path, 'voxel_features_array.npy'))
        voxel_data = voxel_data[:NUM_STIMULI]
        voxel_metadata = pd.read_pickle(Path(fmri_path, 'df_roi_meta.pkl'))
        stimuli_metadata = loadmat(Path(fmri_path, 'stim_info_v4.mat'))['stim_info']
        stimuli_metadata = {'cat_assignment': [stimuli_metadata[0][0][0][idx-1][0][0][0] for idx in stimuli_metadata[0][0][1][:NUM_STIMULI]],
                            'cat_assignment_idx': stimuli_metadata[0][0][1][:NUM_STIMULI,0],
                            'fmri_stim_num': stimuli_metadata[0][0][3][:NUM_STIMULI,0],
                            'filename': [x[0][0]+'.wav' for x in stimuli_metadata[0][0][7][:NUM_STIMULI]],
                            'embedded-wav-end-idx': 32000,
                            'embedded-wav-start-idx': 0,
                            'fmri_stim_idx': np.arange(NUM_STIMULI),
                            'id': np.arange(NUM_STIMULI)}
    elif dataset == 'NH2015comp':
        data = loadmat(Path(fmri_path, 'components.mat'))
        response = data['R']
        stim_names = [x[0]+'.wav' for x in data['stim_names'][0]]
        stim_order = {x: i for i,x in enumerate(stim_names)}
        stim_reorder = [stim_order[s.decode('utf8')] for s in stim_data['filename'][:]]
        
        voxel_data = response[stim_reorder]
        voxel_metadata = pd.read_pickle(Path(fmri_path,'df_roi_meta.pkl'))
        stimuli_metadata = stim_data
    elif dataset == 'NH2015':
        voxel_data = np.load(Path(fmri_path, 'voxel_features_array.npy'))
        voxel_metadata = np.load(Path(fmri_path, 'voxel_features_meta.npy'))
        stimuli_metadata = np.load(Path(fmri_path, 'neural_stim_meta.npy'))
    else:
        raise Exception(f'Unknown dataset name: {dataset}')
    return {'voxel_features': voxel_data,
            'voxel_metadata': pd.DataFrame(voxel_metadata),
            'stimuli_metadata': pd.DataFrame(stimuli_metadata).set_index('id')}
    
def load_activations(path, stimuli_metadata, layer_filter=None):
    activations = []
    activations_ = {}
    for idx, row in stimuli_metadata.iterrows():
        filename = row['filename']
        if not isinstance(filename, str):
            filename = filename.decode('utf-8')
        filename = filename.replace('.wav', '.pkl')
        activations.append(joblib.load(Path(path, filename)))
    for k in activations[0].keys():
        activations_[k] = [a[k] for a in activations]
    if layer_filter is not None:
        activations_ = {k:v for k,v in activations_.items() if k in layer_filter}
    return activations_
    # h5_path = Path(path, 'natsound_activations.h5')
    # activations_ = {}
    # if h5_path.exists():
    #     with h5py.File(h5_path, 'r') as f:
    #         if layer_filter is not None:
    #             keys = layer_filter
    #         else:
    #             keys = f.keys()
    #         for k in keys:
    #             try:
    #                 activations_[k] = f[k][:]
    #             except:
    #                 from IPython import embed; embed()
    # else:
    #     activations = []
    #     for idx, row in stimuli_metadata.iterrows():
    #         filename = row['filename']
    #         if not isinstance(filename, str):
    #             filename = filename.decode('utf-8')
    #         filename = filename.split('.')[0] + '.pkl'
    #         with open(Path(path, filename), 'rb') as f:
    #             activations.append(pickle.load(f))
    #     for k in activations[0].keys():
    #         activations_[k] = np.array([a[k] for a in activations])
    #     if layer_filter is not None:
    #         activations_ = {k:v for k,v in activations_.items() if k in layer_filter}

    #if layer_filter is not None:
    #
    # activations_ = {k:v for k,v in activations_.items() if k in layer_filter}
    return activations_

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, labels):
        super().__init__()
        self.embeddings = embeddings
        self.labels = labels

    def __getitem__(self, idx):
        return {'embeddings': self.embeddings[idx],
                'y': self.labels[idx]}

    def __len__(self):
        return len(self.embeddings)
        