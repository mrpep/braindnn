from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
from loguru import logger
from tqdm import tqdm
from invariant.representations import AudioFeature
import torch
from pathlib import Path
import librosa
import joblib

from get_stimuli_activations import extract_activations
from run_rsa import run_rsa
from specs import layer_map

output_dir = 'from_scratch'
roi = 'Lateral'
if roi == 'all':
    roi_suffix = ''
else:
    roi_suffix = roi

fs = HfFileSystem()
#Gather finegrained checkpoints
all_ckpts_hf = []
for x in fs.ls('lpepino/encodecmae-pretrained/full_exps/base_models/mel256-ec-base/pretrain_checkpoints/finegrained', refresh=True):
    if x['name'].endswith('.ckpt'):
        all_ckpts_hf.append('/'.join(x['name'].split('/')[2:]))
logger.info('Found {} checkpoints!'.format(len(all_ckpts_hf)))
stimuli_dir = Path(__file__, '../../data/stimuli/165_natural_sounds').resolve()

#Run RSA with each checkpoint
model = AudioFeature('mel256-ec-base', device='cuda:0')
layer_filter = layer_map['mel256-ec-base']
rsa_datasets = ['NH2015','B2021']
for ckpt_filename in tqdm(all_ckpts_hf):
    model_id = Path(ckpt_filename).stem.split('=')[-1]
    if not Path(output_dir, 'mel256-ec-base-dynamic', model_id, 'activations').exists():
        ckpt_file = hf_hub_download(repo_id='lpepino/encodecmae-pretrained', filename=ckpt_filename)
        ckpt = torch.load(ckpt_file, map_location=model.device)
        model._model._model.load_state_dict(ckpt['state_dict'])
        activation_dir = Path(output_dir, 'mel256-ec-base-dynamic', model_id, 'activations', 'natural_sounds')
        activation_dir.mkdir(parents=True, exist_ok=True)
        for wav_f in stimuli_dir.rglob('*.wav'):
            out_path = Path(activation_dir, wav_f.stem + '.pkl')
            if not out_path.exists():
                out_path.parent.mkdir(parents=True, exist_ok=True)
                x, fs = librosa.core.load(wav_f, sr=model.sr)
                feats = model(x)
                feats = {k: v for k,v in feats.items() if k in layer_filter}
                
                assert len(feats) == len(layer_filter)
                joblib.dump(feats, out_path)
        Path(ckpt_file).unlink()
    else:
        logger.info('Activations already extracted')
    for d in rsa_datasets:
        if not Path(output_dir, 'mel256-ec-base-dynamic', model_id, f'RSA_{d}{roi_suffix}.pkl').exists():
            run_rsa(model_id, output_dir=Path(output_dir, 'mel256-ec-base-dynamic'), method='rdm_layerwise', dataset=d, roi=roi)
        else:
            logger.info(f'Skipping RSA for step {model_id} as it has already been computed')
    
    
    
    