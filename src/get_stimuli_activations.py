from invariant.representations import AudioFeature
from pathlib import Path
import librosa
import joblib
from tqdm import tqdm
import fire
from specs import m_to_invariant_key, layer_map
from loguru import logger

def extract_activations(model, output_dir='results'):
    stimuli_dir = Path(__file__, '../../data/stimuli').resolve()
    if model in m_to_invariant_key:
        model = m_to_invariant_key[model]
    m = AudioFeature(model, device='cuda:0')
    
    layers = layer_map[model]
    logger.info('Extracting activations from {} layers.'.format(len(layers)))
    
    for fname in tqdm(Path(stimuli_dir, '165_natural_sounds').rglob('*.wav')):
        out_path = Path(output_dir, model, 'activations', 'natural_sounds', fname.stem + '.pkl')
        if not out_path.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            x, fs = librosa.core.load(fname, sr=m.sr)
            feats = m(x)
            feats = {k: v for k,v in feats.items() if k in layers}
            assert len(feats) == len(layers)
            joblib.dump(feats, out_path)

if __name__ == '__main__':
    fire.Fire(extract_activations)



