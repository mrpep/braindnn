from invariant.representations import AudioFeature
from pathlib import Path
import librosa
import joblib
from tqdm import tqdm
import fire

def extract_activations(model, output_dir='results'):
    stimuli_dir = Path(__file__, '../../data/stimuli').resolve()
    m = AudioFeature(model)
    for fname in tqdm(Path(stimuli_dir, '165_natural_sounds').rglob('*.wav')):
        x, fs = librosa.core.load(fname, sr=m.sr)
        feats = m(x)
        out_path = Path(output_dir, model, 'activations', 'natural_sounds', fname.stem + '.pkl')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(feats, out_path)

if __name__ == '__main__':
    fire.Fire(extract_activations)



