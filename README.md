### Setup

1) Clone this repository
2) Create a conda environment
```bash
conda env create -f environment.yml
```
3) Install this repository
```bash
pip install -e .
```
4) Download data (fMRI measurements from NH2015 and B2021, and HEAREval datasets for downstream evaluation)
⚠️ ~70 GB of disk space will be used
```bash
python scripts/download_data.py
```

5) This command will run the RSA, voxel regression, component regression, and downstream performance measurements by default:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/run_full_analysis.py --model [MODEL] --output_dir [RESULTS_FOLDER]
```

The env variable is to avoid some potential OOM errors when extracting activations during downstream training.

Some analysis can be deactivated with flags --downstream=False, --rsa=False or --regression=False

Models used in the paper are listed in the ALL_MODELS variable inside src/specs.py
