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
