import requests
import tarfile
import fire
from pathlib import Path
from tqdm import tqdm
import os

NEURAL_DATA_URL = 'https://mcdermottlab.mit.edu//tuckute_feather_2023/data.tar'
HEAREVAL_URLS = ['https://zenodo.org/records/6332517/files/hear2021-esc50-v2.0.0-full-48000.tar.gz',
                 'https://zenodo.org/records/6332517/files/hear2021-fsd50k-v1.0-full-48000.tar.gz',
                 'https://zenodo.org/records/6332517/files/hear2021-nsynth_pitch-v2.2.3-50h-48000.tar.gz',
                 'https://zenodo.org/records/6332517/files/hear2021-speech_commands-v0.0.2-full-48000.tar.gz',
                 'https://zenodo.org/records/6332517/files/hear2021-tfds_crema_d-1.0.0-full-48000.tar.gz',
                 'https://zenodo.org/records/6332517/files/hear2021-tfds_gtzan-1.0.0-full-48000.tar.gz']
def download_file(url, output_path):
    """
    Download a tar file from a URL with a progress bar.
    
    Args:
        url (str): The URL of the tar file.
        output_path (str): The local path where the file will be saved.
    """
    
    if isinstance(output_path, Path):
        output_path = str(output_path.resolve())
        
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Total size in bytes
    total_size = int(response.headers.get("content-length", 0))
    block_size = 8192  # 8 KB chunks

    with open(output_path, "wb") as f, tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc=output_path,
        ascii=True,
    ) as bar:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    print(f"Downloaded: {output_path}")
    
def extract_tar(file, destination_path):
    """
    Extract a .tar, .tar.gz, or .tgz file to the given destination.

    Args:
        file (str): Path to the tar file.
        destination_path (str): Directory where the contents will be extracted.
    """
    if isinstance(destination_path, Path):
        destination_path = str(destination_path.resolve())
    Path(destination_path).mkdir(parents=True, exist_ok=True)
    with tarfile.open(file, "r:*") as tar:
        tar.extractall(path=destination_path)

    print(f"Extracted {file} to {destination_path}")

def download_data(neural=True, heareval=True):
    if neural:
        output_path = Path('.','neural.tar')
        download_file(NEURAL_DATA_URL, output_path.resolve())
        extract_tar(output_path, output_path.parent)
        os.remove(str(output_path.resolve()))
    if heareval:
        for f in HEAREVAL_URLS:
            tar_path = Path('.', f.split('/')[-1])
            download_file(f, tar_path)
            dest_path = Path('.', 'data', 'heareval')
            extract_tar(tar_path, dest_path)
            os.remove(str(tar_path.resolve()))
        
    
if __name__ == '__main__':
    fire.Fire(download_data)