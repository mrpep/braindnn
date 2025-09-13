from run_rsa import run_rsa
from get_stimuli_activations import extract_activations
from run_voxel_regression import run_regression
from run_downstream import run_downstream

import fire
from loguru import logger

def run_all(model, output_dir, rsa=True, regression=True, downstream=True):
    print(regression)
    logger.info(f'Full analysis for {model}. Results will be stored in {output_dir}')
    logger.info('Extracting activations for stimuli')
    extract_activations(model, output_dir)
    if rsa:
        for dataset in ['B2021', 'NH2015']:
            logger.info(f'Running RSA with {dataset}')
            run_rsa(model, dataset, output_dir)
    if regression:
        for dataset in ['NH2015', 'NH2015comp']:
            logger.info(f'Learning regressors for {dataset}')
            run_regression(model, dataset, output_dir)
    if downstream:
        logger.info('Running downstream evaluation')
        run_downstream(model, output_dir=output_dir)
    
if __name__ == '__main__':
    fire.Fire(run_all)