import yaml
from pathlib import Path
import os
import shutil


def load_filepaths(filepath):
    with open(filepath, 'rb') as file:
        data = yaml.safe_load(file)
    path_to_file = Path(filepath).parents[0]
    for key, value in data.items():
        data[key] = Path(path_to_file / Path(value)).resolve()
    return data


def add_run_specific_filepaths(filepaths, exp_name, fold, seed):
    filepaths.run_dir = filepaths.models_dir / exp_name.split('_')[0]
    
    filepaths.checkpoint_path = filepaths.run_dir / 'chkp' / f'fold_{fold}_{seed}_chkp.pth'
    filepaths.best_model_path = filepaths.run_dir / 'models' / f'fold_{fold}_{seed}_best.pth'
    filepaths.log_path = filepaths.run_dir / 'logs' / f'fold-{fold}.log'
    
    filepaths.config_path = filepaths.run_dir / 'config.yaml'
    filepaths.tokenizer_path = filepaths.run_dir / 'tokenizer'
    filepaths.backbone_config_path = filepaths.run_dir
    return filepaths


def create_run_folder(filepath, debug):
    if debug and os.path.isdir(filepath):
        shutil.rmtree(filepath)

    if not os.path.isdir(filepath):
        os.makedirs(filepath)

        logs_dir = filepath / 'logs'
        os.mkdir(logs_dir)

        checkpoints_dir = filepath / 'chkp'
        os.mkdir(checkpoints_dir)

        models_dir = filepath / 'models'
        os.mkdir(models_dir)

    return True


