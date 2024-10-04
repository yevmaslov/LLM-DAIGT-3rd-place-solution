import yaml
from .utils import dictionary_to_namespace, namespace_to_dictionary
import copy
from pathlib import Path


def load_config(filepath):
    with open(filepath, 'rb') as file:
        data = yaml.safe_load(file)
    # data = dictionary_to_namespace(data)
    return data


def save_config(config, path):
    config_out = copy.deepcopy(config)
    config_out.tokenizer = None
    config_out = namespace_to_dictionary(config_out)
    for key, value in config_out.items():
        if type(value) == type(Path()):
            config_out[key] = str(value)
    with open(path, 'w') as file:
        yaml.dump(config_out, file, default_flow_style=False)


def concat_configs(
    config,
    filepaths
):  
    config.update(filepaths)
    
    config = dictionary_to_namespace(config)

    if config.debug:
        config.exp_name = 'test'
        config.logger.use_wandb = False

        config.dataset.train_batch_size = 2
        config.dataset.valid_batch_size = 2

    config.run_name = config.exp_name + f'_fold{config.fold}'
    config.run_id = config.run_name
    return config