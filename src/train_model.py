import pandas as pd
import numpy as np
import random
from nltk.tokenize import sent_tokenize

import torch

from tqdm.auto import tqdm
tqdm.pandas()

from transformers import AutoTokenizer

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore")

import hashlib
import pickle
import argparse

from config import load_filepaths, load_config, create_run_folder, save_config, add_run_specific_filepaths, concat_configs
from callbacks import Callbacks, FileLogger, MetricsHandler, MetricMeter
from data import make_text, clean_text, CustomDataset, Collator
from training import seed_everything, get_optimizer, get_scheduler, get_valid_steps, metrics, criterion, Trainer
from models import CustomModel

device = "cuda" if torch.cuda.is_available() else "cpu"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str)
    parser.add_argument('--debug', type=str2bool)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    config = load_config(args.config_path)
    
    filepaths = load_filepaths('filepaths.yaml')
    config = concat_configs(config, filepaths)
    config = add_run_specific_filepaths(config, config.exp_name, 0, config.seed)
    config.debug = args.debug

    seed_everything(config.seed)
    create_run_folder(config.run_dir, debug=config.debug)
    logger = FileLogger(output_fn=config.log_path)

    labels = ['generated']
    config.dataset.labels = labels
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.backbone_type, use_fast=False)
    config.tokenizer = tokenizer
    
    train_folds = pd.read_parquet(config.dataset.train_path)
    valid_folds = pd.read_parquet(config.dataset.valid_path)
    
    if config.debug:
        train_folds = train_folds.sample(500)
        valid_folds = valid_folds.sample(500)
    
    valid_folds['full_text'] = valid_folds['text'].copy()
    valid_folds['full_text'] = valid_folds['full_text'].progress_apply(clean_text)
    
    train_folds['full_text'] = train_folds['text'].copy()
    train_folds['full_text'] = train_folds['full_text'].progress_apply(clean_text)
    
    train_dataset = CustomDataset(train_folds, config, train=True)
    valid_dataset = CustomDataset(valid_folds, config, train=True)
    data_collator = Collator(pad_to_multiple_of=0)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.dataset.train_batch_size,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        collate_fn=data_collator,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.dataset.valid_batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        collate_fn=data_collator,
    )
    train_steps_per_epoch = int(len(train_dataset) / config.dataset.train_batch_size)
    valid_steps_per_epoch = int(len(valid_dataset) / config.dataset.valid_batch_size)
    n_train_steps = train_steps_per_epoch * config.training.epochs
    n_valid_steps = valid_steps_per_epoch * config.training.epochs
    eval_steps = get_valid_steps(train_steps_per_epoch, config.training.evaluate_n_times_per_epoch)
    

    model_criterion = criterion.get_criterion(config)
    model = CustomModel(config, init_from_config=False,criterion=model_criterion)
    model.backbone.resize_token_embeddings(len(tokenizer))
    
    if config.model.state_from_model != 'None':
        fp = f'models/{config.model.state_from_model}/models/fold_0_42_best.pth'
        fp = fp if os.path.isfile(fp) else f'models/{config.model.state_from_model}/models/fold_0_10_best.pth'
        
        state = torch.load(fp)
        logger.log(f'State from model: {fp}')
        if config.model.load_parts:
            model.load_parts_of_state_dict(state['model'])
        else:
            model.load_state_dict(state['model'])
            
    model.to(device)

    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config, n_train_steps)

    save_config(config, path=config.config_path)
    tokenizer.save_pretrained(config.tokenizer_path)
    model.backbone_config.save_pretrained(config.backbone_config_path)
    
    metrics_handler = MetricsHandler(
        train_batch_size=config.dataset.train_batch_size,
        valid_batch_size=config.dataset.valid_batch_size,
        train_steps_per_epoch=train_steps_per_epoch,
        valid_steps_per_epoch=valid_steps_per_epoch,
        train_print_frequency=config.logger.train_print_frequency,
        valid_print_frequency=config.logger.valid_print_frequency,
        n_evaluations_per_epoch=len(eval_steps),
        epochs=config.training.epochs,
    )
    metrics_handler.add_metric_to_watch(
        MetricMeter(
            compute_fn=metrics.compute_roc_auc, 
            metric_name='roc_auc', 
            direction='maximize'
        )
    )
    metrics_handler.set_main_metric('roc_auc')
    logger.set_metrics_handler(metrics_handler)
    
    callbacks = Callbacks([metrics_handler, logger])
    
    logger.log(f'Train folds shape: {train_folds.shape}')
    logger.log(f'Valid folds shape: {valid_folds.shape}')
    
    encoded_sample = train_dataset[0]['input_ids']
    encoded_sample_mask = train_dataset[0]['attention_mask']
    encoded_sample_out = [token for token, mask in zip(encoded_sample, encoded_sample_mask) if mask == 1]
    logger.log(tokenizer.decode(encoded_sample_out))
    
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        eval_steps=eval_steps,
        callbacks=callbacks,
    )

    trainer.train()
