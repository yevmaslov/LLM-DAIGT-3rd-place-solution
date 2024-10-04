# import torch
import torch.nn as nn
from types import SimpleNamespace
import numpy as np
import gc
from tqdm import tqdm
import torch
import random
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def batch_to_device(batch):
    for k, v in batch.items():
        if type(v) == dict:
            for _k, _v in v.items():
                if len(v) == 1:
                    v = v[0].unsqueeze(0)
                v[_k] = _v.to(device)
            batch[k] = v
        else:
            if len(v) == 1:
                v = v[0].unsqueeze(0)
            batch[k] = v.to(device)
    return batch


class Trainer:

    def __init__(
            self,
            model: nn.Module,
            config: SimpleNamespace,
            train_dataloader: torch.utils.data.DataLoader=None,
            valid_dataloader: torch.utils.data.DataLoader=None,
            optimizer: torch.optim.Optimizer=None,
            scheduler: torch.optim.lr_scheduler=None,
            eval_steps=None,
            callbacks=None,
    ) -> None:

        self.model = model

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.config = config

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.training.apex)
        self.eval_steps = eval_steps
        
        self.callbacks = callbacks

    def validate(self):
        self.model.eval()
        
        self.callbacks.on_valid_epoch_start()

        predictions = []
        target = []

        for step, inputs in enumerate(self.valid_dataloader):
            inputs = batch_to_device(inputs)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    y_pred, loss = self.model(inputs)

            y_pred = torch.sigmoid(y_pred)
            predictions.append(y_pred.detach().to('cpu').numpy())
            target.append(inputs['labels'].cpu().numpy())
            
            self.callbacks.on_valid_step_end(loss)

        predictions = np.concatenate(predictions)
        target = np.concatenate(target)
        return predictions, target
    
    
    def predict(self, test_loader):
        predictions = []
        self.model.eval()
        self.model.to(device)
        
        for inputs in tqdm(test_loader, total=len(test_loader)):
            inputs = batch_to_device(inputs)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    y_preds, _ = self.model(inputs)
            
            y_preds = torch.sigmoid(y_preds)
            predictions.append(y_preds.detach().to('cpu').numpy())
        predictions = np.concatenate(predictions)
        return predictions
    
    def get_embeddings(self, test_loader):
        predictions = []
        self.model.eval()
        self.model.to(device)
        
        for inputs in tqdm(test_loader, total=len(test_loader)):
            inputs = batch_to_device(inputs)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    outputs = self.model.backbone(inputs['input_ids'], inputs['attention_mask'])
                    embeddings = self.model.pooling(inputs, outputs)
            
            predictions.append(embeddings.detach().to('cpu').numpy())
        predictions = np.concatenate(predictions)
        return predictions

    def train(self):
        self.model.to(device)
        
        self.callbacks.on_training_start()
        
        for epoch in range(self.config.training.epochs):
            self.model.train()

            self.callbacks.on_train_epoch_start()
            for step, inputs in enumerate(self.train_dataloader):
                inputs = batch_to_device(inputs)

                if self.config.training.apex:
                    with torch.cuda.amp.autocast():
                        y_pred, loss = self.model(inputs)
                        raw_loss = loss.item()

                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                               self.config.training.max_grad_norm)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                else:
                    y_pred, loss, skip = self.model(inputs)
                    raw_loss = loss.item()

                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                               self.config.training.max_grad_norm)
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.scheduler.step()
                
                learning_rates = self.scheduler.get_last_lr()
                self.callbacks.on_train_step_end(raw_loss, grad_norm, learning_rates)
                
                if (step + 1) in self.eval_steps:
                    predictions, target = self.validate()
                    self.model.train()
                    
                    self.callbacks.on_valid_epoch_end(target, predictions)
                    score_improved = self.callbacks.get('MetricsHandler').is_valid_score_improved()
                    if score_improved:
                        self.save_best_model(predictions)
            
            self.callbacks.on_train_epoch_end()
            self.save_checkpoint()
            
        torch.cuda.empty_cache()
        gc.collect()
        return None

    
    def save_best_model(self, predictions):
        torch.save(
            {
                'model': self.model.state_dict(),
                'predictions': predictions
            },
            self.config.best_model_path
        )

    def save_checkpoint(self):
        torch.save(
            {
                'model': self.model.state_dict()
            },
            self.config.checkpoint_path
        )

