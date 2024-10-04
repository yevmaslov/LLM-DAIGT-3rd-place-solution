from .base import BaseCallback
import numpy as np
import time
import math
import time
import torch
from collections import OrderedDict


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


class LossMeter(object):
    def __init__(self, subset, batch_size):
        self.batch_size = batch_size
        self.subset = subset
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        return self

    def update(self, val):
        self.val = val
        self.sum += val * self.batch_size
        self.count += self.batch_size
        self.avg = self.sum / self.count
        return self
    
    def get(self):
        output = {'loss': self.avg}
        output = {f'{self.subset}_'+key: value for key, value in output.items()}
        return output
        
        
class EpochProgressMeter(object):
    def __init__(
            self,
            subset,
            n_epochs,
            n_evaluations_per_epoch,
            n_steps_per_epoch,
            print_frequency
        ):
        self.subset = subset
        self.n_epochs = n_epochs
        self.n_evaluations_per_epoch = n_evaluations_per_epoch
        self.n_steps_per_epoch = n_steps_per_epoch
        self.print_frequency = print_frequency
        
        self.epoch = 0
        self.step = 0
        self.global_step = 0
        
        self.start_time = None
        self.elapsed = 0
        self.remain = 0
        
        self.completion_percent = 0
        
        self.norm_step = 0
        self.its_print_step = False
    
    def reset(self):
        self.step = 0
        self.start_time = time.time()
        self.elapsed = 0
        self.remain = 0
        self.completion_percent = 0
        
        self.epoch += 1
        return self
    
    def update(self):
        self.step += 1
        self.global_step += 1
        self.completion_percent = self.step / self.n_steps_per_epoch
        
        self.elapsed = time.time() - self.start_time
        self.remain = self.elapsed / self.completion_percent
        
        self.norm_step = self.epoch / self.n_evaluations_per_epoch
        
        self.its_print_step = (self.step % self.print_frequency == 0) or \
                              (self.completion_percent == 1) or (self.step == 1)
        return self
    
    def get(self):
        output = {
            'step': self.step,
            'epoch': self.epoch,
            'global_step': self.global_step,
            'total_steps': self.n_steps_per_epoch,
            'total_epochs': self.n_epochs,
            'normalized_step': self.norm_step,
            'elapsed_time': as_minutes(self.elapsed),
            'remain_time': as_minutes(self.remain),
            'its_print_step': self.its_print_step,
        }
        output = {f'{self.subset}_'+key: value for key, value in output.items()}
        return output
        

class MetricMeter(object):
    def __init__(
            self,
            metric_name,
            compute_fn,
            direction,
        ):
        
        self.metric_name = metric_name
        self.compute_fn = compute_fn
        self.direction = direction
        
        self.best_score = np.inf if self.direction == 'minimize' else 0
        self.its_best_step = True
        
    def reset(self):
        return self
    
    def update(self, target, predictions):
        self.its_best_step = False
        self.score = self.compute_fn(target, predictions)
        
        if (self.direction == 'minimize' and self.score <= self.best_score) or \
           (self.direction == 'maximize' and self.score >= self.best_score):
               self.best_score = self.score
               self.its_best_step = True
                
        return self
    
    def get(self):
        output = {
            f'{self.metric_name}': self.score,
            f'{self.metric_name}_best': self.best_score,
        }  
        return output
    
    
class TrainingParamMeter(object):
    def __init__(
            self,
            metric_name
        ):
        self.metric_name = metric_name
        self.val = None
        
    def reset(self):
        return self
    
    def update(self, val):
        self.val = val
        return self
    
    def get(self):
        output = {f'{self.metric_name}': self.val}  
        return output
        

class MetricsHandler(BaseCallback):
    def __init__(
        self,
        train_batch_size,
        valid_batch_size,
        train_steps_per_epoch,
        valid_steps_per_epoch,
        train_print_frequency, 
        valid_print_frequency,
        n_evaluations_per_epoch,
        epochs
    ):
        
        self.train_losses = LossMeter(subset='train', batch_size=train_batch_size)
        self.valid_losses = LossMeter(subset='valid', batch_size=valid_batch_size)
        
        self.train_steps = EpochProgressMeter(
            subset='train',
            n_epochs=epochs,
            n_evaluations_per_epoch=1,
            n_steps_per_epoch=train_steps_per_epoch,
            print_frequency=train_print_frequency
        )
        self.valid_steps = EpochProgressMeter(
            subset='valid',
            n_epochs=epochs,
            n_evaluations_per_epoch=n_evaluations_per_epoch,
            n_steps_per_epoch=valid_steps_per_epoch,
            print_frequency=valid_print_frequency
        )
        
        self.grad_meter = TrainingParamMeter(metric_name='grad_norm')
        self.lr_meter = TrainingParamMeter(metric_name='learning_rate')
        
        self.eval_metrics = []
        
    def add_metric_to_watch(self, metric):
        self.eval_metrics.append(metric)
        
    def set_main_metric(self, metric):
        self.main_metric = metric
        
    def on_train_epoch_start(self):
        self.train_losses.reset()
        self.train_steps.reset()
        
    def on_valid_epoch_start(self):
        self.valid_losses.reset()
        self.valid_steps.reset()
        
    def on_valid_epoch_end(self, target, predictions):
        for metric in self.eval_metrics:
            metric.update(target, predictions)
    
    def on_train_step_end(self, loss, grad_norm, learning_rates):
        self.train_steps.update()
        self.train_losses.update(loss)
        self.grad_meter.update(grad_norm)
        self.lr_meter.update(learning_rates[0])
    
    def on_valid_step_end(self, loss):
        self.valid_steps.update()
        self.valid_losses.update(loss)
        
    def get_model_metrics(self):
        metrics = {}
        metrics.update(self.grad_meter.get())
        metrics.update(self.lr_meter.get())
        return metrics
                
    def get_progress_metrics(self):
        metrics = {}
        metrics.update(self.train_losses.get())
        metrics.update(self.train_steps.get())
        metrics.update(self.valid_losses.get())
        metrics.update(self.valid_steps.get())
        return metrics
    
    def get_eval_metrics(self):
        metrics = {}
        for metric in self.eval_metrics:
            metrics.update(metric.get())
        return metrics
    
    def is_valid_score_improved(self):
        metrics = self.get_eval_metrics()
        return metrics[f'{self.main_metric}_best'] == metrics[self.main_metric]
