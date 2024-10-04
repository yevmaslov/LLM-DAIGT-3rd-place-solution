from .base import BaseCallback
import time
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
import math

loggers = {}
class FileLogger(BaseCallback):
    def __init__(
        self, 
        output_fn
    ):
        self.init_file_logger(output_fn)
    
    def set_metrics_handler(self, handler):
        self.metrics = handler
    
    def init_file_logger(self, filename):
        if loggers.get(filename):
            self.logger = loggers.get(filename)
        else:
            logger = getLogger(__name__)
            logger.setLevel(INFO)
            if (logger.hasHandlers()):
                logger.handlers.clear()
            handler1 = StreamHandler()
            handler1.setFormatter(Formatter("%(message)s"))
            handler2 = FileHandler(filename=filename)
            handler2.setFormatter(Formatter("%(message)s"))
            logger.addHandler(handler1)
            logger.addHandler(handler2)
            logger.propagate = False
            self.logger = logger
            loggers[filename] = self.logger
        
    def log(self, text):
        self.logger.info(text)
        
    def on_training_start(self):
        self.log(f"{'='*30} Training Start {'='*30}")
        
    def get_train_epoch_end_log_template(self):
        template = ''
        template += 'Epoch {train_epoch} train_loss: {train_loss:.4f} '
        template += 'valid_loss: {valid_loss:.4f} time: {train_elapsed_time:s}'
        template += '\n'
        template += '='*50
        template += '\n'
        return template

    def get_valid_epoch_end_log_template(self):
        template = '\t'
        template += 'Epoch {valid_epoch} valid_loss: {valid_loss:.4f} time: {valid_elapsed_time:s}'
        return template
    
    def get_train_step_end_log_template(self):
        template = ''
        template += 'Epoch: [{train_epoch}/{train_total_epochs}][{train_step}/{train_total_steps}] '
        template += 'Elapsed {train_elapsed_time:s} Remain {train_remain_time:s} Loss: {train_loss:.4f} '
        template += 'LR {learning_rate:.8f} Grad: {grad_norm:.2f} '
        return template
    
    def get_valid_step_end_log_template(self):
        template = '\t'
        template += 'EVAL: [{valid_epoch}][{valid_step}/{valid_total_steps}] '
        template += 'Elapsed {valid_elapsed_time:s} Remain {valid_remain_time:s} Loss: {valid_loss:.4f} '
        return template

    def on_train_epoch_end(self):
        progress_metrics = self.metrics.get_progress_metrics()
        template = self.get_train_epoch_end_log_template()
        self.log(template.format(**progress_metrics))
        
    def on_valid_epoch_end(self, target, predictions):
        progress_metrics = self.metrics.get_progress_metrics()
        eval_metrics = self.metrics.get_eval_metrics()
        
        template = self.get_valid_epoch_end_log_template()
        template = template.format(**progress_metrics)
        
        unique_eval_metrics = set([key.replace('_best', '') for key in eval_metrics.keys()])
        for metric in unique_eval_metrics:
            score = eval_metrics[metric]
            template += f' {metric}: {score:.4f} '
        self.log(template)
        
        main_score = eval_metrics[self.metrics.main_metric]
        main_score_best = eval_metrics[f'{self.metrics.main_metric}_best']
        if main_score == main_score_best:
            self.log(f'\tSave Best Score: ({main_score:.6f})\n')
            
    def on_train_step_end(self, loss, grad_norm, learning_rates):
        progress_metrics = self.metrics.get_progress_metrics()
        progress_metrics.update(self.metrics.get_model_metrics())
        
        its_print_step = progress_metrics['train_its_print_step']
        
        if its_print_step:
            template = self.get_train_step_end_log_template()
            template = template.format(**progress_metrics)
            self.log(template)
    
    def on_valid_step_end(self, loss):
        progress_metrics = self.metrics.get_progress_metrics()
        its_print_step = progress_metrics['valid_its_print_step']
        
        if its_print_step:
            template = self.get_valid_step_end_log_template()
            template = template.format(**progress_metrics)
            self.log(template)
        