

class BaseCallback:
    def on_training_start(self):
        pass
    
    def on_train_epoch_start(self):
        pass
    
    def on_train_epoch_end(self):
        pass
    
    def on_valid_epoch_start(self):
        pass
    
    def on_valid_epoch_end(self, target, predictions):
        pass
    
    def on_train_step_start(self):
        pass
    
    def on_train_step_end(self, loss, grad_norm, learning_rates):
        pass
    
    def on_valid_step_start(self):
        pass
    
    def on_valid_step_end(self, loss):
        pass
        