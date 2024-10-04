

class Callbacks:
    
    def __init__(self, callbacks):
        self.callbacks = callbacks
        
    def on_training_start(self):
        for callback in self.callbacks:
            callback.on_training_start()
        
    def on_train_epoch_start(self):
        for callback in self.callbacks:
            callback.on_train_epoch_start()
            
    def on_train_epoch_end(self):
        for callback in self.callbacks:
            callback.on_train_epoch_end()
            
    def on_valid_epoch_start(self):
        for callback in self.callbacks:
            callback.on_valid_epoch_start()
            
    def on_valid_epoch_end(self, target, predictions):
        for callback in self.callbacks:
            callback.on_valid_epoch_end(target, predictions)
            
    def on_train_step_start(self):
        for callback in self.callbacks:
            callback.on_train_step_start()
            
    def on_train_step_end(self, loss, grad_norm, learning_rates):
        for callback in self.callbacks:
            callback.on_train_step_end(loss, grad_norm, learning_rates)
            
    def on_valid_step_start(self):
        for callback in self.callbacks:
            callback.on_valid_step_start()
            
    def on_valid_step_end(self, loss):
        for callback in self.callbacks:
            callback.on_valid_step_end(loss)
            
    def get(self, callback_name):
        for callback in self.callbacks:
            if callback_name == callback.__class__.__name__:
                return callback
        raise ValueError(f'Callback {callback_name} not found')
        
    