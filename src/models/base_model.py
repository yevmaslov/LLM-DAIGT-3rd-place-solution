import torch
import torch.nn as nn
from .poolers import get_pooling_layer
from transformers import AutoModel, AutoConfig
from torch.utils.checkpoint import checkpoint
from copy import deepcopy
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomModel(nn.Module):
    def __init__(
            self, 
            config, 
            backbone_config=None,
            init_from_config=False,
            criterion=None
        ):
        super().__init__()
        self.config = config
        
        if backbone_config is None:
            self.init_backbone_config(config)
        else:
            self.backbone_config = backbone_config
        self.init_backbone(init_from_config)
        
        self.pooling = get_pooling_layer(config, self.backbone_config)
        self.fc = nn.Linear(self.backbone_config.hidden_size, len(self.config.dataset.labels))
        self._init_weights(self.fc)
        self.criterion = criterion
        
    def init_backbone_config(self, config):
        self.backbone_config = AutoConfig.from_pretrained(config.model.backbone_type, output_hidden_states=False)
        self.backbone_config.hidden_dropout = config.model.dropout
        self.backbone_config.hidden_dropout_prob = config.model.dropout
        self.backbone_config.attention_dropout = config.model.attention_dropout
        self.backbone_config.attention_probs_dropout_prob = config.model.attention_dropout
        
    def init_backbone(self, init_from_config):
        if init_from_config:
            self.backbone = AutoModel.from_config(self.backbone_config)
        else:
            self.backbone = AutoModel.from_pretrained(self.config.model.backbone_type, config=self.backbone_config)

        self.backbone.resize_token_embeddings(len(self.config.tokenizer))
        
        if self.config.model.gradient_checkpointing:
            self.enable_gradient_checkpointing()
            
        if self.config.model.freeze_embeddings:
            self.freeze(self.backbone.embeddings)
        
        if self.config.model.freeze_n_layers > 0:
            self.freeze(self.backbone.encoder.layer[:self.config.model.freeze_n_layers])
                
        if self.config.model.reinitialize_n_layers > 0:
            for module in self.backbone.encoder.layer[-self.config.model.reinitialize_n_layers:]:
                self._init_weights(module)
                
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.backbone_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.backbone_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def enable_gradient_checkpointing(self):
        if self.backbone.supports_gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
        else:
            print(f'{self.config.model.backbone_type} does not support gradient checkpointing')
            
    def freeze(self, module):
        for parameter in module.parameters():
            parameter.requires_grad = False
            
    def load_parts_of_state_dict(self, state):
        print(f'Loading only {self.config.model.load_n_layers} layers...')
        
        embeddings_state = {
            key.replace('backbone.embeddings.', ''): val 
            for key, val in state.items() 
            if 'backbone.embeddings.' in key
        }
        layers_state = {
            key.replace('backbone.encoder.layer.', ''): val 
            for key, val in state.items() 
            if 'backbone.encoder.layer.' in key and int(key.split('.')[3]) < self.config.model.load_n_layers
        }
        head_state = {
            key.replace('fc.', ''): val 
            for key, val in state.items() 
            if 'fc.' in key
        }
        
        if self.config.model.load_embeddings:
            self.backbone.embeddings.load_state_dict(embeddings_state)
            
        if self.config.model.load_n_layers > 0:
            self.backbone.encoder.layer[:self.config.model.load_n_layers].load_state_dict(layers_state)
            
        if self.config.model.load_head > 0:
            self.fc.load_state_dict(head_state)
            
    def forward(self, inputs):
        backbone_outputs = self.backbone(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
        )
        feature = self.pooling(inputs, backbone_outputs)
        outputs = self.fc(feature)
        
        loss = None
        if 'labels' in inputs.keys():
            loss = self.criterion(outputs, inputs['labels'])
        
        return outputs, loss
