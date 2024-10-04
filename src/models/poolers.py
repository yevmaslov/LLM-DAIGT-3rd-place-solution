import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from transformers.activations import ACT2FN
import numpy as np


def get_last_hidden_state(backbone_outputs):
    last_hidden_state = backbone_outputs[0]
    return last_hidden_state


def get_all_hidden_states(backbone_outputs):
    all_hidden_states = torch.stack(backbone_outputs[1])
    return all_hidden_states


def get_input_ids(inputs):
    return inputs['input_ids']


def get_attention_mask(inputs):
    return inputs['attention_mask']


class MeanPooling(nn.Module):
    def __init__(self, backbone_config, pooling_config):
        super(MeanPooling, self).__init__()
        self.output_dim = backbone_config.hidden_size

    def forward(self, inputs, backbone_outputs):
        attention_mask = get_attention_mask(inputs)
        last_hidden_state = get_last_hidden_state(backbone_outputs)

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class LSTMPooling(nn.Module):
    def __init__(self, backbone_config, pooling_config, is_lstm=True):
        super(LSTMPooling, self).__init__()

        self.num_hidden_layers = backbone_config.num_hidden_layers
        self.hidden_size = backbone_config.hidden_size
        self.hidden_lstm_size = pooling_config.hidden_size
        self.dropout_rate = pooling_config.dropout_rate
        self.bidirectional = pooling_config.bidirectional

        self.is_lstm = is_lstm
        self.output_dim = pooling_config.hidden_size*2 if self.bidirectional else pooling_config.hidden_size

        if self.is_lstm:
            self.lstm = nn.LSTM(self.hidden_size,
                                self.hidden_lstm_size,
                                bidirectional=self.bidirectional,
                                batch_first=True)
        else:
            self.lstm = nn.GRU(self.hidden_size,
                               self.hidden_lstm_size,
                               bidirectional=self.bidirectional,
                               batch_first=True)

        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, inputs, backbone_outputs):
        all_hidden_states = get_all_hidden_states(backbone_outputs)

        hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(1, self.num_hidden_layers + 1)], dim=-1)
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out, _ = self.lstm(hidden_states, None)
        out = self.dropout(out[:, -1, :])
        return out


class WeightedLayerPooling(nn.Module):
    def __init__(self, backbone_config, pooling_config):
        super(WeightedLayerPooling, self).__init__()

        self.num_hidden_layers = backbone_config.num_hidden_layers
        self.layer_start = pooling_config.layer_start
        self.layer_weights = pooling_config.layer_weights if pooling_config.layer_weights is not None else \
            nn.Parameter(torch.tensor([1] * (self.num_hidden_layers + 1 - self.layer_start), dtype=torch.float))

        self.output_dim = backbone_config.hidden_size

    def forward(self, inputs, backbone_outputs):
        all_hidden_states = get_all_hidden_states(backbone_outputs)

        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average[:, 0]


class ConcatPooling(nn.Module):
    def __init__(self, backbone_config, pooling_config):
        super(ConcatPooling, self, ).__init__()

        self.n_layers = pooling_config.n_layers
        self.output_dim = backbone_config.hidden_size*pooling_config.n_layers

    def forward(self, inputs, backbone_outputs):
        all_hidden_states = get_all_hidden_states(backbone_outputs)

        concatenate_pooling = torch.cat([all_hidden_states[-(i + 1)] for i in range(self.n_layers)], -1)
        concatenate_pooling = concatenate_pooling[:, 0]
        return concatenate_pooling


class AttentionPooling(nn.Module):
    def __init__(self, backbone_config, pooling_config):
        super(AttentionPooling, self).__init__()
        self.num_hidden_layers = backbone_config.num_hidden_layers
        self.hidden_size = backbone_config.hidden_size
        self.hiddendim_fc = pooling_config.hiddendim_fc
        self.dropout = nn.Dropout(pooling_config.dropout)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.hidden_size))
        self.q = nn.Parameter(torch.from_numpy(q_t)).float().to(self.device)
        w_ht = np.random.normal(loc=0.0, scale=0.1, size=(self.hidden_size, self.hiddendim_fc))
        self.w_h = nn.Parameter(torch.from_numpy(w_ht)).float().to(self.device)

        self.output_dim = self.hiddendim_fc

    def forward(self, inputs, backbone_outputs):
        all_hidden_states = get_all_hidden_states(backbone_outputs)

        hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(1, self.num_hidden_layers + 1)], dim=-1)
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out = self.attention(hidden_states)
        out = self.dropout(out)
        return out

    def attention(self, h):
        v = torch.matmul(self.q, h.transpose(-2, -1)).squeeze(1)
        v = F.softmax(v, -1)
        v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
        v = torch.matmul(self.w_h.transpose(1, 0), v_temp).squeeze(2)
        return v
    

class GemPooling(nn.Module):
    def __init__(self, backbone_config, pooling_config):
        super(GemPooling, self).__init__()

        self.dim = backbone_config.hidden_size
        self.eps = pooling_config.eps
        self.p = Parameter(torch.ones(1) * pooling_config.p)

        self.output_dim = backbone_config.hidden_size

    def forward(self, inputs, backbone_output):
        attention_mask = get_attention_mask(inputs)
        x = get_last_hidden_state(backbone_output)
        
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(x.size())
        x = torch.sum((x.clamp(min=self.eps) * attention_mask_expanded).pow(self.p), 1)
        ret = x / attention_mask_expanded.sum(1).clip(min=self.eps)
        ret = ret.pow(1 / self.p)
        return ret


class ContextPooling(nn.Module):
    def __init__(self, config, pooling_config):
        super().__init__()
        self.dense = nn.Linear(config.pooler_hidden_size, config.pooler_hidden_size)
        self.dropout = StableDropout(config.pooler_dropout)
        self.config = config

    def forward(self, inputs, backbone_output):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        context_token = get_last_hidden_state(backbone_output)[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.config.pooler_hidden_act](pooled_output)
        return pooled_output

    @property
    def output_dim(self):
        return self.config.hidden_size


def get_pooling_layer(config, backbone_config):
    if config.model.pooling_type == 'MeanPooling':
        return MeanPooling(backbone_config, config.model.gem_pooling)

    elif config.model.pooling_type == 'GRUPooling':
        return LSTMPooling(backbone_config, config.model.gru_pooling, is_lstm=False)

    elif config.model.pooling_type == 'LSTMPooling':
        return LSTMPooling(backbone_config, config.model.lstm_pooling, is_lstm=True)

    elif config.model.pooling_type == 'WeightedLayerPooling':
        return WeightedLayerPooling(backbone_config, config.model.weighted_pooling)

    elif config.model.pooling_type == 'ConcatPooling':
        return ConcatPooling(backbone_config, config.model.concat_pooling)

    elif config.model.pooling_type == 'AttentionPooling':
        return AttentionPooling(backbone_config, config.model.attention_pooling)
    
    elif config.model.pooling_type == 'GemPooling':
        return GemPooling(backbone_config, config.model.gem_pooling)
    
    elif config.model.pooling_type == 'ContextPooling':
        return ContextPooling(backbone_config, config.model.gem_pooling)

    else:
        raise ValueError(f'Invalid pooling type: {config.model.pooling_type}')
