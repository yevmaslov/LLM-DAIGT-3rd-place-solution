import torch.nn as nn

def get_criterion(config):
    return nn.BCEWithLogitsLoss(reduction='mean')
