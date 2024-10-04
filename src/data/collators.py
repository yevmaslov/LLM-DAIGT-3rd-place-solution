import torch
import random
from typing import Dict, List, Tuple, Any, Optional

    
class Collator:
    def __init__(self, pad_to_multiple_of):
        self.ignore_truncation_cols = ['labels', 'tfidf_vec', 'natural_vec', 'generated_vec', 'embeddings', 'weight']
        self.pad_to_multiple_of = 8

    def __call__(self, batch):
        output = dict()
        max_length = max([sum(sample['attention_mask']) for sample in batch])
        if self.pad_to_multiple_of:
            max_length = max_length + (self.pad_to_multiple_of - (max_length % self.pad_to_multiple_of))
        
        for k, v in batch[0].items():
            if k in self.ignore_truncation_cols:
                output[k] = torch.tensor([sample[k] for sample in batch], dtype=torch.float)
            else:
                # output[k] = torch.tensor([sample[k] for sample in batch], dtype=torch.long)
                output[k] = torch.tensor([sample[k][:max_length] for sample in batch], dtype=torch.long)
            
        return output
