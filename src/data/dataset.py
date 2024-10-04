import torch
from torch.utils.data import Dataset
# import json
import numpy as np
# from language_tool_python import LanguageTool

# import torch
# from torch.utils.data import Dataset
# import json
# import numpy as np
# from spellchecker import SpellChecker
# import re
# import spacy
# from textblob import Word
# from nltk import sent_tokenize, word_tokenize
# import nltk
# nltk.download('punkt')
# from textblob import TextBlob
# import nltk
# nltk.download('averaged_perceptron_tagger')
# nltk.download('brown')
# from copy import deepcopy
# from spellchecker import SpellChecker
# from transformers import AutoTokenizer
# from nltk import word_tokenize


class CustomDataset(Dataset):
    def __init__(self, df, cfg, train=True):
        self.cfg = cfg
        self.df = df
        self.texts = self.df['full_text'].values

        self.train = train
        self.labels = None
        if self.train:
            self.labels = df[cfg.dataset.labels].values
        

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        
        ml = self.cfg.dataset.max_length
        
        if hasattr(self.cfg.dataset, 'time_shift') and self.cfg.dataset.time_shift:
            out = self.cfg.tokenizer.encode_plus(text, add_special_tokens=False)['input_ids']
            offset = np.random.randint(0, max(1, len(out) - ml - 1))
            out = out[offset:offset+ml]
            text = self.cfg.tokenizer.decode(out)
        
        inputs = self.cfg.tokenizer.encode_plus(
            text,
            return_tensors=None,
            add_special_tokens=True,
            max_length=ml,
            pad_to_max_length=True,
            truncation=True,
        )
        
        if self.train:
            inputs['labels'] = self.labels[item]
            
        return inputs
