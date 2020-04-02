"""
Module to define our model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers as hug

class model(nn.Module):
    """
    Define the architecture of the model
    """
    def __init__(self, ndim, nout, vocab_size, meta):
        # PLN definition task
        self.classification = nn.Sequential(
                nn.Linear(ndim, nout),
                nn.ReLU
                )
        
        self.MLM = nn.Softmax(vocab_size)
        
        # RLN definiton as BERT or meta-BERT
        self.representation = temp #BERT idk syntax
        
        # Load weights
        if meta:
            #load meta BERT weights to self.representation
        else:
            #load regular BERT weights
        
    def forward(self, data, mask_idx):
        embeddings = self.representation(data)
        
        # do task
        out = self.classification(embeddings)
        
        #do MLM
        masked_embedding = embeddings[mask_idx]
        mlm_out = self.MLM(masked_embedding)
        return out, mlm_out

class test_mode(nn.Module):
    """
    Small model for utils, io, continual learning testing
    """
    
    def __init__(self, ndim, nout, meta):
    
    def forward(self, data):
        