"""
Module for general utility methods like generalized training loop, evaluation,
fine-tuning, and getting data
"""

import torch
import torch.nn as nn
import torch.optim as opt

def train(model, loss, optimizer):
    """
    Training loop
    """
    
def evaluate(model, data, loss):
    """
    Evaluation model on data
    """
    
def fine_tune(model, train_data, val_data, loss, optimizer):
    """
    Fine-tune model on task with train and val data
    """