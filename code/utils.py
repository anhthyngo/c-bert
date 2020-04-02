"""
Module for general utility methods like generalized training loop, evaluation,
fine-tuning, and getting data
"""

import torch
import torch.nn as nn
import torch.optim as opt

def train(model, data, loss, optimizer, device):
    """
    Training loop
    """
    for i, batch in enumerate(data):
        model.train() #puts in train mode
        data, label = data[0], data[1]
        
        #zero gradients
        model.zero_grad()
        
        #send data through forward
        out = model(data)
        
        #calculate loss
        l = loss(out, label)
        
        #calculate gradients through back prop
        l.backward()
        
        #take a step in gradient descent
        optimizer.step()
    
        #put something for annealing adjusting learning rate
    
def evaluate(model, data, loss):
    """
    Evaluation model on data
    """
    model.test() #puts in test mode
    data, label = data[0], data[1]
    out = model(data)
    l = loss(out,label)
    
def fine_tune(model, train_data, val_data, loss, optimizer):
    """
    Fine-tune model on task with train and val data
    """
    
    for _ in epochs:
        train(model, train_data, loss, optimizer, device)
        evaluate(model, val_data, loss)