"""
Module to contain continual learning class
"""

import torch
import utils
import io

def c_learn(model, loss):
    """
    Method to run continual learning to Fine-Tune
    
    Will most likely use methods implemented in utils
    
    This is for the testing our experiment to get results on how well
    'model' can do continual learning.
    """
    
    #curriculum is an iterable for the task names (i.e. squad and triviaqa)
    for task in curriculum:
        data = io.tasks(task)
        utils.fine_tune(model, data)