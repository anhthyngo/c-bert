"""
Module for meta learning
"""

import torch
import utils

def OML(model):
    """
    Method to conduct OML
    
    This is to train BERT to Meta-BERT.
    """
    
    #Do OML loop
    
    #Save final weights