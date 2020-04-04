"""
Module for meta learning
"""

import torch
import utils

def OML(model):
    """
    Method to conduct OML
    
    Probably won't use utils and will need its own training script
    
    This is to train BERT to Meta-BERT.
    """
    
    #Do OML loop
    
    #Save final weights