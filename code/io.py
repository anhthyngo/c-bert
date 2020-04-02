"""
Module with class io containing methods for importing and exporting data
"""

import os
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

class io:
    """
    Object to store task data
    """
    def __init__(self,data_dir, task_dir):
        data_dir = data_dir
        task_dir = task_dir
        tasks = {
            "squad"             : 1,
            "triviaqa"          : 2,
            "newsqa"            : 3,
            "searchqa"          : 4,
            "hotpotqa"          : 5,
            "naturalquestions"  : 6
            }

    def import_data(self):
        """
        Import testing data
        """    

def export_results():
    """
    Export results of analysis
    """