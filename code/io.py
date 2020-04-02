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
    Object to store:
        Import/Output Methods
        Task Data in dictionary 'tasks' as DataLoader objects
    """
    def __init__(self,
                 data_dir, # String of the directory storing all tasks
                 task_dir  # Array of task directories, should match 'tasks' keys
                 ):
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
        Import data, preprocess, and store as DataLoader objects in
        'self.tasks' dictionary
        
        Can reference MRQA script
        """
        
        for task in self.task_dir:
            #[Implement load data]
            
            temp = DataLoader(...)
            self.tasks[task] = temp

    def export_results(self):
        """
        Export results of analysis
        """