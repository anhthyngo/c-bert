"""
Module with class io containing methods for importing and exporting data
"""

import os
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
import gzip
import json
import pandas as pd
import logging as log
import transformers.data.processors.squad as sq
    
class IO:
    """
    Object to store:
        Import/Output Methods
    
    All data are stored in dictionary 'tasks' as DataLoader objects keyed by
    file name.
    
    Dataloading uses Huggingface transformer's squad processor. All data should
    be formatted as original SQuAD v1.1 data from:
    https://github.com/rajpurkar/SQuAD-explorer/tree/master/dataset
    
    Data from dataloader is a list with each entry containing a tensor of
    batched information. The attributes of the list are as follows.
    
     Index  |          Dimension           |               Attribute
    data[0] | [batch_size, max_seq_length] | input_ids
    data[1] | [batch_size, max_seq_length] | attention_mask to mask pad tokens
    data[2] | [batch_size, max_seq_length] | token_type_ids for [CLS] Question [SEP] Context [SEP]
    data[3] |         [batch_size]         | start_positions
    data[4] |         [batch_size]         | end_positions
    data[5] |         [batch_size]         | cls_index for xlnet or xlm (NOT USED)
    data[6] | [batch_size, max_seq_length] | p_mask for xlnet or xlm    (NOT USED)
    data[7] |         [batch_size]         | is_impossible for v2       (NOT USED)
        
    """
    def __init__(self,
                 data_dir,         # name of the directory storing all tasks
                 task_names,       # list of task directories, should match 'tasks' keys
                 tokenizer,        # tokenizer to use
                 max_seq_length, # maximum length of question
                 doc_stride,       # length of sliding window for context
                 max_query_length,   # maximum length of context sequence per window
                 threads,          # number of threads per GPU
                 batch_size=32,    # batch size for training
                 shuffle=True      # whether to shuffle train sampling
                 ):
        self.data_dir = data_dir
        self.task_names = task_names
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.pad_token = tokenizer.pad_token_id
        self.threads = threads
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.processor = sq.SquadV1Processor()
        self.tasks = {
                'SQuAD'                  : None,
                'TriviaQA-web'           : None,
                'NewsQA'                 : None,
                'SearchQA'               : None,
                'HotpotQA'               : None,
                'NaturalQuestionsShort'  : None,
                'tester'                 : None,
                }
                
# =============================================================================
# read data
# =============================================================================
    def read_tasks(self):
        """
        Method to populate the `self.tasks` attribute with dictionaries
        per task. Dictionaries store DataLoaders for training and
        validation datasets.
        
        DataLoaders are keyed by 'train' and 'dev'.
        """        
        temp_task = {
                'train': None, # dataloader for training data
                'dev'  : None  # dataloader for validation data
                }
        
        for task in tqdm(self.task_names):
            data_dir = os.path.join(self.data_dir,task)
            
            # for train and dev
            for use in temp_task.keys(): 
                # get dataset from .json file with correct formatting
                if use == 'train':
                    training = True
                    examples = self.processor.get_train_examples(data_dir)
                elif use == 'dev':
                    training = False
                    examples = self.processor.get_dev_examples(data_dir)
                
                # convert data to squad objects
                features, dataset = sq.squad_convert_examples_to_features(
                    examples=examples,
                    tokenizer=self.tokenizer,
                    max_seq_length=self.max_seq_length,
                    doc_stride=self.doc_stride,
                    max_query_length=self.max_query_length,
                    is_training= training,
                    return_dataset = 'pt',
                    threads = self.threads
                )
                
                # wrap dataset with DataLoader object
                if use == 'train' and self.shuffle:
                    sampler = RandomSampler(dataset)
                else:
                    sampler = SequentialSampler(dataset)
                
                dl = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size)
                
                # add DataLoader to task
                temp_task[use] = dl
                
            # add task to `self.tasks`
            self.tasks[task] = temp_task
    
    def export_results(self):
        """
        Export results of analysis
        """
