"""
Module with class io containing methods for importing and exporting data.

Based on run_squad example:
https://github.com/huggingface/transformers/blob/7972a4019f4bc9f85fd358f42249b90f9cd27c68/examples/run_squad.py
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
import time
    
class IO:
    """
    Object to store:
        Import/Output Methods
    
    All data are stored in dictionary 'tasks'. Entries of 'tasks' are dictionaries
    keyed by file name. Each dictionary contains a dictionary keyed by 'train'
    and 'dev' for training and validation data, respectively. The entries for 
    'train' and 'dev' are dictionaries that have entries:
        
    {'data'     : DataLoader of data with output described below
     'features' : SquadFeatures
     'examples' : SquadExample}
    
    SquadFeatures and SquadExample are used at evaluation time.
    
    Refer to squad.py below for more information about SquadFeatures and
    SquadExample objects:
    SquadFeatures : https://github.com/huggingface/transformers/blob/7972a4019f4bc9f85fd358f42249b90f9cd27c68/src/transformers/data/processors/squad.py#L640
    SquadExample  : https://github.com/huggingface/transformers/blob/7972a4019f4bc9f85fd358f42249b90f9cd27c68/src/transformers/data/processors/squad.py#L577
    
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
                 data_dir,                # name of the directory storing all tasks
                 cache_dir,               # directory of cached data if it exists
                 tokenizer,               # tokenizer to use
                 max_seq_length,          # maximum length of total sequence per window
                 doc_stride,              # length of sliding window for context
                 max_query_length,        # maximum length of question
                 batch_size=32,           # batch size for training
                 shuffle=True,            # whether to shuffle train sampling
                 cache=True
                 ):
        
        self.data_dir =  data_dir
        self.cache_dir = cache_dir
        assert os.path.exists(self.data_dir) or os.path.exists(self.cache_dir), "No data"
        
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cache = cache
        self.uses = ['train', 'dev']
        
        # only working with V1 type data per MRQA
        # https://www.cs.princeton.edu/~danqic/papers/mrqa2019.pdf
        self.processor = sq.SquadV1Processor()
        
# =============================================================================
# Methods to read data
# =============================================================================
    def load_and_cache_task(self,
                            task,      # task data to load
                            use,       # use case. either train or dev
                            bs = None, # batch_size override
                            ):
        """
        Load and cache task data. Based on hugging face run_squad example:
        https://github.com/huggingface/transformers/blob/7972a4019f4bc9f85fd358f42249b90f9cd27c68/examples/run_squad.py#L407
        
        --------------------
        Return:
        dl        - PyTorch DataLoader of the data
        features  - Huggingface SquadFeatures 
        examples  - Huggingface SquadExamples
        
        Please refer to the class description for more details on SquadFeatures
        and SquadExamples
        """
        start = time.time()
        
        if bs is None:
            bs = self.batch_size
        
        # make sure use is either train or dev
        try:
            self.uses.index(use)
        except ValueError:
            log.info("Use {} not supported.".format(use))
        
        # name cached file
        cache_file = os.path.join(self.cache_dir,
                                  "cached_{}_{}_{}.pt".format(
                                      use,
                                      task,
                                      self.max_seq_length))
        
        if os.path.exists(cache_file):
            # load dataset from cached file
            
            log.info('Loading {} {} from cached file: {}'.format(
                task, use, cache_file))
            loaded = torch.load(cache_file)
            features, dataset, examples = (
                loaded['features'],
                loaded['dataset'],
                loaded['examples']
                )
        else:
            # get dataset from .json file with correct formatting
            data_dir = os.path.join(self.data_dir, task)
            
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
                return_dataset = 'pt'
                )
                    
            # save cached
            if self.cache:
                log.info('Saving {} processed data into cached file: {}'.format(len(dataset), cache_file))
                torch.save({'features': features, 'dataset': dataset, 'examples': examples}, cache_file)
                
        # wrap dataset with DataLoader object
        if use == 'train' and self.shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
            
        dl = DataLoader(dataset, sampler=sampler, batch_size=bs)
        
        log.info("Task {} took {:.6f}s".format(task, time.time()-start))
        
        return dl, features, examples   
    
    def sample_dl(self, task = None, samples = 100, use = 'train'):
        """
        Sample `samples` number of samples from DataLoader for `task` for both
        trajectory and random
        --------------------
        Return:
        traj      - sample for trajectory
        rnd       - sample random trajectory
        """
        
        traj = []
        rand = []
        
        dl, _, _ = self.load_and_cache_task(task, use)
        
        for i, batch in enumerate(dl):
            if i < samples:
                traj.append(batch)
            elif i >= samples and i < 2*samples:
                rand.append(batch)
            else:
                break
        
        return traj, rand