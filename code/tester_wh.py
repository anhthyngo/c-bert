"""
Testing for implementations - Will
"""

import torch
import analyze
import numpy as np
from datetime import datetime as dt
import os
import json
import logging as log
import transformers
import myio

if __name__ == '__main__':
    wd = os.getcwd()
    
    # for Spyder to get logging to work
    root = log.getLogger()
    while len(root.handlers):
        root.removeHandler(root.handlers[0])
    
    # define logger. will produce logs in ./logs directory. make sure to clear
    log_fname = os.path.join(wd, "logs", "log_{}.log".format(
        dt.now().strftime("%Y%m%d_%H%M")))
    log.basicConfig(filename=log_fname,
        format='%(asctime)s - %(name)s - %(message)s',
        level=log.INFO)
    root.addHandler(log.StreamHandler())
    
    # ============================= Testing Analyze ==============================
    # Generate test data
    if False:
        btrivia = {"iter":np.arange(11),"val":np.arange(11)}
        bsquad = {"iter":np.arange(11),"val":0.5*np.arange(11)}
        mtrivia = {"iter":np.arange(11),"val":2*np.arange(11)}
        msquad = {"iter":np.arange(11),"val":1.5*np.arange(11)}
    
        data = {
                "BERT Trivia":btrivia,
                "BERT SQuAD":bsquad,
                "Meta-BERT Trivia":mtrivia,
                "Meta-BERT SQuAD":msquad
                }
    
        # Test plotting
        plot = analyze.plot_learning(data, iterations=10, max_score=20, x_tick_int=2, y_tick_int=10)
    
        # Tryout displaying and saving plot
        #
        # Datetime string formatting:
        # %Y = year
        # %m = month
        # %d = day
        # %H = hour
        # %M = minute
        plot.show
        plot.savefig("./results/test_fig_{}.png".format(dt.now().strftime("%Y%m%d_%H%M")))
    # ============================= Testing Analyze ==============================
        
    # ============================= Testing Data Loading ==============================
    
    if True:
        data_dir = os.path.join(wd,'test_data')
        task_names = ['tester']
        tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased') 
        max_seq_length = 384
        doc_stride = 128
        max_query_length = 64
        threads = 1
        
        # create IO object with batch size 2 for testing
        data_handler = myio2.IO(data_dir,
                                task_names,
                                tokenizer,
                                max_seq_length,
                                doc_stride,
                                max_query_length,
                                threads,
                                batch_size = 2)
        data_handler.read_tasks()
        dl_train = data_handler.tasks.get('tester').get('train')
        dl_dev = data_handler.tasks.get('tester').get('dev')
        
        data_handler2 = myio2.IO(data_dir,
                                 task_names,
                                 tokenizer,
                                 max_seq_length,
                                 doc_stride,
                                 max_query_length,
                                 threads,
                                 batch_size = 2)        
        data_handler2.read_tasks()
        dl_train2 = data_handler2.tasks.get('tester').get('train')
        dl_dev2 = data_handler2.tasks.get('tester').get('dev')
        
        # print training example
        data = next(iter(dl_train))
        print("\n{} {}:\n {}".format('train', 1, tokenizer.decode(data[0][0,:])))
        data = next(iter(dl_dev))
        print("\n{} {}:\n {}".format('train', 2, tokenizer.decode(data[0][0,:])))
        data = next(iter(dl_dev2))
        print("\n{} {}:\n {}".format('dev', 1, tokenizer.decode(data[0][0,:])))
        data = next(iter(dl_train2))
        print("\n{} {}:\n {}".format('dev', 2, tokenizer.decode(data[0][0,:])))
        
    # release logs from Python
    handlers = log.getLogger().handlers
    for handler in handlers:
        handler.close()
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            