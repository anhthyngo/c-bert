"""
Main run script to execute meta learning evaluation
"""

import torch
import torch.nn as nn
import transformers
import os
import logging as log
from datetime import datetime as dt
import random
import numpy as np
import sys
import time
import json
from tqdm import trange

# =============== Self Defined ===============
import myio                        # module for handling import/export of data
import learner                     # module for fine-tuning
import model                       # module to define model architecture
import meta_learning               # module for meta-learning (OML)
import meta_learner                # meta learner object based on Javed and White implementation
from args import args, check_args  # module for parsing arguments for program

def main():
    """
    Main method for meta-learning
    """ 
    start = time.time()
    repository = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    parser = args.parse_args()
    
    # run some checks on arguments
    check_args(parser)
    
    # format logging
    log_name = os.path.join(parser.run_log, '{}_meta_run_log_{}.log'.format(
        parser.experiment,
        dt.now().strftime("%Y%m%d_%H%M")
        )
    )
    log.basicConfig(filename=log_name,
                    format='%(asctime)s | %(name)s -- %(message)s',
                    level=log.DEBUG)
    os.chmod(log_name, parser.access_mode)
    
    # set devise to CPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info("Device is {}".format(device))
    
    # set seed for replication
    random.seed(parser.seed)
    np.random.seed(parser.seed)
    torch.manual_seed(parser.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(parser.seed)
    
    log.info("Starting experiment {} meta learning on {} with model {}".format(
        parser.experiment,
        device,
        parser.model))
    
    # set tokenizer and config from Huggingface
    tokenizer = transformers.AutoTokenizer.from_pretrained(parser.model,
                                                           do_lower_case=parser.do_lower_case)
    config = transformers.AutoConfig.from_pretrained(parser.model)
    
    # create IO object and import data
    cache_head = os.path.join(parser.save_dir, 'cached_data')
    cache_dir = os.path.join(cache_head, parser.model)
    if not os.path.exists(cache_head):
        os.mkdir(cache_head)
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    
    data_handler = myio.IO(parser.data_dir,
                           cache_dir,
                           tokenizer,
                           parser.max_seq_length,
                           parser.doc_stride,
                           parser.max_query_length,
                           batch_size = parser.batch_size,
                           shuffle=True,
                           cache=True
                           )
    
    # set oml
    oml = meta_learner.MetaLearningClassification(update_lr     = parser.meta_update_lr,
                                                  meta_lr       = parser.meta_meta_lr,
                                                  hf_model_name = parser.model,
                                                  config        = config,
                                                  myio          = data_handler,
                                                  max_grad_norm = parser.max_grad_norm,
                                                  device        = device)
    
    # freeze_layers
    oml.freeze_rln()
    
    # do meta_learning
    meta_tasks = parser.meta_tasks.split(',')
    
    meta_steps = trange(0, parser.meta_steps, desc = 'Meta Outer', mininterval=30)
    for step in meta_steps:
        
        # sample tasks
        sample_tasks = np.random.choice(meta_tasks, parser.n_meta_tasks, replace = False)
        
        # sample trajectory
        d_traj = []
        d_rand = []
        for task in sample_tasks:
            task_traj, task_rand = data_handler.sample_dl(task    = task,
                                                          samples = parser.n_meta_task_samples,
                                                          use     = 'train')
            d_traj += task_traj
            d_rand += task_rand
            
        loss = oml(d_traj, d_rand)
    
    # save RLN weights
    meta_RLN_head = os.path.join(parser.save_dir, "meta_weights")
    meta_RLN_weights = os.path.join(meta_RLN_head, parser.exp_name + "_meta_weights.pt")
    if not os.path.exist(meta_RLN_head):
        os.mkdir(meta_RLN_head)
    
    # for multi-GPU
    if isinstance(oml.net, nn.DataParallel):
        weights = oml.net.module.model.bert.state_dict()
    else:
        weights = oml.net.model.bert.state_dict()
    
    torch.save(weights, meta_RLN_weights)
    
    log.info("Meta loss is {}".format(loss))
    log.info("Saved meta weights at {}".format(meta_RLN_weights))
    log.info("Total time is: {} min : {} s".format((time.time()-start)//60, (time.time()-start)%60))
    
if __name__ == "__main__":
    main()