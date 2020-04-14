"""
Main run script to execute experiments and analysis

TO DO: Meta-Learning (OML)
"""

import torch
import transformers
import os
import logging as log
from datetime import datetime as dt
import random
import numpy as np
import sys

# =============== Self Defined ===============
import myio                        # module for handling import/export of data
import learner                     # module for fine-tuning
import model                       # module to define model architecture
import meta_learning               # module for meta-learning (OML)
import cont_learning               # module for continual learning
import analyze                     # module for analyzing results
from args import args, check_args  # module for parsing arguments for program

def main():
    """
    Main method for experiment
    """
    repository = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    parser = args.parse_args()
    
    # run some checks on arguments
    check_args(parser)
    
    # format logging
    log_name = os.path.join(parser.run_log, '{}_run_log_{}.log'.format(
        parser.experiment,
        dt.now().strftime("%Y%m%d_%H%M")
        )
    )
    log.basicConfig(filename=log_name,
                    format='%(asctime)s | %(name)s -- %(message)s',
                    level=log.DEBUG)
    
    # set devise to CPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device is {}".format(device))
    
    # set seed for replication
    random.seed(parser.seed)
    np.random.seed(parser.seed)
    torch.manual_seed(parser.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(parser.seed)
    
    # set tokenizer and config from Huggingface
    tokenizer = transformers.AutoTokenizer.from_pretrained(parser.model)
    config = transformers.AutoConfig.from_pretrained(parser.model)
    
    # create IO object and import data
    cache_dir = os.path.join(parser.save_dir, 'cached_data', parser.model)
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    
    data_handler = myio.IO(parser.data_dir,
                           cache_dir,
                           tokenizer,
                           parser.max_seq_length,
                           parser.doc_stride,
                           parser.max_query_length,
                           parser.threads,
                           batch_size = parser.batch_size,
                           shuffle=True,
                           cache=True
                           )
    
# =============================================================================
# BASELINE
# =============================================================================
    # create BERT model
    BERTmodel = model.QAModel(config)
    
    # create learner object for BERT model
    trainer = learner.Learner(BERTmodel,
                              parser.model,
                              device,
                              data_handler,
                              parser.save_dir,
                              parser.n_best_size,
                              parser.max_answer_length,
                              parser.do_lower_case,
                              parser.verbose_logging,
                              parser.version_2_with_negative,
                              parser.null_score_diff_threshold,
                              max_steps = parser.fine_tune_steps,
                              log_int = parser.logging_steps,
                              best_int = parser.save_steps,
                              verbose_int = parser.verbose_steps,
                              max_grad_norm = parser.max_grad_norm,
                              optimizer = None,
                              scheduler = None,
                              weight_decay = parser.weight_decay,
                              lr = parser.learning_rate,
                              eps = parser.adam_epsilon,
                              warmup_steps = parser.warmup_steps)
    
    # create continual learning object and perform continual learning
    parser.continual_curriculum = parser.continual_curriculum.split(',')
    c_learner = cont_learning.ContLearner(parser.model,
                                          'BERT',
                                          trainer,
                                          curriculum = parser.continual_curriculum)
    
    log.info("Generating Plot")
    # generate BERT plot
    plot = analyze.plot_learning(c_learner.scores)
    plot.show
    plot_name = os.path.join(os.getcwd(),"{}_{}.png".format(self.model,dt.now().strftime("%Y%m%d_%H%M")))
    plot.savefig(plot_name)
    log.info("Plot saved at: {}".format(plot_name))
    
    # exit python
    sys.exit(0)
    
if __name__ == "__main__":
    main()