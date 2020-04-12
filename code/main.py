"""
Main run script to execute experiments and analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import transformers as hug
import os
from tqdm import tqdm, trange
import argparse
import logging as log

# =============== Self Defined ===============
import io            # module for handling import/export of data
import utils         # utility functions for training and and evaluating
import model         # module to define model architecture
import meta_learning # module for meta-learning (OML)
import cont_learning # module for continual learning
import analyze       # module for analyzing results
import args          # module for parsing arguments for program

parser = args.parser



# Set devise to CPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device is {}".format(device))

# create io object to import data

# define models

# continual learning for baseline BERT

# meta-learning to get Meta-BERT

# continual learning for Meta-BERT

# analyze results from continual learning steps

# save results with io


"""
model = model.model(...)
model_meta = model.model(...)
optimizer = opt.Adam(model.parameters, arguments)
optimizer_rln = opt.Adam(model_meta.representation.parameters, arguments)
optimizer_pln = opt.Adam(model_meta.classification.parameters, arguments)
loss = nn.NLLLoss(arguments)
"""