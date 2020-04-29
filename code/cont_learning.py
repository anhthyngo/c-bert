"""
Module to contain continual learning class
"""

import torch
import torch.nn as nn
import numpy as np
import copy
import logging as log
import os

class ContLearner():
    def __init__(self,
                 hf_model_name,                          # model name from hugging face
                 model_name,                             # name of model
                 learner,                                # device to run on
                 curriculum = ['SQuAD', 'TriviaQA-web'], # curriculum for learning
                 fine_tune_prev = True                   # whether to fine-tune previous
                 ):
        """
        Class for continual learning
        """
        self.hf_model_name = hf_model_name
        self.model_name = model_name
        self.learner = learner
        self.unsupervised_model = copy.deepcopy(learner.model)
        self.curriculum = curriculum
        self.fine_tune_prev = fine_tune_prev
        self.max_steps = learner.max_steps
        self.log_int = learner.log_int
        self.log_dir = learner.log_dir
        self.save_dir = learner.save_dir
        self.scores = {}
        
        for task in curriculum:
            self.scores['{} {}'.format(model_name,task)] = {
                'iter' : np.arange(start=0, stop=(self.max_steps+self.log_int), step=self.log_int).tolist(),
                'f1'   : []
                }        

# =============================================================================
# Methods to do continual learning
# =============================================================================
    def c_learn(self,
                rln_only = False # bool whether to only carry RLN weights to next task
                ):
        """
        Method to fine-tune continual learning on curriculum.
    
        This is for the testing our experiment to get results on how well
        'model' can do continual learning.
        
        Curriculum: Unsupervised -> SQuAD -> TriviaQA
        """   
        prev_task = None
        
        for i, task in enumerate(self.curriculum):
            if (i == len(self.curriculum) - 1) or self.fine_tune_prev:
                log.info("Fine-Tuning: {}".format(task))
                
                # load best weights from previous task
                if not prev_task is None:
                    # load best weights for previous task
                    prev_task_weights_pre = os.path.join(self.log_dir,
                                                         self.hf_model_name,
                                                         prev_task)
                    self.load_unsupervised_weights(prev_task_weights_pre, "{}_{}_best".format(self.hf_model_name, prev_task), ".pt", rln_only)
                    
                    log.info("Reset learner object for task {}".format(task))
                    self.learner.model = copy.deepcopy(self.unsupervised_model)
                    self.learner.set_optimizer()
                
                # fine tune model on task
                paths, rln_paths, f1, best_path = self.learner.fine_tune(task)
                
                if i == len(self.curriculum) - 1:
                    # we want rln weights for trivia
                    self.scores['{} {}'.format(self.model_name, task)]['f1'] = f1
            
            prev_task = task                    
    
    def evaluate_forgetting(self, rln_only = False):
        """
        Method to evaluate forgetting
        """
        # get previous tasks and last-task name
        prev_tasks = self.curriculum[0:len(self.curriculum)-1]
        last_task = self.curriculum[len(self.curriculum)-1]
        
        for i, prev_task in enumerate(prev_tasks):
            # load best weights of previous task
            best_prev_weights = os.path.join(self.log_dir, self.hf_model_name, prev_task, "{}_{}_best.pt".format(self.hf_model_name, prev_task))
            
            if isinstance(self.unsupervised_model, nn.DataParallel):
                self.unsupervised_model.module.model.load_state_dict(torch.load(best_prev_weights))
            else:
                self.unsupervised_model.model.load_state_dict(torch.load(best_prev_weights))
            
            # evaluate forgetting for each log_int of last task
            for j in np.arange(start=0, stop=(self.max_steps+self.log_int), step=self.log_int).tolist():
                log.info("Evaluating forgetting for {} on iteration: {}".format(prev_task, j))
                
                last_task_weights_pre = os.path.join(self.log_dir, 
                                                     self.hf_model_name, 
                                                     last_task)
                self.load_unsupervised_weights(last_task_weights_pre, "{}".format(j), ".pt", rln_only)
                                    
                zero_results = self.learner.evaluate(prev_task, 
                                                     self.unsupervised_model, 
                                                     prefix = 'forget_{}_{}'.format(prev_task, self.hf_model_name),
                                                     load=(j==0))
                self.scores['{} {}'.format(self.model_name, prev_task)]['f1'].append(zero_results.get('f1'))
    
    def load_unsupervised_weights(self, pre_name, post_name, file_type, rln_only = False):
        """
        Helper method to load weights to self.unsupervised_model
        """
        if isinstance(self.unsupervised_model, nn.DataParallel):
            temp_model = self.unsupervised_model.module
        else:
            temp_model = self.unsupervised_model
        
        if rln_only:
            # only load RLN weights
            last_task_weights = os.path.join(pre_name, post_name + "_rln" + file_type)
            temp_model.model.bert.load_state_dict(torch.load(last_task_weights))
            
# =============================================================================
#             if isinstance(self.unsupervised_model, nn.DataParallel):
#                 self.unsupervised_model.module.model.bert.load_state_dict(torch.load(last_task_weights))
#             else:
#                 self.unsupervised_model.model.bert.load_state_dict(torch.load(last_task_weights))
# =============================================================================
        else:
            # load entire model
            last_task_weights = os.path.join(pre_name, post_name + file_type)
            temp_model.model.load_state_dict(torch.load(last_task_weights))
            
# =============================================================================
#             if isinstance(self.unsupervised_model, nn.DataParallel):
#                 self.unsupervised_model.module.load_state_dict(torch.load(last_task_weights))
#             else:
#                 self.unsupervised_model.load_state_dict(torch.load(last_task_weights))
# =============================================================================
