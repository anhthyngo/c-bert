"""
Module to contain continual learning class
"""

import torch
import torch.nn as nn
import numpy as np
import copy
import logging as log

class ContLearner():
    def __init__(self,
                 hf_model_name,                         # model name from hugging face
                 model_name,                            # name of model
                 learner,                               # device to run on
                 curriculum = ['SQuAD', 'TriviaQA-web'] # curriculum for learning
                 ):
        """
        Class for continual learning
        """
        self.hf_model_name = hf_model_name
        self.model_name = model_name
        self.learner = learner
        self.model = copy.deepcopy(learner.model)
        self.curriculum = curriculum
        self.scores = {}
        
        for task in curriculum:
            self.scores['{} {}'.format(model_name,task)] = {
                'iter' : None,
                'f1'   : []
                }
        
        # do continual learning
        log.info("Starting Continual Learning")
        self.c_learn()

# =============================================================================
# Methods to do continual learning
# =============================================================================
    def c_learn(self):
        """
        Method to run continual learning on curriculum.
    
        This is for the testing our experiment to get results on how well
        'model' can do continual learning.
        
        Curriculum: Unsupervised -> SQuAD -> TriviaQA
        """
        best_prev_weights = []
        prev_tasks = []
        
        for i, task in enumerate(self.curriculum):               
            log.info("Fine-Tuning: {}".format(task))
            # fine tune model on task
            paths, f1, best_path = self.learner.fine_tune(task)
            
            temp_iter = self.learner.log_int*np.arange(len(f1)) # every 10k
            
            if i == len(self.curriculum) - 1:
                # we want rln weights for trivia
                self.scores['{} {}'.format(self.model_name, task)]['f1'] = f1
                self.scores['{} {}'.format(self.model_name, task)]['iter'] = temp_iter
                
                
                # loading best models for previous tasks
                for j, prev_task in enumerate(prev_tasks):
                    # for multi-gpu
                    if isinstance(self.model, nn.DataParallel):
                        self.model.module.load_state_dict(torch.load(best_prev_weights[j]))
                    else:
                        self.model.load_state_dict(torch.load(best_prev_weights[j]))
                        
                    for k, path in enumerate(paths):
                        log.info("Evaluating forgetting for {} on iteration: {}".format(prev_task, k))
                        
                        # get validation scores through zero-shot replacing RLN weights
                        # for multi-gpu
                        if isinstance(self.model, nn.DataParallel):
                            self.model.module.model.bert.load_state_dict(torch.load(path))
                        else:
                            self.model.model.bert.load_state_dict(torch.load(path))
                            
                        zero_results = self.learner.evaluate(prev_task, self.model, prefix = 'forget_{}_{}'.format(prev_task, self.hf_model_name))
                        self.scores['{} {}'.format(self.model_name, prev_task)]['f1'].append(zero_results.get('f1'))
                        
                        self.scores['{} {}'.format(self.model_name, prev_task)]['iter'] = temp_iter
            else:
                # store best SQuAD weights
                best_prev_weights.append(best_path)
                prev_tasks.append(task)
    
    def test_learn(self):
        best_prev_weights = []
        prev_tasks = []
        inc = 0.5
        
        for i, task in enumerate(self.curriculum):
            paths, f1, best_path = [1], [inc*(i+1)], None
            
            temp_iter = np.arange(len(f1)) # every 10k
            
            if i == len(self.curriculum) - 1:
                # we want rln weights for trivia
                self.scores['{} {}'.format(self.model_name, task)]['f1'] = f1
                self.scores['{} {}'.format(self.model_name, task)]['iter'] = temp_iter
                
                
                # loading best models for previous tasks
                for j, prev_task in enumerate(prev_tasks):
                    for k, path in enumerate(paths):
                        self.scores['{} {}'.format(self.model_name, prev_task)]['f1'].append(inc*(k+1))
                        
                        self.scores['{} {}'.format(self.model_name, prev_task)]['iter'] = temp_iter
            else:
                # store best SQuAD weights
                best_prev_weights.append(best_path)
                prev_tasks.append(task)