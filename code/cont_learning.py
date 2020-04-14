"""
Module to contain continual learning class
"""

import torch
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
        best_squad_weights = None
        
        for task in self.curriculum:
            
            log.info("Fine-Tuning: {}".format(task))
            # fine tune model on task
            paths, f1, best_path = self.learner.fine_tune(task)
            
            temp_iter = self.learner.log_int*np.arange(len(f1)+1) # every 10k
            
            if task == 'SQuAD':
                # store best SQuAD weights
                best_squad_weights = best_path
                
            elif task == 'TriviaQA-web':
                # we want rln weights for trivia
                self.scores['TriviaQA-web']['f1'] = f1
                self.scores['TriviaQA-web']['iter'] = temp_iter
                self.scores['SQuAD']['iter'] = temp_iter
                
                # loading RLN and classification for SQuAD
                self.model.load_state_dict(torch.load(best_squad_weights))
                for path in paths:
                    # get validation scores through zero-shot replacing RLN weights
                    self.model.model.bert.load_state_dict(torch.load(path))
                    _ , zero_f1 = self.learner.evaluate(task, self.model, prefix = 'forget_SQuAD_{}'.format(self.hf_model_name))
                    self.scores['SQuAD']['f1'].append(zero_f1)
                    
            