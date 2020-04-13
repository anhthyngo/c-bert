"""
Module to contain continual learning class
"""

import torch
import numpy as np

class ContLearner():
    def __init__(self,
                 model,                                 # model to train
                 model_name,                            # name of model
                 learner,                               # device to run on
                 curriculum = ['SQuAD', 'TriviaQA-web'] # curriculum for learning
                 ):
        """
        Class for continual learning
        """
        self.model = model
        self.model_name = model_name
        self.learner = learner
        self.curriculum = curriculum
        self.scores = {}
        
        for task in curriculum:
            self.scores[task] = {
                'iter' : None,
                'f1'   : []
                }
  
    def c_learn(self):
        """
        Method to run continual learning on curriculum.
    
        This is for the testing our experiment to get results on how well
        'model' can do continual learning.
        
        Curriculum: Unsupervised -> SQuAD -> TriviaQA
        """
        best_squad_weights = None
        
        for task in self.curriculum:
            
            # fine tune model on task
            paths, f1, best_path = self.learner.fine_tune(task)
            
            temp_iter = self.learner.log_int*np.arange(len(f1)+1) # every 10k
            
            if task == 'TriviaQA-web':
                # we want rln weights for trivia
                self.scores['TriviaQA-web']['f1'] = f1
                self.scores['TriviaQA-web']['iter'] = temp_iter
                self.scores['SQuAD']['iter'] = temp_iter
                
                # loading RLN and classification for SQuAD
                self.model.load_state_dict(torch.load(best_squad_weights))
                for path in paths:
                    # get validation scores through zero-shot replacing RLN weights
                    self.model.bert.load_state_dict(torch.load(path))
                    _ , zero_f1 = self.learner.evaluate(task, self.model, prefix = 'forget_SQuAD_{}'.format(self.model_name))
                    self.scores['SQuAD']['f1'].append(zero_f1)
                    
            elif task == 'SQuAD':
                # store best SQuAD weights
                best_squad_weights = best_path