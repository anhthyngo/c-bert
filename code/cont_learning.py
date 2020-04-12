"""
Module to contain continual learning class
"""

import torch
import utils
import myio




class CLearner():
    def __init__(self,
                 model,                     # model to train
                 #device,                    # device to run on
                 #train_data,                # training dataloader
                 #test_data,                 # testing dataloader
                 save_path                  # path to save best weights
                 ):
        """
        Class for continual learning
        """
        self.model = model
        self.train_data = train_data
        self.save_path = save_path
        
# =============================================================================
#     Helper methods
# =============================================================================
   
  
    def c_learner(self,
                  model,                   # mod
                  train_loader,            # data loader for train
                  val_loader,              # data loader for val
                  weight_paths,            # list of weight paths
                  loss,
                  curriculum,
                  save_path
                  ):
        """
        Method to run continual learning to Fine-Tune
    
        Will most likely use methods implemented in utils
    
        This is for the testing our experiment to get results on how well
        'model' can do continual learning.
        """
        
       
        for task in curriculum:
            
            ## this already saves weights and returns the paths
            paths = utils.fine_tune(model,
                                    train_loader,
                                    val_loader,
                                    val_label,
                                    loss,
                                    optimizer,
                                    device,
                                    n_epochs,
                                    anneal_threshold,
                                    anneal_r,
                                    save_path,
                                    model_name = task)
            
            if task == 'TriviaQA-web':
                ## zero-shot SQuAD validation w/ trivia weights.
                zero_shot(model,paths)
                
              
                

    
            trivia_scores.append(trivia_f1)
    
    def zero_shot(self,
                  model,                   #
                  w_paths                  #
                  ):
        
        squad_f1 = {
                    "iter"       : [],
                    "f1_score"   : []
                    }
        
        for path in w_paths:
            model.load_state_dict(torch.load(path))
            zero_shot_squad_f1 = utils.eval(model,'???')
            #squad_f1['iter'].append(iter_)
            squad_f1['f1_score'].append(zero_shot_squad_f1)
        
        return squad_f1
            
 
        
        
        
def main():
    import torch
    import analyze
    import numpy as np
    from datetime import datetime as dt
    import os
    import json
    import transformers
    import myio

    # set parameters for IO object
    wd = os.getcwd()
    data_dir = os.path.join(wd, r'test_data')
    task_names = ['tester']
    tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
    max_length = 512

    data_handler = myio.IO(data_dir, task_names, tokenizer, max_length, batch_size = 2)
    data_handler.read_task(True) 
    train_loader = data_handler.tasks.get('tester').get('train')
    test_loader = data_handler.tasks.get('tester').get('dev')
    
    for i,(data, labels) in enumerate(test_loader):
        for k, obs in enumerate(data):
                print(r'{} batch {} obs {} decoded: {}'.format('tester', i, k, tokenizer.decode(obs.tolist())))
                print('---------------')

if __name__ == '__main__':
    main()


'''      


def c_learn(model, loss):
    """
    Method to run continual learning to Fine-Tune
    
    Will most likely use methods implemented in utils
    
    This is for the testing our experiment to get results on how well
    'model' can do continual learning.
    """
    
    #curriculum is an iterable for the task names (i.e. squad and triviaqa)
    for task in curriculum:
        data = io.tasks(task)
        utils.fine_tune(model, data)
'''