"""
Module to contain continual learning class
NOTE: severely unfinished lol
"""

import torch
import utils
import myio


class CLearner():
    def __init__(self,
                 model,                     # model to train
                 #device,                   # device to run on
                 train_data,                # training dataloader
                 val_data,                  # val dataloader
                 #save_path                  # path to save best weights
                 ):
        """
        Class for continual learning
        """
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.scores = {
                    'TriviaQA-web' : {'iter' : [], 'f1': []},
                    'SQuAD'   : {'iter' : [], 'f1': []}
                 }
        #self.save_path = save_path
        
# =============================================================================
#     Helper methods
# =============================================================================
   
  
    def c_learner(self,
                  model,                   # mod
                  train_loader,            # data loader for train
                  val_loader,              # data loader for val
                  #weight_paths,            # list of weight paths
                  loss,
                  save_path,
                  curriculum = ['SQuAD','TriviaQA-web']
                  ):
        """
        Method to run continual learning to Fine-Tune
    
        Will most likely use methods implemented in utils
    
        This is for the testing our experiment to get results on how well
        'model' can do continual learning.
        
        Curriculum: Unsupervised -> SQuAD -> TriviaQA
        
        NOT TESTED YET
        """
        
        
        best_squad_weights = None
        
        for task in curriculum:
            
            ## this will save the weights for the model at every interval 
            ## probably 10k
            
            # getting RLN weights
            paths, f1 = utils.fine_tune(model,
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
            
            temp_iter = 1e4*np.arange(len(f1))
            
            if task == 'TriviaQA-web':
                ## we want weights for trivia
                self.scores['TriviaQA-web']['f1'] = f1
                self.scores['TriviaQA-web']['iter'] =temp_iter
                
                # loading RLN and classification for SQuAD
                model.load_state_dict(torch.load(best_squad_weights))
                for path in paths:
                    ## get vali scores
                    
                    model.bert.load_state_dict(torch.load(path))
                    _,f1 = utils.evaluate(model,data = val_loader,device)
                    
                    self.scores['SQuAD']['f1'].append(f1)
                
                self.scores['SQuAD']['iter'] = temp_iter
            else:
                ## save squad weights
                # saving classification layer + RLN weights
                
                # set best_squad_weights = os.path.join()
                torch.save(model.state_dict(), best_squad_weights)
            
        
        return scores
    
              

    
        
        
        

