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
        scores = {
                    trivia_f1 : [],
                    zero_f1   : []
                 }
     
        for task in curriculum:
            
            ## this will save the weights for the model at every interval 
            ## probably 10k
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
                ## we want weights for trivia
                for path in paths:
                    ## get vali scores
                    
                    model.load_state_dict(torch.load(path))
                    _,f1 = utils.evaluate(model,data = val_loader,device)
    
                    scores['trivia_f1'].append(f1)
            else:
                for path in paths:
                    ## zero shot SQuAD validation
                    cores['zero_f1'].append(zero_shot(model,path,val_loader))
                    
        
        return scores
    
              

            
    def zero_shot(self,
              model,                   # model to eval
              path,                    # paths of weight
              dataloader               # squad val loader
              ):
        '''
        method to zero-shot SQuAD validation data with saved weights
        
        NOT TESTED YET
        '''
    
    
      
        model.load_state_dict(torch.load(path))
        _,f1 = utils.evaluate(model,data = dataloader,device)
        
        return f1
            
    
    # def zero_shot(self,
    #               model,                   # model to eval
    #               w_paths                  # paths of weights
    #               ):
    #     '''
    #     method to zero-shot SQuAD validation data with saved weights
        
    #     NOT TESTED YET
    #     '''
    #     ## create dictionary to store f1/iters
    #     squad_f1 = {
    #                 "iter"       : [],
    #                 "f1_score"   : []
    #                 }
        
    #     for path in w_paths:
    #         model.load_state_dict(torch.load(path))
    #         zero_shot_squad_f1 = utils.evaluate(model,data = val_loader,device)
    #         squad_f1['iter'].append(iter_)
    #         squad_f1['f1_score'].append(zero_shot_squad_f1)
        
    #     return squad_f1
            
    
        
        
        
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
    data_dir = 'data'
    task_names = ['tester']
    tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased') 
    max_seq_length = 384
    doc_stride = 128
    max_query_length = 64
    threads = 1

    # create IO object with batch size 2 for testing
    data_handler = myio.IO(task_names,
                                tokenizer,
                                max_seq_length,
                                doc_stride,
                                max_query_length,
                                threads,
                                batch_size = 2,
                                data_dir = data_dir)
    data_handler.read_tasks()
    dl_train = data_handler.tasks.get('tester').get('train')
    dl_dev = data_handler.tasks.get('tester').get('dev')
    
    # print training example
    data = next(iter(dl_train))
    print("\n{} {}:\n {}".format('train', 1, tokenizer.decode(data[0][0,:])))

    data = next(iter(dl_dev))
    print("\n{} {}:\n {}".format('dev', 1, tokenizer.decode(data[0][0,:])))

 
if __name__ == '__main__':
    main()


      


# def c_learn(model, loss):
#     """
#     Method to run continual learning to Fine-Tune
    
#     Will most likely use methods implemented in utils
    
#     This is for the testing our experiment to get results on how well
#     'model' can do continual learning.
#     """
    
#     #curriculum is an iterable for the task names (i.e. squad and triviaqa)
#     for task in curriculum:
#         data = io.tasks(task)
#         utils.fine_tune(model, data)
