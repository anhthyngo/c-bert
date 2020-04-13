"""
Module for general utility methods like generalized training loop, evaluation,
fine-tuning, and getting data.

Mainly for continual learning.

Code based off Huggingface's run_squad example
https://github.com/huggingface/transformers/blob/7972a4019f4bc9f85fd358f42249b90f9cd27c68/examples/run_squad.py#L386
"""

import torch
import tqdm
import torch.nn as nn
import torch.optim as opt
import numpy as np
from sklearn import metrics
import transformers
import transformers.data.processors.squad as sq
import transformers.data.metrics.squad_metrics as sq_metrics
import os
import copy
from tqdm import tqdm, trange
import logging as log

class Learner():
    def __init__(self,
                 model,
                 model_name,
                 device,
                 myio,
                 save_dir,
                 n_best,
                 max_answer_length,
                 do_lower_case,
                 verbose_logging,
                 version_2_with_negative,
                 null_score_diff_threshold,
                 max_steps = 1e5,
                 log_int = 1e4,
                 best_int = 500,
                 verbose_int = 1000,
                 max_grad_norm = 1.0,
                 optimizer = None,
                 scheduler = None,
                 weight_decay = 0.0,
                 lr = 5e-3,
                 eps = 1e-8,
                 warmup_steps = 0
                 ):
        """
        Object to store learning. Used for fine-tuning.
        
        Data stored in myio.IO object called myio.
        """
        
        self.model = model
        self.model_name = model_name
        self.device = device
        self.IO = myio
        self.save_dir = save_dir
        self.max_steps = max_steps
        self.log_int = log_int
        self.best_int = best_int
        self.verbose_int = verbose_int
        self.max_grad_norm = max_grad_norm
        self.log_dir = os.path.join(self.save_dir, 'logged')
        
        # make directory for recorded weights if doesn't already exist
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        
        # for evaluation
        self.n_best = n_best
        self.max_answer_length = max_answer_length
        self.do_lower_case = do_lower_case
        self.verbose_logging = verbose_logging
        self.version_2_with_negative = version_2_with_negative
        self.null_score_diff_threshold = null_score_diff_threshold
        
        # set optimizer and scheduler
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        if optimizer is None:
            assert scheduler is None, "Optimizer not defined, so scheduler should not be defined."
            
            # don't apply weight decay to bias and LayerNorm weights
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {
                    "params"       : [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay" : weight_decay
                },
                {
                    "params"       : [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay" : 0.0
                }
            ]
            
            self.optimizer = opt.AdamW(optimizer_grouped_parameters, lr=lr, eps=eps)
            
        if scheduler is None:
            self.scheduler = transformers.optimization.get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps
            )
        
        
    def train_step(self,
                   batch,
                   idx
                   ):
        """
        Training for a single batch.
        
        --------------------
        Returns:
        loss - average of start and end cross entropy loss
        https://huggingface.co/transformers/_modules/transformers/modeling_bert.html#BertForQuestionAnswering
        """
        # unpack batch data
        batch = tuple(t.to(self.device) for t in batch)
        
        inputs = {
            "input_ids"       : batch[0],
            "attention_mask"  : batch[1],
            "token_type_ids"  : batch[2],
            "start_positions" : batch[3],
            "end_positions"   : batch[4]
            }
            
        # zero gradients related to optimizer
        self.model.zero_grad()
            
        # send data through model forward
        out = self.model(**inputs)
        # model outputs are always tuple in transformers (see doc)
        loss = out[0]
        
        # clip gradients
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        # calculate gradients through back prop
        loss.backward()
            
        #take a step in gradient descent
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.detach()
        
    def evaluate(self,
                 task,
                 model = None,
                 prefix = ""
                 ):
        """
        Evaluation model on task.
        
        ---------------------
        Returns:
        results - Huggingface SQuAD eval dictionary with keys
            "exact" : exact match score
            "f1"    : F1 score
            total" : number of observations
        https://github.com/huggingface/transformers/blob/7972a4019f4bc9f85fd358f42249b90f9cd27c68/src/transformers/data/metrics/squad_metrics.py#L107
        """
        # assign model if None
        if model is None:
            model = self.model
        
        # puts model in evaluation mode
        model.eval()
        
        # get dictionary of DataLoader, examples, features
        package = self.IO.get(task).get('dev')
        val_dataloader = package.get('data')
        examples = package.get('examples')
        features = package.get('features')
        
        all_results = []
        
        # don't need to track gradient
        for batch in tqdm(val_dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            
            with torch.no_grad():
                inputs = {
                    "input_ids"      : batch[0],
                    "attention_mask" : batch[1],
                    "token_type_ids" : batch[2]
                    }
                
                example_indices = batch[3]
                
                # send data through model forward
                outputs = model(**inputs)
                
            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                
                output = [output[i].detach().cpu().tolist() for output in outputs]
                
                start_logits, end_logits = output
                result = sq.SquadResult(unique_id, start_logits, end_logits)
                
                all_results.append(result)
        
        # compute predictions
        output_prediction_file = os.path.join(self.save_dir, "predictions_{}.json".format(prefix))
        output_nbest_file = os.path.join(self.save_dir, "nbest_predictions_{}.json".format(prefix))
        output_null_log_odds_file = None
        
        predictions = sq_metrics.compute_predictions_logits(
            examples,
            features,
            all_results,
            self.n_best,
            self.max_answer_length,
            self.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            self.verbose_logging,
            self.version_2_with_negative,
            self.null_score_diff_threshold,
            self.IO.tokenizer
            )
        
        # compute F1 and exact scores
        results = sq_metrics.squad_evaluate(examples, predictions)
        
        # put model back in train mode
        model.train()
        
        return results
        
    def fine_tune(self,
                  task,
                  model_name = None,
                  model = None
                  ):
        """
        Fine-tune model on task
            
        --------------------
        Return: 
        logged_rln_paths - list of best RLN weight paths
        logged_f1        - list of best validation f1 scores
        best_path        - path of best weights
        """
        if model_name is None:
            model_name = self.model_name
            
        if model is None:
            model = self.model
        
        cum_loss =  0.0
        best_f1 = 0
        best_iter = 0
        logged_rln_paths = []
        logged_f1s = []
        task_log_dir = os.path.join(self.log_dir, model_name, task)
        
        # make directory for model weights for given task if doesn't exist
        if not os.path.exists(task_log_dir):
            os.mkdir(task_log_dir)
        
        best_path = os.path.join(self.save_dir, model_name + '_{}_best.pt'.format(task))
        best_model = copy.deepcopy(model)
        
        train_package = self.IO.get(task).get('train')
        train_dataloader = train_package.get('data')
        
        # set number of epochs based on number of iterations
        max_epochs = self.max_steps // len(train_dataloader) + 1
        

        # train
        global_step = 0
        epochs_trained = 0
        
        train_iterator = trange(epochs_trained, int(max_epochs), desc = 'Epoch')
        
        model.zero_grad()
        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc='Iteration')
            for step, batch in enumerate(epoch_iterator):
                
                model.train()
                iter_loss = self.train_step(batch, step)
                cum_loss += iter_loss
                
                # check for best every best_int
                if global_step % self.best_int == 0:
                    val_results = self.evaluate(task, prefix = '{}_current'.format(task))
                    current_f1 = val_results.get('f1')
                    
                    if current_f1 > best_f1:
                        best_f1 = current_f1
                        best_iter = global_step
                        torch.save(model.state_dict(), best_path)
                        best_model.load_state_dict(torch.load(best_path))
                
                # write to log every verbose_int
                if global_step % self.verbose_int == 0:
                    log.info('\nIteration {} of {} | Average Training Loss {:.6f} |'\
                             ' Best Val F1 {} | Best Iteration {} |'.format(
                                 global_step,
                                 self.max_steps,
                                 cum_loss/global_step,
                                 best_f1,
                                 best_iter
                        )
                    )
                
                # save every log_int
                if global_step % self.log_int == 0:
                    log_results = self.evaluate(task, best_model, prefix = '{}_log'.format(task))
                    log_f1 = log_results.get('f1')
                    
                    log_rln_weights = os.path.join(task_log_dir, '{}.pt'.format(global_step))
                    
                    # save weights
                    # only supports bert right now
                    torch.save(model.bert.state_dict(), log_rln_weights)
                    
                    # record f1 and rln weights path
                    logged_rln_paths.append(log_rln_weights)
                    logged_f1s.append(log_f1)
                
                global_step += 1
                
                # break training if max steps reached (-1 to match Huggingface)
                if global_step > self.max_steps:
                    epoch_iterator.close()
                    break
            if global_step > self.max_steps:
                train_iterator.close()
                break
            
            # final check for best if not already checked
            if global_step % self.best_int != 0:
                val_results = self.evaluate(task, prefix = '{}_current'.format(task))
                current_f1 = val_results.get('f1')
                    
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_iter = global_step
                    torch.save(model.state_dict(), best_path)
                    best_model.load_state_dict(torch.load(best_path))
            
            # log finished results
            log.info('\nFinished | Average Training Loss {:.6f} |'\
                     ' Best Val F1 {} | Best Iteration {} |'.format(
                         cum_loss/global_step,
                         best_f1,
                         best_iter
                    )
                )
        
        return rln_paths, best_f1s, best_path