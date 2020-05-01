"""
Module for general utility methods like generalized training loop, evaluation,
fine-tuning, and getting data.

Mainly for continual learning.

Code based off Huggingface's run_squad example
https://github.com/huggingface/transformers/blob/7972a4019f4bc9f85fd358f42249b90f9cd27c68/examples/run_squad.py
"""

import torch
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
import time

class Learner():
    def __init__(self,
                 access_mode,
                 fp16,
                 fp16_opt_level,
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
                 weight_decay = 0.0,
                 lr = 5e-3,
                 eps = 1e-8,
                 warmup_steps = 0,
                 freeze_embeddings = False,
                 ):
        """
        Object to store learning. Used for fine-tuning.
        
        Data stored in myio.IO object called myio.
        """
        self.fp16 = fp16
        self.fp16_opt_level = fp16_opt_level
        self.access_mode = access_mode
        
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.IO = myio
        self.save_dir = save_dir
        self.max_steps = max_steps
        self.log_int = log_int
        self.best_int = best_int
        self.verbose_int = verbose_int
        self.max_grad_norm = max_grad_norm
        self.weight_decay = weight_decay
        self.lr = lr
        self.eps = eps
        self.warmup_steps = warmup_steps
        self.freeze = freeze_embeddings
        
        # make directory for recorded weights if doesn't already exist
        self.log_dir = os.path.join(self.save_dir, 'logged')
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        
        # for evaluation
        self.n_best = n_best
        self.max_answer_length = max_answer_length
        self.do_lower_case = do_lower_case
        self.verbose_logging = verbose_logging
        self.version_2_with_negative = version_2_with_negative
        self.null_score_diff_threshold = null_score_diff_threshold
        
        # data
        self.train_dataloader = None
        self.val_dataloader = None
        self.val_examples = None
        self.val_features = None
        
        # stop embedding weight grad tracking
        if self.freeze:
            self.no_embedding_grads()
        
        # set optimizer
        self.optimizer = optimizer
        
        if optimizer is None:
            self.set_optimizer()
        
        # use mixed precision if needed
        if self.fp16:
            from apex import amp
            amp.register_half_function(torch, "einsum")
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=self.fp16_opt_level)
        
        # if multiple GPUs on single device
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(model)
            self.model.to(self.device)
    
    def no_embedding_grads(self):
        """
        Method to freeze embedding weights
        """
        for param in self.model.model.bert.parameters():
            param.requires_grad = False
    
    def set_optimizer(self):
        """
        Set optimizer for learner object using model.
        """
        # only adjust qa_outputs if doing feature extraction
        if self.freeze:
            named_params = self.model.model.qa_outputs.named_parameters()
        else:
            named_params = self.model.named_parameters()
        
        # don't apply weight decay to bias and LayerNorm weights     
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                "params"       : [p for n, p in named_params if not any(nd in n for nd in no_decay)],
                "weight_decay" : self.weight_decay
            },
            {
                "params"       : [p for n, p in named_params if any(nd in n for nd in no_decay)],
                "weight_decay" : 0.0
            }
        ]
            
        self.optimizer = opt.AdamW(optimizer_grouped_parameters, lr=self.lr, eps=self.eps)
    
    def save_log_weights(self, task_log_dir, idx):
        """
        Helper method to save down logged weights
        """
        log_weights = os.path.join(task_log_dir, '{}.pt'.format(idx))
        log_rln_weights = os.path.join(task_log_dir, '{}_rln.pt'.format(idx))
        
        # for mulit-gpu
        if isinstance(self.model, nn.DataParallel):
            state_dict = self.model.module.state_dict()
            rln_state_dict = self.model.module.model.bert.state_dict()
        else:
            state_dict = self.model.state_dict()
            rln_state_dict = self.model.model.bert.state_dict()
        
        torch.save(state_dict, log_weights)
        torch.save(rln_state_dict, log_rln_weights)
        os.chmod(log_weights, self.access_mode)
        os.chmod(log_rln_weights, self.access_mode)
        
        return log_weights, log_rln_weights
    
    def train_step(self,
                   batch,
                   idx,
                   scheduler,
                   ):
        """
        Training for a single batch.
        
        --------------------
        Returns:
        loss - average of start and end cross entropy loss
        https://huggingface.co/transformers/_modules/transformers/modeling_bert.html#BertForQuestionAnswering
        """
        if self.fp16:
            from apex import amp
        
        # unpack batch data
        batch = tuple(t.to(self.device) for t in batch)
        
        inputs = {
            "input_ids"       : batch[0],
            "attention_mask"  : batch[1],
            "token_type_ids"  : batch[2],
            "start_positions" : batch[3],
            "end_positions"   : batch[4]
            }
            
        # zero gradients
        self.model.zero_grad()
            
        # send data through model forward
        out = self.model(**inputs)
        # model outputs are always tuple in transformers (see doc)
        loss = out[0]
        
        # for multi-gpu
        if isinstance(self.model, nn.DataParallel):
            loss = loss.mean() # average on multi-gpu parallel training
        
        # calculate gradients through back prop
        if self.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        
        # clip gradients
        if self.fp16:
            nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.max_grad_norm)
        else:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
        #take a step in gradient descent
        self.optimizer.step()
        scheduler.step()
        
        # zero gradients
        self.model.zero_grad()
        
        return loss.detach()
        
    def evaluate(self,
                 task,
                 model = None,
                 prefix = "",
                 load = False
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
        else:
            if torch.cuda.is_available() and torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
                # multiple GPUs
                model = torch.nn.DataParallel(model)
            model.to(self.device)
        
        # puts model in evaluation mode
        model.eval()
        
        # get dictionary of DataLoader, examples, features  
        if load:
            val_dataloader, features, examples = self.IO.load_and_cache_task(task, 'dev')
            self.val_dataloader, self.val_features, self.val_examples = val_dataloader, features, examples
        else:
            val_dataloader, features, examples = self.val_dataloader, self.val_features, self.val_examples
        
        all_results = []
        
        # don't need to track gradient
        for i, batch in enumerate(val_dataloader):
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
            self.null_score_diff_threshold)

        # compute F1 and exact scores
        results = sq_metrics.squad_evaluate(examples, predictions)
        
        # put model back in train mode
        model.train()
        
        return results
        
    def fine_tune(self,
                  task,
                  model_name = None,
                  scheduler = None,
                  ):
        """
        Fine-tune model on task
            
        --------------------
        Return: 
        logged_rln_paths - list of best RLN weight paths
        logged_f1        - list of best validation f1 scores
        best_path        - path of best weights
        """
        # set up learning rate scheduler
        if scheduler is None:
            scheduler = transformers.optimization.get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.max_steps
                )
        
        if model_name is None:
            model_name = self.model_name
        
        current_f1 = None
        cum_loss =  0.0
        best_f1 = 0
        best_iter = 0
        logged_rln_paths = []
        logged_paths = []
        logged_f1s = []
        model_log_dir = os.path.join(self.log_dir, model_name)
        task_log_dir = os.path.join(model_log_dir, task)
        
        # make directory for model weights for given task if doesn't exist
        if not os.path.exists(model_log_dir):
            os.mkdir(model_log_dir)
        if not os.path.exists(task_log_dir):
            os.mkdir(task_log_dir)
        
        best_path = os.path.join(task_log_dir, model_name + '_{}_best.pt'.format(task))
        best_rln_path = os.path.join(task_log_dir, model_name + '_{}_best_rln.pt'.format(task))
        best_model = copy.deepcopy(self.model)
        
        # load data
        self.train_dataloader,_,_ = self.IO.load_and_cache_task(task, 'train')
        self.val_dataloader, self.val_features, self.val_examples = self.IO.load_and_cache_task(task, 'dev')
        
        # set number of epochs based on number of iterations
        max_epochs = self.max_steps // len(self.train_dataloader) + 1
        

        # train
        global_step = 1
        
        train_iterator = trange(0, int(max_epochs), desc = 'Epoch', mininterval=30)
        start = time.time()
        # log baseline zero-shot
        log.info("Storing results for zero-shot on task: {}".format(task))
        zero_shot = self.evaluate(task, prefix = '{}_current'.format(task))
        
        log_weights, log_rln_weights = self.save_log_weights(task_log_dir, 0)
        logged_rln_paths.append(log_rln_weights)
        logged_paths.append(log_weights)
        
        logged_f1s.append(zero_shot.get('f1'))
        best_f1 = zero_shot.get('f1')
        
        self.model.zero_grad()
        for epoch in train_iterator:
            epoch_iterator = tqdm(self.train_dataloader, desc='Epoch Iteration', mininterval=30)
            for step, batch in enumerate(epoch_iterator):
                
                self.model.train()
                iter_loss = self.train_step(batch, step, scheduler)
                cum_loss += iter_loss
                
                # check for best every best_int
                if global_step % self.best_int == 0:
                    log.info("="*40+" Evaluating {} on step: {}".format(task, global_step))
                    val_results = self.evaluate(task, prefix = '{}_current'.format(task))
                    current_f1 = val_results.get('f1')
                    
                    log.info("="*40+" Current Score {}, Step = {} | Previous Best Score {}, Step = {}".format(
                        current_f1,
                        global_step,
                        best_f1,
                        best_iter))
                    
                    if current_f1 > best_f1:
                        best_f1 = current_f1
                        best_iter = global_step
                        
                        # for multi-gpu
                        if isinstance(self.model, nn.DataParallel):
                            best_state_dict = self.model.module.state_dict()
                            best_rln_state_dict = self.model.module.model.bert.state_dict()
                        else:
                            best_state_dict = self.model.state_dict()
                            best_rln_state_dict = self.model.model.bert.state_dict()
                        
                        torch.save(best_state_dict, best_path)
                        torch.save(best_rln_state_dict, best_rln_path)
                        os.chmod(best_path, self.access_mode)
                        
                        # for multi-gpu
                        if isinstance(best_model, nn.DataParallel):
                            best_model.module.load_state_dict(torch.load(best_path))
                        else:
                            best_model.load_state_dict(torch.load(best_path))
                
                # write to log every verbose_int
                if global_step % self.verbose_int == 0:
                    log.info('='*40+' Iteration {} of {} | Average Training Loss {:.6f} |'\
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
                    log.info("="*40+" Storing data for plotting on task {} step {}".format(task, global_step))
                    if self.log_int != self.best_int:
                        log_results = self.evaluate(task, prefix = '{}_log'.format(task))
                        log_f1 = log_results.get('f1')
                    elif not current_f1 is None:
                        log_f1 = current_f1
                    
                    log_weights, log_rln_weights = self.save_log_weights(task_log_dir, global_step)
                    
                    # record f1 and rln weights path
                    logged_paths.append(log_weights)
                    logged_rln_paths.append(log_rln_weights)
                    logged_f1s.append(log_f1)
                
                global_step += 1
                
                # break training if max steps reached (+1 to get max_step)
                if global_step > self.max_steps+1:
                    epoch_iterator.close()
                    break
            if global_step > self.max_steps+1:
                train_iterator.close()
                break
            
        # log finished results
        log.info('Finished | Average Training Loss {:.6f} |'\
                 ' Best Val F1 {} | Best Iteration {} | Time Completed {:.2f}s'.format(
                     cum_loss/global_step,
                     best_f1,
                     best_iter,
                     time.time()-start
                )
            )
        
        return logged_paths, logged_rln_paths, logged_f1s, best_path