import logging as log

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm, trange
import copy

import model

class MetaLearningClassification(nn.Module):
    """
    MetaLearningClassification Learner
    """

    def __init__(self,
                 update_lr = 1e-4,     # task-level inner update learning rate
                 meta_lr = 0.01,       # meta-level outer learning rate
                 hf_model_name = None, # Hugginface model name
                 config = None,        # Huggingface configuration object
                 myio = None,          # myio object
                 max_grad_norm = 1,    # max gradient norms
                 device = None,        # device for calculation
                 ):

        super(MetaLearningClassification, self).__init__()

        self.update_lr = update_lr
        self.meta_lr = meta_lr

        self.net = model.QAModel(hf_model_name, config)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.IO = myio
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.net = nn.DataParallel(self.net)

        self.fast_net = self.net

        # send to device
        self.net.to(device)
        self.fast_net.to(device)

    def reset_layer(self):
        """
        Resets classification head
        """
        if isinstance(self.net, nn.DataParallel):
            model = self.net.module.model
        else:
            model = self.net.model
        
        for weight in model.qa_outputs.parameters():
            if len(weight.shape)>1:
                torch.nn.init.kaiming_normal_(weight)
            else:
                torch.nn.init.zeros_(weight)
    
    def freeze_rln(self):
        """
        Method to freeze RLN parameters
        """
        if isinstance(self.net, nn.DataParallel):
            model = self.net.module.model
        else:
            model = self.net.model

        for name, param in model.named_parameters():
            param.learn = True

        for name, param in model.bert.named_parameters():
            param.learn = True

        for name, param in model.bert.named_parameters():
            param.learn = False

    def unfreeze_rln(self):
        """
        Method to freeze RLN parameters
        """
        if isinstance(self.net, nn.DataParallel):
            model = self.net.module.model
        else:
            model = self.net.model

        for name, param in model.named_parameters():
            param.learn = True

        for name, param in model.bert.named_parameters():
            param.learn = True

        for name, param in model.bert.named_parameters():
            param.learn = True


    def send_batch(self, batch, fast_weights):
        # unpack batch data
        batch = tuple(t.to(self.device) for t in batch)

        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "start_positions": batch[3],
            "end_positions": batch[4],
        }

        if fast_weights is None:
            net = self.net
        else:
            net = self.fast_net
            for fast_net_param, fast_weight in zip(net.parameters(), fast_weights):
                if fast_weight.learn == True:
                    fast_net_param = fast_weight

        net.train()
        net.to(self.device)

        out = net(**inputs)
        loss = out[0]

        # for multi-gpu
        if isinstance(self.net, nn.DataParallel):
            loss = loss.mean()  # average on multi-gpu parallel training

        return loss

    def inner_update(self, batch, fast_weights):

        loss = self.send_batch(batch, fast_weights)

        if fast_weights is None:
            fast_weights = self.net.parameters()

        grad = torch.autograd.grad(loss, fast_weights, allow_unused=True)

        fast_weights = list(
            map(lambda p: p[1] - self.update_lr * p[0] if p[1].learn else p[1], zip(grad, fast_weights)))

        for params_old, params_new in zip(self.net.parameters(), fast_weights):
            params_new.learn = params_old.learn

        return fast_weights

    def meta_loss(self, batches, fast_weights):

        losses = 0
        for i, batch in enumerate(batches):
            losses += self.send_batch(batch, fast_weights)
            
        return losses/(i+1)

# =============================================================================
#     def eval_accuracy(self, logits, y):
#         TODO: evaluate accuracy
# =============================================================================

    def forward(self, d_traj, d_rand):
        """

        :param d_traj:   Batched data of sampled trajectory
        :param d_rand:   Input data of the random batch of data
        :return: meta-loss
        """
        # reset classification layer
        self.reset_layer()

        # Doing a single inner update to get updated weights
        fast_weights = self.inner_update(d_traj[0], None)

        for k in trange(1, len(d_traj), desc='Meta Inner', mininterval=30):
            # Doing inner updates using fast weights
            fast_weights = self.inner_update(d_traj[k], fast_weights)

        # Computing meta-loss with respect to latest weights
        meta_loss = self.meta_loss(d_rand, fast_weights)

        # Taking the meta gradient step
        self.optimizer.zero_grad()

        self.unfreeze_rln()
        meta_loss.backward()

        nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)

        self.optimizer.step()
        self.freeze_rln()

        return meta_loss

def main():
    pass


if __name__ == '__main__':
    main()
