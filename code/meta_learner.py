import logging as log

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

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
            
        self.net.to(device)

# =============================================================================
#     def reset_classifer(self, class_to_reset):
#         bias = self.net.model.parameters()[-1]
#         weight = self.net.model.parameters()[-2]
#         torch.nn.init.kaiming_normal_(weight[class_to_reset].unsqueeze(0))
# =============================================================================

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


    def inner_update(self, batch, fast_weights):
        self.net.train()

        # unpack batch data
        batch = tuple(t.to(self.device) for t in batch)
        
        inputs = {
            "input_ids"       : batch[0],
            "attention_mask"  : batch[1],
            "token_type_ids"  : batch[2],
            "start_positions" : batch[3],
            "end_positions"   : batch[4],
            "fast_weights"    : fast_weights,
            }
        
        out = self.net(**inputs)
        loss = out[0]
        
        # for multi-gpu
        if isinstance(self.net, nn.DataParallel):
            loss = loss.mean() # average on multi-gpu parallel training

        if fast_weights is None:
            if isinstance(self.net, nn.DataParallel):
                model = self.net.module.model
            else:
                model = self.net.model
            fast_weights = model.parameters()
        #
        grad = torch.autograd.grad(loss, fast_weights, allow_unused=True)

        fast_weights = list(
            map(lambda p: p[1] - self.update_lr * p[0] if p[1].learn else p[1], zip(grad, fast_weights)))

        for params_old, params_new in zip(self.net.parameters(), fast_weights):
            params_new.learn = params_old.learn

        return fast_weights

    def meta_loss(self, batches, fast_weights):
        
        losses = 0
        for i, batch in enumerate(batches):
            inputs = {
                "input_ids"       : batch[0],
                "attention_mask"  : batch[1],
                "token_type_ids"  : batch[2],
                "start_positions" : batch[3],
                "end_positions"   : batch[4],
                "fast_weights"    : fast_weights,
                }
            
            out = self.net(**inputs)
            loss = out[0]
            
            # for multi-gpu
            if isinstance(self.net, nn.DataParallel):
                loss = loss.mean() # average on multi-gpu parallel training
            
            losses += loss
            
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

        meta_losses = [0 for _ in range(len(d_traj) + 1)]  # losses_q[i] is the loss on step i
        # f1_meta_set = [0 for _ in range(self.update_step + 1)]

        # Doing a single inner update to get updated weights
        fast_weights = self.inner_update(d_traj[0], None)

# =============================================================================
#         with torch.no_grad():
#             # Meta loss before any inner updates
#             meta_loss, last_layer_logits = self.meta_loss(d_rand[0])
#             meta_losses[0] += meta_loss
# 
#             classification_accuracy = self.eval_accuracy(last_layer_logits, y_rand[0])
#             accuracy_meta_set[0] = accuracy_meta_set[0] + classification_accuracy
# 
#             # Meta loss after a single inner update
#             meta_loss, last_layer_logits = self.meta_loss(x_rand[0], fast_weights, y_rand[0], False)
#             meta_losses[1] += meta_loss
# 
#             classification_accuracy = self.eval_accuracy(last_layer_logits, y_rand[0])
#             accuracy_meta_set[1] = accuracy_meta_set[1] + classification_accuracy
# =============================================================================

        for k in range(1, len(d_traj)):
            # Doing inner updates using fast weights
            fast_weights = self.inner_update(d_traj[k], fast_weights)

        # Computing meta-loss with respect to latest weights
        meta_loss = self.meta_loss(d_rand, fast_weights)

# =============================================================================
#             # Computing accuracy on the meta and traj set for understanding the learning
#             with torch.no_grad():
#                 pred_q = F.softmax(logits, dim=1).argmax(dim=1)
#                 classification_accuracy = torch.eq(pred_q, y_rand[0]).sum().item()  # convert to numpy
#                 accuracy_meta_set[k + 1] = accuracy_meta_set[k + 1] + classification_accuracy
# =============================================================================

        # Taking the meta gradient step
        self.optimizer.zero_grad()
        meta_loss.backward()

        nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)

        self.optimizer.step()
# =============================================================================
#         accuracies = np.array(accuracy_meta_set) / len(x_rand[0])
# =============================================================================

        return meta_loss

# =============================================================================
#     def finetune(self, x_traj, y_traj, x_rand, y_rand):
#         """
# 
#         :param x_traj:   Input data of sampled trajectory
#         :param y_traj:   Ground truth of the sampled trajectory
#         :param x_rand:   Input data of the random batch of data
#         :param y_rand:   Ground truth of the random batch of data
#         :return:
#         """
# 
#         # print(y_traj)
#         # print(y_rand)
#         meta_losses = [0 for _ in range(10 + 1)]  # losses_q[i] is the loss on step i
#         accuracy_meta_set = [0 for _ in range(10 + 1)]
# 
#         # Doing a single inner update to get updated weights
# 
#         fast_weights = self.inner_update(x_traj[0], None, y_traj[0], False)
# 
#         with torch.no_grad():
#             # Meta loss before any inner updates
#             meta_loss, last_layer_logits = self.meta_loss(x_rand[0], self.net.parameters(), y_rand[0], False)
#             meta_losses[0] += meta_loss
# 
#             classification_accuracy = self.eval_accuracy(last_layer_logits, y_rand[0])
#             accuracy_meta_set[0] = accuracy_meta_set[0] + classification_accuracy
# 
#             # Meta loss after a single inner update
#             meta_loss, last_layer_logits = self.meta_loss(x_rand[0], fast_weights, y_rand[0], False)
#             meta_losses[1] += meta_loss
# 
#             classification_accuracy = self.eval_accuracy(last_layer_logits, y_rand[0])
#             accuracy_meta_set[1] = accuracy_meta_set[1] + classification_accuracy
# 
#         for k in range(1, 10):
#             # Doing inner updates using fast weights
#             fast_weights = self.inner_update(x_traj[0], fast_weights, y_traj[0], False)
# 
#             # Computing meta-loss with respect to latest weights
#             meta_loss, logits = self.meta_loss(x_rand[0], fast_weights, y_rand[0], False)
#             meta_losses[k + 1] += meta_loss
# 
#             # Computing accuracy on the meta and traj set for understanding the learning
#             with torch.no_grad():
#                 pred_q = F.softmax(logits, dim=1).argmax(dim=1)
#                 classification_accuracy = torch.eq(pred_q, y_rand[0]).sum().item()  # convert to numpy
#                 # print("Accuracy at step", k, "= classification_accuracy")
#                 accuracy_meta_set[k + 1] = accuracy_meta_set[k + 1] + classification_accuracy
# 
# 
#         accuracies = np.array(accuracy_meta_set) / len(x_rand[0])
# 
#         return accuracies, meta_losses
# =============================================================================

def main():
    pass


if __name__ == '__main__':
    main()
