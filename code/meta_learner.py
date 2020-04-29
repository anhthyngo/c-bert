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
                 update_step = 10,     # task-level inner update steps
                 hf_model_name = None, # Hugginface model name
                 config = None,        # Huggingface configuration object
                 myio = None,          # myio object
                 max_grad_norm = 1,    # max gradient norms
                 device = None,        # device for calculation
                 ):

        super(MetaLearningClassification, self).__init__()

        self.update_lr = update_lr
        self.meta_lr = meta_lr
        self.update_step = update_step

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
        #bias = self.net.model.qa_outputs.parameters()[-1]
        if isinstance(self.net, nn.DataParallel):
            model = self.net.module.model
        else:
            model = self.net.model
        
        weight = model.qa_outputs.parameters()
        torch.nn.init.kaiming_normal_(weight)
        #torch.nn.init.kaiming_normal_(bias)
    
    def freeze_rln(self):
        """
        Method to freeze RLN parameters
        """
        if isinstance(self.net, nn.DataParallel):
            model = self.net.module.model
        else:
            model = self.net.model
        
        for name, param in model.bert.named_parameters():
            param.learn = False
    
# =============================================================================
#     def clip_grad_params(self, params, norm=1):
# 
#         for p in params.parameters():
#             g = p.grad
#             # print(g)
#             g = (g * (g < norm).float()) + ((g > norm).float()) * norm
#             g = (g * (g > -norm).float()) - ((g < -norm).float()) * norm
#             # print(g)
#             p.grad = g
# =============================================================================

# =============================================================================
#     def sample_training_data(self, iterators, it2, steps=2, reset=True):
# 
#         # Sample data for inner and meta updates
# 
#         x_traj, y_traj, x_rand, y_rand, x_rand_temp, y_rand_temp = [], [], [], [], [], []
# 
#         counter = 0
#         #
#         x_rand_temp = []
#         y_rand_temp = []
# 
#         class_counter = 0
#         for it1 in iterators:
#             assert(len(iterators)==1)
#             rand_counter = 0
#             for img, data in it1:
#                 class_to_reset = data[0].item()
#                 if reset:
#                     # Resetting weights corresponding to classes in the inner updates; this prevents
#                     # the learner from memorizing the data (which would kill the gradients due to inner updates)
#                     self.reset_layer()
# 
#                 counter += 1
#                 if not counter % int(steps / len(iterators)) == 0:
#                     x_traj.append(img)
#                     y_traj.append(data)
#                     # if counter % int(steps / len(iterators)) == 0:
#                     #     class_cur += 1
#                     #     break
# 
#                 else:
#                     x_rand_temp.append(img)
#                     y_rand_temp.append(data)
#                     rand_counter += 1
#                     if rand_counter==5:
#                         break
#             class_counter += 1
# 
# 
#         # Sampling the random batch of data
#         counter = 0
#         for img, data in it2:
#             if counter == 1:
#                 break
#             x_rand.append(img)
#             y_rand.append(data)
#             counter += 1
# 
#         y_rand_temp = torch.cat(y_rand_temp).unsqueeze(0)
#         x_rand_temp = torch.cat(x_rand_temp).unsqueeze(0)
#         x_traj, y_traj, x_rand, y_rand = torch.stack(x_traj), torch.stack(y_traj), torch.stack(x_rand), torch.stack(
#             y_rand)
# 
#         x_rand = torch.cat([x_rand, x_rand_temp], 1)
#         y_rand = torch.cat([y_rand, y_rand_temp], 1)
#         # print(y_traj)
#         # print(y_rand)
#         return x_traj, y_traj, x_rand, y_rand
# 
#     def sample_few_shot_training_data(self, iterators, it2, steps=2, reset=True):
# 
#         # Sample data for inner and meta updates
# 
#         x_traj, y_traj, x_rand, y_rand, x_rand_temp, y_rand_temp = [], [], [], [], [], []
# 
#         counter = 0
#         #
#         x_rand_temp = []
#         y_rand_temp = []
# 
#         class_counter = 0
#         for it1 in iterators:
#             # print("Itereator no ", class_counter)
#             rand_counter = 0
#             flag=True
#             inner_counter=0
#             for img, data in it1:
# 
#                 class_to_reset = data[0].item()
#                 data[0] = class_counter
#                 # print(data[0])
# 
#                 # if reset:
#                 #     # Resetting weights corresponding to classes in the inner updates; this prevents
#                 #     # the learner from memorizing the data (which would kill the gradients due to inner updates)
#                 #     self.reset_classifer(class_to_reset)
# 
#                 counter += 1
#                 # print((counter % int(steps / len(iterators))) != 0)
#                 # print(counter)
#                 if inner_counter < 5:
#                     x_traj.append(img)
#                     y_traj.append(data)
# 
#                 else:
#                     flag = False
#                     x_rand_temp.append(img)
#                     y_rand_temp.append(data)
#                     rand_counter += 1
#                     if rand_counter==5:
#                         break
#                 inner_counter+=1
#             class_counter += 1
# 
# 
#         # Sampling the random batch of data
#         # counter = 0
#         # for img, data in it2:
#         #     if counter == 1:
#         #         break
#         #     x_rand.append(img)
#         #     y_rand.append(data)
#         #     counter += 1
# 
#         y_rand_temp = torch.cat(y_rand_temp).unsqueeze(0)
#         x_rand_temp = torch.cat(x_rand_temp).unsqueeze(0)
# 
# 
#         x_rand = x_rand_temp
#         y_rand = y_rand_temp
# 
#         x_traj, y_traj = torch.cat(x_traj).unsqueeze(0), torch.cat(y_traj).unsqueeze(0)
# 
#         # print(x_traj.shape, y_traj.shape)
# 
#         x_traj, y_traj = x_traj.expand(25, -1, -1, -1,-1)[0:5], y_traj.expand(25, -1)[0:5]
# 
#         # print(y_traj)
#         # print(y_rand)
#         # print(x_traj.shape, y_traj.shape, x_rand.shape, y_rand.shape)
#         return x_traj, y_traj, x_rand, y_rand
# =============================================================================


    def inner_update(self, batch, fast_weights):
        
        # unpack batch data
        self.net.train()
        
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
            fast_weights = self.net.parameters()
        #
        grad = torch.autograd.grad(loss, fast_weights)

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
#         pred_q = F.softmax(logits, dim=1).argmax(dim=1)
#         correct = torch.eq(pred_q, y).sum().item()
#         return correct
# 
#     def clip_grad(self, grad, norm=50):
#         grad_clipped = []
#         for g, p in zip(grad, self.net.parameters()):
#             g = (g * (g < norm).float()) + ((g > norm).float()) * norm
#             g = (g * (g > -norm).float()) - ((g < -norm).float()) * norm
#             grad_clipped.append(g)
#         return grad_clipped
# =============================================================================

    def forward(self, d_traj, d_rand):
        """

        :param d_traj:   Batched data of sampled trajectory
        :param d_traj:   Input data of the random batch of data
        :return:
        """

        meta_losses = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        f1_meta_set = [0 for _ in range(self.update_step + 1)]

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

        for k in range(1, self.update_step):
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
