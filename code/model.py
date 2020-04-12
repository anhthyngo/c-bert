"""
    Module to define our model (RLN, PLN)

    BERT-M -> SQuAD -> TriviaQA
    BERT -> SQuAD -> TriviaQA

    ref code:
    https://colab.research.google.com/drive/1pTuQhug6Dhl9XalKB0zUGf4FIdYFlpcX#scrollTo=1sjzRT1V0zwm
    https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_bert.py

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
import logging
import tensorflow as tf

import meta_learning


# ============================ Set up Trainer model ============================

class train_model(nn.Module):
    """
        Define the architecture of the model

        set config = BertConfig.from_pretrained('bert-base-uncased')
    """
    def __init__(self, config, meta):
        super().__init__(config) #take transformers.BertConfig
        self.num_labels = config.num_labels
        # === PLN definition task ===
        self.classification = nn.Sequential(
                nn.Linear(config.hidden_size, config.num_labels),
                nn.ReLU(),
                )
        #self.softmax = nn.Softmax(config.vocab_size)

        # === RLN definiton as BERT or meta-BERT ===
        # Load weights
        if meta:
            #load meta BERT weights to self.model
            self.model = meta_learning.from_pretrained('bert_m') #call bert_m from meta_learning.py, change syntax
        else:
            #load regular bert weights to self.model
            self.model = BertModel.from_pretrained('bert-base-uncased')

        self.init_weights() #initialize weights, defined in class BertPreTrainedModel in modeling_bert.py

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        head_mask = None,
        inputs_embeds = None,
        start_positions = None,
        end_positions = None,
    ):
        outputs = self.model(
            input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            position_ids = position_ids,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds, #these are defined in transformers.BertModel
        )

        sequence_output = outputs[0]

        logits = self.classification(sequence_output)
        
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)

    def print_param(self):

        model = self.model
        params = list(model.named_parameters())

        print('The BERT model has {:} different named parameters.\n'.format(len(params)))

        print('==== Embedding Layer ====\n')

        for p in params[0:5]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        print('\n==== First Transformer ====\n')

        for p in params[5:21]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        print('\n==== Output Layer ====\n')

        for p in params[-4:]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
