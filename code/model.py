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
import transformers
import os


# ============================ Set up a QA Model from Huggingface ============================

class QAModel(nn.Module):
    def __init__(self,
                 config,              # Huggingface configuration object for model
                 load_rln = False,    # bool whether to load preloaded rln
                 rln_weights = None   # name of state dict with rln weights
                 ):
        
        """
        Define the architecture of the model

        set config = BertConfig.from_pretrained('bert-base-uncased')

        config = transformers.AutoConfig.from_pretrained(rep_name)
        """
        
        super(QAModel, self).__init__()
        self.num_labels = config.num_labels

        # Define model
        self.model = transformers.AutoModelForQuestionAnswering.from_config(config)
        
        # load meta learned weights 
        # only supporting BERT for now
        if load_rln:
            assert os.path.exist(rln_weights), "Path for meta-learned weights does not exist"
            self.model.bert.load_state_dict(torch.load(rln_weights))

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
        """
        Forward used to feed Huggingface model.
        
        Returns output as specified below

        """
        outputs = self.model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            position_ids = position_ids,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
            start_positions = start_positions,
            end_positions = end_positions
            )
    
        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)

    def print_param(self):

        model = self.model
        params = list(model.named_parameters())

        print('The BERT model has {:} different named parameters.\n'.format(len(params)))

        print('==== Embedding Layer ====\n')
        
        # might need to base indexing off self.model.config
        for p in params[0:5]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        print('\n==== First Transformer ====\n')
        
        # might need to base indexing off self.model.config
        for p in params[5:21]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        print('\n==== Output Layer ====\n')
        
        # might need to base indexing off self.model.config
        for p in params[-4:]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
