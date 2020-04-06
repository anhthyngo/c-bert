"""
Module with class io containing methods for importing and exporting data
"""

import os
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import gzip
import json
import logging
logger = logging.getLogger(__name__)

class io:
    """
    Object to store:
        Import/Output Methods
        Task Data in dictionary 'tasks' as DataLoader objects
    """
    def __init__(self,
                 data_dir, # String of the directory storing all tasks
                 task_dir  # Array of task directories, should match 'tasks' keys
                 ):
        data_dir = data_dir
        task_dir = task_dir    
        tasks = {
            "squad"             : 1,
            "triviaqa"          : 2,
            "newsqa"            : 3,
            "searchqa"          : 4,
            "hotpotqa"          : 5,
            "naturalquestions"  : 6
            }

        
    def read_mrqa_examples(input_file, is_training):
        
    """
    Adapted from: https://github.com/facebookresearch/SpanBERT/
    Read a MRQA json file into a list of MRQAExample.
    Returns nested list.
    NOTES: still trying to figure out dataloader.
    """
    with gzip.GzipFile(input_file, 'r') as reader:
        # skip header
        content = reader.read().decode('utf-8').strip().split('\n')[1:]
        input_data = [json.loads(line) for line in content]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    num_answers = 0
    for i, entry in enumerate(input_data):
        if i % 1000 == 0:
            logger.info("Processing %d / %d.." % (i, len(input_data)))
        paragraph_text = entry["context"]
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in paragraph_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        for qa in entry["qas"]:
            qas_id = qa["qid"]
            question_text = qa["question"]
            start_position = None
            end_position = None
            orig_answer_text = None
            if is_training:
                answers = qa["detected_answers"]
                # import ipdb
                # ipdb.set_trace()
                spans = sorted([span for spans in answers for span in spans['char_spans']])
                # take first span
                char_start, char_end = spans[0][0], spans[0][1]
                orig_answer_text = paragraph_text[char_start:char_end+1]
                start_position, end_position = char_to_word_offset[char_start], char_to_word_offset[char_end]
                num_answers += sum([len(spans['char_spans']) for spans in answers])
            '''
                example = MRQAExample(
                qas_id=qas_id,
                question_text=question_text,
                doc_tokens=doc_tokens,
                orig_answer_text=orig_answer_text,
                start_position=start_position,
                end_position=end_position)
            '''
            
            examples.append([qas_id,question_text,doc_tokens,orig_answer_text,start_position,end_position])
            #examples.append([example])
    logger.info('Num avg answers: {}'.format(num_answers / len(examples)))
    return examples

    def import_data(self):
        """
        Import data, preprocess, and store as DataLoader objects in
        'self.tasks' dictionary
        
        Can reference MRQA script
        """
        
        for task in self.task_dir:
            #[Implement load data]
            
            temp = DataLoader(...)
            self.tasks[task] = temp

    def export_results(self):
        """
        Export results of analysis
        """
