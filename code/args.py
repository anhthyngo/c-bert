"""
Module for argument parcer
"""
import argparse
import os
import logging as log

args = argparse.ArgumentParser(description='Continual BERT')




args.add_argument('--save_dir', type=str, default='/results',
                    help='directory to save results')
args.add_argument('--meta_epochs', type=int, default=100,
                  help='number of epochs for meta-learning')
args.add_argument('--fine_tune_epochs', type=int, default=100000,
                  help='number of epochs for fine-tuning')
args.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
args.add_argument('--batch_size', type=int, default=20,
                    help='batch size')
#args.add_argument('--dropout', type=float, default=0.2,
#                    help='dropout applied to layers (0 = no dropout)')
args.add_argument('--seed', type=int, default=1111,
                    help='random seed')
args.add_argument('--model', type=str, default='BERT',
                    help='name of RLN network. default is BERT')

# used in huggingface run_squad example
args.add_argument('--local_rank', ) # FINISH
args.add_argument('--overwrite_cache', ) # FINISH
args.add_argument('--predict_file', ) # FINISH
args.add_argument('--train_file', ) # FINISH
args.add_argument('--version_2_with_negative', type=bool, default=False,
                  help='whether negative examples exist like in SQuADv2')
args.add_argument('--max_seq_length', ) # FINISH
args.add_argument('--doc_stride', ) # FINISH
args.add_argument('--max_query_length', ) # FINISH
args.add_argument('--threads', ) # FINISH
args.add_argument('--local_rank', type=int, default=-1) # FINISH


args.add_argument('--data_dir', type=str, default='data',
                  help='directory storing all data')