"""
Module for argument parcer
"""
import argparse
import os
import logging as log

args = argparse.ArgumentParser(description='Continual BERT')

args.add_argument('--data', type=str, default='/data',
                  help='directory storing all data')
args.add_argument('--save_dir', type=str, default='/results',
                    help='directory to save results')
args.add_argument('--meta_epochs', type=int, default=100,
                  help='number of epochs for meta-learning')
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