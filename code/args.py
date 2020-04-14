"""
Module for argument parcer.

Many of the arguments are from Huggingface's run_squad example:
https://github.com/huggingface/transformers/blob/7972a4019f4bc9f85fd358f42249b90f9cd27c68/examples/run_squad.py
"""
import argparse
import os

args = argparse.ArgumentParser(description='Continual BERT')



args.add_argument('--experiment',
                  type=str,
                  default='testing',
                  help='name of experiment')
args.add_argument('--save_dir',
                  type=str, 
                  default='results',
                  help='directory to save results')
args.add_argument('--seed',
                  type=int,
                  default=1111,
                  help='random seed')
args.add_argument('--run_log',
                  type=str,
                  default=os.path.join(os.getcwd(),'log'),
                  help='where to print run log')
args.add_argument('--meta_tasks',
                  type=str,
                  default='NewsQA,SearchQA,HotpotQA,NaturalQuestionsShort',
                  help='tasks for meta learning, separated by ,')
args.add_argument('--continual_curriculum',
                  type=str,
                  default='SQuAD,TriviaQA-web',
                  help='tasks in order for continual learning, separated by ,')

# =============================================================================
# for model definition
# =============================================================================
args.add_argument('--model',
                  type=str,
                  default='bert-base-uncased',
                  help='name of RLN network. default is bert-base-uncased',
                  choices={'bert-base-uncased',
                           'bert-base-cased',
                           'bert-large-uncased',
                           'bert-large-cased'})

# =============================================================================
# for dataloading
# =============================================================================
args.add_argument('--local_rank', 
                  type=int, 
                  default=-1,
                  help='local_rank for distributed training on gpus')
args.add_argument('--overwrite_cache', 
                  action='store_true',
                  help='overwrite the cached data sets')
args.add_argument('--max_seq_length',
                  type=int,
                  default=384,
                  help='maximum total input sequence length after tokenization.'
                  'longer sequences truncated. shorter sequences padded.')
args.add_argument('--doc_stride',
                  type=int,
                  default=128,
                  help='when chunking, how much stride between chunks')
args.add_argument('--max_query_length',
                  type=int,
                  default=64,
                  help='maximum number of tokens in a question. longer questions'
                  'will be truncated.')
args.add_argument('--threads',
                  type=int,
                  default=1,
                  help='multiple threads for converting example to features')
args.add_argument('--data_dir',
                  type=str,
                  default='data',
                  help='directory storing all data')
args.add_argument('--batch_size', 
                  type=int, 
                  default=8,
                  help='batch size')

# =============================================================================
# for training
# =============================================================================
args.add_argument('--fp16',
                  action='store_true',
                  help='whether to use 16-bit precision (through NVIDIA apex)')
args.add_argument('--meta_steps',
                  type=int,
                  default=100,
                  help='number of updates for meta-learning')
args.add_argument('--fine_tune_steps', 
                  type=int, 
                  default=100000,
                  help='number of updates for fine-tuning')
args.add_argument('--learning_rate', 
                  type=float, 
                  default=5e-3,
                  help='initial learning rate for Adam')
args.add_argument("--weight_decay",
                  type=float,
                  default=0.0,
                  help='weight decay if applied')
args.add_argument('--adam_epsilon',
                  type=float,
                  default=1e-8,
                  help='epsilon for Adam optimizer')
args.add_argument('--max_grad_norm',
                  type=float,
                  default=1.0,
                  help='max gradient norm for clipping')
args.add_argument('--warmup_steps',
                  type=int,
                  default=0,
                  help='linear warmup over warmup steps')
args.add_argument('--n_best_size',
                  type=int,
                  default=20,
                  help='total number of n-best predictions to generate')
args.add_argument('--max_answer_length',
                  type=int,
                  default=30,
                  help='max length of answer. needed because start'
                  'and end not conditioned on eachother')
args.add_argument('--logging_steps',
                  type=int,
                  default=1e4,
                  help='saves best weights every X update steps')
args.add_argument('--save_steps',
                  type=int,
                  default=500,
                  help='Save best weights every X update steps')
args.add_argument('--verbose_steps',
                  type=int,
                  default=1000,
                  help='Log results ever X update steps')
args.add_argument('--verbose_logging',
                  action='store_true',
                  help='whether to store verbose logging in evaluation')
args.add_argument("--null_score_diff_threshold",
                  type=float,
                  default=0.0,
                  help="If null_score - best_non_null is greater than the threshold predict null.")
args.add_argument("--do_lower_case",
                  action="store_true",
                  help="Set this flag if you are using an uncased model.")
args.add_argument('--version_2_with_negative',
                  action='store_true',
                  help='whether negative examples exist like in SQuADv2')


def check_args(parser):
    """
    make sure directories exist
    """
    assert os.path.exists(parser.data_dir), "Data directory does not exist"
    assert os.path.exists(parser.save_dir), "Save directory does not exist"
    assert os.path.exists(parser.run_log),  "Run logging directory does not exist"
    
    assert parser.version_2_with_negative == False, "Only supports version 1 without negatives"
    assert (parser.do_lower_case and parser.model.find('uncased') != -1) or (not parser.do_lower_case and parser.model.find('uncased') == -1), "do_lower_case associated with uncased model"