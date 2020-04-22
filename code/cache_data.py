import myio
import transformers
import os
from datetime import datetime as dt
import logging as log
import time
import argparse

def main(parser):
    if True:
        tasks = ['SQuAD', 'TriviaQA-web', 'SearchQA', 'NewsQA', 'NaturalQuestionsShort', 'HotpotQA']
        
        
        task = tasks[parser.idx]
        
        
        model = 'bert-base-uncased'
        do_lower_case = True
        
        data_dir = parser.data_dir
        cache_dir = parser.cache_dir
        log_dir = parser.log_dir
        tokenizer = transformers.AutoTokenizer.from_pretrained(model,
                                                               do_lower_case=do_lower_case)
        
        log_name = os.path.join(log_dir, '{}_{}_cache_data_{}.log'.format(
            task,
            model,
            dt.now().strftime("%Y%m%d_%H%M")
            )
        )
        log.basicConfig(filename=log_name,
                        format='%(asctime)s | %(name)s -- %(message)s',
                        level=log.DEBUG)
        
        
        max_seq_length = 384
        doc_stride = 128
        max_query_length = 64    
        
        IO = myio.IO(data_dir,
                     cache_dir,
                     tokenizer,
                     max_seq_length,
                     doc_stride,
                     max_query_length,
                     batch_size=32,
                     shuffle=True,
                     cache=True
                     )
        
        start = time.time()
        
        log.info("="*40 + " Loading {} {} ".format(task,'train') + "="*40)
        _, _, _ = IO.load_and_cache_task(task, 'train')
        
        log.info("="*40 + " Loading {} {} ".format(task,'dev') + "="*40)
        _, _, _ = IO.load_and_cache_task(task, 'dev')
        
        log.info("Task {} took {:.6f}s".format(task, time.time()-start))
        
        # release logs from Python
        handlers = log.getLogger().handlers
        for handler in handlers:
            handler.close()
            

if __name__=="__main__":
    args = argparse.ArgumentParser(description='Cache Data')
    args.add_argument('--idx', type=int)
    args.add_argument('--data_dir', type=str, default=r'C:\Users\Willi\Documents\NYU\2020 Spring\NLU\Project\Projects\MetaLearning\Data_Exploration\data')
    args.add_argument('--cache_dir', type=str,default= r'C:\Users\Willi\Documents\NYU\2020 Spring\NLU\Project\Projects\MetaLearning\Data_Exploration\cached_data')
    args.add_argument('--log_dir', type=str, default = r'C:\Users\Willi\Documents\NYU\2020 Spring\NLU\Project\Projects\MetaLearning\Data_Exploration\cache_log')
    parser = args.parse_args()
    main(parser)