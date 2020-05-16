# cBERT: A Meta-Learning Approach to Contextual Representations for Continual Learning

<img src="https://github.com/anhthyngo/meta-bert/blob/master/img/bert-img.jpeg " width="900">



Instructions for running on New York University's Prince computer cluster.



# Setup

1. Clone repository

   `git clone https://github.com/wh629/c-bert.git`

2. Perform the following commands

   `module purge`
   `module load anaconda3/5.3.1`
   `module load cuda/10.0.130`
   `module load gcc/6.3.0`

3. In cloned repository, create anaconda environment `cbert` from `environment.yml`

   `conda env create -f environment.yml`

4. In repository, setup directories for

   a. `data`

   b. `log`

   c. `results` (For cached data, place in `results/cached_data/<model-name>/`)

5. Load [data](https://drive.google.com/drive/folders/1fKtUxuZPddzKxdvuaarao0JN34kQbtMl?usp=sharing) into `data`

6. For faster runs, load [cached data](https://drive.google.com/drive/folders/1-4Kvwl6nurE3AiEm__oG8Av_C6E_L2pN?usp=sharing) into `results/cached_data/bert-base-uncased/` folder

7. Load [meta weights](https://drive.google.com/drive/folders/1DnsyJY4WEOGKb4Te3QW4E3xEyPCflj-z?usp=sharing) into `results/meta_weights/`

# Train BERT on SQuAD

1. Train on SQuAD either using frozen embeddings or fine-tuning

   a. Fill out `PROJECT=<Repository Directory>` in desired `.sbatch` file

   * For frozen, use `sbatch baseline_SQuAD_frozen.sbatch`

   * For fine-tuning, use `sbatch baseline_SQuAD_finetune.sbatch`

2. Outputs will be found in `results` in the following sub-directories

   a. `cached_data`                                            -  cached data as `.pt` files

   b. `logged/<model-name>/<task-name>`   -  Model state dictionaries as `.pt` files

3. Monitor run using `log/baseline_SQuAD_<frozen/finetune>_run_log_<date>_<time>.log`

# Train BERT on TriviaQA and Evaluate Continual Learning

1. Train on TriviaQA either using frozen embeddings or fine-tuning and Evaluate Continual Learning

   a. Fill out `PROJECT=<Repository Directory>` in desired `.sbatch` file

   * For frozen, use `sbatch baseline_TriviaQA_ContinualLearning_frozen.sbatch`

   * For fine-tuning, use `sbatch baseline_TriviaQA_ContinualLearning_finetune.sbatch`

2. Outputs will be found in `results` in the following sub-directories

   a. `cached_data`                                            -  cached data as `.pt` files

   b. `json_results`                                          -  F1 scores for plotting in `.json` files

   c. `logged/<model-name>/<task-name>`   -  Model state dictionaries as `.pt` files

   d. `plots`                                                         -  Plots of results in `.png` files

3. Monitor run using `log/baseline_TriviaQA_ContinualLearning_<frozen/finetune>_run_log_<date>_<time>.log`

# Perform Meta-Learning

1. Perform meta-learning with `sbatch Meta.sbatch`

2. Meta-learned weights can be found in `results/meta_weights/meta_meta_weights.pt`
3. Monitor run using `log/meta_meta_run_log_<date>_<time>.log`

# Train cBERT on SQuAD

1. Train on SQuAD either using frozen embeddings or fine-tuning

   a. Fill out `PROJECT=<Repository Directory>` in desired `.sbatch` file

   * For frozen, use `sbatch cBERT_SQuAD_frozen.sbatch`

   * For fine-tuning, use `sbatch cBERT_SQuAD_finetune.sbatch`

2. Outputs will be found in `results` in the following sub-directories

   a. `cached_data`                                            -  cached data as `.pt` files

   b. `logged/<model-name>/<task-name>`   -  Model state dictionaries as `.pt` files

3. Monitor run using `log/cbert_SQuAD_<frozen/finetune>_run_log_<date>_<time>.log`

# Train cBERT on TriviaQA and Evaluate Continual Learning

1. Train on TriviaQA either using frozen embeddings or fine-tuning and Evaluate Continual Learning

   a. Fill out `PROJECT=<Repository Directory>` in desired `.sbatch` file

   * For frozen, use `sbatch cBERT_TriviaQA_ContinualLearning_frozen.sbatch`

   * For fine-tuning, use `sbatch cBERT_TriviaQA_ContinualLearning_finetune.sbatch`

2. Outputs will be found in `results` in the following sub-directories

   a. `cached_data`                                            -  cached data as `.pt` files

   b. `json_results`                                          -  F1 scores for plotting in `.json` files

   c. `logged/<model-name>/<task-name>`   -  Model state dictionaries as `.pt` files

   d. `plots`                                                         -  Plots of results in `.png` files

3. Monitor run using `log/cbert_TriviaQA_ContinualLearning_<frozen/finetune>_run_log_<date>_<time>.log`
