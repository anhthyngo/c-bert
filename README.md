# Meta-BERT: Learning Robust Contextual Representations for Continual Learning

<img src="https://github.com/anhthyngo/meta-bert/blob/master/img/bert-img.jpeg " width="900">



# Setup

0. Clone repository

   `git clone https://github.com/anhthyngo/c-bert.git`

1. In cloned repository, create anaconda environment `cbert` from `environment.yml`

   `conda env create -f environment.yml`

2. In directory storing repository setup directories for

   a. `data`

   b. `log`

   c. `results` (For cached data, place in `results/cached_data/<model-name>/`)

3. Submit jobs using `.sbatch` example in `./examples/Run_Script/`

4. Outputs will be found in `results` in the following sub-directories

   a. `cached_data`                                            -  cached data as `.pt` files

   b. `json_results`                                          -  F1 scores for plotting in `.json` files

   c. `logged/<model-name>/<task-name>`   -  Model state dictionaries as `.pt` files

   d. `plots`                                                         -  Plots of results in `.png` files

