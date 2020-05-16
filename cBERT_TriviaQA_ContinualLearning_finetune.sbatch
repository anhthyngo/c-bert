#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --partition=v100_sxm2_4,p40_4,p100_4,k80_4
#SBATCH --mem=64000
#SBATCH --time=24:00:00
#SBATCH --job-name="cbert_triviaqa_cont_finetune"
#SBATCH --output=cbert_triviaqa_cont_finetune.out

module purge
module load anaconda3/5.3.1
module load cuda/10.0.130
module load gcc/6.3.0

source activate cbert

# Set project working directory
PROJECT=<fill repository path>

# Set arguments
STUDY_NAME=cbert_triviaqa_cont_finetune   				# name of experiment
N_TRIALS=10000                     						# number of fine tuning steps
LOGGING_STEPS=1000                 						# number of steps between logging steps for ex$
SAVE_STEPS=1000                    						# number of steps to test for best weights
VERBOSE_STEPS=100                 						# number of steps to record results to run log
SAVE_DIR=${PROJECT}/results	  							# directory for results
DATA_DIR=${PROJECT}/data          						# directory for data
MODEL=bert-base-uncased           						# name of model from Huggingface
BATCH=24                          						# batch-size, will be split over number of GPUs
SEED=42                           						# seed for experiment, Huggingface default is 42
MAX_SEQ=384                       						# maximum sequence length for input
DOC_STRIDE=128                    						# stride between windows for Huggingface slidin$
LR=0.00005                      						# learning rate
WARMUP=1000												# warm-up for 10% of updates
RLN_WEIGHTS=${PROJECT}/results/meta_1bs_meta_weights.pt	# path to meta weights

cd ${PROJECT}
python ./code/main.py \
    --experiment ${STUDY_NAME} \
    --fine_tune_steps ${N_TRIALS} \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --verbose_steps ${VERBOSE_STEPS} \
    --save_dir ${SAVE_DIR} \
    --data_dir ${DATA_DIR} \
    --model ${MODEL} \
    --batch_size ${BATCH} \
    --seed ${SEED} \
    --learning_rate ${LR} \
    --max_seq_length ${MAX_SEQ} \
    --doc_stride ${DOC_STRIDE} \
    --no_prev_fine_tune \
    --do_lower_case \
    --load_rln \
    --warmup_steps ${WARMUP} \
    --rln_weights ${RLN_WEIGHTS}