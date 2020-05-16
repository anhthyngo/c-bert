#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=v100_sxm2_4,p40_4,p100_4
#SBATCH --mem=64000
#SBATCH --time=48:00:00
#SBATCH --job-name="meta"
#SBATCH --output=meta.out

module purge
module load anaconda3/5.3.1
module load cuda/10.0.130
module load gcc/6.3.0

source activate cbert

# Set project working directory
PROJECT=<fill repository path>

# Set arguments
STUDY_NAME=meta                  						# name of experiment
SAVE_DIR=${PROJECT}/results       						# directory for results
DATA_DIR=${PROJECT}/data          						# directory for data
MODEL=bert-base-uncased           						# name of model from Huggingface
BATCH=1                           						# batch-size, will be split over number of GPUs
SEED=42                           						# seed for experiment, Huggingface default is 42
MAX_SEQ=384                       						# maximum sequence length for input
DOC_STRIDE=128                    						# stride between windows for Huggingface sliding window
IN_LR=0.003                       						# inner learning rate
OUT_LR=0.00003                    						# meta learning rate
STEPS=5000                        						# number of meta steps
N_META_TASKS=2                    						# number of tasks for per update
N_META_SAMPLES=15                 						# number of samples per task per inner update
META_TASKS=NewsQA,HotpotQA,NaturalQuestionsShort     	# tasks to sample from

cd ${PROJECT}
python ./code/meta_learning.py \
	--experiment ${STUDY_NAME} \
	--verbose_steps ${VERBOSE} \
	--save_dir ${SAVE_DIR} \
	--data_dir ${DATA_DIR} \
	--model ${MODEL} \
	--batch_size ${BATCH} \
	--seed ${SEED} \
	--max_seq_length ${MAX_SEQ} \
	--doc_stride ${DOC_STRIDE} \
	--meta_update_lr ${IN_LR} \
	--meta_meta_lr ${OUT_LR} \
	--meta_steps ${STEPS} \
	--n_meta_tasks ${N_META_TASKS} \
	--n_meta_task_samples ${N_META_SAMPLES} \
	--freeze_embeddings \
	--do_lower_case \
	--meta_tasks ${META_TASKS}

	
