#!/bin/bash

#SBATCH --output=/cluster/project/sachan/jiaxie/results/IFEval_it_mean_diff_act_9b_it.out
#SBATCH --error=/cluster/project/sachan/jiaxie/results/IFEval_it_mean_diff_act_9b_it.err
#SBATCH --mem-per-cpu=20G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_3090:2
#SBATCH --time=3:00:00

module load eth_proxy
export HF_HOME=/cluster/scratch/jiaxie/.cache/huggingface
export TRANSFORMERS_CACHE=/cluster/scratch/jiaxie/.cache
export TRITON_CACHE_DIR=/cluster/scratch/jiaxie/.triton_cache


cd /cluster/scratch/jiaxie/
source sae/bin/activate

cd /cluster/project/sachan/jiaxie/SAE_Math
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


#Settings
MODEL_NAME_OR_PATH="google/gemma-2-9b-it"
DATA_ROOT="/cluster/project/sachan/jiaxie/SAE_Math/instruct_data"
CACHE_DIR="/cluster/scratch/jiaxie/models/google/gemma-2-9b-it"
LAYER_IDX=31
N_DEVICES=2
PLOT_NUM=5
K=10
TYPE="inference"
TRANSFORMER_LENS=True
DATASET="all_base_x_all_instructions_filtered"
steer_vec_base_directory="/cluster/project/sachan/jiaxie/SAE_Math/mean_vec"
NUM_SAE=500
INSTRUCT_TYPE="response_language_it"
MODE="train"
TRAIN_SIZE=0.3
TEST_SIZE=0.5
VALID_SIZE=0.2


python -u train/sae_instruct_follow.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_root ${DATA_ROOT} \
    --cache_dir ${CACHE_DIR} \
    --layer_idx ${LAYER_IDX} \
    --plot_num ${PLOT_NUM} \
    --K ${K} \
    --type ${TYPE} \
    --transformer_lens \
    --dataset ${DATASET} \
    --steer_vec_base_directory ${steer_vec_base_directory} \
    --calculate_mean_diff \
    --steer_vec_baseline \
    --NUM_SAE ${NUM_SAE} \
    --devices ${N_DEVICES} \
    --bfloat16 \
    --instruct_type ${INSTRUCT_TYPE} \
    --mode ${MODE} \
    --train_size ${TRAIN_SIZE} \
    --test_size ${TEST_SIZE} \
    --valid_size ${VALID_SIZE} \
