#!/bin/bash

#SBATCH --output=/cluster/project/sachan/jiaxie/results/mean_diff_act_2b.out
#SBATCH --error=/cluster/project/sachan/jiaxie/results/mean_diff_act_2b.err
#SBATCH --mem-per-cpu=20G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=3:00:00

module load eth_proxy
export TRANSFORMERS_CACHE=/cluster/scratch/jiaxie/.cache
export TRITON_CACHE_DIR=/cluster/scratch/jiaxie/.triton_cache

#export TRANSFORMERS_OFFLINE=1
cd /cluster/scratch/jiaxie/
source sae/bin/activate

cd /cluster/project/sachan/jiaxie/SAE_Math


#Settings
MODEL_NAME_OR_PATH="google/gemma-2-2b"
DATA_ROOT="/cluster/project/sachan/jiaxie/SAE_Math/data"
CACHE_DIR="/cluster/scratch/jiaxie/models/google/gemma-2-2b"
LAYER_IDX=21
PLOT_NUM=5
K=10
TYPE="inference"
TRANSFORMER_LENS=True
DATASET="gsm8k_train"
SEED=0
steer_vec_base_directory="/cluster/project/sachan/jiaxie/SAE_Math/mean_vec"
N_SHOTS=8

python -u train/sae.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_root ${DATA_ROOT} \
    --cache_dir ${CACHE_DIR} \
    --layer_idx ${LAYER_IDX} \
    --plot_num ${PLOT_NUM} \
    --K ${K} \
    --type ${TYPE} \
    --transformer_lens \
    --seed ${SEED} \
    --dataset ${DATASET} \
    --steer_vec_base_directory ${steer_vec_base_directory} \
    --calculate_mean_diff \
    --steer_vec_baseline \
    --n_shot ${N_SHOTS} \
