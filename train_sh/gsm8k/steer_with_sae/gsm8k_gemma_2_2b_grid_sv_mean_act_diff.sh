#!/bin/bash

#SBATCH --output=/cluster/project/sachan/jiaxie/results/gsm8k_0shot_C5_T2_omega1_mean_act_vec.out
#SBATCH --error=/cluster/project/sachan/jiaxie/results/gsm8k_0shot_C5_T2_omega1_mean_act_vec.err
#SBATCH --mem-per-cpu=20G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=10:00:00

module load eth_proxy
export TRANSFORMERS_CACHE=/cluster/scratch/jiaxie/.cache
export TRITON_CACHE_DIR=/cluster/scratch/jiaxie/.triton_cache


cd /cluster/scratch/jiaxie/
source sae/bin/activate

cd /cluster/project/sachan/jiaxie/SAE_Math

#Settings alphabetically
CACHE_DIR="/cluster/scratch/jiaxie/models/google/gemma-2-2b"
COEFF=(5 900)
DATA_ROOT="/cluster/project/sachan/jiaxie/SAE_Math/data"
K=10
LAYER_IDX=20
MODEL_NAME_OR_PATH="google/gemma-2-2b"
PARAM_FILE="layer_20/width_16k/average_l0_71/params.npz"
PLOT_NUM=5
SAE_FILE="gemma-scope-2b-pt-res-canonical"
SAE_ID="20-gemmascope-res-16k"
SAE_IDX=(15153)
TRANSFORMER_LENS=True
TYPE="inference"
N_SHOT=0
T=2
OMEGA=1
STEERING_TYPE="mean_act_diff"
steer_vec_base_directory="/cluster/project/sachan/jiaxie/SAE_Math/mean_vec"

python -u train/sae.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_root ${DATA_ROOT} \
    --cache_dir ${CACHE_DIR} \
    --layer_idx ${LAYER_IDX} \
    --plot_num ${PLOT_NUM} \
    --K ${K} \
    --type ${TYPE} \
    --sae_file ${SAE_FILE} \
    --param_file ${PARAM_FILE} \
    --transformer_lens \
    --sae_id ${SAE_ID} \
    --grid_search \
    --steer_vec_sae \
    --n_shot ${N_SHOT}\
    --sae_idx ${SAE_IDX[@]} \
    --coeff ${COEFF[@]} \
    --T ${T} \
    --omega ${OMEGA} \
    --steering_type ${STEERING_TYPE} \
    --steer_vec_base_directory ${steer_vec_base_directory} \
