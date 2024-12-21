#!/bin/bash

#SBATCH --output=/cluster/project/sachan/jiaxie/results/sae_9b_mawps_0shot_C200_T1_omega1_12946.out
#SBATCH --error=/cluster/project/sachan/jiaxie/results/sae_9b_mawps_0shot_C200_T1_omega1_12946.err
#SBATCH --mem-per-cpu=20G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_3090:2
#SBATCH --time=08:00:00

module load eth_proxy
export TRANSFORMERS_CACHE=/cluster/scratch/jiaxie/.cache
export TRITON_CACHE_DIR=/cluster/scratch/jiaxie/.triton_cache


cd /cluster/scratch/jiaxie/
source sae/bin/activate

cd /cluster/project/sachan/jiaxie/SAE_Math

#Settings alphabetically
CACHE_DIR="/cluster/scratch/jiaxie/models/google/gemma-2-9b"
COEFF=(200)
DATA_ROOT="/cluster/project/sachan/jiaxie/SAE_Math/data"
K=10
LAYER_IDX=31
MODEL_NAME_OR_PATH="google/gemma-2-9b"
PARAM_FILE="layer_31/width_16k/average_l0_63/params.npz"
PLOT_NUM=5
SAE_FILE="gemma-scope-9b-pt-res-canonical"
SAE_ID="31-gemmascope-res-16k"
SAE_IDX=(12946)
TRANSFORMER_LENS=True
TYPE="inference"
N_SHOT=0
T=1
OMEGA=1
DATASET="mawps"

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
    --bfloat16 \
    --omega ${OMEGA} \
    --dataset  ${DATASET} \
