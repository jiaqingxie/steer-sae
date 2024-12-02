#!/bin/bash

#SBATCH --output=/cluster/project/sachan/jiaxie/results/%j.out
#SBATCH --error=/cluster/project/sachan/jiaxie/results/%j.err
#SBATCH --mem-per-cpu=20G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_3090:2
#SBATCH --time=5:00:00

module load eth_proxy
export TRANSFORMERS_CACHE=/cluster/scratch/jiaxie/.cache
export TRITON_CACHE_DIR=/cluster/scratch/jiaxie/.triton_cache

cd /cluster/scratch/jiaxie/
source sae/bin/activate

cd /cluster/project/sachan/jiaxie/SAE_Math

#Settings alphabetically
CACHE_DIR="/cluster/scratch/jiaxie/models/google/gemma-2-9b"
COEFF=800
DATA_ROOT="/cluster/project/sachan/jiaxie/SAE_Math/data"
K=10

LAYER_IDX=31
N_DEVICES=2
MODEL_NAME_OR_PATH="google/gemma-2-9b"
PARAM_FILE="layer_31/width_16k/average_l0_63/params.npz"
PLOT_NUM=5
SAE_FILE="gemma-scope-9b-pt-res"
SAE_ID="31-gemmascope-res-16k"
SAE_IDX=7137
TRANSFORMER_LENS=True
TYPE="inference"

python train/sae.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_root ${DATA_ROOT} \
    --cache_dir ${CACHE_DIR} \
    --layer_idx ${LAYER_IDX} \
    --plot_num ${PLOT_NUM} \
    --K ${K} \
    --type ${TYPE} \
    --devices ${N_DEVICES} \
    --sae_file ${SAE_FILE} \
    --param_file ${PARAM_FILE} \
    --transformer_lens \
    --sae_id ${SAE_ID} \
    --grid_search \
    --steer_vec_sae \
    --sae_idx ${SAE_IDX} \
    --coeff ${COEFF} \
