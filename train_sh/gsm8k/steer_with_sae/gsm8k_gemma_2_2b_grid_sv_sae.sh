#!/bin/bash

#SBATCH --output=/cluster/project/sachan/jiaxie/results/%j.out
#SBATCH --error=/cluster/project/sachan/jiaxie/results/%j.err
#SBATCH --mem-per-cpu=20G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=3:00:00

module load eth_proxy
#export TRANSFORMERS_CACHE=/cluster/scratch/jiaxie/.cache
export TRITON_CACHE_DIR=/cluster/scratch/jiaxie/.triton_cache
export TRANSFORMERS_OFFLINE=1

cd /cluster/scratch/jiaxie/
source sae/bin/activate

cd /cluster/project/sachan/jiaxie/SAE_Math

#Settings alphabetically
CACHE_DIR="/cluster/scratch/jiaxie/models/google/gemma-2-2b"
COEFF=1000
DATA_ROOT="/cluster/project/sachan/jiaxie/SAE_Math/data"
K=10
LAYER_IDX=20
MODEL_NAME_OR_PATH="google/gemma-2-2b"
PARAM_FILE="layer_20/width_16k/average_l0_71/params.npz"
PLOT_NUM=5
SAE_FILE="google/gemma-scope-2b-pt-res"
SAE_ID="20-gemmascope-res-16k"
SAE_IDX=15153
TRANSFORMER_LENS=True
TYPE="inference"

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
    --sae_idx ${SAE_IDX} \
    --coeff ${COEFF} \
