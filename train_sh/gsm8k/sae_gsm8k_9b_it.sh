#!/bin/bash

#SBATCH --output=/cluster/project/sachan/jiaxie/results/%j.out
#SBATCH --error=/cluster/project/sachan/jiaxie/results/%j.err
#SBATCH --mem-per-cpu=20G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_3090:2
#SBATCH --time=3:00:00

module load eth_proxy
#export TRANSFORMERS_CACHE=/cluster/scratch/jiaxie/.cache
export TRITON_CACHE_DIR=/cluster/scratch/jiaxie/.triton_cache


cd /cluster/scratch/jiaxie/
source sae/bin/activate

cd /cluster/project/sachan/jiaxie/SAE_Math


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#Settings
MODEL_NAME_OR_PATH="google/gemma-2-9b-it"
DATA_ROOT="/cluster/project/sachan/jiaxie/SAE_Math/data"
CACHE_DIR="/cluster/scratch/jiaxie/models/google/gemma-2-9b-it"
LAYER_IDX=31
PLOT_NUM=5
N_DEVICES=2
K=10
TYPE="sae"
SAE_FILE="google/gemma-scope-9b-it-res"
SAE_ID="31-gemmascope-res-16k"
PARAM_FILE="layer_31/width_16k/average_l0_76/params.npz"
TRANSFORMER_LENS=True

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
    --devices ${N_DEVICES} \
    --bfloat16 \
