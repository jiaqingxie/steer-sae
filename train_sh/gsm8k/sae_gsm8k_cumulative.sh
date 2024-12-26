#!/bin/bash

#SBATCH --output=/cluster/project/sachan/jiaxie/results/2b_cumulative.out
#SBATCH --error=/cluster/project/sachan/jiaxie/results/2b_cumulative.err
#SBATCH --mem-per-cpu=20G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=3:00:00

module load eth_proxy
export TRANSFORMERS_CACHE=/cluster/scratch/jiaxie/.cache
export TRITON_CACHE_DIR=/cluster/scratch/jiaxie/.triton_cache

cd /cluster/scratch/jiaxie/
source sae/bin/activate

cd /cluster/project/sachan/jiaxie/SAE_Math


#Settings
MODEL_NAME_OR_PATH="google/gemma-2-2b"
DATA_ROOT="/cluster/project/sachan/jiaxie/SAE_Math/data"
CACHE_DIR="/cluster/scratch/jiaxie/models/google/gemma-2-2b"
LAYER_IDX=20
PLOT_NUM=5
K=10
TYPE="sae"
SAE_FILE="gemma-scope-2b-pt-res-canonical"
SAE_ID="20-gemmascope-res-16k"
TRANSFORMER_LENS=True
DATASET="gsm8k_train"
NUM_SAE=1000
N_SHOT=8

python -u train/sae.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_root ${DATA_ROOT} \
    --cache_dir ${CACHE_DIR} \
    --layer_idx ${LAYER_IDX} \
    --plot_num ${PLOT_NUM} \
    --K ${K} \
    --type ${TYPE} \
    --sae_file ${SAE_FILE} \
    --transformer_lens \
    --sae_id ${SAE_ID} \
    --dataset ${DATASET} \
    --NUM_SAE ${NUM_SAE} \
    --n_shot ${N_SHOT} \
    --cumulative \
