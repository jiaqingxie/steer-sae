#!/bin/bash

#SBATCH --output=/cluster/project/sachan/jiaxie/results/sae_9b_svamp_inference_to.out
#SBATCH --error=/cluster/project/sachan/jiaxie/results/sae_9b_svamp_inference_to.err
#SBATCH --mem-per-cpu=20G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=3:00:00

module load eth_proxy
export HF_HOME=/cluster/scratch/jiaxie/.cache/huggingface
export TRANSFORMERS_CACHE=/cluster/scratch/jiaxie/.cache
export TRITON_CACHE_DIR=/cluster/scratch/jiaxie/.triton_cache


cd /cluster/scratch/jiaxie/
source sae/bin/activate

cd /cluster/project/sachan/jiaxie/SAE_Math

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#Settings alphabetically
CACHE_DIR="/cluster/scratch/jiaxie/models/google/gemma-2-9b"
DATA_ROOT="/cluster/project/sachan/jiaxie/SAE_Math/data"
MODEL_NAME_OR_PATH="google/gemma-2-9b"
TYPE="inference"
N_SHOT=0
DATASET="svamp"
SAE_WORD="to"

python -u train/sae.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_root ${DATA_ROOT} \
    --cache_dir ${CACHE_DIR} \
    --type ${TYPE} \
    --grid_search \
    --n_shot ${N_SHOT}\
    --dataset ${DATASET} \
    --vllm \
    --bfloat16 \
    --sae_word ${SAE_WORD} \
