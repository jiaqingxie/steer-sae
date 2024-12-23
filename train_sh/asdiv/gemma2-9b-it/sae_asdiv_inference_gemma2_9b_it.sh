#!/bin/bash

#SBATCH --output=/cluster/project/sachan/jiaxie/results/sae_9b_it_asdiv_inference_cot.out
#SBATCH --error=/cluster/project/sachan/jiaxie/results/sae_9b_it_asdiv_inference_cot.err
#SBATCH --mem-per-cpu=20G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=3:00:00

module load eth_proxy
export HF_HOME=/cluster/scratch/jiaxie/.cache/huggingface
export TRANSFORMERS_CACHE=/cluster/scratch/jiaxie/.cache
export TRITON_CACHE_DIR=/cluster/scratch/jiaxie/triton_cache


cd /cluster/scratch/jiaxie/
source sae/bin/activate

cd /cluster/project/sachan/jiaxie/SAE_Math

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#Settings alphabetically
CACHE_DIR="/cluster/scratch/jiaxie/models/google/gemma-2-9b-it"
DATA_ROOT="/cluster/project/sachan/jiaxie/SAE_Math/data"


MODEL_NAME_OR_PATH="google/gemma-2-9b-it"
PARAM_FILE="layer_31/width_16k/average_l0_63/params.npz"

TYPE="inference"
N_SHOT=8
DATASET="asdiv"

python -u train/sae.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_root ${DATA_ROOT} \
    --cache_dir ${CACHE_DIR} \
    --type ${TYPE} \
    --grid_search \
    --n_shot ${N_SHOT}\
    --dataset ${DATASET} \
    --vllm \
    --cot_flag \
