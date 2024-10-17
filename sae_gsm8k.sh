#!/bin/bash

#SBATCH --output=/cluster/project/sachan/jiaxie/results/%j.out
#SBATCH --error=/cluster/project/sachan/jiaxie/results/%j.err
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=3:00:00

module load eth_proxy
#export TRANSFORMERS_CACHE=/cluster/scratch/jiaxie/.cache
export TRITON_CACHE_DIR=/cluster/scratch/jiaxie/.triton_cache


cd /cluster/scratch/jiaxie/
source sae/bin/activate

cd /cluster/project/sachan/jiaxie/SAE_Math

#python gsm8k/vllm_main.py --model_name_or_path=google/gemma-2-9b-it --cache_dir=/cluster/scratch/jiaxie/models/google/gemma-2-9b-it
#python gsm8k/vllm_main.py --model_name_or_path=google/gemma-2-9b --cache_dir=/cluster/scratch/jiaxie/models/google/gemma-2-9b



#Settings
MODEL_NAME_OR_PATH="google/gemma-2-2b"
DATA_ROOT="/cluster/project/sachan/jiaxie/SAE_Math/gsm8k/data"
#DEBUG=False
CACHE_DIR="/cluster/scratch/jiaxie/models/google/gemma-2-2b"
#COT_FLAG=False
LAYER_IDX=20
K=5
TYPE="sae"
SAE_FILE="google/gemma-scope-2b-pt-res"
PARAM_FILE="layer_20/width_16k/average_l0_71/params.npz"
#VLLM=False
TRANSFORMER_LENS=True

python -u train/sae.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_root ${DATA_ROOT} \
    --cache_dir ${CACHE_DIR} \
    --layer_idx ${LAYER_IDX} \
    --K ${K} \
    --type ${TYPE} \
    --sae_file ${SAE_FILE} \
    --param_file ${PARAM_FILE} \
    --transformer_lens \
