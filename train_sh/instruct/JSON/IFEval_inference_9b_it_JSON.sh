#!/bin/bash

#SBATCH --output=/cluster/project/sachan/jiaxie/results/IFEval_inference_9b_it_JSON.out
#SBATCH --error=/cluster/project/sachan/jiaxie/results/IFEval_inference_9b_it_JSON.err
#SBATCH --mem-per-cpu=20G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_3090:2
#SBATCH --time=3:00:00

module load eth_proxy
export TRANSFORMERS_CACHE=/cluster/scratch/jiaxie/.cache
export TRITON_CACHE_DIR=/cluster/scratch/jiaxie/.triton_cache

cd /cluster/scratch/jiaxie/
source sae/bin/activate

cd /cluster/project/sachan/jiaxie/SAE_Math


#Settings
MODEL_NAME_OR_PATH="google/gemma-2-9b-it"
DATA_ROOT="/cluster/project/sachan/jiaxie/SAE_Math/instruct_data"
CACHE_DIR="/cluster/scratch/jiaxie/models/google/gemma-2-9b-it"
LAYER_IDX=31
PLOT_NUM=5
K=10
TYPE="inference"
SAE_FILE="gemma-scope-9b-it-res-canonical"
SAE_ID="31-gemmascope-res-16k"
PARAM_FILE="layer_31/width_16k/average_l0_71/params.npz"
DATASET="all_base_x_all_instructions_filtered"
INSTRUCT_TYPE="json_format"
SAE_WORD="JSON"
MODE="test"

python -u train/sae_instruct_follow.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_root ${DATA_ROOT} \
    --cache_dir ${CACHE_DIR} \
    --layer_idx ${LAYER_IDX} \
    --plot_num ${PLOT_NUM} \
    --K ${K} \
    --type ${TYPE} \
    --sae_file ${SAE_FILE} \
    --sae_id ${SAE_ID} \
    --vllm \
    --dataset ${DATASET} \
    --instruct_type ${INSTRUCT_TYPE} \
    --bfloat16 \
    --sae_word ${SAE_WORD} \
    --mode ${MODE} \
