#!/bin/bash

#SBATCH --output=/cluster/project/sachan/jiaxie/results/IFEval_C200_T0_omega1_6098_bn.out
#SBATCH --error=/cluster/project/sachan/jiaxie/results/IFEval_C200_T0_omega1_6098_bn.err
#SBATCH --mem-per-cpu=20G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_3090:2
#SBATCH --time=04:00:00

module load eth_proxy
export TRANSFORMERS_CACHE=/cluster/scratch/jiaxie/.cache
export TRITON_CACHE_DIR=/cluster/scratch/jiaxie/.triton_cache


cd /cluster/scratch/jiaxie/
source sae/bin/activate

cd /cluster/project/sachan/jiaxie/SAE_Math
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#Settings alphabetically
CACHE_DIR="/cluster/scratch/jiaxie/models/google/gemma-2-9b-it"
COEFF=(200)
DATA_ROOT="/cluster/project/sachan/jiaxie/SAE_Math/instruct_data"
K=10
LAYER_IDX=31
MODEL_NAME_OR_PATH="google/gemma-2-9b-it"
PARAM_FILE="layer_31/width_16k/average_l0_63/params.npz"
PLOT_NUM=5
SAE_FILE="gemma-scope-9b-it-res-canonical"
SAE_ID="31-gemmascope-res-16k"
SAE_IDX=(6098)
TRANSFORMER_LENS=True
TYPE="inference"
T=0
OMEGA=1
DATASET="all_base_x_all_instructions_filtered"
INSTRUCT_TYPE="response_language_bn"
N_DEVICES=2
MODE="test"
TRAIN_SIZE=0.3
TEST_SIZE=0.5
VALID_SIZE=0.2

python -u train/sae_instruct_follow.py \
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
    --sae_idx ${SAE_IDX[@]} \
    --coeff ${COEFF[@]} \
    --T ${T} \
    --bfloat16 \
    --dataset ${DATASET} \
    --instruct_type ${INSTRUCT_TYPE} \
    --omega ${OMEGA} \
    --devices ${N_DEVICES} \
    --mode ${MODE} \
    --train_size ${TRAIN_SIZE} \
    --test_size ${TEST_SIZE} \
    --valid_size ${VALID_SIZE} \
