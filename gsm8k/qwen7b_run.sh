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

#python gsm8k/vllm_main.py --model_name_or_path=Qwen/Qwen2.5-Math-7B-Instruct --cache_dir=/cluster/scratch/jiaxie/models/Qwen/Qwen2.5-Math-7B-Instruct

python gsm8k/vllm_main.py --model_name_or_path=google/gemma-2-9b --cache_dir=/cluster/scratch/jiaxie/models/google/gemma-2-9b




# Settings
#PROMPT_TYPE="qwen25-math-cot"
#MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-7B"
#OUTPUT_DIR="/cluster/scratch/jiaxie/Qwen2.5/output"
#CACHE_DIR="/cluster/scratch/jiaxie/models/Qwen/Qwen2.5-Math-7B"
#
#SPLIT="test"
#NUM_TEST_SAMPLE=-1
#
## English open datasets
##gsm8k,math,svamp,asdiv,mawps,carp_en,tabmwp,minerva_math,gaokao2023en,olympiadbench,college_math
#DATA_NAME="gsm8k"
#TOKENIZERS_PARALLELISM=false \
#python3 -u math_eval.py \
#    --model_name_or_path ${MODEL_NAME_OR_PATH} \
#    --data_name ${DATA_NAME} \
#    --output_dir ${OUTPUT_DIR} \
#    --split ${SPLIT} \
#    --prompt_type ${PROMPT_TYPE} \
#    --num_test_sample ${NUM_TEST_SAMPLE} \
#    --seed 0 \
#    --temperature 0 \
#    --n_sampling 1 \
#    --top_p 1 \
#    --start 0 \
#    --end -1 \
#    --use_vllm \
#    --save_outputs \
#    --overwrite \
#    --cache_dir ${CACHE_DIR} \
