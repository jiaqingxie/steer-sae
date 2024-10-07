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
source sae/bin/activatege



cd /cluster/project/sachan/jiaxie/SAE_Math
#python gsm8k/vllm_main.py --model_name_or_path=Qwen/Qwen2.5-Math-7B --cache_dir=/cluster/scratch/jiaxie/models/Qwen/Qwen2.5-Math-7B
#python gsm8k/vllm_main.py --model_name_or_path=meta-llama/Meta-Llama-3-8B --cache_dir=/cluster/scratch/jiaxie/models/meta-llama/Meta-Llama-3-8B
python gsm8k/vllm_main.py --model_name_or_path=mistralai/Mistral-7B-v0.1 --cache_dir=/cluster/scratch/jiaxie/models/mistralai/Mistral-7B-v0.1