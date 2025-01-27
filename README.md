# SAE_Math
 
Training sparse autoencoders on some SOTA math LLMs.

News
- 10.3.2024 Online
#SBATCH --gpus=rtx_3090:1


```bash
python -m venv sae
pip install vllm==0.6.3
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers
pip install transformer_lens sae_lens

```


