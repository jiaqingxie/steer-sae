# SAE_Math
 
Training sparse autoencoders on some SOTA math LLMs.

News
- 2024.10.3 Online
- 2025.1.10 Finish Analysis I


```bash
python -m venv sae
pip install vllm==0.6.3
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers
pip install transformer_lens sae_lens

```


Analysis I 

Analysis II

Train and test on: ifeval_wo_instructions.jsonl, ifeval_single_keyword_include.jsonl, and ifeval_single_keyword_exclude.jsonl

Length constraints: Answer using {at most} {K} sentences.

Format constraints: On lowercase / JSON format / Highlight sentences

Dataset: IFEval



