import transformer_lens
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

# Define model and tokenizer
model = transformer_lens.HookedTransformer.from_pretrained("google/gemma-2-2b", cache_dir="/cluster/scratch/jiaxie/models/google/gemma-2-2b")
tokenizer =  AutoTokenizer.from_pretrained("google/gemma-2-2b", cache_dir="/cluster/scratch/jiaxie/models/google/gemma-2-2b")
device = "cuda:0" if torch.cuda.is_available() else "cpu"


# Count the number of layers
layer_count = len(list(model.children()))
layer_2_3 = len(2/3 * layer_count)

## Loading some gsm-8k prompts
prompt = ("Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' "
          "market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?")
prm
inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device)
_, cache = model.run_with_cache(inputs)


# target
target_act = cache[f'blocks.{layer_2_3}.hook_resid_post'].squeeze()



### Vanilla SAE
class JumpReLUSAE(nn.Module):
  def __init__(self, d_model, d_sae):
    # Note that we initialise these to zeros because we're loading in pre-trained weights.
    # If you want to train your own SAEs then we recommend using blah
    super().__init__()
    self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
    self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
    self.threshold = nn.Parameter(torch.zeros(d_sae))
    self.b_enc = nn.Parameter(torch.zeros(d_sae))
    self.b_dec = nn.Parameter(torch.zeros(d_model))

  def encode(self, input_acts):
    pre_acts = input_acts @ self.W_enc + self.b_enc
    mask = (pre_acts > self.threshold)
    acts = mask * torch.nn.functional.relu(pre_acts)
    return acts

  def decode(self, acts):
    return acts @ self.W_dec + self.b_dec

  def forward(self, acts):
    acts = self.encode(acts)
    recon = self.decode(acts)
    return recon


# downlowd SAE weights

path_to_params = hf_hub_download(
    repo_id="google/gemma-scope-2b-pt-res",
    filename="layer_20/width_16k/average_l0_71/params.npz",
    force_download=False,
    cache_dir="/cluster/scratch/jiaxie/models/google/gemma-2-2b"
)

params = np.load(path_to_params)
pt_params = {k: torch.from_numpy(v).cuda() for k, v in params.items()}

sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
sae.load_state_dict(pt_params)

sae_acts = sae.encode(target_act.to(torch.float32))
# recon = sae.decode(sae_acts)

values, inds = sae_acts.max(-1)
print(inds)