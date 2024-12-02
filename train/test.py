import transformer_lens
from sae_lens import SAE
llm = transformer_lens.HookedTransformer.from_pretrained("gemma-2-2b", device = "cuda:0")
print("start!")
sae = SAE.from_pretrained(
            release="gemma-scope-2b-pt-res",
            sae_id=f"layer_20/width_16k/average_l0_63",  # won't always be a hook point
            device="cuda:0"
)[0]