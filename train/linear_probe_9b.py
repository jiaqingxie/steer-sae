import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import random

device = "cuda"
model_name_or_path = "google/gemma-2-9b"
n_devices = 2
llm = HookedTransformer.from_pretrained(model_name_or_path, n_devices=n_devices, torch_dtype=torch.bfloat16)

# 加载SAE
sae_1, cfg_dict, _ = SAE.from_pretrained(
    release="gemma-scope-9b-pt-res-canonical", sae_id="layer_31/width_16k/canonical", device=device
)


def generate_balanced_data(samples_per_class):
    operations = ['+', '-', '*', '/']
    data = []
    for idx, op in enumerate(operations):
        for _ in range(samples_per_class):
            a, b = random.randint(1, 100), random.randint(1, 100)
            if op == '/':
                b = random.randint(1, 10)
                a = b * random.randint(1, 10)
            expression = f"{a} {op} {b}"
            data.append((expression, idx))
    return data


full_data = generate_balanced_data(300)
texts, labels = zip(*full_data)
labels = np.array(labels)


texts_train, texts_val, labels_train, labels_val = train_test_split(
    texts, labels, test_size=0.5, random_state=42, stratify=labels
)


W_dec_feature = sae_1.W_dec[12946].to("cuda:1")


# W_dec_feature = torch.load("/cluster/project/sachan/jiaxie/SAE_Math/mean_vec/gemma-2-9b_mean_diff.pt", map_location="cuda:1")

def get_Wdec_feature(text, model, W_dec_feature):
    tokens = model.to_tokens(text).to(device)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
        hidden_acts = cache["blocks.31.hook_resid_post"]
        feature_values = torch.einsum("bsd,d->bs", hidden_acts, W_dec_feature.to(hidden_acts.dtype))
        feature_scalar = feature_values.mean(dim=1).item()
    return feature_scalar


X_train = np.array([get_Wdec_feature(text, llm, W_dec_feature) for text in texts_train]).reshape(-1, 1)
X_val = np.array([get_Wdec_feature(text, llm, W_dec_feature) for text in texts_val]).reshape(-1, 1)


clf = LogisticRegression(multi_class="multinomial", max_iter=1000)
clf.fit(X_train, labels_train)

predictions = clf.predict(X_val)
print(classification_report(labels_val, predictions, target_names=["+", "-", "*", "/"]))
