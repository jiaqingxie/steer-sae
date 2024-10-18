import copy
import os
import ssl
import urllib.request

import os.path as osp
import gzip
import json
import re
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def download_url(url: str, folder="folder"):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition("/")[2]
    file = file if file[0] == "?" else file.split("?")[0]
    path = osp.join(folder, file)
    if osp.exists(path):
        print(f"File {file} exists, use existing file.")
        return path

    print(f"Downloading {url}")
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, "wb") as f:
        f.write(data.read())

    return path


def load_jsonl(
    file_path,
    instruction="instruction",
    input="input",
    output="output",
    category="category",
    is_gzip=False,
):
    # Format of each line:
    # {'instruction': ..., 'input': ..., 'output':...}
    list_data_dict = []
    open_func = open if not is_gzip else gzip.open
    with open_func(file_path, "r") as f:
        for line in f:
            item = json.loads(line)
            new_item = dict(
                instruction=item[instruction] if instruction in item else None,
                input=item[input] if input in item else None,
                output=item[output] if output in item else None,
                category=item[category] if category in item else None,
            )
            item = new_item
            list_data_dict.append(item)
    return list_data_dict


def TopK(a: dict, k: int):
    sorted_list = sorted(a.items(), key=lambda x: x[1], reverse=True)
    top_k = sorted_list[:k]
    return top_k

def extract_explanation(idx):
    def get_dashboard_html(sae_release="gemma-2-2b", sae_id="20-gemmascope-res-16k", feature_idx=6868):
        return html_template.format(sae_release, sae_id, feature_idx)

    html_template = "https://neuronpedia.org/{}/{}/{}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"
    html_add = get_dashboard_html(sae_release="gemma-2-2b", sae_id="20-gemmascope-res-16k", feature_idx=idx)

    response = requests.get(html_add)
    html_content = response.content.decode('utf-8')
    start_pos = html_content.find('"explanations')
    start_pos += len('\"explanations\"')
    extracted_content = html_content[start_pos:start_pos + 1000]

    start_pos = extracted_content.find('\\\"description\\\"')
    start_pos += len('\\\"description\\\":\\\"')
    end_pos = extracted_content.find('\\\"authorId\\\"')
    result = extracted_content[start_pos:end_pos - 3]
    return result

def plot_SAE_barplot(input, top_n):

    df = pd.DataFrame(list(input.items()), columns=['Feature', 'Count'])
    df = df.sort_values(by='Count', ascending=False)
    df['Feature'] = df['Feature'].astype(str) # Important as type of keys are int.

    df_top = df.head(top_n)
    sns.set_theme(style='whitegrid', context='paper', font_scale=1.2)
    blue_gradient = sns.color_palette("Blues_r", n_colors=top_n)

    plt.figure(figsize=(6, 5))
    ax = sns.barplot(
        x='Feature',
        y='Count',
        data=df_top,
        palette=blue_gradient,
        edgecolor='black'
    )

    plt.xticks(rotation=45, ha='center', fontsize=12)

    for p in ax.patches:
        ax.annotate(
            format(p.get_height(), '.0f'),
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center',
            va='bottom',
            fontsize=10,
            xytext=(0, 5),
            textcoords='offset points'
        )

    ax.set_title(f'Top {top_n} SAE Feature Count Distribution', fontsize=16, weight='bold')
    ax.set_xlabel('SAE Feature Index', fontsize=16)
    ax.set_ylabel('SAE Feature Count', fontsize=16)

    sns.despine()
    plt.tight_layout()
    plt.savefig("/cluster/project/sachan/jiaxie/SAE_Math/output/SAE_bar_{}.pdf".format(top_n))

