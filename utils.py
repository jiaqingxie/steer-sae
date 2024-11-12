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
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


test_url = {
    "gsm8k": "https://raw.githubusercontent.com/openai/grade-school-math/"
             "2909d34ef28520753df82a2234c357259d254aa8/grade_school_math/data/test.jsonl",
    "mawps": "https://raw.githubusercontent.com/QwenLM/Qwen2.5-Math/"
              "e128aa06871c7a0eeedc2ab21b69459bcb24c4fb/evaluation/data/mawps/test.jsonl",
    "svamp": "https://raw.githubusercontent.com/QwenLM/Qwen2.5-Math/"
             "e128aa06871c7a0eeedc2ab21b69459bcb24c4fb/evaluation/data/svamp/test.jsonl",
    "aqua":  "https://raw.githubusercontent.com/QwenLM/Qwen2.5-Math/"
             "e128aa06871c7a0eeedc2ab21b69459bcb24c4fb/evaluation/data/aqua/test.jsonl",
    "asdiv":  "https://raw.githubusercontent.com/QwenLM/Qwen2.5-Math/"
             "e128aa06871c7a0eeedc2ab21b69459bcb24c4fb/evaluation/data/asdiv/test.jsonl"
}

pair = {
    "gsm8k": ["question", "answer"],
    "mawps": ["input", "target"],
    "svamp": ["Question", "Body", "Answer"],
    "aqua": ["question", "options", "correct"],
    "asdiv": ["question", "body", "answer"]

}


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
            instruction_value = item.get(instruction, "")
            input_value = item.get(input, "")

            if type(input_value) == list:
                input_value = ", ".join(input_value)

            # Prepend input to instruction if input is not empty
            if input_value:
                instruction_value = input_value + " " + instruction_value

            new_item = dict(
                instruction=instruction_value,
                input=item.get(input, None),
                output=item.get(output, None),
                category=item.get(category, None),
            )
            list_data_dict.append(new_item)
    return list_data_dict


def TopK(a: dict, k: int):
    sorted_list = sorted(a.items(), key=lambda x: x[1], reverse=True)
    top_k = sorted_list[:k]
    return top_k

def extract_explanation(idx, model, sae_id):
    def get_dashboard_html(sae_release="gemma-2-2b", sae_id="20-gemmascope-res-16k", feature_idx=6868):
        return html_template.format(sae_release, sae_id, feature_idx)

    html_template = "https://neuronpedia.org/{}/{}/{}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"
    html_add = get_dashboard_html(sae_release=model, sae_id=sae_id, feature_idx=idx)

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

def plot_SAE_barplot(input_dict, top_n, cot_flag, model, path):
    plt.rcParams['font.family'] = 'Arial'

    df = pd.DataFrame(list(input_dict.items()), columns=['Feature', 'Count'])
    df['Feature'] = df['Feature'].astype(str)

    df = df.sort_values(by='Count', ascending=False).reset_index(drop=True)
    df = df.head(60)

    if top_n == -1:
        top_n = len(input_dict)
    df_top = df.head(top_n)

    sns.set_theme(style='whitegrid', context='paper', font_scale=1.2)

    num_features = len(df)
    full_gradient = sns.color_palette("Blues_r", n_colors=num_features)

    feature_to_color = dict(zip(df['Feature'], full_gradient))

    main_colors = [feature_to_color[feature] for feature in df['Feature']]

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        x='Feature',
        y='Count',
        data=df,
        palette=main_colors,
        edgecolor='black',
        ax=ax
    )

    plt.xticks(rotation=90, ha='center', fontsize=6)

    ax.set_xlabel('Sorted SAE Feature Index', fontsize=14)
    ax.set_ylabel('SAE Feature Frequency', fontsize=14)

    ax.set_title('SAE Feature Distribution', fontsize=16, weight='bold', y=1.05)

    sns.despine()
    plt.tight_layout()

    ax_inset = inset_axes(ax, width="35%", height="35%", loc='upper right', borderpad=1)

    top_gradient = sns.color_palette("Blues_r", n_colors=top_n)

    sns.barplot(
        x='Feature',
        y='Count',
        data=df_top,
        ax=ax_inset,
        palette=top_gradient,
        edgecolor='black'
    )



    ax_inset.set_xticklabels(ax_inset.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax_inset.set_xlabel('Top {} SAE Features'.format(top_n), fontsize=12)
    ax_inset.set_ylabel('SAE Feature Frequency', fontsize=12)
    ax_inset.set_title('Top {} SAE Feature Distribution'.format(top_n), fontsize=14)

    ax_inset.tick_params(axis='both', which='major', labelsize=10)

    # for p in ax_inset.patches:
    #     ax_inset.annotate(
    #         format(p.get_height(), '.0f'),
    #         (p.get_x() + p.get_width() / 2., p.get_height()),
    #         ha='center',
    #         va='bottom',
    #         fontsize=10,
    #         xytext=(0, 5),
    #         textcoords='offset points'
    #     )

    feature_positions = dict(zip(df['Feature'], range(len(df))))
    x_positions = [feature_positions[feature] for feature in df_top['Feature']]
    x_min = min(x_positions) - 0.5
    x_max = max(x_positions) + 0.5
    y_min = 0
    y_max = df['Count'].max()

    rect = Rectangle((x_min, y_min), x_max - x_min, y_max,
                     linewidth=1, edgecolor='red', facecolor='none', linestyle='--')
    ax.add_patch(rect)

    if not cot_flag:
        filename = f"{path}/SAE_{model}_barplot_{top_n}.pdf"
    else:
        filename = f"{path}/SAE_{model}_barplot_{top_n}_COT.pdf"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Plot saved as {filename}")


