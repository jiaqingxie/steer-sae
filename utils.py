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
import sympy as sp
import torch

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
             "e128aa06871c7a0eeedc2ab21b69459bcb24c4fb/evaluation/data/asdiv/test.jsonl",
    "gsm8k_train": "https://raw.githubusercontent.com/QwenLM/Qwen2.5-Math/"
             "e128aa06871c7a0eeedc2ab21b69459bcb24c4fb/evaluation/data/gsm8k/train.jsonl",
    "math": "https://raw.githubusercontent.com/QwenLM/Qwen2.5-Math/"
             "e128aa06871c7a0eeedc2ab21b69459bcb24c4fb/evaluation/data/math/test.jsonl",

}

pair = {
    "gsm8k": ["question", "answer"],
    "gsm8k_train": ["question", "answer"],
    "mawps": ["input", "target"],
    "svamp": ["Question", "Body", "Answer"],
    "aqua": ["question", "options", "correct"],
    "asdiv": ["question", "body", "answer"],
    "math": ["problem", "solution"],
    "instruct_format_length":["prompt", "prompt_without_instruction", "key", "instruction_id_list_for_eval"],
    "all_base_x_all_instructions_filtered":["model_output", "prompt_without_instruction", "icl_key", "single_instruction_id"]
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
    input="inputs",
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
                if "aqua" in file_path:
                    instruction_value = input_value + ". " + instruction_value
                else:
                    instruction_value = input_value + " " + instruction_value

            new_item = dict(
                instruction=instruction_value,
                input=item.get(input, None),
                output=item.get(output, None),
                category=item.get(category, None),
            )
            list_data_dict.append(new_item)
    return list_data_dict


def load_jsonl_instruct(
    file_path,
    prompt="inputs",
    prompt_without_instruct="base_without_instruct",
    id="id",
    type="type",
    is_gzip=False,
):

    list_data_dict = []
    open_func = open if not is_gzip else gzip.open
    with open_func(file_path, "r") as f:
        for line in f:
            item = json.loads(line)
            new_item = dict(
                prompt=item.get(prompt, None),
                prompt_without_instruct=item.get(prompt_without_instruct, None),
                category=item.get(id, None),
                type=item.get(type, None),
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

def plot_SAE_barplot(input_dict, top_n, cot_flag, model, path, cumulative):
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
    if cumulative:
        ax.set_ylabel('Average Activation Value', fontsize=14)
    else:
        ax.set_ylabel('Top-K SAE Feature Frequency', fontsize=14)

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


####### adapted from Qwen 2.5
direct_answer_trigger_for_fewshot = ("choice is", "answer is")
def choice_answer_clean(model_pred, vllm):
    if vllm:
        pred = model_pred.outputs[0].text
    else:
        pred = model_pred
    pred = pred.strip("\n")

    # Determine if this is ICL, if so, use \n\n to split the first chunk.
    ICL = False
    for trigger in direct_answer_trigger_for_fewshot:
        if pred.count(trigger) > 1:
            ICL = True
    if ICL:
        pred = pred.split("\n\n")[0]

    # Split the trigger to find the answer.
    preds = re.split("|".join(direct_answer_trigger_for_fewshot), pred)
    if len(preds) > 1:
        answer_flag = True
        pred = preds[-1]
    else:
        answer_flag = False

    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")

    # Clean the answer based on the dataset
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]

    if len(pred) == 0:
        pred = ""
    else:
        if answer_flag:
            # choose the first element in list ...
            pred = pred[0]
        else:
            # choose the last e
            pred = pred[-1]

    # Remove the period at the end, again!
    pred = pred.rstrip(".").rstrip("/")

    return pred




def extract_latex_or_number(model_pred, vllm):
    """
    Extract the desired output based on the updated rules:
    1. If '=' is present:
       - Return everything after the last '=' until the next $ or $$.
       - If no $ or $$ after '=', return everything after the last '='.
    2. If no '=':
       - For the last $...$ or $$...$$:
         - If a number exists inside, return that number.
         - Otherwise, return the content inside the last $...$ or $$...$$.
    3. If no '=' or $/$, return the last number in the string.
    This version ensures no trailing $ or $$ in the output and corrects for cases without '='.
    """
    if vllm:
        input_str = model_pred.outputs[0].text
    else:
        input_str = model_pred
    input_str = input_str.strip("\n")
    # Regex patterns for $$...$$ and $...$
    double_dollar_pattern = r'\$\$(.*?)\$\$'
    single_dollar_pattern = r'\$(.*?)\$'

    # Find all $$...$$ and $...$ matches
    double_dollar_matches = re.findall(double_dollar_pattern, input_str)
    single_dollar_matches = re.findall(single_dollar_pattern, input_str)

    # Get the last $$...$$ and $...$ if they exist
    last_double_dollar = double_dollar_matches[-1] if double_dollar_matches else None
    last_single_dollar = single_dollar_matches[-1] if single_dollar_matches else None

    # Determine the positions of the last $$ $$ and $ $
    last_double_dollar_pos = input_str.rfind(f"$${last_double_dollar}$$") if last_double_dollar else -1
    last_single_dollar_pos = input_str.rfind(f"${last_single_dollar}$") if last_single_dollar else -1

    # Determine the most relevant LaTeX block (whichever comes last)
    if last_double_dollar_pos > last_single_dollar_pos:
        last_block = last_double_dollar
    else:
        last_block = last_single_dollar

    # Find the position of the last '='
    if "=" in input_str:
        equal_pos = input_str.rfind("=")
        # Extract content after the last '=' until the next $ or $$
        after_equal = input_str[equal_pos + 1:].split("$")[0].split("$$")[0].strip()
        return after_equal

    # If no '=' is present
    if last_block:
        return last_block.strip()  # Return the entire content of the last $...$ or $$...$$

    # If no $/$ or '=' is present, return the last number
    numbers = re.findall(r'\b\d+\.?\d*\b', input_str)
    return numbers[-1] if numbers else None

def simplify_latex_expression(expression):
    """
    Simplify a LaTeX mathematical expression according to the rules:
    1. If there is \sqrt{}, calculate the square root if it contains only numbers or \pi.
    2. If there is \frac{}{}, calculate the fraction if it contains only numbers or \pi.
    3. Replace \pi with 3.14 for numerical evaluation.
    4. Handle expressions with numeric coefficients (e.g., 10\sqrt{} or 10\frac{}{}), treating missing * as multiplication.
    5. Skip simplification if variables (letters) are present inside \sqrt{} or \frac{}{}.
    """
    # Replace \pi with 3.14 in the expression
    if not isinstance(expression, (int, str)):
        return "Invalid"
    expression = expression.replace(r"\pi", "3.14")

    # Ensure proper multiplication when numeric coefficient is followed by \ (e.g., 3\sqrt{} → 3*\sqrt{})
    expression = re.sub(r"(\d)(\\)", r"\1*\2", expression)

    # Match and simplify \sqrt{} if it contains only numbers or \pi
    sqrt_matches = re.findall(r"\\sqrt\{([^\}]+)\}", expression)
    for match in sqrt_matches:
        try:
            value = sp.sqrt(eval(match, {"__builtins__": None}, {"pi": 3.14}))
            expression = expression.replace(rf"\sqrt{{{match}}}", str(float(value)))
        except Exception:
            continue  # Skip if the content is non-numeric or contains variables

    # Match and simplify \frac{}{} if both parts are numeric or contain \pi
    frac_matches = re.findall(r"\\frac\{([^\}]+)\}\{([^\}]+)\}", expression)
    for numerator, denominator in frac_matches:
        try:
            value = sp.Rational(eval(numerator, {"__builtins__": None}, {"pi": 3.14}),
                                eval(denominator, {"__builtins__": None}, {"pi": 3.14}))
            expression = expression.replace(rf"\frac{{{numerator}}}{{{denominator}}}", str(float(value)))
        except Exception:
            continue  # Skip if the content is non-numeric or contains variables

    # Evaluate the resulting mathematical expression with numeric coefficients
    try:
        evaluated_expression = eval(expression, {"__builtins__": None}, {"pi": 3.14})
        return str(evaluated_expression)
    except Exception:
        return expression


def load_tensor(base_directory, file_name):
    # 构建文件名
    file_path = os.path.join(base_directory, file_name)

    if os.path.exists(file_path):
        print(f"Loading tensor from {file_path}")
        tensor = torch.load(file_path)
        return tensor
    else:
        print(f"File {file_name} not found in {base_directory}, skipping.")
        return None