import torch
import re
import os
import random
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer
import vllm
from vllm import LLM, SamplingParams

from utils import download_url, load_jsonl
import argparse

transformers.logging.set_verbosity(40)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 8
COT_FLAG = True
DEBUG = False
ANSWER_TRIGGER = "The answer is"


def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_answer, answer):
    gt_answer = extract_answer_from_output(answer)
    assert gt_answer != INVALID_ANS
    if model_answer == INVALID_ANS:
        return False
    else:
        return model_answer == gt_answer or float(model_answer) == float(gt_answer)


def create_demo_text(n_shot=8, cot_flag=True):
    question, chain, answer = [], [], []
    question.append(
        "There are 15 trees in the grove. "
        "Grove workers will plant trees in the grove today. "
        "After they are done, there will be 21 trees. "
        "How many trees did the grove workers plant today?"
    )
    chain.append(
        "There are 15 trees originally. "
        "Then there were 21 trees after some more were planted. "
        "So there must have been 21 - 15 = 6."
    )
    answer.append("6")

    question.append(
        "If there are 3 cars in the parking lot and 2 more cars arrive, "
        "how many cars are in the parking lot?"
    )
    chain.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
    answer.append("5")

    question.append(
        "Leah had 32 chocolates and her sister had 42. If they ate 35, "
        "how many pieces do they have left in total?"
    )
    chain.append(
        "Originally, Leah had 32 chocolates. "
        "Her sister had 42. So in total they had 32 + 42 = 74. "
        "After eating 35, they had 74 - 35 = 39."
    )
    answer.append("39")

    question.append(
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
        "has 12 lollipops. How many lollipops did Jason give to Denny?"
    )
    chain.append(
        "Jason started with 20 lollipops. Then he had 12 after giving some "
        "to Denny. So he gave Denny 20 - 12 = 8."
    )
    answer.append("8")

    question.append(
        "Shawn has five toys. For Christmas, he got two toys each from his "
        "mom and dad. How many toys does he have now?"
    )
    chain.append(
        "Shawn started with 5 toys. If he got 2 toys each from his mom and "
        "dad, then that is 4 more toys. 5 + 4 = 9."
    )
    answer.append("9")

    question.append(
        "There were nine computers in the server room. Five more computers "
        "were installed each day, from Monday to Thursday. "
        "How many computers are now in the server room?"
    )
    chain.append(
        "There were originally 9 computers. For each of 4 days, 5 more "
        "computers were added. So 5 * 4 = 20 computers were added. "
        "9 + 20 is 29."
    )
    answer.append("29")

    question.append(
        "Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On "
        "Wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of Wednesday?"
    )
    chain.append(
        "Michael started with 58 golf balls. After losing 23 on Tuesday, "
        "he had 58 - 23 = 35. After losing 2 more, "
        "he had 35 - 2 = 33 golf balls."
    )
    answer.append("33")

    question.append(
        "Olivia has $23. She bought five bagels for $3 each. "
        "How much money does she have left?"
    )
    chain.append(
        "Olivia had 23 dollars. "
        "5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
        "So she has 23 - 15 dollars left. 23 - 15 is 8."
    )
    answer.append("8")

    # randomize order of the examples ...
    index_list = list(range(len(question)))
    random.shuffle(index_list)

    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list[:n_shot]:
        if cot_flag:
            # demo_text += (
            #     "Q: "
            #     + question[i]
            #     + "\nA: "
            #     + chain[i]
            #     + " "
            #     + ANSWER_TRIGGER
            #     + " "
            #     + answer[i]
            #     + ".\n\n"
            # )
            demo_text += (
                    "<|im_start|>system\n"
                    "Please reason step by step, and put your final answer within \\boxed{{}}."
                    "<|im_end|>\n"
                    "<|im_start|>user\n"
                    + question[i] +
                    "<|im_end|>\n"
                    "<|im_start|>assistant\n"
                    + chain[i] +
                    " \\boxed{" + answer[i] + "}."
                    "\n\n"
            )
        else:
            demo_text += (
                "Q: "
                + question[i]
                + "\nA: "
                + ANSWER_TRIGGER
                + " "
                + answer[i]
                + ".\n\n"
            )
    return demo_text


def build_prompt(input_text, n_shot, cot_flag):
    demo = create_demo_text(n_shot, cot_flag)
    if cot_flag:
        input_text_prompt = demo + ( "<|im_start|>system\n"
                    "Please reason step by step, and put your final answer within \\boxed{{}}."
                    "<|im_end|>\n"
                    "<|im_start|>user\n\n"
                    + input_text
                    )
    else:
        input_text_prompt = demo + "Q: " + input_text + "\n" + "A:"
    return input_text_prompt


def clean_answer(model_pred):
    generated_text = model_pred.outputs[0].text

    generated_text = generated_text.lower()
    # preds = generated_text.split(ANSWER_TRIGGER.lower())

    match = re.search(r'\\boxed{(.*?)}', generated_text)

    if match:
        preds = match.group(1)
    else:
        preds = None

    if preds is not None:
        answer_flag = True if len(preds) > 1 else False
    else:
        return INVALID_ANS

    # if answer_flag:
    #     pred = preds[1]
    # else:
    #     pred = preds[-1]


    pred = preds.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]

    if len(pred) == 0:
        return INVALID_ANS

    if answer_flag:
        pred = pred[0]
    else:
        pred = pred[-1]

    if pred[-1] == ".":
        pred = pred[:-1]

    return pred


def seed_everything(seed: int):
    import random
    import os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load(model_name_or_path, cache_dir):
    print(f"Loading model from {model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir, trust_remote_code=True)
    llm = vllm.LLM(model_name_or_path, download_dir = cache_dir, gpu_memory_utilization = 0.95, max_model_len=4096, trust_remote_code=True)  # Initialize vLLM
    return llm, tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/dataset/llama2/llama-2-7b-hf",
        help="The model checkpoint for weights initialization.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="The root folder of the data.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="local directory where the model will be saved."
    )
    parser.add_argument("--load", type=str, default=None, help="load quantized model")

    args = parser.parse_args()
    return args


def generate(model, tokenizer, input_text, generate_kwargs):
    # Input text as is, no need for attention mask or input ids in vLLM
    input_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": input_text.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
    response = model.generate([input_text], generate_kwargs)

    if len(response) > 1:
        return response

    return response[0]


def main():
    args = parse_args()

    seed_everything(args.seed)

    test_filepath = os.path.join(args.data_root, "gsm8k_test.jsonl")
    if not os.path.exists(test_filepath):
        download_url(
            "https://raw.githubusercontent.com/openai/"
            "grade-school-math/2909d34ef28520753df82a2234c357259d254aa8/"
            "grade_school_math/data/test.jsonl",
            args.data_root,
        )
        os.rename(os.path.join(args.data_root, "test.jsonl"), test_filepath)

    list_data_dict = load_jsonl(test_filepath, instruction="question", output="answer")

    model, tokenizer = load(args.model_name_or_path, args.cache_dir)

    answers = []
    for sample in tqdm(list_data_dict):
        input_text = build_prompt(sample["instruction"], N_SHOT, COT_FLAG)
        input_text = input_text.strip(" ")

        sampling_params = SamplingParams(
            max_tokens=2048,
            temperature=0,
            top_p=1,
            stop = ["</s>", "<|im_end|>", "<|endoftext|>", "\n\nQ"],
            stop_token_ids=(
                [151645, 151643])
        )
        model_completion = generate(model, tokenizer, input_text, sampling_params)
        model_answer = clean_answer(model_completion)

        is_cor = is_correct(model_answer, sample["output"])
        answers.append(is_cor)
        if DEBUG:
            print(f"Full input_text:\n{input_text}\n\n")
        print(
            f'Question: {sample["instruction"]}\n\n'
            f'Answers: {extract_answer_from_output(sample["output"])}\n\n'
            f"Model Answers: {model_answer}\n\n"
            f"Model Completion: {model_completion}\n\n"
            f"Is correct: {is_cor}\n\n"
        )

        print(
            f"Num of total question: {len(answers)}, "
            f"Correct num: {sum(answers)}, "
            f"Accuracy: {float(sum(answers))/len(answers)}."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "results.txt"), "w") as f:
        for answer in answers:
            print(answer, file=f)

    with open(os.path.join(args.output_dir, "scores.txt"), "w") as f:
        print(
            f"Num of total question: {len(answers)}, "
            f"Correct num: {sum(answers)}, "
            f"Accuracy: {float(sum(answers))/len(answers)}.",
            file=f,
        )

if __name__ == "__main__":
    main()
