import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import re
import os
import random
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import vllm
from vllm import LLM, SamplingParams
import transformer_lens
from huggingface_hub import hf_hub_download
from models.SAE.JumpReLU import JumpReLUSAE
import gc
from prompts.prompts import get_examples




from utils import *
import argparse
import numpy as np



transformers.logging.set_verbosity(40)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
N_SHOT = 8
ANSWER_TRIGGER = "The answer is"
ANSWER_TRIGGER_2 = "kasarigan: "
ANSWER_TRIGGER_3 = "automáticamente se le da la respuesta de"

def extract_answer_from_output(completion, dataset):
    if dataset == "gsm8k":
        match = ANS_RE.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return INVALID_ANS
    elif dataset in ["svamp", "mawps", "aqua"]:
        return completion
    elif dataset == "asdiv":
        match = re.match(r'\d+', completion)
        if match:
            match_str = match.group()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return INVALID_ANS


def is_correct(model_answer, answer, dataset):
    gt_answer = extract_answer_from_output(answer, dataset)
    assert gt_answer != INVALID_ANS

    if model_answer == INVALID_ANS:
        return False

    try:
        model_answer_float = float(model_answer)
        gt_answer_float = float(gt_answer)
        return model_answer_float == gt_answer_float
    except (ValueError, TypeError):
        return model_answer == gt_answer # deal with multiple-choice questions

def create_demo_text(n_shot=8, cot_flag=True, dataset="gsm8k"):
    examples = get_examples(dataset)
    question, chain, answer = [], [], []
    for q, c, a in examples[dataset]:
        question.append(q)
        chain.append(c)
        answer.append(str(a))

    # randomize order of the examples ...
    index_list = list(range(len(question)))
    random.shuffle(index_list)

    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list[:n_shot]:
        if cot_flag:
            demo_text += (
                "Q: "
                + question[i]
                + "\nA: "
                + chain[i]
                + " "
                + ANSWER_TRIGGER
                + " "
                + answer[i]
                + ".\n\n"
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


def build_prompt(input_text, n_shot, cot_flag, dataset):
    demo = create_demo_text(n_shot, cot_flag, dataset)
    if cot_flag:
        input_text_prompt = demo + "Q: " + input_text + "\n Please reason step by step. " + "A:"

    else:
        input_text_prompt = demo + "Q: " + input_text + "\n" + "A:"
    return input_text_prompt


def clean_answer(model_pred, sae=False):
    if not sae:
        generated_text = model_pred.outputs[0].text
    else:
        generated_text = model_pred
    generated_text = generated_text.lower()
    if ANSWER_TRIGGER in generated_text:
        preds = generated_text.split(ANSWER_TRIGGER.lower())
    elif ANSWER_TRIGGER in generated_text:
        preds = generated_text.split(ANSWER_TRIGGER_2.lower())
    elif ANSWER_TRIGGER_3 in generated_text:
        preds = generated_text.split(ANSWER_TRIGGER_3.lower())
    else:
        preds = generated_text.split(ANSWER_TRIGGER.lower())



    if preds is not None:
        answer_flag = True if len(preds) > 1 else False
    else:
        return INVALID_ANS

    if answer_flag:
        preds = preds[1]
    else:
        preds = preds[-1]


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
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load(model_name_or_path, cache_dir, use_vllm, use_transformer_lens, n_devices, bfloat16):
    print(f"Loading model from {model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir, trust_remote_code=True)
    if use_vllm:
        if bfloat16:
            llm = vllm.LLM(model_name_or_path, download_dir=cache_dir, gpu_memory_utilization=1, max_model_len=4096,
                           trust_remote_code=True, dtype=torch.bfloat16)
        else:
            llm = vllm.LLM(model_name_or_path, download_dir = cache_dir, gpu_memory_utilization = 1, max_model_len=4096, trust_remote_code=True)  # Initialize vLLM
    elif use_transformer_lens:
        torch.set_grad_enabled(False)
        #with torch.no_grad():
        if bfloat16:
            llm = transformer_lens.HookedTransformer.from_pretrained(model_name_or_path, cache_dir=cache_dir, center_unembed=False, n_devices = n_devices, torch_dtype=torch.bfloat16)
        else:
            llm = transformer_lens.HookedTransformer.from_pretrained(model_name_or_path, cache_dir=cache_dir, center_unembed=False, n_devices = n_devices)
    else:
        llm = AutoModelForCausalLM.from_pretrained(model_name_or_path,device_map="auto",torch_dtype=torch.bfloat16, cache_dir=cache_dir, trust_remote_code=True)
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
        "--debug",
        action="store_true",
        help="Check full input text",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="local directory where the model will be saved."
    )
    parser.add_argument(
        "--cot_flag",
        action="store_true",
        help="use chain of thought or not"
    )
    parser.add_argument(
        "--bfloat16",
        action="store_true",
        help="floating point precision 16 bits"
    )
    parser.add_argument(
        "--layer_idx",
        type=int,
        default=20,
        help="layer to choose from"
    )
    parser.add_argument(
        "--coeff",
        type=int,
        default=400,
        help="coefficient for steering vectors"
    )
    parser.add_argument(
        "--K",
        type=int,
        default=5,
        help="Top K SAE features"
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="number of GPUs to use"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="inference",
        help="inference/sae"
    )
    parser.add_argument(
        "--sae_file",
        type=str,
        default="google/gemma-scope-2b-pt-res",
        help="pretrained sae name on huggingface"
    )
    parser.add_argument(
        "--param_file",
        type=str,
        default="layer_20/width_16k/average_l0_71/params.npz",
        help="parameter file name on huggingface"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        help="Evaluation Dataset Name"
    )
    parser.add_argument(
        "--plot_num",
        type=int,
        default=15,
        help="number of SAE features"
    )
    parser.add_argument(
        "--sae_idx",
        type=int,
        default=15153,
        help="SAE feature index"
    )
    parser.add_argument(
        "--sae_id",
        type=str,
        default="20-gemmascope-res-16k",
        help="sae id in html"
    )
    parser.add_argument(
        "--vllm",
        action="store_true",
        help="use vllm or not"
    )
    parser.add_argument(
        "--transformer_lens",
        action="store_true",
        help="use Transformer Lens or not"
    )
    parser.add_argument(
        "--cumulative",
        action="store_true",
        help="use cumulative sae features or not"
    )
    parser.add_argument(
        "--steer_vec_sae",
        action="store_true",
        help="use sae vectors as steering vectors"
    )
    parser.add_argument(
        "--grid_search",
        action="store_true",
        help="grid searching coefficients for sae"
    )
    parser.add_argument("--load", type=str, default=None, help="load quantized model")

    args = parser.parse_args()
    return args


def generate(model, tokenizer, input_text, generate_kwargs):
    # Input text as is, no need for attention mask or input ids in vLLM
    response = model.generate([input_text], generate_kwargs)

    if len(response) > 1:
        return response

    return response[0]


def main():
    args = parse_args()


    seed_everything(args.seed)

    test_filepath = os.path.join(args.data_root, args.dataset + "_test.jsonl")
    if not os.path.exists(test_filepath):
        download_url(
            test_url[args.dataset],
            args.data_root,
        )
        os.rename(os.path.join(args.data_root, "test.jsonl"), test_filepath)

    list_data_dict = []
    if len(pair[args.dataset]) > 2:
        question, body, answer = pair[args.dataset]
        list_data_dict = load_jsonl(test_filepath, instruction=question, input=body, output=answer)
    else:
        question, answer = pair[args.dataset]
        list_data_dict = load_jsonl(test_filepath, instruction=question, output=answer)


    model, tokenizer = load(args.model_name_or_path, args.cache_dir, args.vllm, args.transformer_lens, args.devices, args.bfloat16)

    path_to_params = hf_hub_download(
        repo_id=args.sae_file,
        filename=args.param_file,
        force_download=False,
        cache_dir=args.cache_dir,
    )

    params = np.load(path_to_params)
    pt_params = {k: torch.from_numpy(v).cuda() for k, v in params.items()}

    sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
    sae.load_state_dict(pt_params)
    sae = sae.to("cuda:0")

    cum = []
    cnt = 0
    top_k_values, top_k_indices = 0, 0

    answers = [] if args.type == "inference" else {}
    for sample in tqdm(list_data_dict):

        input_text = build_prompt(sample["instruction"], N_SHOT, args.cot_flag, args.dataset)
        input_text = input_text.strip(" ")

        if args.vllm:
            sampling_params = SamplingParams(
                max_tokens=2048,
                temperature=0,
                top_p=1,
                stop = ["</s>", "<|im_end|>", "<|endoftext|>", "\n\nQ"],
                stop_token_ids=(
                    [151645, 151643]
                    if "qwen2" in args.model_name_or_path.lower()
                    else None
                )

            )
        else:
            sampling_params = dict(max_new_tokens=256, top_p=1, temperature=0)

        if args.type == "inference":
            if args.steer_vec_sae:
                steering_vector = sae.W_dec[args.sae_idx]
                if args.grid_search:
                    coeff = args.coeff
                sampling_kwargs = sampling_params
                steering_on = True
                eos_token_id = tokenizer.encode("\n\n", add_special_tokens=False)

                def steering_hook(resid_pre, hook):
                    if resid_pre.shape[1] == 1:
                        return

                    if steering_on:
                        # using our steering vector and applying the coefficient
                        resid_pre[:, :, :] += coeff * steering_vector

                def hooked_generate(prompt_batch, fwd_hooks=[], seed=None, **kwargs):
                    if seed is not None:
                        torch.manual_seed(seed)

                    with model.hooks(fwd_hooks=fwd_hooks):
                        tokenized = model.to_tokens(prompt_batch)
                        result = model.generate(
                            stop_at_eos=True,
                            input=tokenized,
                            do_sample=True,
                            eos_token_id=eos_token_id,
                            **kwargs
                        )
                    return result

                def run_generate(input_text):
                    model.reset_hooks()
                    editing_hooks = [(f"blocks.{args.layer_idx}.hook_resid_post", steering_hook)]
                    res = hooked_generate(
                        [input_text], editing_hooks, seed=None, **sampling_kwargs
                    )

                    # Print results, removing the ugly beginning of sequence token
                    token_len = model.to_tokens(input_text).size(1)
                    answer = model.to_string(res[:, token_len:])
                    answer = "".join(answer)
                    return answer

                model_answer = clean_answer(run_generate(input_text), True)
            else:
                model_completion = generate(model, tokenizer, input_text, sampling_params)
                model_answer = clean_answer(model_completion)

            is_cor = is_correct(model_answer, sample["output"], args.dataset)
            answers.append(is_cor)
            if args.debug:
                print(f"Full input_text:\n{input_text}\n\n")
            print(
                f'Question: {sample["instruction"]}\n\n'
                f'Answers: {extract_answer_from_output(sample["output"], args.dataset)}\n\n'
                f"Model Answers: {model_answer}\n\n"
                f"Is correct: {is_cor}\n\n"
            )

            print(
                f"Num of total question: {len(answers)}, "
                f"Correct num: {sum(answers)}, "
                f"Accuracy: {float(sum(answers))/len(answers)}."
            )
        elif args.type == "sae":

            with torch.no_grad():
                inputs = tokenizer.encode(
                    input_text, return_tensors="pt", add_special_tokens=True
                ).to("cuda")

                _, cache = model.run_with_cache(
                    inputs,
                    names_filter=lambda name: name == f'blocks.{args.layer_idx}.hook_resid_post'
                )

                target_act = cache[f'blocks.{args.layer_idx}.hook_resid_post'].squeeze().detach()

                target_act = target_act.to("cuda:0")
                sae_acts = sae.encode(target_act.to(torch.float32))
                target_act = target_act.cpu()
                sae_acts = sae_acts.cpu()

                if args.cumulative:
                    if cnt:
                        cum += sae_acts[-1]
                    else:
                        cum = sae_acts[-1]
                else:
                    top_k_values, top_k_indices = torch.topk(sae_acts[-1], args.K)
                    for ind in top_k_indices:
                        answers[ind.item()] = answers.get(ind.item(), 0) + 1


                del cache
                del target_act
                del sae_acts
                gc.collect()
                torch.cuda.empty_cache()

        cnt += 1
    if args.cumulative:
        answers = {}
        top_k_values, top_k_indices = torch.topk(cum, args.K)
        for ind, val in zip(top_k_indices, top_k_values):
            answers[ind.item()] = val.item()

    os.makedirs(args.output_dir, exist_ok=True)
    name = args.model_name_or_path.split('/')[1] if '/' in args.model_name_or_path else None

    if args.type == "inference":
        if args.steer_vec_sae:
            output_path = os.path.join(args.output_dir, args.dataset)
            output_path = os.path.join(output_path, "steer_vec")
            os.makedirs(output_path, exist_ok=True)  # Ensure the directory exists
            with open(os.path.join(output_path, f"scores_{name}_{args.cot_flag}_{args.sae_idx}_{args.coeff}.txt"), "w") as f:
                print(
                    f"Num of total question: {len(answers)}, "
                    f"Correct num: {sum(answers)}, "
                    f"Accuracy: {float(sum(answers)) / len(answers)}.",
                    file=f,
                )
            with open(os.path.join(output_path, f"results_{name}_{args.cot_flag}_{args.sae_idx}_{args.coeff}.txt"), "w") as f:
                for answer in answers:
                    print(answer, file=f)
        else:
            output_path = os.path.join(args.output_dir, args.dataset)
            os.makedirs(output_path, exist_ok=True)  # Ensure the directory exists
            with open(os.path.join(output_path, f"scores_{name}_{args.cot_flag}.txt"), "w") as f:
                print(
                    f"Num of total question: {len(answers)}, "
                    f"Correct num: {sum(answers)}, "
                    f"Accuracy: {float(sum(answers)) / len(answers)}.",
                    file=f,
                )
            with open(os.path.join(output_path, f"results_{name}_{args.cot_flag}.txt"), "w") as f:
                for answer in answers:
                    print(answer, file=f)

    elif args.type == "sae":


        Top_K = TopK(answers, args.K)
        output_path = os.path.join(args.output_dir, args.dataset)
        if args.cumulative:
            output_path= os.path.join(output_path, "cumulative")
        else:
            output_path= os.path.join(output_path, "non-cumulative")
        os.makedirs(output_path, exist_ok=True)  # Ensure the directory exists
        with open(os.path.join(output_path, f"results_{name}_{args.cot_flag}_{args.K}.txt"), "w") as f:
            print(f"Top {args.K} SAE features")
            for index, value in Top_K:
                print(
                    f"SAE Index: {index}, Cardinality: {value}, "
                    f"Description: {extract_explanation(index, name, args.sae_id)}",
                    file=f,
                )
        print(f"Cardinality of SAE features: {len(answers)}")
        plot_SAE_barplot(answers, args.plot_num, args.cot_flag, name, output_path)




if __name__ == "__main__":
    main()





