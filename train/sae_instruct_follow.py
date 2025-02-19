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
import gc
from prompts.prompts import get_examples
from transformers import  GenerationConfig
from sae_lens import SAE
from functools import partial
import math
from utils import *
from parser import *
import argparse
import numpy as np
import json

transformers.logging.set_verbosity(40)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
ANSWER_TRIGGER = "The answer is"





def contains_valid_json(s: str) -> bool:
    """
    判断字符串 s 中是否包含可被 json.loads 成功解析的 JSON 子串。
    只要找到一段可成功解析的 JSON，就返回 True，否则返回 False。
    """
    brackets_map = {'{': '}', '[': ']'}

    length = len(s)
    for i in range(length):
        if s[i] in brackets_map:  # 可能是 JSON 的起始符号
            open_bracket = s[i]
            close_bracket = brackets_map[open_bracket]

            # 用一个计数器追踪括号匹配层级
            level = 1
            for j in range(i + 1, length):
                if s[j] == open_bracket:
                    level += 1
                elif s[j] == close_bracket:
                    level -= 1

                # 当level回到0，说明找到了完整的匹配区间
                if level == 0:
                    # 取出这段子串
                    candidate = s[i:j + 1]
                    # 尝试解析
                    try:
                        json.loads(candidate)
                        return True
                    except json.JSONDecodeError:
                        # 解析失败就继续找下一个可能的区间
                        pass
                    break  # 同一个起始只尝试匹配到第一个对应结束符，匹配完就跳出

    # 如果所有可能的子串都没成功解析，返回 False
    return False

def build_prompt(prompt_without_instruct, prompt, type, num_sentences, least, most, model_name, sae_word):
    if ("json_format" in type) or ("lowercase" in type) or ("highlight" in type):
        if "it" not in model_name:
            prompt_without_instruct = "Question: " + prompt_without_instruct + "\nAnswer:"
            prompt = "Question: " + prompt + "\nAnswer:"
            #TODO: add sae_words at the end of the input
            if sae_word != "":
                prompt += " " + sae_word

        return prompt_without_instruct, prompt
    elif "length_constraints" in type:
        if num_sentences == 1:
            if least:
                if "it" in model_name:
                    prompt = prompt_without_instruct + f" Answer using at least {num_sentences} sentence."
                else:
                    prompt = "Question:" + prompt_without_instruct + f" Answer using at least {num_sentences} sentence.\nAnswer:"
                    prompt_without_instruct = "Question: " + prompt_without_instruct + "\nAnswer:"
            elif most:
                if "it" in model_name:
                    prompt = prompt_without_instruct + f" Answer using at most {num_sentences} sentence."
                else:
                    prompt = "Question:" + prompt_without_instruct + f" Answer using at most {num_sentences} sentence.\nAnswer:"
                    prompt_without_instruct = "Question: " + prompt_without_instruct + "\nAnswer:"
        else:
            if least:
                if "it" in model_name:
                    prompt = prompt_without_instruct + f" Answer using at least {num_sentences} sentences."
                else:
                    prompt = "Question:" + prompt_without_instruct + f" Answer using at least {num_sentences} sentences.\nAnswer:"
                    prompt_without_instruct = "Question: " + prompt_without_instruct + "\nAnswer:"
            elif most:
                if "it" in model_name:
                    prompt = prompt_without_instruct + f" Answer using at most {num_sentences} sentences."
                else:
                    prompt = "Question:" + prompt_without_instruct + f" Answer using at most {num_sentences} sentences.\nAnswer:"
                    prompt_without_instruct = "Question: " + prompt_without_instruct + "\nAnswer:"
        return prompt_without_instruct, prompt
    else:
        raise ValueError("this type is not supported")


def clean_answer(model_pred, sae, vllm, dataset="gsm8k", steer_vec_sae=False):
    if not sae:
        if vllm:
            generated_text = model_pred.outputs[0].text
        else:
            generated_text = model_pred
    else:
        generated_text = model_pred

    return generated_text


def seed_everything(seed: int):

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load(model_name_or_path, cache_dir, use_vllm, use_transformer_lens, n_devices, bfloat16):
    print(f"Loading model from {model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if use_vllm:
        if bfloat16:
            llm = vllm.LLM(model_name_or_path, gpu_memory_utilization=1, max_model_len=4096,
                           trust_remote_code=True, dtype=torch.bfloat16)
        else:
            llm = vllm.LLM(model_name_or_path, gpu_memory_utilization = 1, max_model_len=4096, trust_remote_code=True)  # Initialize vLLM
    elif use_transformer_lens:
        torch.set_grad_enabled(False)
        if bfloat16:
            llm = transformer_lens.HookedTransformer.from_pretrained(model_name_or_path, n_devices = n_devices, torch_dtype=torch.bfloat16)
        else:
            llm = transformer_lens.HookedTransformer.from_pretrained(model_name_or_path, n_devices = n_devices)
    else:
        if bfloat16:
            llm = AutoModelForCausalLM.from_pretrained(model_name_or_path,device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True)
        else:
            llm = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto",
                                                       trust_remote_code=True)
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
        "--steering_type",
        type=str,
        default="sae",
        help="sae / mean_act_diff"
    )

    parser.add_argument(
        "--cot_flag",
        action="store_true",
        help="use chain of thought or not"
    )
    parser.add_argument(
        "--sae_word",
        type=str,
        default="",
        help="Word that is added after Answer: {} which is indicated by SAE"
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
        "--n_shot",
        type=int,
        default=8,
        help="number of shots"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        help="train/valid/test"
    )

    parser.add_argument(
        "--coeff",
        nargs='+',
        type=int,
        default=[700, 900],
        help="coefficient for steering vectors"
    )
    parser.add_argument(
        "--K",
        type=int,
        default=5,
        help="Top K SAE features"
    )
    parser.add_argument(
        "--NUM_SAE",
        type=int,
        default=500,
        help="Number of samples considered in inferring most important SAE features"
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
        default="./vec",
        help="inference/sae"
    )

    parser.add_argument(
        "--steer_vec_base_directory",
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
        nargs='+',
        type=int,
        help='List of integers'

    )
    parser.add_argument(
        "--omega",
        type=float,
        default=1,
        help="weight of location diff"
    )
    parser.add_argument(
        "--T",
        type=float,
        default=1,
        help="Current temperature. Higher temperature leads to more decay "
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
        "--add_instruction",
        action="store_true",
        help="add Please reason step by step in the input or not"
    )
    parser.add_argument(
        "--calculate_mean_diff",
        action="store_true",
        help="if calculate mean diff, then not infer, if not and steer vec baseline is true, then infer"
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
        help="use sae (or mean_act_diff) vectors as steering vectors"
    )
    parser.add_argument(
        "--steer_vec_baseline",
        action="store_true",
        help="use act difference vectors as steering vectors"
    )
    parser.add_argument(
        "--grid_search",
        action="store_true",
        help="grid searching coefficients for sae"
    )
    parser.add_argument(
        "--least",
        action="store_true",
        help="length constraint: at least"
    )
    parser.add_argument(
        "--most",
        action="store_true",
        help="length constraint: at most"
    )
    parser.add_argument(
        "--num_sentences",
        type=int,
        default=1,
        help="length constraint: {K} sentence(s)"
    )
    parser.add_argument(
        "--instruct_type",
        type=str,
        default="length_constraints",
        help="length constraint: {K} sentence(s)"
    )
    parser.add_argument("--load", type=str, default=None, help="load quantized model")

    args = parser.parse_args()
    return args


def generate(model, tokenizer, input_text, generate_kwargs, vllm):
    # Input text as is, no need for attention mask or input ids in vLLM
    if vllm:
        response = model.generate([input_text], generate_kwargs)

        if len(response) > 1:
            return response

        return response[0]
    else:
        input_ids = input_text.input_ids.cuda()
        attention_mask = input_text.attention_mask.cuda()

        output_ids = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs
        )

        response = []
        for i in range(output_ids.shape[0]):
            response.append(
                tokenizer.decode(
                    output_ids[i][input_ids.shape[1]:],
                    skip_special_tokens=True,
                    ignore_tokenization_space=True,
                )
            )

        if len(response) > 1:
            return response
        return response[0]


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    seed_everything(0)

    test_filepath = None
    # load ../input_data_single_instr.jsonl
    if args.dataset == "instruct_format_length":
        test_filepath = os.path.join(args.data_root, "input_data_single_instr.jsonl")
    elif args.dataset == "all_base_x_all_instructions_filtered":
        test_filepath = os.path.join(args.data_root, "all_base_x_all_instructions_filtered.jsonl")

    # prompt with no instruct, prompt with instruct, key, and type of instruction
    base, base_without_instruct, id, type = pair[args.dataset]
    list_data_dict = load_jsonl_instruct(test_filepath, base, base_without_instruct, id, type)


    # load model
    model, tokenizer = load(args.model_name_or_path, args.cache_dir, args.vllm, args.transformer_lens, args.devices, args.bfloat16)

    if args.type == "sae" or args.steer_vec_sae:
        if "9b" in args.model_name_or_path:
            sae, cfg_dict, sparsity = SAE.from_pretrained(
                release=args.sae_file,  # see other options in sae_lens/pretrained_saes.yaml
                sae_id=f"layer_{args.layer_idx}/width_16k/canonical",  # won't always be a hook point
                device="cuda:0"
            )
        else:
            sae, cfg_dict, sparsity = SAE.from_pretrained(
                release=args.sae_file,  # see other options in sae_lens/pretrained_saes.yaml
                sae_id=f"layer_{args.layer_idx}/width_16k/canonical",  # won't always be a hook point
                device="cuda:0"
            )

    cum = []
    cnt = 0

    steering_vec = None
    if args.steer_vec_baseline:
        name = args.model_name_or_path.split('/')[1] if '/' in args.model_name_or_path else None

        if "json" in args.instruct_type:
            file_name = f"{name}_mean_diff_instruct_json_format.pt"
        else:
            file_name = f"{name}_mean_diff.pt"
        steering_vec = load_tensor(args.steer_vec_base_directory, file_name)



    answers = [] if args.type == "inference" else {}


    train_data_dict, test_data_dict, valid_data_dict = filter_and_split_list(
                list_data_dict, instruct_type=args.instruct_type, dataset_type=args.dataset, random_state=args.seed)


    if args.mode == "train":
        list_data_dict = train_data_dict
    elif args.mode == "test":
        list_data_dict = test_data_dict
    elif args.mode == "valid":
        list_data_dict = valid_data_dict
    else:
        raise ValueError("Mode should be 'train' or 'test' or 'valid'.")

    for sample in tqdm(list_data_dict):
        ###### if test, then sample["type"][0], else: sample["type"]
        if args.dataset == "instruct_format_length":
            _type = sample["type"][0]
        else:
            _type = sample["type"]
        if args.instruct_type not in _type:
            continue

        ####### build prompt
        prompt_without_instruct, prompt = build_prompt(sample["prompt_without_instruct"], sample["prompt"],
                                                           _type, args.num_sentences, args.least, args.most,
                                                           args.model_name_or_path, args.sae_word)

        ####### apply chat template to instruction tuned models
        if "it" in args.model_name_or_path:
            chat = [
                {"role": "user", "content": prompt_without_instruct},
            ]
            prompt_without_instruct = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            chat = [
                {"role": "user", "content": prompt},
            ]
            prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        ##### set vllm hyper-params
        if args.vllm:
            sampling_params = SamplingParams(
                max_tokens=1024,
                temperature=0.05,
                top_p=1,
                stop = ["</s>", "<|im_end|>", "<|endoftext|>", "\n\nQ", "Question"],
                stop_token_ids=(
                    [151645, 151643]
                    if "qwen2" in args.model_name_or_path.lower()
                    else None
                )

            )
        else:
            if args.steer_vec_sae or args.steer_vec_baseline:
                sampling_params = dict(top_p=1, temperature=0.05, freq_penalty=0)
            else:
                sampling_params = dict(top_p=1, temperature=0, max_length=2048, do_sample=True)

        if args.type == "inference":
            input_text = prompt_without_instruct.strip(" ")
            if args.steer_vec_sae:
                if args.steering_type == "sae":
                    steering_vector = sae.W_dec[args.sae_idx[0]]
                elif args.steering_type == "mean_act_diff":
                    name = args.model_name_or_path.split('/')[1] if '/' in args.model_name_or_path else None
                    file_name = f"{name}_mean_diff.pt"
                    file_path = os.path.join(args.steer_vec_base_directory, file_name)
                    steering_vector = torch.load(file_path)

                if args.devices == 2:
                    steering_vector = steering_vector.to("cuda:1")

                sampling_kwargs = sampling_params
                steering_on = True
                eos_token_id = tokenizer.encode("Question", add_special_tokens=False)[0]
                eos_token_id_2 = tokenizer.encode("<eos>", add_special_tokens=False)[0]

                def steering_hook(resid_pre, hook, coeff1_dynamic):
                    # print(resid_pre.shape)
                    if resid_pre.shape[1] == 1:
                        return

                    if steering_on:
                        if "9b" in args.model_name_or_path and args.steering_type == "mean_act_diff":
                            resid_pre[:, :, :] = resid_pre[:, :, :].to("cuda:1")
                        tilde_pre = resid_pre[:, :, :] + args.coeff[0] * math.pow(1/ (args.omega * coeff1_dynamic()), args.T) * steering_vector

                        resid_pre[:, :, :] = tilde_pre * torch.norm(resid_pre[:, :, :], p=2) / torch.norm(tilde_pre, p=2)


                def dynamic_coeff1():
                    dynamic_coeff1.counter += 1

                    return dynamic_coeff1.counter

                dynamic_coeff1.counter = 0
                dynamic_hook = partial(steering_hook, coeff1_dynamic=dynamic_coeff1)

                def hooked_generate(prompt_batch, fwd_hooks=[], seed=None, **kwargs):
                    if seed is not None:
                        torch.manual_seed(seed)

                    tokenized = model.to_tokens(prompt_batch)
                    generated_tokens = tokenized

                    for _ in range(kwargs.get('max_new_tokens', 1024)):
                        with model.hooks(fwd_hooks=fwd_hooks):
                            result = model.generate(
                                stop_at_eos=True,
                                input=generated_tokens,
                                max_new_tokens=1,
                                do_sample=True,
                                eos_token_id=[eos_token_id, eos_token_id_2],
                                **kwargs,
                                verbose=False,
                            )

                        new_token = result[:, -1:]
                        generated_tokens = torch.cat([generated_tokens, new_token], dim=1)

                        if new_token[0].item() in [9413, 235368, 187]:
                            break

                        if "<eos>" in model.to_string(new_token):
                            break

                    return generated_tokens


                def run_generate(input_text):
                    model.reset_hooks()
                    editing_hooks = [(f"blocks.{args.layer_idx}.hook_resid_post", dynamic_hook)]
                    res = hooked_generate(
                        [input_text], editing_hooks, seed=args.seed, **sampling_kwargs
                    )

                    # Print results, removing the ugly beginning of sequence token
                    token_len = model.to_tokens(input_text).size(1)
                    answer = model.to_string(res[:, token_len:])
                    answer = "".join(answer)

                    return answer
                with torch.no_grad():
                    seed_everything(0)
                    dynamic_coeff1.counter = 0
                    model_completion = run_generate(input_text)
                gc.collect()
                torch.cuda.empty_cache()
                model_answer = clean_answer(model_completion, True, args.vllm, args.dataset, args.steer_vec_sae)

            elif args.steer_vec_baseline:
                if cnt == args.NUM_SAE:
                    break

                input_text = prompt_without_instruct.strip(" ")
                input_text_instruct = prompt.strip(" ")


                sampling_kwargs = sampling_params
                eos_token_id = tokenizer.encode("Q", add_special_tokens=False)

                cache_name = f"blocks.{args.layer_idx}.hook_resid_post"
                _, cache = model.run_with_cache(input_text,
                                                names_filter=lambda name: name == f'blocks.{args.layer_idx}.hook_resid_post')
                if args.devices == 2:
                    act_original = cache[cache_name].to("cuda:1")
                else:
                    act_original =cache[cache_name]
                _, cache = model.run_with_cache(input_text_instruct,
                                                names_filter=lambda name: name == f'blocks.{args.layer_idx}.hook_resid_post')
                if args.devices == 2:
                    act_cot = cache[cache_name].to("cuda:1")
                else:
                    act_cot = cache[cache_name]


                if args.calculate_mean_diff:
                    if cnt:
                        steering_vec += act_cot[:, -1, :] - act_original[:, -1, :]
                    else:
                        steering_vec = act_cot[:, -1, :] - act_original[:, -1, :]

                    del cache
                    del act_original
                    del act_cot
                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    continue
            else:
                input_text = prompt_without_instruct.strip(" ")
                if not args.vllm:
                    input_text = tokenizer(
                        input_text,
                        padding=False,
                        add_special_tokens=True,
                        return_tensors="pt",
                    ).to("cuda:0")
                    sampling_kwargs = GenerationConfig(**sampling_params)


                model_completion = generate(model, tokenizer, input_text, sampling_params, args.vllm)
                output_text = None
                if args.vllm:
                    output_text = model_completion.outputs[0].text

                if args.vllm:
                    model_completion = output_text
                print(f"Question: {prompt}")
                print(f"Answer: {model_completion}")
                model_answer = model_completion
                gc.collect()
                torch.cuda.empty_cache()

            if not args.calculate_mean_diff:
                if "json" in args.instruct_type:
                    is_cor = contains_valid_json(model_answer)
                    answers.append(is_cor)
                if args.debug:
                    print(f"Full input_text:\n{input_text}\n\n")
                print(
                    f'Question: {sample["prompt_without_instruct"]}\n\n'
                    f"Model Answers: {model_answer}\n\n"
                    f"Model Completion: {model_completion}\n\n"
                    f"Is correct: {is_cor}\n\n"
                )

                print(
                    f"Num of total question: {len(answers)}, "
                    f"Correct num: {sum(answers)}, "
                    f"Accuracy: {float(sum(answers))/len(answers)}."
                )
        elif args.type == "sae":
            if cnt == args.NUM_SAE:
                break
            input_text = prompt
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
                    top_k_values, top_k_indices = torch.topk(sae_acts[-1], args.K)
                    # print(top_k_indices)
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

    if args.steer_vec_baseline and args.calculate_mean_diff:
        steering_vec = steering_vec.squeeze()
        steering_vec /= cnt
        name = args.model_name_or_path.split('/')[1] if '/' in args.model_name_or_path else None
        file_name = f"{name}_mean_diff_instruct_{args.instruct_type}.pt"
        file_path = os.path.join(args.steer_vec_base_directory, file_name)
        if not os.path.exists(args.steer_vec_base_directory):
            os.makedirs(args.steer_vec_base_directory)
        torch.save(steering_vec, file_path)

    if args.cumulative:
        answers = {}
        top_k_values, top_k_indices = torch.topk(cum, cum.size(0))
        for ind, val in zip(top_k_indices, top_k_values):
            answers[ind.item()] = val.item() / args.NUM_SAE

    os.makedirs(args.output_dir, exist_ok=True)
    name = args.model_name_or_path.split('/')[1] if '/' in args.model_name_or_path else None

    if args.type == "inference":
        if args.steer_vec_sae:
            output_path = os.path.join(args.output_dir, args.dataset)
            output_path = os.path.join(output_path, "steer_vec")
            os.makedirs(output_path, exist_ok=True)  # Ensure the directory exists
            with open(os.path.join(output_path, f"scores_{name}_{args.cot_flag}_{args.sae_idx}_{args.coeff[0]}.txt"), "w") as f:
                print(
                    f"Num of total question: {len(answers)}, "
                    f"Correct num: {sum(answers)}, "
                    f"Accuracy: {float(sum(answers)) / len(answers)}.",
                    file=f,
                )
            with open(os.path.join(output_path, f"results_{name}_{args.cot_flag}_{args.sae_idx}_{args.coeff[0]}.txt"), "w") as f:
                for answer in answers:
                    print(answer, file=f)
        elif args.steer_vec_baseline:
            if not args.calculate_mean_diff:
                output_path = os.path.join(args.output_dir, args.dataset)
                output_path = os.path.join(output_path, "steer_vec")
                os.makedirs(output_path, exist_ok=True)  # Ensure the directory exists
                with open(
                        os.path.join(output_path, f"scores_{name}_{args.cot_flag}_mean_diff_steering_{args.coeff[0]}.txt"),
                        "w") as f:
                    print(
                        f"Num of total question: {len(answers)}, "
                        f"Correct num: {sum(answers)}, "
                        f"Accuracy: {float(sum(answers)) / len(answers)}.",
                        file=f,
                    )
                with open(
                        os.path.join(output_path, f"results_{name}_{args.cot_flag}_mean_diff_steering_{args.coeff[0]}.txt"),
                        "w") as f:
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
        with open(os.path.join(output_path, f"results_{name}_{args.cot_flag}_{args.K}_{args.layer_idx}.txt"), "w") as f:
            print(f"Top {args.K} SAE features")
            for index, value in Top_K:
                print(
                    f"SAE Index: {index}, Cardinality: {value}, "
                    f"Description: {extract_explanation(index, name, args.sae_id)}",
                    file=f,
                )
        print(f"Cardinality of SAE features: {len(answers)}")
        plot_SAE_barplot(answers, args.plot_num, args.cot_flag, name, output_path, args.cumulative)




if __name__ == "__main__":
    main()
