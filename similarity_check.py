import numpy as np
import torch
from models.SAE.JumpReLU import JumpReLUSAE
from huggingface_hub import hf_hub_download

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
    parser.add_argument("--load", type=str, default=None, help="load quantized model")

    args = parser.parse_args()
    return args

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

def similarity(idx_1, idx_2):
    # Compute the dot product between the vectors
    vec1 = sae.W_dec[idx_1]
    vec2 = sae.W_dec[idx_2]
    dot_product = torch.dot(vec1, vec2)

    norm_vec1 = torch.norm(vec1, p=2)
    norm_vec2 = torch.norm(vec2, p=2)

    cos_sim = dot_product / (norm_vec1 * norm_vec2)

    print("Cosine Similarity: ", cos_sim.item())