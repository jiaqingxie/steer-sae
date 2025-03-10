import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import *

test_filepath = 'E:\\SAE_Math\\instruct_data\\all_base_x_all_instructions_filtered.jsonl'
dataset = "all_base_x_all_instructions_filtered"
base, base_without_instruct, id, type = pair[dataset]
instruct_type = "json_format"

list_data_dict = load_jsonl_instruct(test_filepath, base, base_without_instruct, id, type)

train_data_dict, test_data_dict, valid_data_dict = filter_and_split_list(
    list_data_dict, instruct_type=instruct_type, dataset_type=dataset, random_state=42)

print(len(train_data_dict))
print(len(test_data_dict))
print(len(valid_data_dict))
