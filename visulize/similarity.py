from sae_lens import SAE
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns


def plot_similarity_matrix(tensor_list, name, title='Similarity Matrix', id=None):
    # assert len(tensor_list) == 5, "输入必须是包含5个Tensor的list"

    # 确保所有tensor是1D向量，如果是2D的（例如[1, hidden_size]），就 squeeze 一下
    tensor_list = [t.squeeze() for t in tensor_list]

    # 计算相似度矩阵
    similarity_matrix = torch.zeros((len(tensor_list), len(tensor_list)))
    for i in range(len(tensor_list)):
        for j in range(len(tensor_list)):
            similarity_matrix[i][j] = F.cosine_similarity(tensor_list[i], tensor_list[j], dim=0)

    # 转成numpy数组用于可视化
    similarity_matrix = similarity_matrix.detach().numpy()

    # 画图
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', xticklabels=id,
                yticklabels=id)
    plt.title(title)
    plt.savefig(name)

def single_pair_similarity(tensor_1, tensor_2):
    tensor_1, tensor_2 = tensor_1.squeeze(), tensor_2.squeeze()
    print(tensor_1.shape, tensor_2.shape)
    return F.cosine_similarity(tensor_1, tensor_2, dim=0)


if __name__ == '__main__':

    # 加载两个SAE
    sae_1, cfg_dict, _ = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res-canonical", sae_id=f"layer_20/width_16k/canonical", device="cuda"
    )
    sae_2, cfg_dict, _ = SAE.from_pretrained(
        release="gemma-scope-9b-it-res-canonical", sae_id=f"layer_31/width_16k/canonical", device="cuda"
    )
    # id_list1 = [6631, 15153, 1858, 1642, 2899]
    # id_list2 = [7007, 1126, 9005, 6782, 12946]
    # # 获取指定位置的解码向量
    # gemma2_2b = [sae_1.W_dec[i] for i in id_list1]
    # gemma2_9b = [sae_2.W_dec[i] for i in id_list2]
    #
    # # 画两个模型的相似度矩阵
    # plot_similarity_matrix(gemma2_2b, name="C:\\Users\\嘉庆\\Desktop\\2b_sim.png", title='Gemma2-2B Top5 SAE Similarity Matrix', id=id_list1)
    # plot_similarity_matrix(gemma2_9b, name="C:\\Users\\嘉庆\\Desktop\\9b_sim.png", title='Gemma2-9B Top5 SAE Similarity Matrix', id=id_list2)
    #### Gemma-2-2b
    # MeanActDiff_2B = torch.load("E:\SAE_Math\mean_vec\gemma-2-2b_mean_diff_500.pt").detach().cpu()
    # SAE_2B = sae_1.W_dec[15153].detach().cpu()
    #### Gemma-2-9b
    # MeanActDiff_9B = torch.load("E:\SAE_Math\mean_vec\gemma-2-9b_mean_diff.pt", map_location="cuda").detach().cpu()
    # SAE_9B = sae_2.W_dec[6782].detach().cpu()
    # id_list = [5340, 16364, 14689, 12251, 4427, 31, 15526, 16267, 14869, 3579, 6536, 2967]
    # gemma2_9b_it = [sae_2.W_dec[i] for i in id_list]
    # plot_similarity_matrix(gemma2_9b_it, name="C:\\Users\\嘉庆\\Desktop\\9b_it_sim.png", title='Gemma2-9B-it SAE Similarity Matrix', id=id_list)


    # plot_similarity_matrix(gemma2_9b, name="C:\\Users\\嘉庆\\Desktop\\9b_sim.png", title='Gemma2-9B Top5 SAE Similarity Matrix', id=id_list2)

    # print(single_pair_similarity(MeanActDiff_2B, SAE_2B))
    # print(single_pair_similarity(MeanActDiff_9B, SAE_9B))

    ### Gemma-2-9b-it
    MeanActDiff_9B_it = torch.load("E:\SAE_Math\mean_vec\instruct\gemma-2-9b-it_mean_diff_instruct_response_language_vi.pt", map_location="cuda").detach().cpu()
    SAE_9B_it = sae_2.W_dec[3579].detach().cpu()
    print(single_pair_similarity(MeanActDiff_9B_it, SAE_9B_it))