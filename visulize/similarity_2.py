import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

def plot_similarity_matrix(tensor_list, name, title='Similarity Matrix', id=None):
    tensor_list = [t.squeeze() for t in tensor_list]

    similarity_matrix = torch.zeros((len(tensor_list), len(tensor_list)))
    for i in range(len(tensor_list)):
        for j in range(len(tensor_list)):
            similarity_matrix[i][j] = F.cosine_similarity(tensor_list[i], tensor_list[j], dim=0)

    similarity_matrix = similarity_matrix.detach().numpy()

    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', xticklabels=id, yticklabels=id)
    plt.title(title)
    plt.savefig(name)
    plt.close()

# === 主逻辑 ===

folder_path = "E:\\SAE_Math\\mean_vec\\instruct"
tensor_list = []
label_list = []

for file in os.listdir(folder_path):
    if file.endswith(".pt"):
        full_path = os.path.join(folder_path, file)
        tensor = torch.load(full_path, map_location="cuda").detach().cpu()
        tensor_list.append(tensor)

        # 从文件名中提取最后一个下划线后的部分作为方法名
        method = file.split("_")[-1].replace(".pt", "")
        label_list.append(method)

# 输出路径
output_path = os.path.join(folder_path, "C:\\Users\\嘉庆\\Desktop\\similarity_matrix_mean.png")

# 画图
plot_similarity_matrix(
    tensor_list=tensor_list,
    name=output_path,
    title="Cosine Similarity of MeanActDiff Vectors",
    id=label_list
)

print(f"图像已保存到: {output_path}")
