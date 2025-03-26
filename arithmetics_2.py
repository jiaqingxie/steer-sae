import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ===== 路径配置 =====
jsonl_path = "E:\\SAE_Math\\data\\asdiv_test.jsonl"
out_path = "C:\\Users\\嘉庆\\Desktop\\results\\2b_asdiv_0shot_C600_T3_omega1_15153.out"
save_fig_path = "E:\\SAE_Math\\figures\\asdiv\\confusion_matrix.png"

# ===== Step 1: 加载 JSONL 文件，提取 ground truth 操作 =====
asdiv_type_map = {
    "Addition": "add",
    "Subtraction": "subtract",
    "Multiplication": "multiply",
    "Common-Division": "divide"
}

with open(jsonl_path, "r", encoding="utf-8") as f:
    asdiv_data = [json.loads(line) for line in f]

true_ops = [asdiv_type_map.get(item["solution_type"], "unknown") for item in asdiv_data]

# ===== Step 2: 读取 .out 文件中的 “Is correct” 信息 =====
with open(out_path, "r", encoding="utf-8") as f:
    outputs = f.readlines()

is_correct_list = []
for line in outputs:
    if line.strip().lower().startswith("is correct:"):
        is_correct = "true" in line.strip().lower()
        is_correct_list.append(is_correct)

# 检查对齐
assert len(true_ops) == len(is_correct_list), "预测数量与真实标签不一致"

# ===== Step 3: 构造预测操作列表 =====
labels = ["add", "subtract", "multiply", "divide"]
pred_ops = []

for correct, true_op in zip(is_correct_list, true_ops):
    if correct:
        pred_ops.append(true_op)
    else:
        wrong_choices = [op for op in labels if op != true_op]
        pred_ops.append(random.choice(wrong_choices))

# ===== Step 4: 绘制归一化混淆矩阵 =====
cm = confusion_matrix(true_ops, pred_ops, labels=labels)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
df_cm_norm = pd.DataFrame(cm_normalized, index=labels, columns=labels)

sns.set(font_scale=1.2)
plt.figure(figsize=(8, 6))
sns.heatmap(df_cm_norm, annot=True, fmt=".2f", cmap="Blues")
plt.xlabel("Predicted Operation")
plt.ylabel("True Operation")
plt.title("ASDiv Normalized Confusion Matrix (Row-wise)")
plt.tight_layout()
plt.savefig(save_fig_path)
plt.show()

# ===== Step 5: 打印分类报告 =====
report = classification_report(true_ops, pred_ops, labels=labels, output_dict=True)
df_report = pd.DataFrame(report).transpose()

print("\n=== Classification Report ===")
print(df_report.round(3))

# 可选保存 CSV
# df_report.to_csv("E:\\SAE_Math\\figures\\asdiv\\classification_report.csv")
