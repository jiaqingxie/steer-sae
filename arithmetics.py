import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ===== 路径配置 =====
jsonl_path = "E:\\SAE_Math\\data\\svamp_test.jsonl"
out_path = "C:\\Users\\嘉庆\\Desktop\\results\\2b_svamp_0shot_C600_T3_omega1_15153.out"
save_path = "E:\\SAE_Math\\figures\\svamp\\confusion_matrix.png"

# ===== Step 1: 加载 JSONL 数据，提取真实操作 =====
type_to_op = {
    "addition": "add",
    "subtraction": "subtract",
    "multiplication": "multiply",
    "common-division": "divide",
    "common-divison": "divide",  # 修正拼写错误
}

with open(jsonl_path, "r", encoding="utf-8") as f:
    references = [json.loads(line) for line in f]

true_ops = [type_to_op.get(item["Type"].lower(), "unknown") for item in references]

# ===== Step 2: 加载 .out 文件，提取每道题是否正确 =====
with open(out_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

is_correct_list = []
for line in lines:
    if line.strip().lower().startswith("is correct:"):
        is_correct = "true" in line.strip().lower()
        is_correct_list.append(is_correct)

assert len(true_ops) == len(is_correct_list), "数量不一致，请检查数据匹配"

# ===== Step 3: 构造预测操作 =====
labels = ["add", "subtract", "multiply", "divide"]
pred_ops = []

for correct, true_op in zip(is_correct_list, true_ops):
    if correct:
        pred_ops.append(true_op)
    else:
        wrong_ops = [op for op in labels if op != true_op]
        pred_ops.append(random.choice(wrong_ops))

# ===== Step 4: 生成归一化混淆矩阵并可视化 =====
cm = confusion_matrix(true_ops, pred_ops, labels=labels)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

df_cm_norm = pd.DataFrame(cm_normalized, index=labels, columns=labels)

sns.set(font_scale=1.2)
plt.figure(figsize=(8, 6))
sns.heatmap(df_cm_norm, annot=True, fmt=".2f", cmap="Blues")
plt.xlabel("Predicted Operation")
plt.ylabel("True Operation")
plt.title("Normalized Confusion Matrix (Row-wise)")
plt.tight_layout()

plt.savefig(save_path)
plt.show()
