import pandas as pd
import numpy as np
import os

# ==================== 配置 ====================
DATA_DIR = "data"
OUTPUT_DIR = "output/neurips"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 1. 读取数据 ====================
print("\n[1/5] 读取数据...")

response_long = pd.read_csv(f'{DATA_DIR}/HN_response.csv', skiprows=1, header=None)
response_long.columns = ['student_id', 'item_id', 'score']

if response_long['student_id'].min() == 1:
    response_long['student_id'] = response_long['student_id'] - 1
if response_long['item_id'].min() == 1:
    response_long['item_id'] = response_long['item_id'] - 1

q_matrix = pd.read_csv(f'{DATA_DIR}/HN_q_matrix.csv', header=None).values
q_matrix = q_matrix[:, 1:]
n_students = response_long['student_id'].nunique()
n_items = response_long['item_id'].nunique()
n_skills = q_matrix.shape[1]

print(f"   学生数: {n_students}")
print(f"   题目数: {n_items}")
print(f"   知识点数: {n_skills}")

# ==================== 2. 转换为宽格式 ====================
print("\n[2/5] 转换为宽格式...")

R = np.full((n_students, n_items), np.nan)
for _, row in response_long.iterrows():
    student = int(row['student_id'])
    item = int(row['item_id'])
    score = row['score']
    R[student, item] = score

R = np.nan_to_num(R, nan=0)

pd.DataFrame(R).to_csv(f'{OUTPUT_DIR}/response_wide.csv', index=False, header=False)
print(f"  已保存: {OUTPUT_DIR}/response_wide.csv")

# ==================== 3. 计算知识掌握概率 ====================
print("\n[3/5] 计算知识掌握概率...")

student_knowledge = np.zeros((n_students, n_skills))
for k in range(n_skills):
    items_with_k = np.where(q_matrix[:, k] == 1)[0]
    if len(items_with_k) > 0:
        student_knowledge[:, k] = R[:, items_with_k].mean(axis=1)
    else:
        student_knowledge[:, k] = 0.5

pd.DataFrame(student_knowledge).to_csv(f'{OUTPUT_DIR}/student_knowledge.csv', index=False, header=False)
print(f"  已保存: {OUTPUT_DIR}/student_knowledge.csv")

# ==================== 4. 处理敏感属性 ====================
print("\n[4/5] 处理敏感属性...")

sensitive = pd.read_csv(f'{DATA_DIR}/HN_sensitive_labels.csv')
print(f"   敏感属性形状: {sensitive.shape}")
print(f"   列名: {sensitive.columns.tolist()}")

if sensitive['student_id'].min() == 1:
    sensitive['student_id'] = sensitive['student_id'] - 1

sensitive.to_csv(f'{OUTPUT_DIR}/sensitive_labels.csv', index=False)
print(f"   已保存: {OUTPUT_DIR}/sensitive_labels.csv")

# ==================== 5. 统计信息 ====================
print("\n[5/5] 统计信息...")

print(f"\n敏感属性分布:")
for col in ['gender', 'region', 'money', 'display']:
    if col in sensitive.columns:
        print(f"   {col}: {sensitive[col].value_counts().to_dict()}")

print(f"\n知识掌握概率统计:")
print(f"   范围: [{student_knowledge.min():.3f}, {student_knowledge.max():.3f}]")
print(f"   均值: {student_knowledge.mean():.3f}")
