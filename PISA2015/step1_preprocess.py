import pandas as pd
import numpy as np
import os

print("=" * 60)
print("Step 1：数据预处理（PISA 2015）")
print("=" * 60)

DATA_DIR = "data"
OUTPUT_DIR = "output/neurips"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 1. 读取数据 ====================
print("\n[1/5] 读取数据...")

response_long = pd.read_csv(f'{DATA_DIR}/PISA_response.csv')
print(f"   response.csv 形状: {response_long.shape}")
print(f"   前5行:\n{response_long.head()}")

response_long.columns = ['student_id', 'item_id', 'score']

# 检查 score 分布
print(f"\n   score 分布: {response_long['score'].value_counts().to_dict()}")

# 转换 ID 为 0 索引
student_ids = response_long['student_id'].unique()
student_id_map = {old: new for new, old in enumerate(sorted(student_ids))}
response_long['student_id'] = response_long['student_id'].map(student_id_map)

item_ids = response_long['item_id'].unique()
item_id_map = {old: new for new, old in enumerate(sorted(item_ids))}
response_long['item_id'] = response_long['item_id'].map(item_id_map)

# 读取 Q 矩阵
q_matrix = pd.read_csv(f'{DATA_DIR}/PISA_q_matrix.csv', header=None).values
print(f"\n   Q矩阵形状: {q_matrix.shape}")

n_students = response_long['student_id'].nunique()
n_items = response_long['item_id'].nunique()
n_skills = q_matrix.shape[1]

print(f"\n   学生数: {n_students}")
print(f"   题目数: {n_items}")
print(f"   知识点数: {n_skills}")
print(f"   总作答记录: {len(response_long)}")

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
print(f"   已保存: {OUTPUT_DIR}/response_wide.csv")

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
print(f"   已保存: {OUTPUT_DIR}/student_knowledge.csv")

# ==================== 4. 处理敏感属性 ====================
print("\n[4/5] 处理敏感属性...")

sensitive = pd.read_csv(f'{DATA_DIR}/PISA_sensitive_labels.csv')
print(f"   敏感属性形状: {sensitive.shape}")
print(f"   列名: {sensitive.columns.tolist()}")

if 'student_id' in sensitive.columns:
    sensitive['student_id'] = sensitive['student_id'].map(student_id_map)

sensitive.to_csv(f'{OUTPUT_DIR}/sensitive_labels.csv', index=False)
print(f"   已保存: {OUTPUT_DIR}/sensitive_labels.csv")

# ==================== 5. 统计信息 ====================
print("\n[5/5] 统计信息...")

print(f"\n数据统计:")
print(f"   学生数: {n_students}")
print(f"   题目数: {n_items}")
print(f"   知识点数: {n_skills}")
print(f"   总作答次数: {len(response_long)}")
print(f"   平均每题作答数: {len(response_long) / n_items:.1f}")
print(f"   平均每学生作答数: {len(response_long) / n_students:.1f}")

print("\nStep 1 完成！")