# 数据预处理（宽格式 + DINA）

import pandas as pd
import numpy as np
import os

if os.path.exists('output/neurips/response_wide.csv'):
    os.remove('output/neurips/response_wide.csv')
    print("   已删除旧文件")

print("=" * 60)
print("Step 1：数据预处理")
print("=" * 60)

# ==================== 处理 NeurIPS2020 ====================
print("\n[1/2] 处理 NeurIPS2020 (训练集)")

response_long = pd.read_csv('data/neurips/response.csv', header=None)
response_long.columns = ['student_id', 'item_id', 'score']

q_matrix = pd.read_csv('data/neurips/q_matrix.csv', header=None).values

n_students = response_long['student_id'].nunique()
n_items = response_long['item_id'].nunique()
n_skills = q_matrix.shape[1]

print(f"   学生数: {n_students}, 题目数: {n_items}, 知识点数: {n_skills}")

# 转换为宽格式
R = np.full((n_students, n_items), np.nan)
for _, row in response_long.iterrows():
    R[int(row['student_id']), int(row['item_id'])] = row['score']
R = np.nan_to_num(R, nan=0)

# 保存作答矩阵
os.makedirs('output/neurips', exist_ok=True)
pd.DataFrame(R).to_csv('output/neurips/response_wide.csv', index=False)
print(f"   已保存: output/neurips/response_wide.csv")

# 计算知识掌握概率
print("   计算知识掌握概率...")
student_knowledge = np.zeros((n_students, n_skills))
for k in range(n_skills):
    items_with_k = np.where(q_matrix[:, k] == 1)[0]
    if len(items_with_k) > 0:
        student_knowledge[:, k] = R[:, items_with_k].mean(axis=1)
    else:
        student_knowledge[:, k] = 0.5

pd.DataFrame(student_knowledge).to_csv('output/neurips/student_knowledge.csv', index=False)
print(f"   已保存: output/neurips/student_knowledge.csv")

# ==================== 处理 Assist0910 ====================
print("\n[2/2] 处理 Assist0910 (备选集)")

response_long = pd.read_csv('data/assist/response.csv', header=None)
response_long.columns = ['student_id', 'item_id', 'score']

q_matrix = pd.read_csv('data/assist/q_matrix.csv', header=None).values

n_students = response_long['student_id'].nunique()
n_items = response_long['item_id'].nunique()
n_skills = q_matrix.shape[1]

print(f"   学生数: {n_students}, 题目数: {n_items}, 知识点数: {n_skills}")

R = np.full((n_students, n_items), np.nan)
for _, row in response_long.iterrows():
    R[int(row['student_id']), int(row['item_id'])] = row['score']
R = np.nan_to_num(R, nan=0)

pd.DataFrame(R).to_csv('output/assist/response_wide.csv', index=False)
print(f"   已保存: output/assist/response_wide.csv")

print("   计算知识掌握概率...")
student_knowledge = np.zeros((n_students, n_skills))
for k in range(n_skills):
    items_with_k = np.where(q_matrix[:, k] == 1)[0]
    if len(items_with_k) > 0:
        student_knowledge[:, k] = R[:, items_with_k].mean(axis=1)
    else:
        student_knowledge[:, k] = 0.5

pd.DataFrame(student_knowledge).to_csv('output/assist/student_knowledge.csv', index=False)
print(f"   已保存: output/assist/student_knowledge.csv")

print("\nStep 1 完成！")