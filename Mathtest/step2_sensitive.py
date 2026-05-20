import pandas as pd
import os

print("=" * 60)
print("Step 2：敏感属性验证（数学实验）")
print("=" * 60)

# ==================== 配置 ====================
OUTPUT_DIR = "output/neurips"

# ==================== 1. 读取敏感属性 ====================
print("\n[1/2] 读取敏感属性...")

sensitive = pd.read_csv(f'{OUTPUT_DIR}/sensitive_labels.csv')
print(f"   敏感属性形状: {sensitive.shape}")
print(f"   列名: {sensitive.columns.tolist()}")

# ==================== 2. 统计分布 ====================
print("\n[2/2] 统计分布...")

print("\n敏感属性分布:")
for col in ['gender', 'region', 'money', 'display']:
    if col in sensitive.columns:
        counts = sensitive[col].value_counts().sort_index()
        print(f"  {col}: {counts.to_dict()}")


print(f"\nstudent_id 范围: {sensitive['student_id'].min()} - {sensitive['student_id'].max()}")
print(f"student_id 数量: {sensitive['student_id'].nunique()}")
