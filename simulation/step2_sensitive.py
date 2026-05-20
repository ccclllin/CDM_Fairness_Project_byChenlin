# 生成敏感属性标签

import pandas as pd
import numpy as np
import os

print("=" * 60)
print("Step 2：准备敏感属性")
print("=" * 60)

# ==================== NeurIPS2020 ====================
print("\n[1/2] NeurIPS2020 敏感属性")

sensitive = pd.read_csv('data/neurips/NeurIPS_sensitive.csv')
print(f"   原始形状: {sensitive.shape}")
print(f"   列名: {sensitive.columns.tolist()}")

# 确保 student_id 列存在
if 'student_id' not in sensitive.columns:
    sensitive['student_id'] = range(len(sensitive))

# 按 student_id 排序
sensitive = sensitive.sort_values('student_id').reset_index(drop=True)

# 保存到 output
os.makedirs('output/neurips', exist_ok=True)
sensitive.to_csv('output/neurips/sensitive_labels.csv', index=False)
print(f"   ✓ 已保存: output/neurips/sensitive_labels.csv")

# 显示分布
print(f"\n   敏感属性分布:")
for col in ['gender', 'age', 'region', 'money']:
    if col in sensitive.columns:
        print(f"      {col}: {sensitive[col].value_counts().to_dict()}")

# ==================== Assist0910 ====================
print("\n[2/2] Assist0910 敏感属性")

sensitive = pd.read_csv('data/assist/Assist_sensitive.csv')
print(f"   原始形状: {sensitive.shape}")

if 'student_id' not in sensitive.columns:
    sensitive['student_id'] = range(len(sensitive))

sensitive = sensitive.sort_values('student_id').reset_index(drop=True)

os.makedirs('output/assist', exist_ok=True)
sensitive.to_csv('output/assist/sensitive_labels.csv', index=False)
print(f"   已保存: output/assist/sensitive_labels.csv")

print("\nStep 2 完成！")