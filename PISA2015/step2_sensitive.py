import pandas as pd
import os

print("=" * 60)
print("Step 2：清理敏感属性")
print("=" * 60)

OUTPUT_DIR = "output/neurips"

sensitive = pd.read_csv(f'{OUTPUT_DIR}/sensitive_labels.csv')
print(f"原始形状: {sensitive.shape}")
print(f"原始列名: {sensitive.columns.tolist()}")

# 保留需要的列
keep_cols = ['student_id', 'gender', 'immigrant_status', 'school_location', 'school_ownership']
available_cols = [col for col in keep_cols if col in sensitive.columns]
sensitive_clean = sensitive[available_cols]

print(f"\n保留的列: {available_cols}")

# 重命名
rename_map = {
    'immigrant_status': 'immigrant',
    'school_location': 'region',
    'school_ownership': 'school_type'
}
sensitive_clean = sensitive_clean.rename(columns=rename_map)

# 只保留前 2484 行对齐
if len(sensitive_clean) > 2484:
    sensitive_clean = sensitive_clean.iloc[:2484]

print("\n敏感属性分布:")
for col in sensitive_clean.columns:
    if col != 'student_id':
        print(f"   {col}: {sensitive_clean[col].value_counts().to_dict()}")

sensitive_clean.to_csv(f'{OUTPUT_DIR}/sensitive_labels_clean.csv', index=False)
print(f"\n已保存: {OUTPUT_DIR}/sensitive_labels_clean.csv")

print("\nStep 2 完成！")