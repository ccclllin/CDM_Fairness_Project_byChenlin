# 注入偏见 模型学习差异

import pandas as pd
import numpy as np
import shutil
import random

# 固定随机种子，确保结果可复现
np.random.seed(42)
random.seed(42)

print("=" * 60)
print("多属性偏见注入")
print("=" * 60)

# ==================== 配置 ====================
BIAS_CONFIG = {
    'money': {
        'weak_groups': [0, 1],
        'strength': 0.70,
        'weight': 1.2
    },
    'region': {
        'weak_groups': [2],
        'strength': 0.70,
        'weight': 1.2
    },
    'age': {
        'weak_groups': [12, 13],
        'strength': 0.70,
        'weight': 1.2
    },
    'gender': {
        'weak_groups': [0],  # 女
        'strength': 0.70,
        'weight': 1.2
    }
}

USE_CUMULATIVE = True
BOOST_STRONG = True  # 是否反向加强优势群体
BOOST_FACTOR = 1.30  # 优势群体能力提高30%

print("\n偏见配置:")
for attr, config in BIAS_CONFIG.items():
    print(f"  {attr}: 弱势={config['weak_groups']}, 强度={config['strength']}, 权重={config['weight']}")
print(f"  累积效应: {'是' if USE_CUMULATIVE else '否'}")
print(f"  反向加强: {'是' if BOOST_STRONG else '否'}, 加强系数={BOOST_FACTOR}")

# ==================== 加载数据 ====================
print("\n[1/3] 加载数据...")

train_knowledge = pd.read_csv('output/neurips/student_knowledge.csv', header=None).values
train_sensitive = pd.read_csv('output/neurips/sensitive_labels.csv')

print(f"知识掌握概率行数: {train_knowledge.shape[0]}")
print(f"敏感属性行数: {train_sensitive.shape[0]}")

# ==================== 对齐数据 ====================
print("\n[2/3] 对齐数据...")

min_rows = min(train_knowledge.shape[0], len(train_sensitive))
train_knowledge = train_knowledge[:min_rows]
train_sensitive = train_sensitive.iloc[:min_rows].reset_index(drop=True)

print(f"对齐后行数: {train_knowledge.shape[0]}")
print(f"原始知识掌握概率均值: {train_knowledge.mean():.3f}")

# ==================== 计算偏见因子 ====================
print("\n[3/4] 计算偏见因子...")

n_students = len(train_knowledge)
bias_factor = np.ones(n_students)

for attr, config in BIAS_CONFIG.items():
    mask = train_sensitive[attr].isin(config['weak_groups']).values
    count = mask.sum()
    print(f"  {attr}: {count} 人属于弱势群体")

    if USE_CUMULATIVE:
        bias_factor[mask] *= config['strength'] ** config['weight']
    else:
        temp_factor = np.ones(n_students)
        temp_factor[mask] = config['strength']
        bias_factor = np.minimum(bias_factor, temp_factor)

bias_factor = np.clip(bias_factor, 0.4, 1.0)
print(f"\n偏见因子范围: {bias_factor.min():.3f} - {bias_factor.max():.3f}")
print(f"偏见因子均值: {bias_factor.mean():.3f}")

# ==================== 应用偏见 ====================
print("\n[4/4] 应用偏见...")

# 先降低弱势群体
train_knowledge_biased = train_knowledge * bias_factor.reshape(-1, 1)

# 反向加强：提高优势群体
if BOOST_STRONG:
    print(f"\n   反向加强优势群体（提高 {int((BOOST_FACTOR - 1) * 100)}%）...")

    for attr, config in BIAS_CONFIG.items():
        weak_mask = train_sensitive[attr].isin(config['weak_groups']).values
        strong_mask = ~weak_mask

        train_knowledge_biased[strong_mask] = train_knowledge_biased[strong_mask] * BOOST_FACTOR
        print(f"   {attr}: 优势群体能力提高 {int((BOOST_FACTOR - 1) * 100)}%")

# 裁剪到合理范围
train_knowledge_biased = np.clip(train_knowledge_biased, 0.0, 1.0)

print(f"\n整体均值: {train_knowledge.mean():.3f} → {train_knowledge_biased.mean():.3f}")

# 各群体统计
print(f"\n各群体均值对比:")
for attr, config in BIAS_CONFIG.items():
    weak_mask = train_sensitive[attr].isin(config['weak_groups']).values
    strong_mask = ~weak_mask

    weak_biased = train_knowledge_biased[weak_mask].mean()
    strong_biased = train_knowledge_biased[strong_mask].mean()
    gap = strong_biased - weak_biased
    print(f"  {attr}: 弱势={weak_biased:.3f}, 优势={strong_biased:.3f}, 差异={gap:.3f}")

# ==================== 保存 ====================
pd.DataFrame(train_knowledge_biased).to_csv('output/neurips/student_knowledge_biased.csv', index=False)
print("\n已保存: output/neurips/student_knowledge_biased.csv")

# 备份原始数据
shutil.copy('output/neurips/student_knowledge.csv', 'output/neurips/student_knowledge_original.csv')
print("已备份原始数据: output/neurips/student_knowledge_original.csv")

print("\n偏见注入完成！")