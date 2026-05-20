# 评估实验组1公平性

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import time
import warnings

warnings.filterwarnings('ignore')

print("=" * 60)
print("实验组 Step 4-2：语义增强模型公平性评估")
print("=" * 60)

# ==================== 配置 ====================
BATCH_SIZE = 1024
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n设备: {DEVICE}")

# ==================== 1. 定义模型 ====================
print("\n[1/5] 定义模型...")


class SemanticCDM(nn.Module):
    def __init__(self, n_features, n_items, embed_dim=64):
        super(SemanticCDM, self).__init__()
        self.feature_fc = nn.Sequential(
            nn.Linear(n_features, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.item_embedding = nn.Embedding(n_items, embed_dim)
        self.interaction_fc = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, features, item_id):
        f_embed = self.feature_fc(features)
        i_embed = self.item_embedding(item_id)
        combined = torch.cat([f_embed, i_embed], dim=1)
        return self.interaction_fc(combined).squeeze(-1)


# ==================== 2. 读取数据 ====================
print("\n[2/5] 读取数据...")

semantic_vectors = np.load('output/neurips/semantic_vectors.npy')
sensitive = pd.read_csv('output/neurips/sensitive_labels.csv')
response = pd.read_csv('output/neurips/response_wide.csv', header=None).values

# 对齐
min_rows = min(semantic_vectors.shape[0], len(sensitive), response.shape[0])
semantic_vectors = semantic_vectors[:min_rows]
sensitive = sensitive.iloc[:min_rows].reset_index(drop=True)
response = response[:min_rows]

n_students, n_features = semantic_vectors.shape
n_items = response.shape[1]

print(f"   学生数: {n_students}")
print(f"   特征维度: {n_features}")
print(f"   题目数: {n_items}")

# ==================== 3. 划分测试集 ====================
print("\n[3/5] 准备测试数据...")

from sklearn.model_selection import train_test_split

_, test_idx = train_test_split(range(n_students), test_size=0.2, random_state=42)

X_test = semantic_vectors[test_idx]
y_test = response[test_idx]
test_sensitive = sensitive.iloc[test_idx].reset_index(drop=True)

scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

print(f"   测试集学生数: {len(test_idx)}")

# ==================== 4. 加载模型 ====================
print("\n[4/5] 加载模型...")

model = SemanticCDM(n_features=n_features, n_items=n_items).to(DEVICE)
model.load_state_dict(torch.load('output/semantic_model.pth', map_location=DEVICE))
model.eval()
print("   ✓ 模型加载成功")

# ==================== 5. 批量预测 ====================
print("\n[5/6] 批量预测...")

test_samples = []
for student_id in range(len(X_test)):
    for item_id in range(n_items):
        label = y_test[student_id, item_id]
        if not np.isnan(label):
            test_samples.append({
                'features': X_test[student_id],
                'item_id': item_id,
                'label': label,
                'student_id': student_id
            })

print(f"   总预测次数: {len(test_samples)}")

all_preds = []
all_labels = []
all_student_ids = []

start = time.time()
for i in range(0, len(test_samples), BATCH_SIZE):
    batch = test_samples[i:i + BATCH_SIZE]
    batch_f = torch.FloatTensor(np.array([s['features'] for s in batch])).to(DEVICE)
    batch_i = torch.LongTensor([s['item_id'] for s in batch]).to(DEVICE)

    with torch.no_grad():
        preds = model(batch_f, batch_i)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend([s['label'] for s in batch])
        all_student_ids.extend([s['student_id'] for s in batch])

print(f"   预测完成！耗时: {time.time() - start:.1f} 秒")

# ==================== 6. 整体性能 ====================
print("\n[6/7] 整体性能...")

pred_binary = [1 if p > 0.5 else 0 for p in all_preds]
overall_acc = accuracy_score(all_labels, pred_binary)
overall_auc = roc_auc_score(all_labels, all_preds)

print(f"\n   整体准确率: {overall_acc:.4f}")
print(f"   整体 AUC: {overall_auc:.4f}")

# ==================== 7. 各群体准确率 ====================
print("\n[7/7] 各群体准确率...")

# 计算每个学生的平均准确率
student_acc = {}
for i, sid in enumerate(all_student_ids):
    if sid not in student_acc:
        student_acc[sid] = {'correct': 0, 'total': 0}
    student_acc[sid]['correct'] += (1 if all_preds[i] > 0.5 else 0) == all_labels[i]
    student_acc[sid]['total'] += 1

student_acc_mean = {sid: d['correct'] / d['total'] for sid, d in student_acc.items()}

test_sensitive['accuracy'] = test_sensitive.index.map(lambda x: student_acc_mean.get(x, 0))

# 性别
print("\n【性别】")
for gid, gname in [(0, '女'), (1, '男')]:
    mask = test_sensitive['gender'] == gid
    if mask.sum() > 0:
        print(f"   {gname}: {test_sensitive[mask]['accuracy'].mean():.4f} (n={mask.sum()})")

# 年龄
print("\n【年龄】")
for age in range(12, 19):
    mask = test_sensitive['age'] == age
    if mask.sum() > 0:
        print(f"   {age}岁: {test_sensitive[mask]['accuracy'].mean():.4f} (n={mask.sum()})")

# 地区
print("\n【地区】")
region_names = {0: '贫困地区', 1: '一般地区', 2: '较发达地区', 3: '发达地区'}
for rid, rname in region_names.items():
    mask = test_sensitive['region'] == rid
    if mask.sum() > 0:
        print(f"   {rname}: {test_sensitive[mask]['accuracy'].mean():.4f} (n={mask.sum()})")

# 家庭经济
print("\n【家庭经济】")
money_names = {0: '贫穷', 1: '较贫穷', 2: '小康', 3: '较富裕', 4: '富裕'}
for mid, mname in money_names.items():
    mask = test_sensitive['money'] == mid
    if mask.sum() > 0:
        print(f"   {mname}: {test_sensitive[mask]['accuracy'].mean():.4f} (n={mask.sum()})")

# ==================== 8. 计算差异 ====================
print("\n" + "=" * 60)
print("公平性差异分析")
print("=" * 60)

gender_acc = [test_sensitive[test_sensitive['gender'] == gid]['accuracy'].mean() for gid in [0, 1]]
gender_gap = abs(gender_acc[0] - gender_acc[1])

age_acc = [test_sensitive[test_sensitive['age'] == age]['accuracy'].mean() for age in range(12, 19)]
age_gap = max(age_acc) - min(age_acc)

region_acc = [test_sensitive[test_sensitive['region'] == rid]['accuracy'].mean() for rid in range(4)]
region_gap = max(region_acc) - min(region_acc)

money_acc = [test_sensitive[test_sensitive['money'] == mid]['accuracy'].mean() for mid in range(5)]
money_gap = max(money_acc) - min(money_acc)

print(f"\n   性别差异: {gender_gap:.4f} ({gender_gap * 100:.2f}%)")
print(f"   年龄差异: {age_gap:.4f} ({age_gap * 100:.2f}%)")
print(f"   地区差异: {region_gap:.4f} ({region_gap * 100:.2f}%)")
print(f"   家庭经济差异: {money_gap:.4f} ({money_gap * 100:.2f}%)")

# ==================== 9. 保存报告 ====================
report = f"""
============================================================
实验组报告：语义增强模型公平性评估
============================================================

【模型配置】
- 输入特征: 语义向量 ({n_features}维)
- 模型: SemanticCDM

【模型性能】
- 整体准确率: {overall_acc:.4f}
- 整体 AUC: {overall_auc:.4f}

【公平性差异】
- 性别差异: {gender_gap:.4f} ({gender_gap * 100:.2f}%)
- 年龄差异: {age_gap:.4f} ({age_gap * 100:.2f}%)
- 地区差异: {region_gap:.4f} ({region_gap * 100:.2f}%)
- 家庭经济差异: {money_gap:.4f} ({money_gap * 100:.2f}%)

============================================================
"""

print(report)

with open('output/semantic_eval_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("\n报告已保存: output/semantic_eval_report.txt")
print("实验组 Step 4-2 完成！")