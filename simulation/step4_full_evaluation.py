# 评估对照组公平性

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import time
import warnings

warnings.filterwarnings('ignore')

print("=" * 60)
print("Step 4：完整评估（全部测试集）")
print("=" * 60)

# ==================== 配置 ====================
BATCH_SIZE = 1024
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n设备: {DEVICE}")
print(f"批次大小: {BATCH_SIZE}")

# ==================== 1. 导入模型定义 ====================
print("\n[1/5] 导入模型...")


class NeuralCDM(nn.Module):
    def __init__(self, n_skills, n_sensitive, n_items, embed_dim=64):
        super(NeuralCDM, self).__init__()

        self.knowledge_fc = nn.Sequential(
            nn.Linear(n_skills, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.sensitive_fc = nn.Sequential(
            nn.Linear(n_sensitive, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.item_embedding = nn.Embedding(n_items, embed_dim)

        self.interaction_fc = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, knowledge, sensitive, item_id):
        k_embed = self.knowledge_fc(knowledge)
        s_embed = self.sensitive_fc(sensitive)
        i_embed = self.item_embedding(item_id)
        combined = torch.cat([k_embed, s_embed, i_embed], dim=1)
        return self.interaction_fc(combined).squeeze(-1)


# ==================== 2. 读取数据 ====================
print("\n[2/5] 读取数据...")

knowledge = pd.read_csv('output/neurips/student_knowledge.csv', header=None).values
sensitive = pd.read_csv('output/neurips/sensitive_labels.csv')
response = pd.read_csv('output/neurips/response_wide.csv', header=None).values

# 对齐
min_rows = min(knowledge.shape[0], len(sensitive), response.shape[0])
knowledge = knowledge[:min_rows]
sensitive = sensitive.iloc[:min_rows].reset_index(drop=True)
response = response[:min_rows]

n_students, n_skills = knowledge.shape
n_items = response.shape[1]

print(f"   总学生数: {n_students}")
print(f"   总题目数: {n_items}")
print(f"   知识点数: {n_skills}")

# ==================== 3. 准备训练/测试划分 ====================
print("\n[3/5] 准备训练/测试数据...")

from sklearn.model_selection import train_test_split

sensitive_cols = ['gender', 'age', 'region', 'money']
sensitive_values = sensitive[sensitive_cols].values.astype(np.float32)

scaler = StandardScaler()
sensitive_values = scaler.fit_transform(sensitive_values)

# 使用与训练时相同的划分
train_idx, test_idx = train_test_split(
    range(n_students),
    test_size=0.2,
    random_state=42
)

test_knowledge = knowledge[test_idx]
test_sensitive = sensitive_values[test_idx]
test_response = response[test_idx]
test_sensitive_df = sensitive.iloc[test_idx].reset_index(drop=True)

print(f"   测试集学生数: {len(test_idx)}")
print(f"   测试集题目数: {n_items}")

# ==================== 4. 加载模型 ====================
print("\n[4/5] 加载模型...")

model = NeuralCDM(
    n_skills=n_skills,
    n_sensitive=len(sensitive_cols),
    n_items=n_items,
    embed_dim=64
).to(DEVICE)

model.load_state_dict(torch.load('output/neuralcdm_model.pth', map_location=DEVICE))
model.eval()
print("   ✓ 模型加载成功")

# ==================== 5. 批量预测 ====================
print("\n[5/6] 批量预测（完整测试集）...")

# 构建所有测试样本
test_samples = []
for student_id in range(len(test_knowledge)):
    for item_id in range(n_items):
        label = test_response[student_id, item_id]
        if not np.isnan(label):
            test_samples.append({
                'knowledge': test_knowledge[student_id],
                'sensitive': test_sensitive[student_id],
                'item_id': item_id,
                'label': label,
                'student_id': student_id
            })

print(f"   总预测次数: {len(test_samples)}")

# 分批预测
all_predictions = []
all_labels = []
all_student_ids = []

start_time = time.time()

for i in range(0, len(test_samples), BATCH_SIZE):
    batch = test_samples[i:i + BATCH_SIZE]

    batch_k = torch.FloatTensor(np.array([s['knowledge'] for s in batch])).to(DEVICE)
    batch_s = torch.FloatTensor(np.array([s['sensitive'] for s in batch])).to(DEVICE)
    batch_i = torch.LongTensor([s['item_id'] for s in batch]).to(DEVICE)

    with torch.no_grad():
        batch_pred = model(batch_k, batch_s, batch_i)
        all_predictions.extend(batch_pred.cpu().numpy())
        all_labels.extend([s['label'] for s in batch])
        all_student_ids.extend([s['student_id'] for s in batch])

elapsed = time.time() - start_time
print(f"\n   预测完成！耗时: {elapsed:.1f} 秒 ({elapsed / 60:.1f} 分钟)")

# ==================== 6. 整体性能 ====================
print("\n[6/7] 整体性能...")

predictions_binary = [1 if p > 0.5 else 0 for p in all_predictions]
overall_accuracy = accuracy_score(all_labels, predictions_binary)
overall_auc = roc_auc_score(all_labels, all_predictions)

print(f"\n   整体准确率: {overall_accuracy:.4f}")
print(f"   整体 AUC: {overall_auc:.4f}")

# ==================== 7. 按敏感属性分组评估 ====================
print("\n[7/7] 按敏感属性分组评估...")

# 创建学生预测汇总
student_predictions = {}
for i, student_id in enumerate(all_student_ids):
    if student_id not in student_predictions:
        student_predictions[student_id] = {'pred': [], 'true': []}
    student_predictions[student_id]['pred'].append(all_predictions[i])
    student_predictions[student_id]['true'].append(all_labels[i])

# 计算每个学生的准确率
student_accuracy = {}
for student_id, data in student_predictions.items():
    pred_binary = [1 if p > 0.5 else 0 for p in data['pred']]
    acc = accuracy_score(data['true'], pred_binary)
    student_accuracy[student_id] = acc

# 获取测试集的学生信息
test_student_info = test_sensitive_df.copy()
test_student_info['accuracy'] = test_student_info.index.map(
    lambda x: student_accuracy.get(x, 0)
)

# 定义分组
gender_groups = {0: '女', 1: '男'}
region_groups = {0: '贫困地区', 1: '一般地区', 2: '较发达地区', 3: '发达地区'}
money_groups = {0: '贫穷', 1: '较贫穷', 2: '小康', 3: '较富裕', 4: '富裕'}

# 年龄分组
age_groups = {
    '12岁': [12],
    '13岁': [13],
    '14岁': [14],
    '15岁': [15],
    '16岁': [16],
    '17岁': [17],
    '18岁': [18]
}

# 计算各群体准确率
print("\n【性别】")
for gid, gname in gender_groups.items():
    mask = test_student_info['gender'] == gid
    if mask.sum() > 0:
        acc = test_student_info[mask]['accuracy'].mean()
        print(f"   {gname}: {acc:.4f} (n={mask.sum()})")

print("\n【年龄】")
for age_group, ages in age_groups.items():
    mask = test_student_info['age'].isin(ages)
    if mask.sum() > 0:
        acc = test_student_info[mask]['accuracy'].mean()
        print(f"   {age_group}: {acc:.4f} (n={mask.sum()})")

print("\n【地区】")
for rid, rname in region_groups.items():
    mask = test_student_info['region'] == rid
    if mask.sum() > 0:
        acc = test_student_info[mask]['accuracy'].mean()
        print(f"   {rname}: {acc:.4f} (n={mask.sum()})")

print("\n【家庭经济】")
for mid, mname in money_groups.items():
    mask = test_student_info['money'] == mid
    if mask.sum() > 0:
        acc = test_student_info[mask]['accuracy'].mean()
        print(f"   {mname}: {acc:.4f} (n={mask.sum()})")

# ==================== 8. 计算差异 ====================
print("\n" + "=" * 60)
print("公平性差异分析")
print("=" * 60)

# 性别差异
gender_accs = []
for gid in gender_groups.keys():
    mask = test_student_info['gender'] == gid
    if mask.sum() > 0:
        gender_accs.append(test_student_info[mask]['accuracy'].mean())
gender_gap = max(gender_accs) - min(gender_accs) if len(gender_accs) > 1 else 0

# 年龄差异（每个年龄一组）
age_accs = []
for ages in age_groups.values():
    mask = test_student_info['age'].isin(ages)
    if mask.sum() > 0:
        age_accs.append(test_student_info[mask]['accuracy'].mean())
age_gap = max(age_accs) - min(age_accs) if len(age_accs) > 1 else 0

# 地区差异
region_accs = []
for rid in region_groups.keys():
    mask = test_student_info['region'] == rid
    if mask.sum() > 0:
        region_accs.append(test_student_info[mask]['accuracy'].mean())
region_gap = max(region_accs) - min(region_accs) if len(region_accs) > 1 else 0

# 家庭经济差异
money_accs = []
for mid in money_groups.keys():
    mask = test_student_info['money'] == mid
    if mask.sum() > 0:
        money_accs.append(test_student_info[mask]['accuracy'].mean())
money_gap = max(money_accs) - min(money_accs) if len(money_accs) > 1 else 0

print(f"\n   性别差异: {gender_gap:.4f} ({gender_gap * 100:.2f}%)")
print(f"   年龄差异: {age_gap:.4f} ({age_gap * 100:.2f}%)")
print(f"   地区差异: {region_gap:.4f} ({region_gap * 100:.2f}%)")
print(f"   家庭经济差异: {money_gap:.4f} ({money_gap * 100:.2f}%)")

# ==================== 9. 保存报告 ====================
report = f"""
============================================================
Step 4：完整评估报告
============================================================

【评估规模】
- 测试集学生数: {len(test_idx)}
- 测试集题目数: {n_items}
- 总预测次数: {len(test_samples)}

【模型性能】
- 整体准确率: {overall_accuracy:.4f}
- 整体 AUC: {overall_auc:.4f}

【各群体准确率】
性别:
- 女: {test_student_info[test_student_info['gender'] == 0]['accuracy'].mean() if (test_student_info['gender'] == 0).sum() > 0 else 0:.4f}
- 男: {test_student_info[test_student_info['gender'] == 1]['accuracy'].mean() if (test_student_info['gender'] == 1).sum() > 0 else 0:.4f}

年龄:
"""

for age_group, ages in age_groups.items():
    mask = test_student_info['age'].isin(ages)
    if mask.sum() > 0:
        report += f"- {age_group}: {test_student_info[mask]['accuracy'].mean():.4f} (n={mask.sum()})\n"

report += f"""
地区:
- 贫困地区: {test_student_info[test_student_info['region'] == 0]['accuracy'].mean() if (test_student_info['region'] == 0).sum() > 0 else 0:.4f}
- 一般地区: {test_student_info[test_student_info['region'] == 1]['accuracy'].mean() if (test_student_info['region'] == 1).sum() > 0 else 0:.4f}
- 较发达地区: {test_student_info[test_student_info['region'] == 2]['accuracy'].mean() if (test_student_info['region'] == 2).sum() > 0 else 0:.4f}
- 发达地区: {test_student_info[test_student_info['region'] == 3]['accuracy'].mean() if (test_student_info['region'] == 3).sum() > 0 else 0:.4f}

家庭经济:
- 贫穷: {test_student_info[test_student_info['money'] == 0]['accuracy'].mean() if (test_student_info['money'] == 0).sum() > 0 else 0:.4f}
- 较贫穷: {test_student_info[test_student_info['money'] == 1]['accuracy'].mean() if (test_student_info['money'] == 1).sum() > 0 else 0:.4f}
- 小康: {test_student_info[test_student_info['money'] == 2]['accuracy'].mean() if (test_student_info['money'] == 2).sum() > 0 else 0:.4f}
- 较富裕: {test_student_info[test_student_info['money'] == 3]['accuracy'].mean() if (test_student_info['money'] == 3).sum() > 0 else 0:.4f}
- 富裕: {test_student_info[test_student_info['money'] == 4]['accuracy'].mean() if (test_student_info['money'] == 4).sum() > 0 else 0:.4f}

【公平性差异】
- 性别差异: {gender_gap:.4f} ({gender_gap * 100:.2f}%)
- 年龄差异: {age_gap:.4f} ({age_gap * 100:.2f}%)
- 地区差异: {region_gap:.4f} ({region_gap * 100:.2f}%)
- 家庭经济差异: {money_gap:.4f} ({money_gap * 100:.2f}%)

============================================================
"""

print(report)

with open('output/step4_full_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("\n报告已保存: output/step4_full_report.txt")
print("Step 4 完整评估完成！")