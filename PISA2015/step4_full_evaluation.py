import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import time
import warnings

warnings.filterwarnings('ignore')

print("=" * 60)
print("Step 4：对照组完整评估（PISA 2012）")
print("=" * 60)

# ==================== 配置 ====================
EMBED_DIM = 64
BATCH_SIZE = 1024
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 42

print(f"\n设备: {DEVICE}")
print(f"嵌入维度: {EMBED_DIM}")

# ==================== 1. 定义模型 ====================
print("\n[1/5] 导入模型...")


class NeuralCDM(nn.Module):
    def __init__(self, n_skills, n_sensitive, n_items, embed_dim=32):
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
sensitive = pd.read_csv('output/neurips/sensitive_labels_clean.csv')
response = pd.read_csv('output/neurips/response_wide.csv', header=None).values

min_rows = min(knowledge.shape[0], len(sensitive), response.shape[0])
knowledge = knowledge[:min_rows]
sensitive = sensitive.iloc[:min_rows].reset_index(drop=True)
response = response[:min_rows]

n_students, n_skills = knowledge.shape
n_items = response.shape[1]

print(f"   总学生数: {n_students}")
print(f"   总题目数: {n_items}")
print(f"   知识点数: {n_skills}")

# ==================== 3. 准备测试数据 ====================
print("\n[3/5] 准备测试数据...")

sensitive_cols = ['gender', 'immigrant', 'region', 'school_type']
sensitive_values = sensitive[sensitive_cols].values.astype(np.float32)

for i, col in enumerate(sensitive_cols):
    if col in ['region', 'school_type']:
        le = LabelEncoder()
        sensitive_values[:, i] = le.fit_transform(sensitive[col].values)

scaler = StandardScaler()
sensitive_values = scaler.fit_transform(sensitive_values)

train_idx, test_idx = train_test_split(
    range(n_students),
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=sensitive['gender'].values
)

test_knowledge = knowledge[test_idx]
test_sensitive = sensitive_values[test_idx]
test_response = response[test_idx]
test_sensitive_df = sensitive.iloc[test_idx].reset_index(drop=True)

print(f"   测试集学生数: {len(test_idx)}")

# ==================== 4. 加载模型 ====================
print("\n[4/5] 加载模型...")

model = NeuralCDM(
    n_skills=n_skills,
    n_sensitive=len(sensitive_cols),
    n_items=n_items,
    embed_dim=EMBED_DIM
).to(DEVICE)

model.load_state_dict(torch.load('output/neuralcdm_model.pth', map_location=DEVICE))
model.eval()
print("   模型加载成功")

# ==================== 5. 批量预测 ====================
print("\n[5/6] 批量预测...")

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

all_preds = []
all_labels = []
all_student_ids = []

start = time.time()
for i in range(0, len(test_samples), BATCH_SIZE):
    batch = test_samples[i:i + BATCH_SIZE]
    batch_k = torch.FloatTensor(np.array([s['knowledge'] for s in batch])).to(DEVICE)
    batch_s = torch.FloatTensor(np.array([s['sensitive'] for s in batch])).to(DEVICE)
    batch_i = torch.LongTensor([s['item_id'] for s in batch]).to(DEVICE)

    with torch.no_grad():
        preds = model(batch_k, batch_s, batch_i)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend([s['label'] for s in batch])
        all_student_ids.extend([s['student_id'] for s in batch])

print(f"   预测完成 耗时: {time.time() - start:.1f} 秒")

# ==================== 6. 整体性能 ====================
print("\n[6/7] 整体性能...")

pred_binary = [1 if p > 0.5 else 0 for p in all_preds]
overall_acc = accuracy_score(all_labels, pred_binary)
overall_auc = roc_auc_score(all_labels, all_preds)

print(f"\n   整体准确率: {overall_acc:.4f}")
print(f"   整体 AUC: {overall_auc:.4f}")

# ==================== 7. 各群体准确率 ====================
print("\n[7/7] 各群体准确率...")

student_acc = {}
for i, sid in enumerate(all_student_ids):
    if sid not in student_acc:
        student_acc[sid] = {'correct': 0, 'total': 0}
    student_acc[sid]['correct'] += (1 if all_preds[i] > 0.5 else 0) == all_labels[i]
    student_acc[sid]['total'] += 1

student_acc_mean = {sid: d['correct'] / d['total'] for sid, d in student_acc.items()}

test_sensitive_df['accuracy'] = test_sensitive_df.index.map(lambda x: student_acc_mean.get(x, 0))

print("\n【性别】")
for gid, gname in [(0, '女'), (1, '男')]:
    mask = test_sensitive_df['gender'] == gid
    if mask.sum() > 0:
        print(f"   {gname}: {test_sensitive_df[mask]['accuracy'].mean():.4f} (n={mask.sum()})")

print("\n【移民背景】")
immigrant_names = {0: '本地', 1: '第一代', 2: '第二代'}
for iid, iname in immigrant_names.items():
    mask = test_sensitive_df['immigrant'] == iid
    if mask.sum() > 0:
        print(f"   {iname}: {test_sensitive_df[mask]['accuracy'].mean():.4f} (n={mask.sum()})")

print("\n【地区】")
region_names = {1: '地区1', 2: '地区2', 3: '地区3', 4: '地区4', 5: '地区5'}
for rid, rname in region_names.items():
    mask = test_sensitive_df['region'] == rid
    if mask.sum() > 0:
        print(f"   {rname}: {test_sensitive_df[mask]['accuracy'].mean():.4f} (n={mask.sum()})")

print("\n【学校类型】")
school_names = {1: '公立', 2: '私立', 3: '其他'}
for tid, tname in school_names.items():
    mask = test_sensitive_df['school_type'] == tid
    if mask.sum() > 0:
        print(f"   {tname}: {test_sensitive_df[mask]['accuracy'].mean():.4f} (n={mask.sum()})")

# ==================== 8. 公平性差异 ====================
print("\n" + "=" * 60)
print("公平性差异分析")
print("=" * 60)

gender_acc = [test_sensitive_df[test_sensitive_df['gender'] == gid]['accuracy'].mean() for gid in [0, 1]]
gender_gap = abs(gender_acc[0] - gender_acc[1])

immigrant_acc = [test_sensitive_df[test_sensitive_df['immigrant'] == iid]['accuracy'].mean() for iid in [0, 1, 2]]
immigrant_gap = max(immigrant_acc) - min(immigrant_acc)

region_acc = [test_sensitive_df[test_sensitive_df['region'] == rid]['accuracy'].mean() for rid in [1, 2, 3, 4, 5]]
region_gap = max(region_acc) - min(region_acc)

school_acc = [test_sensitive_df[test_sensitive_df['school_type'] == tid]['accuracy'].mean() for tid in [1, 2, 3]]
school_gap = max(school_acc) - min(school_acc)

print(f"\n   性别差异: {gender_gap:.4f} ({gender_gap * 100:.2f}%)")
print(f"   移民背景差异: {immigrant_gap:.4f} ({immigrant_gap * 100:.2f}%)")
print(f"   地区差异: {region_gap:.4f} ({region_gap * 100:.2f}%)")
print(f"   学校类型差异: {school_gap:.4f} ({school_gap * 100:.2f}%)")

print("\nStep 4 完成！")