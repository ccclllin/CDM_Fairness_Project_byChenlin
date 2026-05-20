import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
import warnings

warnings.filterwarnings('ignore')


# ==================== 配置 ====================
EMBED_DIM = 16
BATCH_SIZE = 1024
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 42

print(f"\n设备: {DEVICE}")
print(f"嵌入维度: {EMBED_DIM}")

# ==================== 1. 定义模型 ====================
print("\n[1/5] 导入模型...")

class SemanticCDM(nn.Module):
    def __init__(self, n_features, n_items, embed_dim=16):
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
response = (response > 0.5).astype(np.float32)

min_rows = min(semantic_vectors.shape[0], len(sensitive), response.shape[0])
semantic_vectors = semantic_vectors[:min_rows]
sensitive = sensitive.iloc[:min_rows].reset_index(drop=True)
response = response[:min_rows]

n_students = semantic_vectors.shape[0]
n_features = semantic_vectors.shape[1]
n_items = response.shape[1]

print(f"   学生数: {n_students}")
print(f"   特征维度: {n_features}")
print(f"   题目数: {n_items}")

# ==================== 3. 划分测试集 ====================
print("\n[3/5] 准备测试数据...")

train_idx, test_idx = train_test_split(
    range(n_students),
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=sensitive['region'].values
)

X_test = semantic_vectors[test_idx]
y_test = response[test_idx]
test_sensitive_df = sensitive.iloc[test_idx].reset_index(drop=True)

scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

print(f"   测试集学生数: {len(test_idx)}")

# ==================== 4. 加载模型 ====================
print("\n[4/5] 加载模型...")

model = SemanticCDM(n_features=n_features, n_items=n_items, embed_dim=EMBED_DIM).to(DEVICE)
model.load_state_dict(torch.load('output/semantic_model.pth', map_location=DEVICE))
model.eval()
print("   模型加载成功")

# ==================== 5. 批量预测 ====================
print("\n[5/6] 批量预测...")

test_samples = []
for student_id in range(len(X_test)):
    feat = X_test[student_id].astype(np.float32)
    for item_id in range(n_items):
        label = y_test[student_id, item_id]
        if not np.isnan(label):
            test_samples.append((feat, item_id, label, student_id))

print(f"   总预测次数: {len(test_samples)}")

all_preds = []
all_labels = []
all_student_ids = []

start = time.time()
for i in range(0, len(test_samples), BATCH_SIZE):
    batch = test_samples[i:i + BATCH_SIZE]
    batch_f = torch.FloatTensor(np.array([s[0] for s in batch])).to(DEVICE)
    batch_i = torch.LongTensor([s[1] for s in batch]).to(DEVICE)

    with torch.no_grad():
        preds = model(batch_f, batch_i)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend([s[2] for s in batch])
        all_student_ids.extend([s[3] for s in batch])

print(f"   预测完成耗时: {time.time() - start:.1f} 秒")

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

print("\n【地区】")
for rid, rname in [(0, '城市'), (1, '县域')]:
    mask = test_sensitive_df['region'] == rid
    if mask.sum() > 0:
        print(f"   {rname}: {test_sensitive_df[mask]['accuracy'].mean():.4f} (n={mask.sum()})")

print("\n【家庭经济】")
money_names = {0: '富裕', 1: '小康', 2: '一般', 3: '贫穷'}
for mid, mname in money_names.items():
    mask = test_sensitive_df['money'] == mid
    if mask.sum() > 0:
        print(f"   {mname}: {test_sensitive_df[mask]['accuracy'].mean():.4f} (n={mask.sum()})")

# ==================== 8. 公平性差异 ====================
print("\n" + "=" * 60)
print("公平性差异分析")
print("=" * 60)

gender_acc = [test_sensitive_df[test_sensitive_df['gender'] == gid]['accuracy'].mean() for gid in [0, 1]]
gender_gap = abs(gender_acc[0] - gender_acc[1])

region_acc = [test_sensitive_df[test_sensitive_df['region'] == rid]['accuracy'].mean() for rid in [0, 1]]
region_gap = abs(region_acc[0] - region_acc[1])

money_acc = [test_sensitive_df[test_sensitive_df['money'] == mid]['accuracy'].mean() for mid in range(4)]
money_gap = max(money_acc) - min(money_acc)

print(f"\n   性别差异: {gender_gap:.4f} ({gender_gap * 100:.2f}%)")
print(f"   地区差异: {region_gap:.4f} ({region_gap * 100:.2f}%)")
print(f"   家庭经济差异: {money_gap:.4f} ({money_gap * 100:.2f}%)")
