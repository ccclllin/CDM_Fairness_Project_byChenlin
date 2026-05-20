# 语义向量 + 敏感属性训练

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

print("=" * 60)
print("实验组 Step 3-3：语义向量 + 敏感属性训练")
print("=" * 60)

# ==================== 配置 ====================
EMBED_DIM = 64
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_SIZE = 0.2

print(f"\n设备: {DEVICE}")
print(f"输入特征: 语义向量 (100维) + 敏感属性 (4维) = 104维")

# ==================== 1. 读取数据 ====================
print("\n[1/5] 读取数据...")

# 语义向量
semantic_vectors = np.load('output/neurips/semantic_vectors.npy')
print(f"   语义向量形状: {semantic_vectors.shape}")

# 敏感属性
sensitive = pd.read_csv('output/neurips/sensitive_labels.csv')
sensitive_cols = ['gender', 'age', 'region', 'money']
sensitive_values = sensitive[sensitive_cols].values.astype(np.float32)

# 作答数据
response = pd.read_csv('output/neurips/response_wide.csv', header=None).values
response = (response > 0.5).astype(np.float32)

# 对齐
min_rows = min(semantic_vectors.shape[0], len(sensitive), response.shape[0])
semantic_vectors = semantic_vectors[:min_rows]
sensitive = sensitive.iloc[:min_rows].reset_index(drop=True)
response = response[:min_rows]

n_students = semantic_vectors.shape[0]
n_items = response.shape[1]

print(f"   学生数: {n_students}")
print(f"   题目数: {n_items}")

# ==================== 2. 标准化敏感属性 ====================
print("\n[2/5] 标准化敏感属性...")

scaler_s = StandardScaler()
sensitive_values = scaler_s.fit_transform(sensitive_values)
print(f"   敏感属性维度: {len(sensitive_cols)}")

# ==================== 3. 特征融合 ====================
print("\n[3/5] 特征融合...")

# 拼接语义向量和敏感属性
features = np.concatenate([semantic_vectors, sensitive_values], axis=1)
n_features = features.shape[1]
print(f"   融合后特征维度: {n_features}")

# 划分数据集
train_idx, test_idx = train_test_split(
    range(n_students), test_size=TEST_SIZE, random_state=42
)

X_train = features[train_idx]
X_test = features[test_idx]
y_train = response[train_idx]
y_test = response[test_idx]
test_sensitive = sensitive.iloc[test_idx].reset_index(drop=True)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"   训练集: {len(train_idx)} 学生")
print(f"   测试集: {len(test_idx)} 学生")

# ==================== 4. 定义模型 ====================
print("\n[4/6] 定义模型...")


class HybridCDM(nn.Module):
    def __init__(self, n_features, n_items, embed_dim=64):
        super(HybridCDM, self).__init__()

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


model = HybridCDM(
    n_features=n_features,
    n_items=n_items,
    embed_dim=EMBED_DIM
).to(DEVICE)

print(f"   模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# ==================== 5. 构建训练样本 ====================
print("\n[5/7] 构建训练样本...")

train_samples = []
for student_id in range(len(X_train)):
    for item_id in range(n_items):
        label = y_train[student_id, item_id]
        if not np.isnan(label):
            train_samples.append((X_train[student_id], item_id, float(label)))

print(f"   训练样本数: {len(train_samples)}")

# ==================== 6. 训练模型 ====================
print("\n[6/7] 训练模型...")

from torch.utils.data import DataLoader, TensorDataset

max_samples = min(50000, len(train_samples))
np.random.shuffle(train_samples)
train_samples = train_samples[:max_samples]

features_list = [s[0] for s in train_samples]
item_list = [s[1] for s in train_samples]
label_list = [s[2] for s in train_samples]

features_tensor = torch.FloatTensor(np.array(features_list)).to(DEVICE)
item_tensor = torch.LongTensor(np.array(item_list)).to(DEVICE)
label_tensor = torch.FloatTensor(np.array(label_list)).to(DEVICE)

dataset = TensorDataset(features_tensor, item_tensor, label_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_f, batch_i, batch_label in dataloader:
        optimizer.zero_grad()
        pred = model(batch_f, batch_i)
        loss = criterion(pred, batch_label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"   Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(dataloader):.4f}")

torch.save(model.state_dict(), 'output/hybrid_model.pth')
print("\n模型已保存: output/hybrid_model.pth")

# ==================== 7. 快速评估 ====================
print("\n[7/7] 快速评估...")

model.eval()
predictions = []
true_labels = []

for student_id in range(min(100, len(X_test))):
    feat = torch.FloatTensor(X_test[student_id]).unsqueeze(0).to(DEVICE)
    for item_id in range(min(200, n_items)):
        label = y_test[student_id, item_id]
        if not np.isnan(label):
            with torch.no_grad():
                pred = model(feat, torch.LongTensor([item_id]).to(DEVICE))
                predictions.append(pred.item())
                true_labels.append(label)

            if len(predictions) >= 5000:
                break
    if len(predictions) >= 5000:
        break

if len(predictions) > 0:
    pred_binary = [1 if p > 0.5 else 0 for p in predictions]
    accuracy = accuracy_score(true_labels, pred_binary)
    auc = roc_auc_score(true_labels, predictions)
    print(f"\n   快速评估结果:")
    print(f"      准确率: {accuracy:.4f}")
    print(f"      AUC: {auc:.4f}")

print("\n实验组 Step 3-3 完成！")