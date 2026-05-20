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

# ==================== 配置（与对照组完全一致）====================
EMBED_DIM = 16
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_SIZE = 0.2
RANDOM_SEED = 42

print(f"\n设备: {DEVICE}")
print(f"嵌入维度: {EMBED_DIM}")
print(f"批次大小: {BATCH_SIZE}")
print(f"训练轮数: {EPOCHS}")
print(f"学习率: {LEARNING_RATE}")

# ==================== 1. 读取数据 ====================
print("\n[1/4] 读取数据...")

semantic_vectors = np.load('output/neurips/semantic_vectors.npy')
sensitive = pd.read_csv('output/neurips/sensitive_labels.csv')
response = pd.read_csv('output/neurips/response_wide.csv', header=None).values
response = (response > 0.5).astype(np.float32)

print(f"   语义向量形状: {semantic_vectors.shape}")
print(f"   作答数据形状: {response.shape}")

# ==================== 2. 数据对齐 ====================
min_rows = min(semantic_vectors.shape[0], len(sensitive), response.shape[0])
semantic_vectors = semantic_vectors[:min_rows]
sensitive = sensitive.iloc[:min_rows].reset_index(drop=True)
response = response[:min_rows]

n_students = semantic_vectors.shape[0]
n_features = semantic_vectors.shape[1]
n_items = response.shape[1]

print(f"   学生数: {n_students}")
print(f"   语义特征维度: {n_features}")
print(f"   题目数: {n_items}")

# ==================== 3. 划分数据集 ====================
print("\n[2/4] 划分训练集和测试集...")

train_idx, test_idx = train_test_split(
    range(n_students),
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=sensitive['region'].values
)

X_train = semantic_vectors[train_idx]
X_test = semantic_vectors[test_idx]
y_train = response[train_idx]
y_test = response[test_idx]
test_sensitive_df = sensitive.iloc[test_idx].reset_index(drop=True)

print(f"   训练集: {len(train_idx)} 学生")
print(f"   测试集: {len(test_idx)} 学生")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==================== 4. 定义模型（与对照组一致）====================
print("\n[3/4] 定义模型...")

class SemanticCDM(nn.Module):
    def __init__(self, n_features, n_items, embed_dim=16):
        super(SemanticCDM, self).__init__()

        # 特征编码器（与对照组的 knowledge_fc 结构一致）
        self.feature_fc = nn.Sequential(
            nn.Linear(n_features, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 题目嵌入层（与对照组一致）
        self.item_embedding = nn.Embedding(n_items, embed_dim)

        # 交互层（与对照组一致，但输入维度不同）
        # 对照组: 知识(16) + 敏感(3) + 题目(16) = 35维，拼接后 16*3=48
        # 实验组: 语义(50) + 题目(16) = 66维，拼接后 16*2=32
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


model = SemanticCDM(
    n_features=n_features,
    n_items=n_items,
    embed_dim=EMBED_DIM
).to(DEVICE)

print(f"   模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# ==================== 5. 构建训练样本 ====================
print("\n[4/5] 构建训练样本...")

train_samples = []
for student_id in range(len(X_train)):
    feat = X_train[student_id].astype(np.float32)
    for item_id in range(n_items):
        label = y_train[student_id, item_id]
        if not np.isnan(label):
            train_samples.append((feat, item_id, float(label)))

print(f"   训练样本数: {len(train_samples)}")

# ==================== 6. 训练模型 ====================
print("\n[5/6] 训练模型...")

from torch.utils.data import DataLoader, TensorDataset

features_list = []
item_list = []
label_list = []

for feat, item_id, label in train_samples:
    features_list.append(feat)
    item_list.append(item_id)
    label_list.append(label)

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

torch.save(model.state_dict(), 'output/semantic_model.pth')
print("\n 模型已保存: output/semantic_model.pth")

# ==================== 7. 快速评估 ====================
print("\n[6/6] 快速评估...")

model.eval()
predictions = []
true_labels = []

for student_id in range(min(100, len(X_test))):
    feat = torch.FloatTensor(X_test[student_id]).unsqueeze(0).to(DEVICE)
    for item_id in range(min(100, n_items)):
        label = y_test[student_id, item_id]
        if not np.isnan(label):
            with torch.no_grad():
                pred = model(feat, torch.LongTensor([item_id]).to(DEVICE))
                predictions.append(pred.item())
                true_labels.append(label)

            if len(predictions) >= 3000:
                break
    if len(predictions) >= 3000:
        break

if len(predictions) > 0:
    pred_binary = [1 if p > 0.5 else 0 for p in predictions]
    accuracy = accuracy_score(true_labels, pred_binary)
    auc = roc_auc_score(true_labels, predictions)

    print(f"\n   快速评估结果:")
    print(f"      评估样本数: {len(predictions)}")
    print(f"      准确率: {accuracy:.4f}")
    print(f"      AUC: {auc:.4f}")
