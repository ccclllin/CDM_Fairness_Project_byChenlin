# 用知识+敏感属性训练模型

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
print("Step 3：NeuralCDM with Sensitive Attributes")
print("=" * 60)

# ==================== 配置 ====================
EMBED_DIM = 64
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_SIZE = 0.2

print(f"\n设备: {DEVICE}")
print(f"使用数据: student_knowledge_biased.csv（已注入偏见）")

# ==================== 1. 读取数据 ====================
print("\n[1/4] 读取 NeurIPS2020 数据...")

train_knowledge = pd.read_csv('output/neurips/student_knowledge_biased.csv', header=None).values
train_sensitive = pd.read_csv('output/neurips/sensitive_labels.csv')
train_response = pd.read_csv('output/neurips/response_wide.csv', header=None).values

print(f"   知识形状: {train_knowledge.shape}")
print(f"   敏感属性形状: {train_sensitive.shape}")
print(f"   作答形状: {train_response.shape}")

# ==================== 2. 数据对齐 ==================== #确保三个数据文件的行数一致=4918
print("\n[2/4] 数据对齐...")

min_rows = min(train_knowledge.shape[0], len(train_sensitive), train_response.shape[0])
train_knowledge = train_knowledge[:min_rows]
train_sensitive = train_sensitive.iloc[:min_rows].reset_index(drop=True)
train_response = train_response[:min_rows]

n_students, n_skills = train_knowledge.shape
n_items = train_response.shape[1]

print(f"   对齐后学生数: {n_students}")
print(f"   题目数: {n_items}")
print(f"   知识点数: {n_skills}")

# ==================== 3. 准备敏感属性 ====================
print("\n[3/4] 准备敏感属性...")

sensitive_cols = ['gender', 'age', 'region', 'money']
train_sensitive_values = train_sensitive[sensitive_cols].values.astype(np.float32)

scaler = StandardScaler()
train_sensitive_values = scaler.fit_transform(train_sensitive_values)

print(f"   敏感属性列: {sensitive_cols}")

# ==================== 4. 划分训练集和测试集 ====================
print("\n[4/5] 划分训练集和测试集...")

train_idx, test_idx = train_test_split(
    range(n_students),
    test_size=TEST_SIZE,
    random_state=42
)

train_knowledge_data = train_knowledge[train_idx]
train_sensitive_data = train_sensitive_values[train_idx]
train_response_data = train_response[train_idx]

test_knowledge = train_knowledge[test_idx]
test_sensitive = train_sensitive_values[test_idx]
test_response = train_response[test_idx]

print(f"   训练集: {len(train_idx)} 学生")
print(f"   测试集: {len(test_idx)} 学生")

# ==================== 5. 构建训练样本 ====================
print("\n[5/6] 构建训练样本...")

train_samples = []
for student_id in range(len(train_knowledge_data)):
    student_k = train_knowledge_data[student_id].astype(np.float32)
    student_s = train_sensitive_data[student_id]
    for item_id in range(n_items):
        label = train_response_data[student_id, item_id]
        if not np.isnan(label):
            train_samples.append({
                'knowledge': student_k,
                'sensitive': student_s,
                'item_id': item_id,
                'label': float(label)
            })

print(f"   训练样本数: {len(train_samples)}")

# ==================== 6. 定义模型 ====================
print("\n[6/7] 定义 NeuralCDM...")


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


model = NeuralCDM(
    n_skills=n_skills,
    n_sensitive=len(sensitive_cols),
    n_items=n_items,
    embed_dim=EMBED_DIM
).to(DEVICE)

print(f"   模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# ==================== 7. 训练模型 ====================
print("\n[7/8] 训练模型...")

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()

from torch.utils.data import DataLoader, TensorDataset

max_samples = min(50000, len(train_samples))
train_samples = train_samples[:max_samples]

knowledge_list = []
sensitive_list = []
item_list = []
label_list = []

for sample in train_samples:
    knowledge_list.append(sample['knowledge'])
    sensitive_list.append(sample['sensitive'])
    item_list.append(sample['item_id'])
    label_list.append(sample['label'])

knowledge_tensor = torch.FloatTensor(np.array(knowledge_list)).to(DEVICE)
sensitive_tensor = torch.FloatTensor(np.array(sensitive_list)).to(DEVICE)
item_tensor = torch.LongTensor(np.array(item_list)).to(DEVICE)
label_tensor = torch.FloatTensor(np.array(label_list)).to(DEVICE)

dataset = TensorDataset(knowledge_tensor, sensitive_tensor, item_tensor, label_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_k, batch_s, batch_i, batch_label in dataloader:
        optimizer.zero_grad()
        pred = model(batch_k, batch_s, batch_i)
        loss = criterion(pred, batch_label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"   Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(dataloader):.4f}")

torch.save(model.state_dict(), 'output/neuralcdm_model.pth')
print("\n模型已保存: output/neuralcdm_model.pth")

# ==================== 8. 评估模型 ====================
print("\n[8/8] 评估模型...")

model.eval()
predictions = []
true_labels = []

max_eval_students = min(100, len(test_knowledge))
max_eval_items = min(200, n_items)

for student_id in range(max_eval_students):
    student_k = torch.FloatTensor(test_knowledge[student_id].astype(np.float32)).to(DEVICE)
    if student_k.dim() == 1:
        student_k = student_k.unsqueeze(0)

    student_s = torch.FloatTensor(test_sensitive[student_id]).to(DEVICE)
    if student_s.dim() == 1:
        student_s = student_s.unsqueeze(0)

    for item_id in range(max_eval_items):
        label = test_response[student_id, item_id]
        if not np.isnan(label):
            with torch.no_grad():
                item_tensor = torch.LongTensor([item_id]).to(DEVICE)
                pred = model(student_k, student_s, item_tensor)
                predictions.append(pred.item())
                true_labels.append(label)

            if len(predictions) >= 5000:
                break
    if len(predictions) >= 5000:
        break

if len(predictions) > 0:
    predictions_binary = [1 if p > 0.5 else 0 for p in predictions]
    accuracy = accuracy_score(true_labels, predictions_binary)
    auc = roc_auc_score(true_labels, predictions)

    print(f"\n   测试集结果:")
    print(f"      评估样本数: {len(predictions)}")
    print(f"      准确率: {accuracy:.4f}")
    print(f"      AUC: {auc:.4f}")
else:
    print("\n   没有有效的评估样本")

print("\nStep 3 完成！")