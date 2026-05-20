# 跨数据集验证 - TF-IDF 通用嵌入方法

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

warnings.filterwarnings('ignore')

print("=" * 60)
print("Step 5：跨数据集验证 - TF-IDF 通用嵌入")
print("=" * 60)

# ==================== 配置 ====================
EMBED_DIM = 64
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0
MAX_FEATURES = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\n设备: {DEVICE}")
print(f"训练集: NeurIPS2020")
print(f"测试集: Assist0910")
print(f"方法: TF-IDF 通用嵌入")
print(f"  - max_features: {MAX_FEATURES}")
print(f"  - weight_decay: {WEIGHT_DECAY}")

# ==================== 1. 定义模型 ====================
print("\n[1/6] 定义模型...")


class UniversalCDM(nn.Module):
    def __init__(self, n_features, n_items, embed_dim=64):
        super(UniversalCDM, self).__init__()

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


# ==================== 2. 生成知识点描述 ====================
print("\n[2/6] 生成知识点文本描述...")


def generate_skill_texts(n_skills, dataset_name):

    texts = []
    for i in range(n_skills):
        texts.append(f"{dataset_name}_skill_{i}")
    return texts


# ==================== 3. 创建通用嵌入 ====================
print("\n[3/6] 创建 TF-IDF 通用嵌入...")


def create_tfidf_embedding(knowledge_matrix, skill_texts, vectorizer=None):
    n_students, n_skills = knowledge_matrix.shape

    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, min_df=1, max_df=0.9)
        vectorizer.fit(skill_texts)

    skill_embeddings = vectorizer.transform(skill_texts).toarray()
    print(f"   知识点嵌入形状: {skill_embeddings.shape}")

    student_embeddings = np.dot(knowledge_matrix, skill_embeddings)
    print(f"   学生嵌入形状: {student_embeddings.shape}")

    return student_embeddings, skill_embeddings, vectorizer


# ==================== 4. 加载数据 ====================
print("\n[4/6] 加载数据...")

train_knowledge = pd.read_csv('output/neurips/student_knowledge.csv', header=None).values
train_response = pd.read_csv('output/neurips/response_wide.csv', header=None).values
train_response = (train_response > 0.5).astype(np.float32)

test_knowledge = pd.read_csv('output/assist/student_knowledge.csv', header=None).values
test_response = pd.read_csv('output/assist/response_wide.csv', header=None).values
test_response = (test_response > 0.5).astype(np.float32)

print(f"   训练集知识: {train_knowledge.shape}")
print(f"   测试集知识: {test_knowledge.shape}")

# ==================== 5. 创建通用嵌入 ====================
print("\n[5/6] 创建通用嵌入...")

train_skill_texts = generate_skill_texts(train_knowledge.shape[1], "neurips")
test_skill_texts = generate_skill_texts(test_knowledge.shape[1], "assist")

print(f"   训练集知识点数: {len(train_skill_texts)}")
print(f"   测试集知识点数: {len(test_skill_texts)}")

# 训练 TF-IDF 向量化器
_, _, tfidf_vectorizer = create_tfidf_embedding(train_knowledge, train_skill_texts)

# 生成嵌入
train_embeddings, _, _ = create_tfidf_embedding(train_knowledge, train_skill_texts, tfidf_vectorizer)
test_embeddings, _, _ = create_tfidf_embedding(test_knowledge, test_skill_texts, tfidf_vectorizer)

print(f"\n   训练集嵌入形状: {train_embeddings.shape}")
print(f"   测试集嵌入形状: {test_embeddings.shape}")

# 标准化
scaler = StandardScaler()
train_embeddings = scaler.fit_transform(train_embeddings)
test_embeddings = scaler.transform(test_embeddings)

# ==================== 6. 题目对齐 ====================
print("\n[6/6] 题目对齐...")

common_items = min(train_response.shape[1], test_response.shape[1])
train_response = train_response[:, :common_items]
test_response = test_response[:, :common_items]

print(f"   共同题目数: {common_items}")

# ==================== 7. 训练模型 ====================
print("\n[7/8] 训练模型...")


def build_samples(embeddings, responses, max_students=500, max_items=300):
    samples = []
    n_students = min(max_students, embeddings.shape[0])
    n_items = min(max_items, responses.shape[1])
    for student_id in range(n_students):
        emb = embeddings[student_id].astype(np.float32)
        for item_id in range(n_items):
            label = responses[student_id, item_id]
            if not np.isnan(label):
                samples.append((emb, item_id, float(label)))
    return samples


train_samples = build_samples(train_embeddings, train_response)
print(f"   训练样本数: {len(train_samples)}")

# 准备训练数据
features_list = []
item_list = []
label_list = []

for emb, item_id, label in train_samples[:20000]:
    features_list.append(emb)
    item_list.append(item_id)
    label_list.append(label)

features_tensor = torch.FloatTensor(np.array(features_list)).to(DEVICE)
item_tensor = torch.LongTensor(np.array(item_list)).to(DEVICE)
label_tensor = torch.FloatTensor(np.array(label_list)).to(DEVICE)

dataset = torch.utils.data.TensorDataset(features_tensor, item_tensor, label_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = UniversalCDM(
    n_features=train_embeddings.shape[1],
    n_items=common_items,
    embed_dim=EMBED_DIM
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
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

torch.save(model.state_dict(), 'output/universal_model.pth')
print("\n模型已保存: output/universal_model.pth")

# ==================== 8. 评估 ====================
print("\n[8/8] 在 Assist0910 上评估...")

model.eval()
predictions = []
true_labels = []

for student_id in range(min(200, test_embeddings.shape[0])):
    emb = torch.FloatTensor(test_embeddings[student_id].astype(np.float32)).unsqueeze(0).to(DEVICE)

    for item_id in range(min(200, common_items)):
        label = test_response[student_id, item_id]
        if not np.isnan(label):
            with torch.no_grad():
                pred = model(emb, torch.LongTensor([item_id]).to(DEVICE))
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

    print(f"\n   跨数据集测试结果（TF-IDF 通用嵌入）:")
    print(f"      评估样本数: {len(predictions)}")
    print(f"      准确率: {accuracy:.4f}")
    print(f"      AUC: {auc:.4f}")

    print(f"\n   随机基线准确率: 0.5000")
    print(f"   提升幅度: +{(accuracy - 0.5) * 100:.2f}%")
else:
    print("   没有有效的评估样本")

print("\nStep 5 完成！")