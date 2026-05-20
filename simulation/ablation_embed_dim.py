import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings('ignore')

print("=" * 60)
print("消融实验：嵌入维度消融")
print("=" * 60)

# ==================== 配置 ====================
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 测试的嵌入维度
EMBED_DIMS = [16, 32, 64, 128, 256]

# 结果存储
results = []

# 固定其他参数
SEMANTIC_DIM = 100
DROPOUT = 0.2

# ==================== 1. 定义模型 ====================
print("\n[1/5] 定义模型...")


class SemanticCDM(nn.Module):
    def __init__(self, n_features, n_items, embed_dim=64):
        super(SemanticCDM, self).__init__()

        self.feature_fc = nn.Sequential(
            nn.Linear(n_features, embed_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )

        self.item_embedding = nn.Embedding(n_items, embed_dim)

        self.interaction_fc = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
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


# ==================== 2. 加载数据 ====================
print("\n[2/5] 加载数据...")

# 读取语义描述
desc_df = pd.read_csv('output/neurips/semantic_descriptions.csv')
if 'semantic_description' in desc_df.columns:
    semantic_descriptions = desc_df['semantic_description'].tolist()
elif 'description' in desc_df.columns:
    semantic_descriptions = desc_df['description'].tolist()
else:
    semantic_descriptions = desc_df.iloc[:, 0].tolist()

print(f"   语义描述数: {len(semantic_descriptions)}")

# 使用偏见注入数据
train_knowledge = pd.read_csv('output/neurips/student_knowledge_biased.csv', header=None).values
response = pd.read_csv('output/neurips/response_wide.csv', header=None).values
response = (response > 0.5).astype(np.float32)
sensitive = pd.read_csv('output/neurips/sensitive_labels.csv')

print(f"   知识掌握概率形状: {train_knowledge.shape}")
print(f"   作答数据形状: {response.shape}")

# ==================== 3. 生成语义向量 ====================
print("\n[3/5] 生成语义向量...")

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=SEMANTIC_DIM, min_df=1, max_df=0.9)
semantic_vectors = vectorizer.fit_transform(semantic_descriptions).toarray()
print(f"   语义向量形状: {semantic_vectors.shape}")

# ==================== 4. 划分数据集 ====================
print("\n[4/5] 划分训练集和测试集...")

n_students = response.shape[0]
n_items = response.shape[1]

train_idx, test_idx = train_test_split(
    range(n_students), test_size=0.2, random_state=42
)

train_vectors = semantic_vectors[train_idx]
test_vectors = semantic_vectors[test_idx]
test_response = response[test_idx]
test_sensitive = sensitive.iloc[test_idx].reset_index(drop=True)

print(f"   训练集: {len(train_idx)} 学生")
print(f"   测试集: {len(test_idx)} 学生")
print(f"   题目数: {n_items}")

# 标准化
scaler = StandardScaler()
train_vectors = scaler.fit_transform(train_vectors)
test_vectors = scaler.transform(test_vectors)

# ==================== 5. 对不同嵌入维度进行消融实验 ====================
print("\n[5/5] 开始嵌入维度消融实验...")
print("-" * 60)

for embed_dim in EMBED_DIMS:
    print(f"\n>>> 测试嵌入维度: {embed_dim}")

    # 构建训练样本
    train_samples = []
    for student_id in range(len(train_vectors)):
        feat = train_vectors[student_id].astype(np.float32)
        student_original_id = train_idx[student_id]
        for item_id in range(n_items):
            label = response[student_original_id, item_id]
            if not np.isnan(label):
                train_samples.append((feat, item_id, float(label)))

    print(f"   训练样本数: {len(train_samples):,}")

    # 训练模型
    model = SemanticCDM(
        n_features=SEMANTIC_DIM,
        n_items=n_items,
        embed_dim=embed_dim
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    from torch.utils.data import DataLoader, TensorDataset

    # 限制样本数避免内存溢出
    max_samples = min(200000, len(train_samples))
    features_list = []
    item_list = []
    label_list = []

    for feat, item_id, label in train_samples[:max_samples]:
        features_list.append(feat)
        item_list.append(item_id)
        label_list.append(label)

    features_tensor = torch.FloatTensor(np.array(features_list)).to(DEVICE)
    item_tensor = torch.LongTensor(np.array(item_list)).to(DEVICE)
    label_tensor = torch.FloatTensor(np.array(label_list)).to(DEVICE)

    dataset = TensorDataset(features_tensor, item_tensor, label_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

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
            print(f"     Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(dataloader):.4f}")

    # 完整评估
    print(f"    完整评估")
    model.eval()

    # 构建测试样本
    test_samples = []
    for student_id in range(len(test_vectors)):
        feat = test_vectors[student_id].astype(np.float32)
        for item_id in range(n_items):
            label = test_response[student_id, item_id]
            if not np.isnan(label):
                test_samples.append((feat, item_id, label, student_id))

    # 批量预测
    all_preds = []
    all_labels = []
    all_student_ids = []
    batch_size_pred = 1024

    for i in range(0, len(test_samples), batch_size_pred):
        batch = test_samples[i:i + batch_size_pred]
        batch_f = torch.FloatTensor(np.array([s[0] for s in batch])).to(DEVICE)
        batch_i = torch.LongTensor([s[1] for s in batch]).to(DEVICE)

        with torch.no_grad():
            preds = model(batch_f, batch_i)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend([s[2] for s in batch])
            all_student_ids.extend([s[3] for s in batch])

    # 计算整体准确率
    pred_binary = [1 if p > 0.5 else 0 for p in all_preds]
    accuracy = accuracy_score(all_labels, pred_binary)
    auc = roc_auc_score(all_labels, all_preds)

    # 计算每个学生的准确率
    student_acc = {}
    for i, sid in enumerate(all_student_ids):
        if sid not in student_acc:
            student_acc[sid] = {'correct': 0, 'total': 0}
        student_acc[sid]['correct'] += (1 if all_preds[i] > 0.5 else 0) == all_labels[i]
        student_acc[sid]['total'] += 1

    student_acc_mean = {sid: d['correct'] / d['total'] for sid, d in student_acc.items()}

    # 按家庭经济分组计算公平性差异
    money_labels = test_sensitive['money'].values
    group_acc = {}

    for g in range(5):
        accs = []
        for sid, acc in student_acc_mean.items():
            if sid < len(money_labels) and money_labels[sid] == g:
                accs.append(acc)
        if accs:
            group_acc[g] = np.mean(accs)

    if group_acc:
        fairness_gap = max(group_acc.values()) - min(group_acc.values())
    else:
        fairness_gap = 0

    print(f"\n   嵌入维度 {embed_dim} 结果:")
    print(f"      测试样本数: {len(all_preds):,}")
    print(f"      准确率: {accuracy:.4f}")
    print(f"      AUC: {auc:.4f}")
    print(f"      公平性差异: {fairness_gap:.4f}")

    results.append({
        'embed_dim': embed_dim,
        'accuracy': accuracy,
        'auc': auc,
        'fairness_gap': fairness_gap
    })

# ==================== 6. 输出汇总结果 ====================
print("\n" + "=" * 60)
print("消融实验汇总结果（嵌入维度消融）")
print("=" * 60)

print("\n| 嵌入维度 | 准确率 | AUC | 公平性差异 |")
print("|:---|:---|:---|:---|")
for r in results:
    print(f"| {r['embed_dim']} | {r['accuracy']:.4f} | {r['auc']:.4f} | {r['fairness_gap']:.4f} |")

if results:
    best_acc = max(results, key=lambda x: x['accuracy'])
    best_fair = min(results, key=lambda x: x['fairness_gap'])

    print(f"\n最佳准确率嵌入维度: {best_acc['embed_dim']} (准确率: {best_acc['accuracy']:.4f})")
    print(f"最佳公平性嵌入维度: {best_fair['embed_dim']} (公平性差异: {best_fair['fairness_gap']:.4f})")

# 保存结果
results_df = pd.DataFrame(results)
results_df.to_csv('output/ablation_embed_dim_results.csv', index=False)
print("\n结果已保存: output/ablation_embed_dim_results.csv")

print("\n消融实验完成！")