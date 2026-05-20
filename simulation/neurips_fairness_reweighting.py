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
print("融合实验：样本重加权 + 语义增强（NeurIPS 2020）")
print("=" * 60)

# ==================== 配置 ====================
EMBED_DIM = 64
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_SIZE = 0.2
RANDOM_SEED = 42

# 样本重加权配置
ALPHA_VALUES = [0, 1.0, 2.0, 3.0, 5.0]  # 0 表示不加权（基线）

# 结果存储
results = []

print(f"\n设备: {DEVICE}")
print(f"嵌入维度: {EMBED_DIM}")
print(f"测试的权重系数: {ALPHA_VALUES}")

# ==================== 1. 定义模型 ====================
print("\n[1/6] 定义模型...")


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


# ==================== 2. 加载数据 ====================
print("\n[2/6] 加载数据...")

# 使用偏见注入数据
train_knowledge = pd.read_csv('output/neurips/student_knowledge_biased.csv', header=None).values
train_sensitive = pd.read_csv('output/neurips/sensitive_labels.csv')
train_response = pd.read_csv('output/neurips/response_wide.csv', header=None).values
train_response = (train_response > 0.5).astype(np.float32)

# 加载语义向量
semantic_vectors = np.load('output/neurips/semantic_vectors.npy')
print(f"   语义向量形状: {semantic_vectors.shape}")

# 数据对齐
min_rows = min(train_knowledge.shape[0], len(train_sensitive), train_response.shape[0], semantic_vectors.shape[0])
train_knowledge = train_knowledge[:min_rows]
train_sensitive = train_sensitive.iloc[:min_rows].reset_index(drop=True)
train_response = train_response[:min_rows]
semantic_vectors = semantic_vectors[:min_rows]

n_students = semantic_vectors.shape[0]
n_features = semantic_vectors.shape[1]
n_items = train_response.shape[1]

print(f"   学生数: {n_students}")
print(f"   语义特征维度: {n_features}")
print(f"   题目数: {n_items}")

# ==================== 3. 划分数据集 ====================
print("\n[3/6] 划分训练集和测试集...")

train_idx, test_idx = train_test_split(
    range(n_students),
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=train_sensitive['region'].values
)

X_train = semantic_vectors[train_idx]
X_test = semantic_vectors[test_idx]
y_train = train_response[train_idx]
y_test = train_response[test_idx]

# 获取训练集的敏感属性用于加权
train_sensitive_data = train_sensitive.iloc[train_idx].reset_index(drop=True)
test_sensitive_df = train_sensitive.iloc[test_idx].reset_index(drop=True)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"   训练集: {len(train_idx)} 学生")
print(f"   测试集: {len(test_idx)} 学生")

# ==================== 4. 构建训练样本 ====================
print("\n[4/6] 构建训练样本...")

train_samples = []
for student_id in range(len(X_train)):
    feat = X_train[student_id].astype(np.float32)
    money_label = train_sensitive_data.iloc[student_id]['money']
    for item_id in range(n_items):
        label = y_train[student_id, item_id]
        if not np.isnan(label):
            train_samples.append({
                'features': feat,
                'item_id': item_id,
                'label': float(label),
                'money': money_label
            })

print(f"   训练样本数: {len(train_samples):,}")

# ==================== 5. 对不同权重进行实验 ====================
print("\n[5/6] 开始融合实验...")
print("-" * 60)

from torch.utils.data import DataLoader, TensorDataset

for alpha in ALPHA_VALUES:
    print(f"\n>>> 测试 alpha={alpha} (弱势群体权重 = {1 + alpha})")

    # 构建带权重的训练样本
    features_list = []
    item_list = []
    label_list = []
    weight_list = []

    for sample in train_samples[:200000]:  # 限制样本数
        features_list.append(sample['features'])
        item_list.append(sample['item_id'])
        label_list.append(sample['label'])

        # 计算权重：贫穷（money=0）群体权重更高
        if alpha > 0:
            if sample['money'] == 0:
                weight = 1 + alpha
            elif sample['money'] == 1:
                weight = 1 + alpha * 0.5
            else:
                weight = 1.0
        else:
            weight = 1.0

        weight_list.append(weight)

    features_tensor = torch.FloatTensor(np.array(features_list)).to(DEVICE)
    item_tensor = torch.LongTensor(np.array(item_list)).to(DEVICE)
    label_tensor = torch.FloatTensor(np.array(label_list)).to(DEVICE)
    weight_tensor = torch.FloatTensor(np.array(weight_list)).to(DEVICE)

    dataset = TensorDataset(features_tensor, item_tensor, label_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化模型
    model = SemanticCDM(
        n_features=n_features,
        n_items=n_items,
        embed_dim=EMBED_DIM
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss(reduction='none')

    # 训练
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        batch_count = 0

        for batch_f, batch_i, batch_label in dataloader:
            batch_start = batch_count * BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, len(weight_tensor))
            batch_weights = weight_tensor[batch_start:batch_end]

            optimizer.zero_grad()
            pred = model(batch_f, batch_i)
            loss = (criterion(pred, batch_label) * batch_weights).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

        if (epoch + 1) % 10 == 0:
            print(f"     Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / batch_count:.4f}")

    # 评估
    print(f"    评估中...")
    model.eval()

    # 构建测试样本
    test_samples = []
    for student_id in range(len(X_test)):
        feat = X_test[student_id].astype(np.float32)
        for item_id in range(n_items):
            label = y_test[student_id, item_id]
            if not np.isnan(label):
                test_samples.append((feat, item_id, label, student_id))

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

    # 整体性能
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
    money_labels = test_sensitive_df['money'].values
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

    print(f"\n    alpha={alpha} 结果:")
    print(f"       准确率: {accuracy:.4f}")
    print(f"       AUC: {auc:.4f}")
    print(f"       公平性差异: {fairness_gap:.4f}")

    results.append({
        'alpha': alpha,
        'weight': 1 + alpha,
        'accuracy': accuracy,
        'auc': auc,
        'fairness_gap': fairness_gap
    })

# ==================== 6. 输出汇总结果 ====================
print("\n" + "=" * 60)
print("融合实验汇总结果")
print("=" * 60)

print("\n| alpha | 权重 | 准确率 | AUC | 公平性差异 |")
print("|:---|:---|:---|:---|:---|")
for r in results:
    print(f"| {r['alpha']} | {r['weight']:.1f} | {r['accuracy']:.4f} | {r['auc']:.4f} | {r['fairness_gap']:.4f} |")

# 保存结果
results_df = pd.DataFrame(results)
results_df.to_csv('output/neurips_reweighting_results.csv', index=False)
print("\n结果已保存: output/neurips_reweighting_results.csv")

# ==================== 7. 与已有结果对比 ====================
print("\n" + "=" * 60)
print("与已有结果对比")
print("=" * 60)

print("\n| 方法 | 准确率 | 公平性差异 |")
print("|:---|:---|:---|")
print("| 对照组（无优化）| 81.28% | 6.80% |")
print("| 实验组一（仅语义）| 79.43% | 1.86% |")

if results:
    best_reweight = min(results, key=lambda x: x['fairness_gap'])
    print(f"| 融合实验（语义+重加权, alpha={best_reweight['alpha']}）| {best_reweight['accuracy']*100:.2f}% | {best_reweight['fairness_gap']*100:.2f}% |")

print("\n融合实验完成！")