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

print("=" * 70)
print("样本重加权公平性优化实验（完整评估 + 多维度加权）")
print("=" * 70)

# ==================== 配置 ====================
EMBED_DIM = 64
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义不同的加权策略
WEIGHT_STRATEGIES = {
    'baseline': {
        'name': '基线（无加权）',
        'weights': {'money': {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}},
        'alpha': 0
    },
    'money_focus': {
        'name': '家庭经济聚焦',
        'weights': {'money': {0: 3.0, 1: 1.5, 2: 1.0, 3: 1.0, 4: 1.0}},
        'alpha': 2.0
    },
    'money_heavy': {
        'name': '家庭经济重度',
        'weights': {'money': {0: 5.0, 1: 2.0, 2: 1.0, 3: 1.0, 4: 1.0}},
        'alpha': 4.0
    },
    'region_focus': {
        'name': '地区聚焦（贫困地区）',
        'weights': {'region': {0: 3.0, 1: 1.0, 2: 1.0, 3: 1.0}},
        'alpha': 2.0
    },
    'age_focus': {
        'name': '年龄聚焦（低龄12-14岁）',
        'weights': {'age': {12: 3.0, 13: 3.0, 14: 3.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0}},
        'alpha': 2.0
    },
    'gender_focus': {
        'name': '性别聚焦（女性）',
        'weights': {'gender': {0: 3.0, 1: 1.0}},
        'alpha': 2.0
    },
    'multi_focus': {
        'name': '多属性综合（贫穷+贫困地区+低龄+女性）',
        'weights': {
            'money': {0: 2.0, 1: 1.5, 2: 1.0, 3: 1.0, 4: 1.0},
            'region': {0: 1.5, 1: 1.0, 2: 1.0, 3: 1.0},
            'age': {12: 1.5, 13: 1.5, 14: 1.5, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0},
            'gender': {0: 1.5, 1: 1.0}
        },
        'alpha': 2.0
    }
}

# 结果存储
results = []

print(f"\n设备: {DEVICE}")
print(f"测试的加权策略: {len(WEIGHT_STRATEGIES)} 种")
print("-" * 70)

# ==================== 1. 定义模型 ====================
print("\n[1/6] 定义模型...")


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


# ==================== 2. 加载数据 ====================
print("\n[2/6] 加载数据...")

train_knowledge = pd.read_csv('output/neurips/student_knowledge_biased.csv', header=None).values
train_sensitive = pd.read_csv('output/neurips/sensitive_labels.csv')
train_response = pd.read_csv('output/neurips/response_wide.csv', header=None).values
train_response = (train_response > 0.5).astype(np.float32)

print(f"   知识形状: {train_knowledge.shape}")
print(f"   敏感属性形状: {train_sensitive.shape}")
print(f"   作答形状: {train_response.shape}")

# ==================== 3. 数据对齐 ====================
print("\n[3/6] 数据对齐...")

min_rows = min(train_knowledge.shape[0], len(train_sensitive), train_response.shape[0])
train_knowledge = train_knowledge[:min_rows]
train_sensitive = train_sensitive.iloc[:min_rows].reset_index(drop=True)
train_response = train_response[:min_rows]

n_students, n_skills = train_knowledge.shape
n_items = train_response.shape[1]

print(f"   对齐后学生数: {n_students}")
print(f"   题目数: {n_items}")
print(f"   知识点数: {n_skills}")

# 准备敏感属性
sensitive_cols = ['gender', 'age', 'region', 'money']
train_sensitive_values = train_sensitive[sensitive_cols].values.astype(np.float32)
scaler = StandardScaler()
train_sensitive_values = scaler.fit_transform(train_sensitive_values)

# ==================== 4. 划分数据集 ====================
print("\n[4/6] 划分训练集和测试集...")

train_idx, test_idx = train_test_split(
    range(n_students), test_size=0.2, random_state=42
)

train_knowledge_data = train_knowledge[train_idx]
train_sensitive_data = train_sensitive_values[train_idx]
train_response_data = train_response[train_idx]

test_knowledge = train_knowledge[test_idx]
test_sensitive = train_sensitive_values[test_idx]
test_response = train_response[test_idx]
test_sensitive_df = train_sensitive.iloc[test_idx].reset_index(drop=True)

print(f"   训练集: {len(train_idx)} 学生")
print(f"   测试集: {len(test_idx)} 学生")

# ==================== 5. 构建训练样本 ====================
print("\n[5/6] 构建训练样本...")

train_samples = []
for student_id in range(len(train_knowledge_data)):
    student_k = train_knowledge_data[student_id].astype(np.float32)
    student_s = train_sensitive_data[student_id]

    original_student_id = train_idx[student_id]
    # 获取该学生的所有敏感属性值
    student_gender = train_sensitive.iloc[original_student_id]['gender']
    student_age = train_sensitive.iloc[original_student_id]['age']
    student_region = train_sensitive.iloc[original_student_id]['region']
    student_money = train_sensitive.iloc[original_student_id]['money']

    for item_id in range(n_items):
        label = train_response_data[student_id, item_id]
        if not np.isnan(label):
            train_samples.append({
                'knowledge': student_k,
                'sensitive': student_s,
                'item_id': item_id,
                'label': float(label),
                'gender': student_gender,
                'age': student_age,
                'region': student_region,
                'money': student_money
            })

print(f"   训练样本数: {len(train_samples):,}")

# ==================== 6. 对不同加权策略进行实验 ====================
print("\n[6/7] 开始样本重加权实验（完整评估）...")
print("-" * 70)

from torch.utils.data import DataLoader, TensorDataset

for strategy_key, strategy in WEIGHT_STRATEGIES.items():
    print(f"\n>>> 测试策略: {strategy['name']}")

    # 构建带权重的训练样本
    features_list = []
    sensitive_list = []
    item_list = []
    label_list = []
    weight_list = []

    max_samples = min(200000, len(train_samples))

    for sample in train_samples[:max_samples]:
        features_list.append(sample['knowledge'])
        sensitive_list.append(sample['sensitive'])
        item_list.append(sample['item_id'])
        label_list.append(sample['label'])

        # 根据加权策略计算样本权重
        weight = 1.0
        weights_config = strategy['weights']

        for attr, weight_map in weights_config.items():
            if attr == 'money':
                weight *= weight_map.get(sample['money'], 1.0)
            elif attr == 'region':
                weight *= weight_map.get(sample['region'], 1.0)
            elif attr == 'age':
                weight *= weight_map.get(sample['age'], 1.0)
            elif attr == 'gender':
                weight *= weight_map.get(sample['gender'], 1.0)

        weight_list.append(weight)

    # 转换为 tensor
    features_tensor = torch.FloatTensor(np.array(features_list)).to(DEVICE)
    sensitive_tensor = torch.FloatTensor(np.array(sensitive_list)).to(DEVICE)
    item_tensor = torch.LongTensor(np.array(item_list)).to(DEVICE)
    label_tensor = torch.FloatTensor(np.array(label_list)).to(DEVICE)
    weight_tensor = torch.FloatTensor(np.array(weight_list)).to(DEVICE)

    dataset = TensorDataset(features_tensor, sensitive_tensor, item_tensor, label_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化模型
    model = NeuralCDM(
        n_skills=n_skills,
        n_sensitive=len(sensitive_cols),
        n_items=n_items,
        embed_dim=EMBED_DIM
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss(reduction='none')

    # 训练模型
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        batch_count = 0

        for batch_k, batch_s, batch_i, batch_label in dataloader:
            batch_start = batch_count * BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, len(weight_tensor))
            batch_weights = weight_tensor[batch_start:batch_end]

            optimizer.zero_grad()
            pred = model(batch_k, batch_s, batch_i)
            loss = (criterion(pred, batch_label) * batch_weights).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

        if (epoch + 1) % 10 == 0:
            print(f"     Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / batch_count:.4f}")

    # ==================== 完整评估 ====================
    print(f"    完整评估中...")

    model.eval()

    # 构建测试样本（全部测试集）
    test_samples = []
    for student_id in range(len(test_knowledge)):
        student_k = test_knowledge[student_id].astype(np.float32)
        student_s = test_sensitive[student_id]

        for item_id in range(n_items):
            label = test_response[student_id, item_id]
            if not np.isnan(label):
                test_samples.append((student_k, student_s, item_id, label, student_id))

    print(f"     测试样本数: {len(test_samples):,}")

    # 批量预测
    all_preds = []
    all_labels = []
    all_student_ids = []
    batch_size_pred = 1024

    for i in range(0, len(test_samples), batch_size_pred):
        batch = test_samples[i:i + batch_size_pred]
        batch_k = torch.FloatTensor(np.array([s[0] for s in batch])).to(DEVICE)
        batch_s = torch.FloatTensor(np.array([s[1] for s in batch])).to(DEVICE)
        batch_i = torch.LongTensor([s[2] for s in batch]).to(DEVICE)

        with torch.no_grad():
            preds = model(batch_k, batch_s, batch_i)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend([s[3] for s in batch])
            all_student_ids.extend([s[4] for s in batch])

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
    money_labels = test_sensitive_df['money'].values
    group_acc = {}

    for g in range(5):
        accs = []
        for sid, acc in student_acc_mean.items():
            if sid < len(money_labels) and money_labels[sid] == g:
                accs.append(acc)
        if accs:
            group_acc[g] = np.mean(accs)

    fairness_gap = max(group_acc.values()) - min(group_acc.values()) if group_acc else 0

    print(f"\n    策略 {strategy['name']} 结果:")
    print(f"       测试样本数: {len(all_preds):,}")
    print(f"       准确率: {accuracy:.4f}")
    print(f"       AUC: {auc:.4f}")
    print(f"       公平性差异: {fairness_gap:.4f} ({fairness_gap * 100:.2f}%)")

    results.append({
        'strategy': strategy['name'],
        'accuracy': accuracy,
        'auc': auc,
        'fairness_gap': fairness_gap
    })

# ==================== 7. 输出汇总结果 ====================
print("\n" + "=" * 70)
print("样本重加权实验汇总结果（完整评估 + 多维度加权）")
print("=" * 70)

print("\n| 加权策略 | 准确率 | AUC | 公平性差异 |")
print("|:---|:---|:---|:---|")
for r in results:
    print(f"| {r['strategy']} | {r['accuracy']:.4f} | {r['auc']:.4f} | {r['fairness_gap']:.4f} |")

# 保存结果
results_df = pd.DataFrame(results)
results_df.to_csv('output/reweighting_experiment_full.csv', index=False)
print("\n✓ 结果已保存: output/reweighting_experiment_full.csv")

# ==================== 8. 与语义增强方法对比 ====================
print("\n" + "=" * 70)
print("与语义增强方法对比")
print("=" * 70)

print("\n| 方法 | 准确率 | 公平性差异 |")
print("|:---|:---|:---|")
print("| 对照组（无优化）| 81.28% | 6.80% |")
print("| 实验组一（仅语义）| 79.43% | 1.86% |")

if results:
    best_reweight = min(results, key=lambda x: x['fairness_gap'])
    print(f"| 样本重加权最佳（{best_reweight['strategy']}）| {best_reweight['accuracy']*100:.2f}% | {best_reweight['fairness_gap']*100:.2f}% |")

print("\n" + "=" * 70)
print("样本重加权实验完成！")
print("=" * 70)