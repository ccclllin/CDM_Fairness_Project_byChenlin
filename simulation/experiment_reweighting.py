"""
样本重加权公平性优化实验
"""

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
print("样本重加权公平性优化实验")
print("=" * 60)

# ==================== 配置 ====================
EMBED_DIM = 64
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 样本重加权参数
ALPHA_VALUES = [0, 2.0, 3.0, 5.0]  # 0 表示不加权（基线）

# 结果存储
results = []

print(f"\n设备: {DEVICE}")
print(f"测试的权重系数: {ALPHA_VALUES}")
print(f"说明: alpha=0 为对照组（无加权）")

# ==================== 1. 定义模型 ====================
print("\n[1/5] 定义模型...")


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
print("\n[2/5] 加载数据...")

# 使用偏见注入数据
train_knowledge = pd.read_csv('output/neurips/student_knowledge_biased.csv', header=None).values
train_sensitive = pd.read_csv('output/neurips/sensitive_labels.csv')
train_response = pd.read_csv('output/neurips/response_wide.csv', header=None).values
train_response = (train_response > 0.5).astype(np.float32)

print(f"   知识形状: {train_knowledge.shape}")
print(f"   敏感属性形状: {train_sensitive.shape}")
print(f"   作答形状: {train_response.shape}")

# ==================== 3. 数据对齐 ====================
print("\n[3/5] 数据对齐...")

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
print("\n[4/5] 划分训练集和测试集...")

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
    money_label = train_sensitive.iloc[original_student_id]['money']

    for item_id in range(n_items):
        label = train_response_data[student_id, item_id]
        if not np.isnan(label):
            train_samples.append({
                'knowledge': student_k,
                'sensitive': student_s,
                'item_id': item_id,
                'label': float(label),
                'money': money_label
            })

print(f"   训练样本数: {len(train_samples)}")

# ==================== 6. 对不同权重进行实验 ====================
print("\n[6/6] 开始样本重加权实验...")
print("-" * 60)

from torch.utils.data import DataLoader, TensorDataset

for alpha in ALPHA_VALUES:
    print(f"\n>>> 测试 alpha={alpha} (弱势群体权重 = {1 + alpha})")

    # 构建带权重的训练样本
    features_list = []
    sensitive_list = []
    item_list = []
    label_list = []
    weight_list = []

    max_samples = min(150000, len(train_samples))

    for sample in train_samples[:max_samples]:
        features_list.append(sample['knowledge'])
        sensitive_list.append(sample['sensitive'])
        item_list.append(sample['item_id'])
        label_list.append(sample['label'])

        # 计算样本权重：贫穷（money=0）群体权重更高
        if alpha > 0:
            if sample['money'] == 0:
                weight = 1 + alpha
            elif sample['money'] == 1:
                weight = 1 + alpha * 0.5  # 较贫穷权重稍低
            else:
                weight = 1.0
        else:
            weight = 1.0

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
    criterion = nn.BCELoss(reduction='none')  # 不自动平均，以便加权

    # 训练模型
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        batch_count = 0

        for batch_k, batch_s, batch_i, batch_label in dataloader:
            # 获取对应批次的权重
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

    # ==================== 评估模型 ====================
    print(f"    评估中")

    model.eval()
    predictions = []
    true_labels = []
    student_ids = []

    for student_id in range(min(200, len(test_knowledge))):
        student_k = torch.FloatTensor(test_knowledge[student_id].astype(np.float32)).unsqueeze(0).to(DEVICE)
        student_s = torch.FloatTensor(test_sensitive[student_id]).unsqueeze(0).to(DEVICE)

        for item_id in range(min(200, n_items)):
            label = test_response[student_id, item_id]
            if not np.isnan(label):
                with torch.no_grad():
                    pred = model(student_k, student_s, torch.LongTensor([item_id]).to(DEVICE))
                    predictions.append(pred.item())
                    true_labels.append(label)
                    student_ids.append(student_id)

                if len(predictions) >= 3000:
                    break
        if len(predictions) >= 3000:
            break

    if len(predictions) > 0:
        pred_binary = [1 if p > 0.5 else 0 for p in predictions]
        accuracy = accuracy_score(true_labels, pred_binary)
        auc = roc_auc_score(true_labels, predictions)

        # 按家庭经济分组计算公平性（简化版）
        test_money_labels = test_sensitive_df['money'].values
        group_correct = {}
        group_total = {}

        for i, (pred, true, sid) in enumerate(zip(predictions, true_labels, student_ids)):
            if sid < len(test_money_labels):
                money = test_money_labels[sid]
                if money not in group_correct:
                    group_correct[money] = 0
                    group_total[money] = 0
                group_correct[money] += (1 if pred > 0.5 else 0) == true
                group_total[money] += 1

        if group_total:
            group_acc = {g: group_correct[g] / group_total[g] for g in group_total}
            if group_acc:
                fairness_gap = max(group_acc.values()) - min(group_acc.values())
            else:
                fairness_gap = 0
        else:
            fairness_gap = 0

        print(f"\n    权重 alpha={alpha} 结果:")
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

# ==================== 7. 输出汇总结果 ====================
print("\n" + "=" * 60)
print("样本重加权实验汇总结果")
print("=" * 60)

print("\n| alpha | 权重 | 准确率 | AUC | 公平性差异 |")
print("|:---|:---|:---|:---|:---|")
for r in results:
    print(f"| {r['alpha']} | {r['weight']:.1f} | {r['accuracy']:.4f} | {r['auc']:.4f} | {r['fairness_gap']:.4f} |")

# 保存结果
results_df = pd.DataFrame(results)
results_df.to_csv('output/reweighting_experiment_results.csv', index=False)
print("\n结果已保存: output/reweighting_experiment_results.csv")

# 对比
print("\n" + "=" * 60)
print("与语义增强方法对比")
print("=" * 60)
print("\n| 方法 | 准确率 | 公平性差异 |")
print("|:---|:---|:---|")
print("| 对照组（无优化）| 81.28% | 6.80% |")
print("| 实验组一（仅语义）| 79.43% | 1.86% |")

if len(results) > 1:
    best_reweight = min(results[1:], key=lambda x: x['fairness_gap'])
    print(f"| 样本重加权（alpha={best_reweight['alpha']}）| {best_reweight['accuracy']*100:.2f}% | {best_reweight['fairness_gap']*100:.2f}% |")

print("\n样本重加权实验完成！")