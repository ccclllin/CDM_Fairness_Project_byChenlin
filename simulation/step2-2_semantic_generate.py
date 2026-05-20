# 调用大模型生成语义描述

import pandas as pd
import numpy as np
import asyncio
import aiohttp
import time
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

print("=" * 60)
print("实验组 Step 2-2：生成无偏语义描述")
print("=" * 60)

# ==================== 配置 ====================
DEEPSEEK_API_KEY = "sk-"

MAX_CONCURRENT = 10  # 并发数
BATCH_SIZE = 100  # 每批保存一次
MAX_RETRIES = 3

KNOWLEDGE_FILE = 'output/neurips/student_knowledge.csv'
OUTPUT_DIR = 'output/neurips'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 1. 读取数据 ====================
print("\n[1/4] 读取学生知识掌握概率...")
student_knowledge = pd.read_csv(KNOWLEDGE_FILE, header=None).values
n_students = student_knowledge.shape[0]
print(f"   总学生数: {n_students}")

# ==================== 2. 构建提示词 ====================
print("\n[2/4] 构建提示词...")


def build_prompt(student_data):
    mastery_rate = student_data.mean()

    top_idx = np.argsort(student_data)[-5:][::-1]
    top_skills = ', '.join([f"知识点{idx}:{student_data[idx]:.0%}" for idx in top_idx])

    bottom_idx = np.argsort(student_data)[:5]
    bottom_skills = ', '.join([f"知识点{idx}:{student_data[idx]:.0%}" for idx in bottom_idx])

    prompt = f"""请根据以下学生的认知诊断结果，生成一段客观的学习能力描述。

【诊断数据】
- 整体知识掌握率：{mastery_rate:.1%}
- 掌握最好的5个知识点：{top_skills}
- 掌握最差的5个知识点：{bottom_skills}

【要求】
1. 使用准确的数字描述
2. 只描述学习表现，不提地域、家庭、性别、年龄
3. 语言客观中性
4. 控制在150字以内

请直接输出描述："""

    return prompt


# 预先生成所有提示词prompts
print("   生成所有 prompts...")
prompts = [build_prompt(s) for s in student_knowledge]
print(f"   ✓ 共 {len(prompts)} 个 prompts")

# ==================== 3. 异步调用 API ====================
print(f"\n[3/4] 异步调用 API（并发数: {MAX_CONCURRENT}）...")

semaphore = asyncio.Semaphore(MAX_CONCURRENT)


async def call_api(session, prompt, student_id):
    async with semaphore:
        url = "https://api.deepseek.com/chat/completions"
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 300
        }

        for attempt in range(MAX_RETRIES):
            try:
                async with session.post(url, headers=headers, json=payload,
                                        timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result['choices'][0]['message']['content'].strip()
                    elif resp.status == 429:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        await asyncio.sleep(1)
            except:
                await asyncio.sleep(1)

        return ""


async def process_all():
    async with aiohttp.ClientSession() as session:
        tasks = [call_api(session, prompt, i) for i, prompt in enumerate(prompts)]

        # 分批显示进度
        descriptions = [""] * len(prompts)
        for i, task in enumerate(asyncio.as_completed(tasks)):
            desc = await task
            descriptions[i] = desc
            if (i + 1) % 100 == 0:
                print(f"   已完成 {i + 1}/{len(prompts)} 个学生")

        return descriptions


# 运行
start_time = time.time()
descriptions = asyncio.run(process_all())
elapsed = time.time() - start_time

print(f"\n   API 调用完成！总耗时: {elapsed:.1f} 秒 ({elapsed / 60:.1f} 分钟)")
print(f"   平均每学生: {elapsed / len(prompts):.2f} 秒")

# ==================== 4. 向量化 ====================
print("\n[4/4] 语义向量化...")

valid_descs = [d if d else "学生对知识点的掌握情况一般" for d in descriptions]

vectorizer = TfidfVectorizer(max_features=100)
semantic_vectors = vectorizer.fit_transform(valid_descs).toarray()

np.save(f'{OUTPUT_DIR}/semantic_vectors.npy', semantic_vectors)
joblib.dump(vectorizer, f'{OUTPUT_DIR}/tfidf_vectorizer.pkl')

print(f"   ✓ 向量化完成: {semantic_vectors.shape}")
print(f"   ✓ 已保存: {OUTPUT_DIR}/semantic_vectors.npy")

# 保存原始描述
pd.DataFrame({
    'student_id': range(len(descriptions)),
    'description': descriptions
}).to_csv(f'{OUTPUT_DIR}/semantic_descriptions.csv', index=False)

print("\n实验组 Step 2-2 完成！")