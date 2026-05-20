import pandas as pd
import numpy as np
import asyncio
import aiohttp
import time
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


# ==================== 配置 ====================
DEEPSEEK_API_KEY = "sk-"

MAX_CONCURRENT = 10
MAX_RETRIES = 3

MAX_FEATURES = 50
TEMPERATURE = 0.2
MAX_TOKENS = 200

KNOWLEDGE_FILE = 'output/neurips/student_knowledge.csv'
OUTPUT_DIR = 'output/neurips'

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n配置:")
print(f"  语义向量维度: {MAX_FEATURES}")
print(f"  temperature: {TEMPERATURE}")
print(f"  max_tokens: {MAX_TOKENS}")

# ==================== 1. 读取数据 ====================
print("\n[1/4] 读取学生知识掌握概率...")
student_knowledge = pd.read_csv(KNOWLEDGE_FILE, header=None).values
n_students = student_knowledge.shape[0]
n_skills = student_knowledge.shape[1]

print(f"   总学生数: {n_students}")
print(f"   知识点数: {n_skills}")

# ==================== 2. 构建提示词（优化版）====================
print("\n[2/4] 构建提示词...")

def build_prompt(student_data):
    mastery_rate = student_data.mean()
    std_dev = student_data.std()

    # 掌握最好的2个知识点
    top_idx = np.argsort(student_data)[-2:][::-1]
    top_skills = '、'.join([f"知识点{idx}" for idx in top_idx])
    top_scores = '、'.join([f"{student_data[idx]:.0%}" for idx in top_idx])

    # 掌握最差的2个知识点
    bottom_idx = np.argsort(student_data)[:2]
    bottom_skills = '、'.join([f"知识点{idx}" for idx in bottom_idx])
    bottom_scores = '、'.join([f"{student_data[idx]:.0%}" for idx in bottom_idx])

    high = np.sum(student_data >= 0.7)
    medium = np.sum((student_data >= 0.4) & (student_data < 0.7))
    low = np.sum(student_data < 0.4)

    prompt = f"""你是一个教育数据分析专家。请根据以下学生的认知诊断结果，生成一段详细、客观的学习能力描述。

【学生诊断数据】
- 整体知识掌握率：{mastery_rate:.1%}
- 知识掌握均衡度：{'均衡' if std_dev < 0.25 else '分化明显' if std_dev > 0.35 else '一般'}
- 掌握最好的2个知识点：{top_skills}（掌握率分别为{top_scores}）
- 掌握最差的2个知识点：{bottom_skills}（掌握率分别为{bottom_scores}）
- 知识点分布：掌握良好（≥70%）{high}个，中等（40%-70%）{medium}个，薄弱（<40%）{low}个

【输出要求】
1. 使用具体数字描述
2. 只描述学习表现，不得提及地域、家庭背景、性别、年龄等信息
3. 语言客观、专业、中性
4. 字数控制在100-150字

请直接输出描述："""

    return prompt


print("   生成所有 prompts...")
prompts = [build_prompt(s) for s in student_knowledge]
print(f"   ✓ 共 {len(prompts)} 个 prompts")

print(f"\n   示例提示词:\n{prompts[0][:500]}...")

# ==================== 3. 异步并发调用 API ====================
print(f"\n[3/4] 异步并发调用 API（并发数: {MAX_CONCURRENT}）...")

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
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS
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

        descriptions = [""] * len(prompts)
        completed = 0

        for task in asyncio.as_completed(tasks):
            desc = await task
            descriptions[completed] = desc
            completed += 1

            if completed % 100 == 0:
                print(f"   已完成 {completed}/{len(prompts)} 个学生")

        return descriptions


start_time = time.time()
descriptions = asyncio.run(process_all())
elapsed = time.time() - start_time

print(f"\n   API 调用完成 总耗时: {elapsed:.1f} 秒 ({elapsed / 60:.1f} 分钟)")
print(f"   平均每学生: {elapsed / len(prompts):.2f} 秒")

# ==================== 4. 向量化 ====================
print("\n[4/4] 语义向量化...")

valid_descs = [d if d else "学生对知识点的掌握情况一般" for d in descriptions]

vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, min_df=1, max_df=0.9)
semantic_vectors = vectorizer.fit_transform(valid_descs).toarray()

print(f"   向量化完成: {semantic_vectors.shape}")

np.save(f'{OUTPUT_DIR}/semantic_vectors.npy', semantic_vectors)
joblib.dump(vectorizer, f'{OUTPUT_DIR}/tfidf_vectorizer.pkl')
print(f"   已保存: {OUTPUT_DIR}/semantic_vectors.npy")

# 保存原始描述
pd.DataFrame({
    'student_id': range(len(descriptions)),
    'description': descriptions
}).to_csv(f'{OUTPUT_DIR}/semantic_descriptions.csv', index=False)

print("\nStep 2-2 完成！")