#!/usr/bin/env python3
"""
🎙️ Podcast-Style Q&A Generator (Overnight Batch Mode)
使用 Ollama Qwen 32B 生成播客风格的蘑菇知识对话数据
"""

import json
import subprocess
import time
import os
from datetime import datetime

# ================= 配置 =================
INPUT_FILE = "data/raw_data_source/raw_mushroom_wiki_data.json"
OUTPUT_DIR = "data/processed/"
OLLAMA_MODEL = "qwen2.5vl:32b"

# 生成配置
TOPICS_PER_MUSHROOM = 5      # 每个蘑菇生成5个话题
QA_PER_TOPIC = 5              # 每个话题生成3个Q&A
SLEEP_BETWEEN_CALLS = 2       # API调用间隔(秒)
# ========================================

def call_ollama(prompt, model=OLLAMA_MODEL, timeout=120):
    """调用 Ollama 并返回结果"""
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        print("⚠️  Timeout - 跳过")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def generate_topics(mushroom_name, wiki_summary):
    """步骤1: 话题裂变 - 生成有趣的播客话题"""
    prompt = f"""System / Role: You are the content producer for a popular science podcast called "The Mushroom Mind".

Task: Generate {TOPICS_PER_MUSHROOM} specific, engaging "spin-off" topics for the mushroom: {mushroom_name}

Context about this mushroom:
{wiki_summary[:500]}

Rules:
1. Go Deep: Focus on History, Pop Culture, Myths, True Crime, Cooking, or Weird Science
2. Catchy Titles: Sound like YouTube video titles
3. Variety: Mix different angles (science, culture, safety, cooking, myths)
4. Output Format: Return ONLY a valid JSON list of topic strings

Examples of good topics:
- "The Viking Berserker Rage: Did Amanita muscaria fuel ancient warriors?"
- "Why Truffles Cost More Than Gold: The Economics of Fungus"
- "Cordyceps: The Real-Life 'Last of Us' Zombie Fungus"
- "Lion's Mane and Brain Health: Miracle Cure or Snake Oil?"
- "Death Cap Mushrooms: The Silent Killer in Your Backyard"

Generate {TOPICS_PER_MUSHROOM} topics now (return ONLY the JSON list):"""

    response = call_ollama(prompt)
    if not response:
        return []

    try:
        # 尝试解析JSON
        topics = json.loads(response)
        if isinstance(topics, list):
            return topics[:TOPICS_PER_MUSHROOM]
    except:
        # 如果JSON解析失败，尝试提取引号内的内容
        import re
        topics = re.findall(r'"([^"]+)"', response)
        return topics[:TOPICS_PER_MUSHROOM]

    return []

def extract_json_from_text(text):
    """从文本中提取JSON（支持markdown代码块等格式）"""
    import re

    # 尝试1: 直接解析
    try:
        return json.loads(text)
    except:
        pass

    # 尝试2: 提取 markdown 代码块中的 JSON
    json_block_match = re.search(r'```json\s*(\[.*?\])\s*```', text, re.DOTALL)
    if json_block_match:
        try:
            return json.loads(json_block_match.group(1))
        except:
            pass

    # 尝试3: 提取任意 [...] 块
    json_array_match = re.search(r'\[.*\]', text, re.DOTALL)
    if json_array_match:
        try:
            return json.loads(json_array_match.group(0))
        except:
            pass

    return None

def generate_podcast_qa(mushroom_name, topic):
    """步骤2: 生成播客风格的Q&A对话（去代词化版本）"""
    prompt = f"""You are "Fungi Phil", a podcast host generating training data.

Task: Create {QA_PER_TOPIC} Q&A pairs about: {mushroom_name} - {topic}

CRITICAL CONSTRAINTS (MUST FOLLOW):
1. **NO PRONOUNS IN QUESTIONS**: User questions MUST explicitly mention the mushroom name.
   - ❌ BAD: "Is it poisonous?" "Can they help with cancer?" "What does it look like?"
   - ✅ GOOD: "Is Amanita muscaria poisonous?" "Can Enoki mushrooms help with cancer?" "What does the Fly Agaric look like?"
   - Use either the scientific name ({mushroom_name}) or common name, but ALWAYS mention it explicitly

2. Answer Requirements:
   - 4-6 sentences, engaging, factual, use analogies
   - Start with hooks: "Oh absolutely!", "Here's the wild part...", etc.
   - Natural spoken-style (but questions must name the mushroom!)

Why this matters: Each Q&A will be trained independently. If you use "it/they", the model won't know what you're referring to.

CRITICAL: Return ONLY a JSON array like this (no markdown, no extra text):
[
  {{"text": "User: Can Amanita phalloides kill you?\\nAssistant: Here's the scary truth - Amanita phalloides is responsible for 90% of mushroom poisoning deaths worldwide. It contains amatoxins that destroy your liver cells slowly over days, like tiny saboteurs. The deadly part is you feel fine for 6-24 hours after eating it, then symptoms hit hard. By the time people seek help, the damage is often irreversible. That's why we say: when in doubt, throw it out!<|im_end|>"}},
  {{"text": "User: Why is the Death Cap mushroom so deadly?\\nAssistant: Oh man, the name is no joke! It got its nickname because it literally caps off people's lives - it's the deadliest mushroom on Earth. The innocent-looking cap contains enough toxin in just half a mushroom to kill an adult human. What makes it extra terrifying is that it looks similar to edible mushrooms like puffballs or Caesar's mushroom. Foragers who make mistakes with this one don't get second chances, which is why mycologists take identification so seriously!<|im_end|>"}}
]

Generate {QA_PER_TOPIC} Q&A pairs now (JSON array only, NO PRONOUNS in questions):"""

    response = call_ollama(prompt, timeout=180)
    if not response:
        return []

    # 使用增强的 JSON 提取
    qa_list = extract_json_from_text(response)

    if qa_list and isinstance(qa_list, list):
        valid_items = [item for item in qa_list if isinstance(item, dict) and "text" in item]
        if valid_items:
            return valid_items

    # 如果还是失败，尝试手动解析
    print(f"⚠️  JSON解析失败，尝试手动提取...")
    print(f"   原始响应前200字符: {response[:200]}")
    return []

# ==================== 主逻辑 ====================
def main():
    print("=" * 80)
    print("🎙️  Podcast-Style Q&A Generator (Overnight Batch Mode)")
    print("=" * 80)

    # 检查 Ollama
    print(f"\n🔧 测试 Ollama 连接...")
    test = call_ollama("Hello", OLLAMA_MODEL)
    if not test:
        print("❌ Ollama 不可用，请检查服务")
        return
    print("✅ Ollama 连接成功\n")

    # 加载蘑菇数据
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            mushrooms = json.load(f)
    except FileNotFoundError:
        print(f"❌ 找不到数据文件: {INPUT_FILE}")
        return

    print(f"📚 加载了 {len(mushrooms)} 种蘑菇")
    print(f"⏰ 预计生成: {len(mushrooms)} × {TOPICS_PER_MUSHROOM} × {QA_PER_TOPIC} = {len(mushrooms) * TOPICS_PER_MUSHROOM * QA_PER_TOPIC} 个 Q&A")
    print(f"⏱️  预计耗时: ~{len(mushrooms) * TOPICS_PER_MUSHROOM * 2 / 60:.1f} 小时\n")

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, f"mushroom_qa_podcast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")

    all_qa = []
    total_generated = 0
    start_time = time.time()

    # 主循环
    for idx, mushroom_data in enumerate(mushrooms, 1):
        mushroom_name = mushroom_data.get('topic', 'Unknown')
        summary = mushroom_data.get('summary', '')

        print(f"\n{'='*80}")
        print(f"[{idx}/{len(mushrooms)}] 🍄 {mushroom_name}")
        print(f"{'='*80}")

        # 步骤1: 生成话题
        print(f"  📋 生成话题中...", end=" ", flush=True)
        topics = generate_topics(mushroom_name, summary)
        print(f"✅ 得到 {len(topics)} 个话题")

        if not topics:
            print(f"  ⚠️  跳过（无法生成话题）")
            continue

        # 显示话题
        for i, topic in enumerate(topics, 1):
            print(f"     {i}. {topic}")

        time.sleep(SLEEP_BETWEEN_CALLS)

        # 步骤2: 为每个话题生成Q&A
        for topic_idx, topic in enumerate(topics, 1):
            print(f"\n  🎙️  [{topic_idx}/{len(topics)}] 生成对话: {topic[:50]}...", end=" ", flush=True)

            qa_pairs = generate_podcast_qa(mushroom_name, topic)

            if qa_pairs:
                all_qa.extend(qa_pairs)
                total_generated += len(qa_pairs)
                print(f"✅ +{len(qa_pairs)} 条")

                # 实时保存（防止中断丢失数据）
                with open(output_file, 'w', encoding='utf-8') as f:
                    for qa in all_qa:
                        json.dump(qa, f, ensure_ascii=False)
                        f.write('\n')
            else:
                print(f"⚠️  跳过")

            time.sleep(SLEEP_BETWEEN_CALLS)

        # 进度报告
        elapsed = time.time() - start_time
        avg_time_per_mushroom = elapsed / idx
        remaining_mushrooms = len(mushrooms) - idx
        eta_seconds = avg_time_per_mushroom * remaining_mushrooms

        print(f"\n  📊 当前进度: {total_generated} 条 Q&A | 已用时: {elapsed/60:.1f}分钟 | 预计剩余: {eta_seconds/60:.1f}分钟")

    # 完成
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("🎉 生成完成！")
    print("=" * 80)
    print(f"📁 输出文件: {output_file}")
    print(f"📊 总计生成: {total_generated} 条 Q&A")
    print(f"⏱️  总耗时: {total_time/3600:.2f} 小时")
    print(f"⚡ 平均速度: {total_generated/(total_time/60):.1f} 条/分钟")
    print("=" * 80)

    # 预览数据
    print("\n📝 数据预览（前3条）:")
    for i, qa in enumerate(all_qa[:3], 1):
        print(f"\n【示例 {i}】")
        print(qa['text'][:300] + "...")

if __name__ == "__main__":
    main()
