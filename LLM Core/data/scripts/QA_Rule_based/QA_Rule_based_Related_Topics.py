#!/usr/bin/env python3
"""
📋 基于规则的通用话题 Q&A 生成器
专门处理概念性话题（真菌、菌丝、栽培、烹饪等）
"""

import json
import random
import re

# ================= 配置 =================
INPUT_FILE = "data/raw_data_source/raw_mushroom_wiki_data.json"
OUTPUT_FILE = "data/processed/mushroom_rule_based_short.jsonl"
# =======================================

# 通用话题模板（适合概念、过程、方法）
TEMPLATES = {
    "summary": [
        "What is {topic}?",
        "Tell me about {topic}.",
        "Can you explain {topic}?",
        "What does {topic} mean?",
        "Could you describe {topic}?"
    ],

    # 特征/性质类
    "characteristics": [
        "What are the characteristics of {topic}?",
        "What are the key features of {topic}?",
        "How does {topic} work?",
        "What makes {topic} unique?"
    ],

    # 过程/方法类
    "process": [
        "How does {topic} occur?",
        "What is the process of {topic}?",
        "How is {topic} done?",
        "What steps are involved in {topic}?"
    ],

    # 功能/用途类
    "function": [
        "What is {topic} used for?",
        "Why is {topic} important?",
        "What role does {topic} play?",
        "How is {topic} applied?"
    ],

    # 识别/检测类
    "identification": [
        "How can you identify {topic}?",
        "What are the signs of {topic}?",
        "How do you recognize {topic}?",
        "What indicates {topic}?"
    ],

    # 类型/分类类
    "types": [
        "What are the types of {topic}?",
        "What are different kinds of {topic}?",
        "How is {topic} classified?",
        "What variations of {topic} exist?"
    ],

    # 安全/风险类
    "safety": [
        "Is {topic} dangerous?",
        "What are the risks of {topic}?",
        "How to stay safe with {topic}?",
        "What precautions should be taken with {topic}?"
    ]
}

# 章节名称到模板类型的映射
SECTION_TO_TEMPLATE = {
    # 生物学/形态学
    "characteristics": ["characteristics", "summary"],
    "morphology": ["characteristics", "summary"],
    "structure": ["characteristics", "summary"],
    "anatomy": ["characteristics", "summary"],
    "biology": ["characteristics", "summary"],

    # 过程/生命周期
    "life cycle": ["process", "summary"],
    "reproduction": ["process", "summary"],
    "development": ["process", "summary"],
    "growth": ["process", "summary"],

    # 分类/类型
    "classification": ["types", "summary"],
    "types": ["types", "summary"],
    "taxonomy": ["types", "summary"],
    "species": ["types", "summary"],

    # 生态/分布
    "ecology": ["characteristics", "function"],
    "habitat": ["characteristics", "summary"],
    "distribution": ["summary"],

    # 识别/检测
    "identification": ["identification", "characteristics"],
    "diagnosis": ["identification", "characteristics"],
    "detection": ["identification", "characteristics"],
    "recognition": ["identification", "characteristics"],

    # 毒性/安全
    "toxicity": ["safety", "summary"],
    "poisoning": ["safety", "summary"],
    "symptoms": ["safety", "identification"],
    "treatment": ["safety", "process"],

    # 用途/应用
    "uses": ["function", "summary"],
    "applications": ["function", "summary"],
    "cultivation": ["process", "function"],
    "cooking": ["process", "function"],
    "preparation": ["process", "summary"],

    # 历史/文化
    "history": ["summary"],
    "etymology": ["summary"],
    "culture": ["summary", "function"]
}

def clean_text(text):
    """清洗文本：去引用、去多余空格"""
    text = re.sub(r'\[\d+\]', '', text)
    text = text.replace('\n', ' ')
    return " ".join(text.split())

def smart_shorten(text, max_sentences=3):
    """
    智能缩短文本到指定句子数
    """
    if not text:
        return ""

    sentences = text.split('. ')
    final_sentences = []

    for s in sentences:
        clean_s = s.strip()
        if not clean_s or len(clean_s) < 20:  # 跳过太短的句子
            continue

        if not clean_s.endswith('.'):
            clean_s += '.'

        final_sentences.append(clean_s)

        if len(final_sentences) >= max_sentences:
            break

    return " ".join(final_sentences)

def match_section_to_templates(section_name):
    """
    根据章节名称匹配合适的问题模板
    """
    section_lower = section_name.lower()

    # 尝试精确匹配
    for keyword, template_types in SECTION_TO_TEMPLATE.items():
        if keyword in section_lower:
            return template_types

    # 默认返回通用模板
    return ["summary", "characteristics"]

# ================= 主逻辑 =================
print("=" * 80)
print("📋 基于规则生成通用话题 Q&A")
print("=" * 80)

formatted_data = []

try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
except FileNotFoundError:
    print(f"❌ 找不到 {INPUT_FILE}")
    print(f"💡 请先运行: python data/scripts/scrape_related_topics.py")
    exit()

for item in raw_data:
    topic = item['topic']

    # --- 1. 处理 Summary (定义) ---
    if item.get('summary'):
        question = random.choice(TEMPLATES["summary"]).format(topic=topic)
        answer = smart_shorten(clean_text(item['summary']), max_sentences=3)

        if answer:
            formatted_data.append({
                "text": f"User: {question}\nAssistant: {answer}<|im_end|>"
            })

    # --- 2. 处理各个 Section ---
    sections = item.get('sections', {})

    for section_name, section_content in sections.items():
        # 根据章节名称选择合适的模板
        template_types = match_section_to_templates(section_name)

        # 随机选择一个模板类型
        template_type = random.choice(template_types)

        # 生成问题
        if template_type in TEMPLATES:
            question_template = random.choice(TEMPLATES[template_type])
            question = question_template.format(topic=topic)

            # 生成答案
            answer = smart_shorten(clean_text(section_content), max_sentences=3)

            if answer:
                formatted_data.append({
                    "text": f"User: {question}\nAssistant: {answer}<|im_end|>"
                })

# ================= 保存数据 =================
print(f"\n✅ 总计生成 {len(formatted_data)} 条 Q&A")

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for item in formatted_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')

print(f"📁 保存到: {OUTPUT_FILE}")

# 预览
print("\n📝 数据预览（前5条）:")
for i, item in enumerate(formatted_data[:5], 1):
    print(f"\n【{i}】")
    print(item['text'][:200] + "...")

print("\n" + "=" * 80)
print("🎉 完成！")
