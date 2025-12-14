import json
import random
import re

# ================= 配置 =================
INPUT_FILE = "data/raw_data_source/related_topics_wiki_data.json"
OUTPUT_FILE = "data/processed/mushroom_rule_based_related_topics.jsonl"
# =======================================

TEMPLATES = {
    "summary": [
        "What is {topic}?",
        "Tell me about {topic}.",
        "Can you explain what {topic} is?"
    ],
    "Description": [
        "What does {topic} look like?",
        "Describe the appearance of {topic}.",
        "How can I identify {topic}?"
    ],
    "Toxicity": [
        "Is {topic} poisonous?",
        "Can I eat {topic}?",
        "Is {topic} safe?"
    ],
    "Habitat": [
        "Where does {topic} grow?",
        "What is the habitat of {topic}?",
        "Where can I find {topic}?"
    ],
    "Edibility": [
        "Is {topic} edible?",
        "Can you eat {topic}?",
        "Is {topic} good to eat?"
    ]
}

def clean_text(text):
    """基础清洗：去引用、去多余空格"""
    # 去除 [1], [12] 这种引用
    text = re.sub(r'\[\d+\]', '', text)
    # 去除换行
    text = text.replace('\n', ' ')
    return " ".join(text.split())

def smart_shorten(text, max_sentences=2):
    """
    核心函数：把长段落变短，变成适合聊天的长度。
    策略：只取前 N 句。
    """
    if not text:
        return ""

    # 1. 按句号+空格切分
    sentences = text.split('. ')

    # 2. 如果第一句太短（比如只是个分类名），可能要多取一句
    final_sentences = []
    current_len = 0

    for s in sentences:
        clean_s = s.strip()
        if not clean_s:
            continue

        # 补回句号
        if not clean_s.endswith('.'):
            clean_s += '.'

        final_sentences.append(clean_s)
        current_len += 1

        # 达到数量限制就停止
        if current_len >= max_sentences:
            break

    return " ".join(final_sentences)

# ================= 主逻辑 =================
print("=" * 60)
print("正在生成精简版对话数据...")
print("=" * 60)

formatted_data = []

try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
except FileNotFoundError:
    print(f"❌ 找不到 {INPUT_FILE}，请先运行爬虫脚本！")
    exit()

for item in raw_data:
    topic = item['topic']

    # --- 1. 处理 Summary (定义) ---
    # 策略：Summary 通常包含定义+分布+历史。我们只要前2句定义。
    if item.get('summary'):
        question = random.choice(TEMPLATES["summary"]).format(topic=topic)

        raw_answer = clean_text(item['summary'])
        short_answer = smart_shorten(raw_answer, max_sentences=2)  # 只取前2句

        if short_answer:
            formatted_data.append({
                "text": f"User: {question}\nAssistant: {short_answer}<|im_end|>"
            })

    # --- 2. 处理各种章节 ---
    sections = item.get('sections', {})

    # 遍历模板，查找对应章节
    for section_type, question_templates in TEMPLATES.items():
        if section_type == "summary":
            continue  # 已经处理过

        # 模糊匹配找到章节（支持子章节，如 "Description > Appearance"）
        matching_keys = [k for k in sections.keys() if section_type in k]

        for key in matching_keys:
            question = random.choice(question_templates).format(topic=topic)

            raw_answer = clean_text(sections[key])

            # 根据类型调整句子数量
            if section_type == "Toxicity":
                short_answer = smart_shorten(raw_answer, max_sentences=2)  # 毒性要简短有力
            elif section_type == "Description":
                short_answer = smart_shorten(raw_answer, max_sentences=3)  # 外观多一句
            else:
                short_answer = smart_shorten(raw_answer, max_sentences=2)

            if short_answer:
                formatted_data.append({
                    "text": f"User: {question}\nAssistant: {short_answer}<|im_end|>"
                })

            # 每个章节类型只取一个匹配（避免重复）
            break

# 确保输出目录存在
import os
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# 保存
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for entry in formatted_data:
        json.dump(entry, f, ensure_ascii=False)
        f.write('\n')

print("=" * 60)
print(f"🎉 处理完成！生成了 {len(formatted_data)} 条【精简版】对话数据。")
print(f"📁 数据已保存到: {OUTPUT_FILE}")
print("=" * 60)
print(f"\n示例预览 (前3条):")
for i, example in enumerate(formatted_data[:3], 1):
    print(f"\n--- 示例 {i} ---")
    print(json.dumps(example, indent=2, ensure_ascii=False))
print("=" * 60)
