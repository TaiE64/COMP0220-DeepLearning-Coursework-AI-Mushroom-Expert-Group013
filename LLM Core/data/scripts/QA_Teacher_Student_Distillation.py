import json
import subprocess
import time
import os

# ================= 配置 =================
INPUT_FILE = "data/raw_data_source/raw_mushroom_wiki_data.json"
OUTPUT_FILE = "data/processed/mushroom_qa_teacher_student.jsonl"
OLLAMA_MODEL = "qwen2.5vl:32b"  # 你的 Ollama 模型名称
MAX_QA_PER_TOPIC = 5  # 每个主题生成多少个 Q&A 对
# =======================================

def call_ollama(prompt, model=OLLAMA_MODEL):
    """调用 Ollama CLI 获取模型回复"""
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=60  # 60秒超时
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        print("⚠️  Ollama 调用超时")
        return None
    except Exception as e:
        print(f"❌ Ollama 调用失败: {e}")
        return None

def generate_qa_for_topic(topic_data):
    """
    基于单个主题的维基数据，使用大模型生成多个 Q&A 对

    策略：
    1. 使用所有可用章节（不过滤）
    2. 让大模型从完整信息中生成多样化的问题和答案
    """
    topic = topic_data['topic']
    summary = topic_data.get('summary', '')
    sections = topic_data.get('sections', {})

    # 构建完整上下文 - 使用所有章节
    context = f"Topic: {topic}\n\n"

    if summary:
        context += f"Summary:\n{summary[:800]}\n\n"  # 摘要限制800字符

    # 添加所有章节内容（放宽限制，让模型看到更完整的信息）
    for section_name, content in sections.items():
        if content.strip():  # 只添加非空章节
            # 增加字符限制到1000，确保信息完整性
            truncated_content = content[:1000] if len(content) > 1000 else content
            context += f"{section_name}:\n{truncated_content}\n\n"

    # 生成 Q&A 的提示词 - 播客风格
    prompt = f"""You are a knowledgeable science communicator explaining {topic} to a curious audience. Based on the following information, generate {MAX_QA_PER_TOPIC} diverse question-answer pairs.

Context:
{context}

Requirements:
1. Generate {MAX_QA_PER_TOPIC} different questions covering various aspects (identification, toxicity, habitat, edibility, taxonomy, etc.)
2. **Answer Style**: Explain clearly like you're having a conversation, not writing a paper
   - Start with the key point
   - Explain HOW and WHY things work
   - Use phrases like "Here's why...", "The reason is...", "This happens because..."
   - Complete the thought - don't stop mid-explanation
3. **Answer Length**: 3-5 sentences (enough to explain the mechanism, not just list facts)
4. **Logic**: If mentioning multiple compounds/reasons, explain their relationship clearly
5. Format each Q&A pair as:
   Q: [question]
   A: [complete, conversational explanation]
   ---

Example of GOOD answer:
"This is a common misconception! While it's called the 'fly agaric', the main toxins are actually Ibotenic acid and Muscimol. When you ingest it, Ibotenic acid converts to Muscimol in your body, which is what causes the hallucinations and poisoning. Although Muscarine is also present, its concentration is very low and doesn't contribute much to the toxicity. So it's mainly those first two compounds attacking your nervous system."

Generate the Q&A pairs now:"""

    print(f"🤖 正在为 {topic} 生成 Q&A...")
    response = call_ollama(prompt)

    if not response:
        return []

    # 解析大模型的回复
    qa_pairs = []
    lines = response.split('\n')
    current_q = None
    current_a = None

    for line in lines:
        line = line.strip()
        if line.startswith('Q:'):
            current_q = line[2:].strip()
        elif line.startswith('A:'):
            current_a = line[2:].strip()
            if current_q and current_a:
                qa_pairs.append({
                    "text": f"User: {current_q}\nAssistant: {current_a}<|im_end|>"
                })
                current_q = None
                current_a = None
        elif line == '---':
            continue

    return qa_pairs

# ================= 主逻辑 =================
print("=" * 60)
print("Teacher-Student 蒸馏：使用 Qwen-32B 生成 Q&A 数据")
print("=" * 60)

# 检查 Ollama 是否可用
print("\n检查 Ollama 是否可用...")
test_result = call_ollama("Hello", OLLAMA_MODEL)
if not test_result:
    print("❌ Ollama 不可用或模型未加载，请检查：")
    print(f"   1. ollama list 查看模型是否存在")
    print(f"   2. ollama run {OLLAMA_MODEL} 测试模型")
    exit(1)
print("✅ Ollama 连接成功")

# 加载原始数据
try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
except FileNotFoundError:
    print(f"❌ 找不到 {INPUT_FILE}，请先运行爬虫脚本！")
    exit(1)

print(f"\n📚 加载了 {len(raw_data)} 个主题")
print(f"⏱️  预计耗时: {len(raw_data) * MAX_QA_PER_TOPIC * 3 / 60:.1f} 分钟\n")

all_qa_pairs = []
failed_topics = []

for i, topic_data in enumerate(raw_data, 1):
    print(f"[{i}/{len(raw_data)}] ", end="")

    try:
        qa_pairs = generate_qa_for_topic(topic_data)
        all_qa_pairs.extend(qa_pairs)
        print(f"   ✅ 生成了 {len(qa_pairs)} 个 Q&A")
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        failed_topics.append(topic_data['topic'])

    # 礼貌调用：每个主题后暂停一下
    time.sleep(1)

# 确保输出目录存在
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# 保存结果
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for qa in all_qa_pairs:
        json.dump(qa, f, ensure_ascii=False)
        f.write('\n')

print("\n" + "=" * 60)
print(f"🎉 生成完成！")
print(f"✅ 总共生成: {len(all_qa_pairs)} 个 Q&A 对")
print(f"📊 平均每个主题: {len(all_qa_pairs) / len(raw_data):.1f} 个")
if failed_topics:
    print(f"❌ 失败主题 ({len(failed_topics)}): {', '.join(failed_topics[:5])}")
print(f"📁 数据已保存到: {OUTPUT_FILE}")
print("=" * 60)

# 显示示例
if all_qa_pairs:
    print(f"\n示例预览 (前3条):")
    for i, example in enumerate(all_qa_pairs[:3], 1):
        print(f"\n--- 示例 {i} ---")
        print(example['text'])
print("=" * 60)
