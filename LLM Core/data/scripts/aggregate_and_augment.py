import json
import random
import os
import glob
from pathlib import Path

# ================= 配置 =================
PROCESSED_DIR = "data/processed/"
OUTPUT_FILE = "data/final_training_data.jsonl"
AUGMENTATION_RATIO = 0.3  # 30% 的数据进行增强
RANDOM_SEED = 42

# 增强策略说明：
# 1. add_typos: 拼写错误（字符交换/删除/重复/键盘误触/随机插入）
# 2. add_noise: 添加噪声标点符号
# 3. remove_words: 随机删除 15% 词语（保留关键疑问词）
# 4. add_case_variation: 大小写变化（全小写/全大写/首字母大写）
# 5. add_garbled_text: 乱码模拟（编码错误、特殊字符、Unicode问题）
# =======================================

random.seed(RANDOM_SEED)

# ==================== 数据增强函数 ====================

def add_typos(text, typo_rate=0.1):
    """添加拼写错误（10%的单词）- 包括常见打字错误"""
    words = text.split()
    num_typos = max(1, int(len(words) * typo_rate))

    # 常见的拼写替换（基于键盘布局的错误）
    keyboard_mistakes = {
        'a': ['s', 'q', 'z'],
        'e': ['w', 'r', 'd'],
        'i': ['u', 'o', 'k'],
        'o': ['i', 'p', 'l'],
        's': ['a', 'd', 'w'],
        't': ['r', 'y', 'g'],
        'n': ['b', 'm', 'h'],
    }

    for _ in range(num_typos):
        if len(words) == 0:
            break
        idx = random.randint(0, len(words) - 1)
        word = words[idx]

        if len(word) > 2:
            # 随机选择一种错误类型
            typo_type = random.choice(['swap', 'delete', 'duplicate', 'keyboard', 'insert'])

            if typo_type == 'swap' and len(word) > 2:
                # 交换相邻字符
                pos = random.randint(0, len(word) - 2)
                word = word[:pos] + word[pos+1] + word[pos] + word[pos+2:]
            elif typo_type == 'delete':
                # 删除一个字符
                pos = random.randint(0, len(word) - 1)
                word = word[:pos] + word[pos+1:]
            elif typo_type == 'duplicate':
                # 重复一个字符
                pos = random.randint(0, len(word) - 1)
                word = word[:pos+1] + word[pos] + word[pos+1:]
            elif typo_type == 'keyboard' and len(word) > 1:
                # 键盘布局错误（按到相邻键）
                pos = random.randint(0, len(word) - 1)
                char = word[pos].lower()
                if char in keyboard_mistakes:
                    replacement = random.choice(keyboard_mistakes[char])
                    word = word[:pos] + replacement + word[pos+1:]
            elif typo_type == 'insert':
                # 随机插入一个字符
                pos = random.randint(0, len(word))
                random_char = random.choice('abcdefghijklmnopqrstuvwxyz')
                word = word[:pos] + random_char + word[pos:]

            words[idx] = word

    return ' '.join(words)

def add_noise(text):
    """添加随机噪声字符"""
    noise_chars = ['!', '?', '.', ',', '...', '??', '!!']
    words = text.split()

    if len(words) > 0:
        # 在随机位置插入噪声
        insert_pos = random.randint(0, len(words))
        noise = random.choice(noise_chars)
        words.insert(insert_pos, noise)

    return ' '.join(words)

def remove_words(text, remove_rate=0.15):
    """随机删除词语（15%）"""
    words = text.split()

    if len(words) <= 2:
        return text  # 太短不删除

    num_remove = max(1, int(len(words) * remove_rate))

    for _ in range(num_remove):
        if len(words) > 2:
            idx = random.randint(0, len(words) - 1)
            # 避免删除重要疑问词
            if words[idx].lower() not in ['what', 'how', 'why', 'where', 'when', 'is', 'are', 'can', 'do', 'does']:
                words.pop(idx)

    return ' '.join(words)

def add_case_variation(text):
    """添加大小写变化"""
    variations = [
        text.lower(),  # 全小写
        text.upper(),  # 全大写
        text.capitalize(),  # 首字母大写
        ' '.join([w.capitalize() for w in text.split()])  # 每个单词首字母大写
    ]
    return random.choice(variations)

def add_garbled_text(text, garble_rate=0.05):
    """添加乱码（模拟编码错误、特殊字符等）"""
    words = text.split()

    if len(words) == 0:
        return text

    # 常见的乱码替换（模拟编码问题）
    garbled_replacements = {
        'a': ['@', 'á', 'à', 'â'],
        'e': ['3', 'é', 'è', 'ê'],
        'i': ['1', '!', 'í', 'ì'],
        'o': ['0', 'ó', 'ò', 'ô'],
        's': ['$', '5'],
        'l': ['1', '|'],
        't': ['7', '+'],
        'g': ['9'],
    }

    # Unicode 特殊字符（模拟乱码）
    special_chars = ['�', '□', '▪', '•', '�', '™', '©', '®']

    num_garbles = max(1, int(len(words) * garble_rate))

    for _ in range(num_garbles):
        idx = random.randint(0, len(words) - 1)
        word = words[idx]

        if len(word) > 2:
            garble_type = random.choice(['char_replace', 'special_insert', 'encoding_error'])

            if garble_type == 'char_replace':
                # 字符替换为相似符号
                pos = random.randint(0, len(word) - 1)
                char = word[pos].lower()
                if char in garbled_replacements:
                    replacement = random.choice(garbled_replacements[char])
                    word = word[:pos] + replacement + word[pos+1:]

            elif garble_type == 'special_insert':
                # 插入特殊乱码字符
                pos = random.randint(0, len(word))
                special = random.choice(special_chars)
                word = word[:pos] + special + word[pos:]

            elif garble_type == 'encoding_error':
                # 模拟编码错误：随机位置插入 ? 或替换为 ?
                pos = random.randint(0, len(word) - 1)
                if random.random() < 0.5:
                    word = word[:pos] + '?' + word[pos+1:]
                else:
                    word = word[:pos] + '??' + word[pos:]

            words[idx] = word

    return ' '.join(words)

def augment_question(question):
    """对问题进行增强（组合多种技术）"""
    augmentation_types = [
        ('typo', add_typos),              # 拼写错误（键盘误触、字符交换等）
        ('noise', add_noise),             # 噪声符号
        ('remove', remove_words),         # 随机删词
        ('case', add_case_variation),     # 大小写变化
        ('garbled', add_garbled_text),    # 🆕 乱码模拟（编码错误、特殊字符）
    ]

    # 随机选择 1-2 种增强方式
    num_augmentations = random.randint(1, 2)
    selected = random.sample(augmentation_types, num_augmentations)

    augmented = question
    for aug_type, aug_func in selected:
        augmented = aug_func(augmented)

    return augmented

def parse_qa_line(line):
    """解析 Q&A 格式的一行数据"""
    try:
        data = json.loads(line)
        text = data.get('text', '')

        # 提取 User 和 Assistant 部分
        if 'User:' in text and 'Assistant:' in text:
            parts = text.split('Assistant:', 1)
            question_part = parts[0].replace('User:', '').strip()
            answer_part = parts[1].strip()
            return question_part, answer_part
    except:
        pass

    return None, None

# ==================== 主逻辑 ====================
print("=" * 60)
print("Data Aggregation & Augmentation")
print("=" * 60)

# 查找所有 .jsonl 文件
jsonl_files = glob.glob(os.path.join(PROCESSED_DIR, "*.jsonl"))

# 排除输出文件自身
jsonl_files = [f for f in jsonl_files if f != OUTPUT_FILE]

print(f"\n📂 找到 {len(jsonl_files)} 个数据文件:")
for f in jsonl_files:
    print(f"   - {os.path.basename(f)}")

if len(jsonl_files) == 0:
    print("❌ 没有找到数据文件")
    exit(1)

# 加载所有数据
all_data = []
file_stats = {}

for filepath in jsonl_files:
    filename = os.path.basename(filepath)
    count = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                all_data.append((line, filename))
                count += 1

    file_stats[filename] = count
    print(f"✅ {filename}: {count} 条")

print(f"\n📊 总计: {len(all_data)} 条原始数据")

# 打乱数据
random.shuffle(all_data)

# 数据增强
augmented_data = []
original_data = []

num_to_augment = int(len(all_data) * AUGMENTATION_RATIO)
print(f"\n🔧 将对 {num_to_augment} 条数据 ({AUGMENTATION_RATIO*100:.0f}%) 进行增强...")

for i, (line, source) in enumerate(all_data):
    # 原始数据保留
    original_data.append(line)

    # 对部分数据进行增强
    if i < num_to_augment:
        question, answer = parse_qa_line(line)

        if question and answer:
            # 只增强问题部分，保持答案不变
            augmented_question = augment_question(question)

            # 重新组合
            augmented_line = json.dumps({
                "text": f"User: {augmented_question}\nAssistant: {answer}"
            }, ensure_ascii=False)

            augmented_data.append(augmented_line)

print(f"✅ 生成了 {len(augmented_data)} 条增强数据")

# 合并原始数据和增强数据
final_data = original_data + augmented_data
random.shuffle(final_data)

print(f"\n📊 最终数据集: {len(final_data)} 条 (原始:{len(original_data)} + 增强:{len(augmented_data)})")

# 保存
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for line in final_data:
        f.write(line + '\n')

print(f"\n✅ 数据已保存到: {OUTPUT_FILE}")

# 显示示例
print("\n" + "=" * 60)
print("示例数据预览")
print("=" * 60)

print("\n【原始数据示例】")
for i in range(min(2, len(original_data))):
    data = json.loads(original_data[i])
    print(f"\n{i+1}. {data['text'][:150]}...")

if len(augmented_data) > 0:
    print("\n【增强数据示例】")
    for i in range(min(2, len(augmented_data))):
        data = json.loads(augmented_data[i])
        print(f"\n{i+1}. {data['text'][:150]}...")

print("\n" + "=" * 60)
print("🎉 完成！")
print("=" * 60)

# 统计信息
print("\n📈 数据统计:")
print(f"   总数据量: {len(final_data)}")
print(f"   原始数据: {len(original_data)} ({len(original_data)/len(final_data)*100:.1f}%)")
print(f"   增强数据: {len(augmented_data)} ({len(augmented_data)/len(final_data)*100:.1f}%)")
print(f"   增强比例: {AUGMENTATION_RATIO*100:.0f}%")
print("\n来源文件统计:")
for filename, count in sorted(file_stats.items(), key=lambda x: x[1], reverse=True):
    print(f"   {filename}: {count} 条 ({count/len(original_data)*100:.1f}%)")
