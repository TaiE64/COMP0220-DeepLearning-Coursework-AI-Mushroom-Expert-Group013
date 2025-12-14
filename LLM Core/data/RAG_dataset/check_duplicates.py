#!/usr/bin/env python3
"""检查数据中的重复项"""

import json
from collections import Counter
import hashlib

def generate_unique_id(prefix: str, text: str) -> str:
    """生成唯一 ID（使用哈希避免重复）"""
    hash_val = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"{prefix}_{hash_val}"

# Load related topics
with open('data/raw_data_source/related_topics_wiki_data.json', 'r', encoding='utf-8') as f:
    topics = json.load(f)

print(f"总话题数: {len(topics)}")

# Check for duplicate topic names
topic_names = [t.get('topic', 'Unknown') for t in topics]
counts = Counter(topic_names)
duplicates = {name: count for name, count in counts.items() if count > 1}

if duplicates:
    print(f"\n发现重复的话题名称:")
    for name, count in duplicates.items():
        print(f"  {name}: {count} 次")

# Check for duplicate IDs that would be generated
all_ids = []
for topic_data in topics:
    topic = topic_data.get('topic', 'Unknown')
    summary = topic_data.get('summary', '')
    sections = topic_data.get('sections', {})
    
    # Summary ID
    if summary:
        doc_text = f"{topic}\n\n{summary}"
        doc_id = generate_unique_id("topic", doc_text)
        all_ids.append((doc_id, topic, "summary"))
    
    # Section IDs
    for section_name, section_content in sections.items():
        if len(section_content) > 100:
            doc_text = f"{topic} - {section_name}\n\n{section_content}"
            doc_id = generate_unique_id("topic", doc_text)
            all_ids.append((doc_id, topic, section_name))

print(f"\n总文档块数: {len(all_ids)}")

# Check for duplicate IDs
id_counts = Counter([doc_id for doc_id, _, _ in all_ids])
duplicate_ids = {doc_id: count for doc_id, count in id_counts.items() if count > 1}

if duplicate_ids:
    print(f"\n发现重复的 ID:")
    for doc_id, count in duplicate_ids.items():
        print(f"\n  ID: {doc_id} (出现 {count} 次)")
        # Show which documents have this ID
        for did, topic, section in all_ids:
            if did == doc_id:
                print(f"    - {topic} / {section}")
else:
    print("\n✅ 没有发现重复的 ID")
