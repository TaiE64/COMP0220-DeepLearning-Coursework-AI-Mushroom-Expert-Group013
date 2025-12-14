#!/usr/bin/env python3
"""
🔍 构建 RAG 向量数据库
将蘑菇知识库转换为可检索的向量数据库
"""

import json
import os
from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions
import hashlib

# ================= 配置 =================
# 原始数据源
MUSHROOM_DATA = "data/raw_data_source/raw_mushroom_wiki_data.json"
RELATED_TOPICS_DATA = "data/raw_data_source/related_topics_wiki_data.json"

# 向量数据库存储位置
CHROMA_DB_PATH = "data/RAG_dataset/chroma_db"

# 嵌入模型（使用免费的 sentence-transformers）
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 轻量级，快速
# 其他选择:
# - "all-mpnet-base-v2"  # 更准确，但稍慢
# - "paraphrase-multilingual-MiniLM-L12-v2"  # 支持中文
# ========================================

def generate_unique_id(prefix: str, text: str) -> str:
    """生成唯一 ID（使用哈希避免重复）"""
    hash_val = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"{prefix}_{hash_val}"

def load_mushroom_data() -> List[Dict]:
    """加载蘑菇数据"""
    print("📚 加载蘑菇数据...")

    documents = []

    # 1. 加载具体蘑菇品种
    try:
        with open(MUSHROOM_DATA, 'r', encoding='utf-8') as f:
            mushrooms = json.load(f)

        # 去重蘑菇（基于话题名称）
        seen_mushrooms = set()
        unique_mushrooms = []
        for mushroom in mushrooms:
            mushroom_name = mushroom.get('topic', 'Unknown')
            if mushroom_name not in seen_mushrooms:
                seen_mushrooms.add(mushroom_name)
                unique_mushrooms.append(mushroom)
        
        if len(mushrooms) != len(unique_mushrooms):
            print(f"  ⚠️  发现 {len(mushrooms) - len(unique_mushrooms)} 个重复蘑菇，已去重")
        
        mushrooms = unique_mushrooms

        for mushroom in mushrooms:
            topic = mushroom.get('topic', 'Unknown')
            summary = mushroom.get('summary', '')
            sections = mushroom.get('sections', {})

            # 添加摘要作为一个文档
            if summary:
                doc_text = f"{topic}\n\n{summary}"
                documents.append({
                    "id": generate_unique_id("mushroom", doc_text),
                    "text": f"{topic}\n\n{summary}",
                    "metadata": {
                        "type": "mushroom",
                        "topic": topic,
                        "section": "summary",
                        "url": mushroom.get('wiki_url', '')
                    }
                })

            # 每个章节作为独立文档
            for section_name, section_content in sections.items():
                if len(section_content) > 100:  # 跳过太短的
                    doc_text = f"{topic} - {section_name}\n\n{section_content}"
                    documents.append({
                        "id": generate_unique_id("mushroom", doc_text),
                        "text": doc_text,
                        "metadata": {
                            "type": "mushroom",
                            "topic": topic,
                            "section": section_name,
                            "url": mushroom.get('wiki_url', '')
                        }
                    })

        print(f"  ✅ 加载了 {len(mushrooms)} 种蘑菇")

    except FileNotFoundError:
        print(f"  ⚠️  找不到 {MUSHROOM_DATA}")

    # 2. 加载延展话题
    try:
        with open(RELATED_TOPICS_DATA, 'r', encoding='utf-8') as f:
            topics = json.load(f)

        # 去重话题（基于话题名称）
        seen_topics = set()
        unique_topics = []
        for topic_data in topics:
            topic_name = topic_data.get('topic', 'Unknown')
            if topic_name not in seen_topics:
                seen_topics.add(topic_name)
                unique_topics.append(topic_data)
        
        if len(topics) != len(unique_topics):
            print(f"  ⚠️  发现 {len(topics) - len(unique_topics)} 个重复话题，已去重")
        
        topics = unique_topics

        for topic_data in topics:
            topic = topic_data.get('topic', 'Unknown')
            summary = topic_data.get('summary', '')
            sections = topic_data.get('sections', {})

            # 添加摘要
            if summary:
                doc_text = f"{topic}\n\n{summary}"
                documents.append({
                    "id": generate_unique_id("topic", doc_text),
                    "text": f"{topic}\n\n{summary}",
                    "metadata": {
                        "type": "general_topic",
                        "topic": topic,
                        "section": "summary",
                        "url": topic_data.get('wiki_url', '')
                    }
                })

            # 章节
            for section_name, section_content in sections.items():
                if len(section_content) > 100:
                    doc_text = f"{topic} - {section_name}\n\n{section_content}"
                    documents.append({
                        "id": generate_unique_id("topic", doc_text),
                        "text": doc_text,
                        "metadata": {
                            "type": "general_topic",
                            "topic": topic,
                            "section": section_name,
                            "url": topic_data.get('wiki_url', '')
                        }
                    })

        print(f"  ✅ 加载了 {len(topics)} 个通用话题")

    except FileNotFoundError:
        print(f"  ⚠️  找不到 {RELATED_TOPICS_DATA}")

    print(f"\n📊 原始文档: {len(documents)} 个")

    # 去重（基于 ID）
    seen_ids = set()
    unique_documents = []
    duplicates = 0

    for doc in documents:
        if doc['id'] not in seen_ids:
            seen_ids.add(doc['id'])
            unique_documents.append(doc)
        else:
            duplicates += 1

    if duplicates > 0:
        print(f"⚠️  发现 {duplicates} 个重复文档，已去重")

    print(f"📊 去重后: {len(unique_documents)} 个文档块")
    return unique_documents

def build_vector_database(documents: List[Dict]):
    """构建向量数据库"""
    print("\n🔧 构建向量数据库...")

    # 创建输出目录
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)

    # 初始化 Chroma 客户端
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # 使用 sentence-transformers 嵌入模型
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # 创建或获取集合
    try:
        # 如果集合已存在，删除它
        client.delete_collection(name="mushroom_knowledge")
        print("  🗑️  删除旧集合")
    except:
        pass

    collection = client.create_collection(
        name="mushroom_knowledge",
        embedding_function=embedding_function,
        metadata={"description": "Mushroom and mycology knowledge base"}
    )

    # 批量添加文档（Chroma 推荐批量插入以提高效率）
    batch_size = 100
    total_batches = (len(documents) + batch_size - 1) // batch_size

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]

        ids = [doc['id'] for doc in batch]
        texts = [doc['text'] for doc in batch]
        metadatas = [doc['metadata'] for doc in batch]

        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )

        batch_num = i // batch_size + 1
        print(f"  ✅ 处理批次 {batch_num}/{total_batches} ({len(batch)} 文档)")

    print(f"\n🎉 向量数据库构建完成！")
    print(f"📁 位置: {CHROMA_DB_PATH}")
    print(f"📊 文档总数: {collection.count()}")

def test_retrieval():
    """测试检索功能"""
    print("\n🧪 测试检索功能...")

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    collection = client.get_collection(
        name="mushroom_knowledge",
        embedding_function=embedding_function
    )

    # 测试查询
    test_queries = [
        "What is Amanita muscaria?",
        "How to identify poisonous mushrooms?",
        "What is mycelium?",
        "Can I eat Death Cap mushrooms?"
    ]

    for query in test_queries:
        print(f"\n❓ 查询: {query}")

        results = collection.query(
            query_texts=[query],
            n_results=3  # 返回前3个最相关的结果
        )

        print("📄 检索结果:")
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
            topic = metadata.get('topic', 'Unknown')
            section = metadata.get('section', 'Unknown')
            print(f"  [{i}] {topic} ({section})")
            print(f"      {doc[:100]}...")

# ==================== 主逻辑 ====================
def main():
    print("=" * 80)
    print("🔍 构建蘑菇知识 RAG 向量数据库")
    print("=" * 80)

    # 1. 加载数据
    documents = load_mushroom_data()

    if not documents:
        print("❌ 没有数据可以处理！")
        print("💡 请先运行:")
        print("   - python data/scripts/scrape_mushroom_wiki.py")
        print("   - python data/scripts/scrape_related_topics.py")
        return

    # 2. 构建向量数据库
    build_vector_database(documents)

    # 3. 测试检索
    test_retrieval()

    print("\n" + "=" * 80)
    print("✅ 完成！现在可以使用 RAG 系统了")
    print("=" * 80)
    print("\n💡 下一步:")
    print("   python data/RAG_dataset/rag_query.py \"What is Amanita muscaria?\"")

if __name__ == "__main__":
    main()
