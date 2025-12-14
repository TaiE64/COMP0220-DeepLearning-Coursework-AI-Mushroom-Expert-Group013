#!/usr/bin/env python3
"""
💬 RAG 交互式查询系统
实时对话式查询蘑菇知识库
"""

import chromadb
from chromadb.utils import embedding_functions
import subprocess

# ================= 配置 =================
CHROMA_DB_PATH = "data/RAG_dataset/chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen2.5vl:32b"
TOP_K = 3  # 检索前3个文档
# ========================================

def retrieve_docs(query: str, collection, top_k: int = TOP_K):
    """检索相关文档"""
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    docs = []
    for doc, metadata, distance in zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ):
        docs.append({
            "text": doc,
            "metadata": metadata,
            "similarity": 1 - distance
        })

    return docs

def generate_answer(query: str, context_docs: list):
    """生成答案"""
    context = "\n\n".join([doc['text'] for doc in context_docs])

    prompt = f"""You are a mushroom expert. Answer based on the context provided.

Context:
{context}

Question: {query}

Answer (be concise and friendly):"""

    try:
        result = subprocess.run(
            ["ollama", "run", OLLAMA_MODEL, prompt],
            capture_output=True,
            text=True,
            timeout=45
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

def main():
    print("=" * 80)
    print("💬 蘑菇知识 RAG 交互式查询")
    print("=" * 80)
    print("\n💡 输入问题进行查询，输入 'quit' 或 'exit' 退出\n")

    # 初始化数据库
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        collection = client.get_collection(
            name="mushroom_knowledge",
            embedding_function=embedding_function
        )
        print(f"✅ 已加载知识库 ({collection.count()} 个文档)\n")
    except Exception as e:
        print(f"❌ 无法加载知识库: {e}")
        print("💡 请先运行: python data/RAG_dataset/build_rag_database.py")
        return

    # 交互循环
    while True:
        try:
            query = input("🍄 你的问题 > ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 再见！")
                break

            # 检索
            print(f"\n🔍 检索中...")
            docs = retrieve_docs(query, collection)

            # 显示检索结果
            print(f"📄 找到 {len(docs)} 个相关文档:")
            for i, doc in enumerate(docs, 1):
                topic = doc['metadata'].get('topic', 'Unknown')
                section = doc['metadata'].get('section', '')
                similarity = doc['similarity']
                print(f"  [{i}] {topic} - {section} ({similarity:.0%})")

            # 生成答案
            print(f"\n🤖 生成答案中...")
            answer = generate_answer(query, docs)

            print(f"\n💬 {answer}\n")
            print("-" * 80 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}\n")

if __name__ == "__main__":
    main()
