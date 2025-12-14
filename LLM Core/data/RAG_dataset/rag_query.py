#!/usr/bin/env python3
"""
🔍 RAG 查询系统
使用向量数据库检索 + LLM 生成答案
"""

import sys
import chromadb
from chromadb.utils import embedding_functions
import subprocess

# ================= 配置 =================
CHROMA_DB_PATH = "data/RAG_dataset/chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen2.5vl:32b"  # 你的本地模型
TOP_K = 5  # 检索前5个最相关的文档
# ========================================

def retrieve_relevant_docs(query: str, top_k: int = TOP_K):
    """从向量数据库检索相关文档"""
    print(f"🔍 检索相关文档...")

    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )

        collection = client.get_collection(
            name="mushroom_knowledge",
            embedding_function=embedding_function
        )

        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )

        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        print(f"✅ 找到 {len(documents)} 个相关文档\n")

        # 显示检索结果
        retrieved_docs = []
        for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances), 1):
            topic = metadata.get('topic', 'Unknown')
            section = metadata.get('section', 'Unknown')
            similarity = 1 - distance  # 转换为相似度

            print(f"📄 [{i}] {topic} - {section} (相似度: {similarity:.2%})")
            print(f"    {doc[:150]}...\n")

            retrieved_docs.append({
                "text": doc,
                "metadata": metadata,
                "similarity": similarity
            })

        return retrieved_docs

    except Exception as e:
        print(f"❌ 检索失败: {e}")
        print("💡 请先运行: python data/RAG_dataset/build_rag_database.py")
        return []

def generate_answer_with_llm(query: str, context_docs: list):
    """使用 LLM 基于检索到的文档生成答案"""
    print("🤖 使用 LLM 生成答案...\n")

    # 构建上下文
    context = "\n\n---\n\n".join([
        f"[Document {i+1}]\n{doc['text']}"
        for i, doc in enumerate(context_docs)
    ])

    # 构建 prompt
    prompt = f"""You are a knowledgeable mushroom expert assistant. Answer the user's question based ONLY on the provided context documents.

Context Documents:
{context}

User Question: {query}

Instructions:
1. Answer based ONLY on the information in the context documents
2. If the context doesn't contain enough information, say "I don't have enough information in my knowledge base to answer that."
3. Be concise but informative (3-5 sentences)
4. Cite which document(s) you used if relevant
5. Use a friendly, conversational tone

Answer:"""

    # 调用 Ollama
    try:
        result = subprocess.run(
            ["ollama", "run", OLLAMA_MODEL, prompt],
            capture_output=True,
            text=True,
            timeout=60
        )

        answer = result.stdout.strip()
        return answer

    except Exception as e:
        return f"❌ LLM 调用失败: {e}"

def main():
    print("=" * 80)
    print("🔍 蘑菇知识 RAG 查询系统")
    print("=" * 80)

    # 获取查询
    if len(sys.argv) < 2:
        print("\n用法: python data/RAG_dataset/rag_query.py \"你的问题\"")
        print("\n示例查询:")
        print('  python data/RAG_dataset/rag_query.py "What is Amanita muscaria?"')
        print('  python data/RAG_dataset/rag_query.py "How to identify poisonous mushrooms?"')
        print('  python data/RAG_dataset/rag_query.py "What is mycelium and how does it grow?"')
        return

    query = " ".join(sys.argv[1:])
    print(f"\n❓ 问题: {query}\n")

    # 1. 检索相关文档
    docs = retrieve_relevant_docs(query)

    if not docs:
        print("❌ 没有找到相关文档")
        return

    # 2. 使用 LLM 生成答案
    answer = generate_answer_with_llm(query, docs)

    # 3. 显示答案
    print("=" * 80)
    print("💬 答案:")
    print("=" * 80)
    print(answer)
    print("\n" + "=" * 80)

    # 4. 显示来源
    print("\n📚 信息来源:")
    for i, doc in enumerate(docs, 1):
        metadata = doc['metadata']
        topic = metadata.get('topic', 'Unknown')
        url = metadata.get('url', '')
        print(f"  [{i}] {topic}")
        if url:
            print(f"      {url}")

if __name__ == "__main__":
    main()
