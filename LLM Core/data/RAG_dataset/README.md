# 🔍 蘑菇知识 RAG 系统

基于向量检索的增强生成系统，可以实时查询蘑菇知识库。

## 📋 系统组成

```
RAG 系统架构:
用户问题 → 向量检索 (Chroma) → 找到相关文档 → 喂给 LLM (Ollama) → 生成答案
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install chromadb sentence-transformers
```

### 2. 构建向量数据库

```bash
python data/RAG_dataset/build_rag_database.py
```

**数据来源**:
- `data/raw_data_source/raw_mushroom_wiki_data.json` (35种蘑菇)
- `data/raw_data_source/related_topics_wiki_data.json` (30+个延展话题)

**输出**:
- 向量数据库: `data/RAG_dataset/chroma_db/`
- 文档总数: ~200-400 个文档块

### 3. 使用方式

#### 方式A: 命令行查询

```bash
python data/RAG_dataset/rag_query.py "What is Amanita muscaria?"
```

#### 方式B: 交互式查询（推荐）

```bash
python data/RAG_dataset/rag_interactive.py
```

然后输入问题进行对话。

## 📊 工作流程详解

### 步骤1: 向量化文档

```python
# 将每个蘑菇的 summary 和 sections 切分成独立文档块
documents = [
    {
        "id": "mushroom_Amanita_muscaria_summary",
        "text": "Amanita muscaria is a basidiomycote mushroom...",
        "metadata": {
            "type": "mushroom",
            "topic": "Amanita muscaria",
            "section": "summary"
        }
    },
    ...
]

# 使用 sentence-transformers 将文本转换为向量
embedding_model = "all-MiniLM-L6-v2"  # 384维向量
```

### 步骤2: 相似度检索

```python
# 用户问题也转换为向量，然后计算余弦相似度
query = "What is Amanita muscaria?"
query_vector = embed(query)

# 找到最相似的前K个文档
top_k_docs = vector_db.search(query_vector, k=5)
```

### 步骤3: LLM 生成答案

```python
# 将检索到的文档作为上下文喂给 LLM
prompt = f"""
Context: {retrieved_docs}
Question: {query}
Answer:
"""

answer = ollama.run(prompt)
```

## 🎯 示例查询

### 查询具体蘑菇

```bash
python data/RAG_dataset/rag_query.py "Is Amanita phalloides deadly?"
```

**输出**:
```
🔍 检索相关文档...
✅ 找到 5 个相关文档

📄 [1] Amanita phalloides - Toxicity (相似度: 95%)
    Amanita phalloides contains amatoxins which are highly toxic...

💬 答案:
Yes, Amanita phalloides (Death Cap mushroom) is extremely deadly.
It contains amatoxins that destroy liver cells, and just half a
mushroom can be fatal. Most deaths from mushroom poisoning worldwide
are caused by this species.

📚 信息来源:
  [1] Amanita phalloides
      https://en.wikipedia.org/wiki/Amanita_phalloides
```

### 查询通用概念

```bash
python data/RAG_dataset/rag_query.py "What is mycelium?"
```

### 查询烹饪相关

```bash
python data/RAG_dataset/rag_query.py "How to cook mushrooms safely?"
```

## ⚙️ 配置选项

### 修改嵌入模型

编辑 `build_rag_database.py`:

```python
# 轻量级（快速）
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 默认

# 高精度（稍慢）
EMBEDDING_MODEL = "all-mpnet-base-v2"

# 支持中文
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
```

### 修改检索数量

编辑 `rag_query.py`:

```python
TOP_K = 5  # 检索前5个文档（默认）
TOP_K = 10 # 检索更多文档（更全面但可能引入噪声）
```

### 修改 LLM 模型

编辑脚本中的:

```python
OLLAMA_MODEL = "qwen2.5vl:32b"  # 你当前的模型
# OLLAMA_MODEL = "llama2:7b"    # 其他模型
```

## 🆚 RAG vs 微调对比

| 特性 | RAG | 微调 |
|------|-----|------|
| **知识更新** | ✅ 实时，只需更新数据库 | ❌ 需要重新训练 |
| **可解释性** | ✅ 可以看到引用来源 | ❌ 黑盒 |
| **准确性** | ✅ 基于真实文档，不易幻觉 | ⚠️ 可能产生幻觉 |
| **成本** | 低（无需训练） | 高（需要 GPU 训练） |
| **响应速度** | 稍慢（需要检索） | 快 |
| **适用场景** | 事实查询、知识问答 | 风格模仿、任务执行 |

## 💡 最佳实践

### 混合使用 RAG + 微调

1. **RAG**: 用于准确的知识查询
   - "What is Amanita muscaria?"
   - "Is this mushroom poisonous?"

2. **微调模型**: 用于对话风格和任务执行
   - 聊天语气
   - 对话上下文理解
   - 个性化回复

### 建议架构

```python
if is_factual_question(user_query):
    # 使用 RAG 查询知识库
    answer = rag_system.query(user_query)
else:
    # 使用微调模型生成回复
    answer = finetuned_model.generate(user_query)
```

## 🔧 故障排查

### 问题1: `ModuleNotFoundError: No module named 'chromadb'`

**解决**:
```bash
pip install chromadb sentence-transformers
```

### 问题2: 检索结果不准确

**解决**:
- 增加 TOP_K 值
- 更换更好的嵌入模型 (`all-mpnet-base-v2`)
- 检查数据库是否正确构建

### 问题3: LLM 响应太慢

**解决**:
- 减少 TOP_K（提供更少的上下文）
- 使用更小的 LLM 模型
- 增加 timeout 时间

## 📈 性能优化

### 1. 使用 GPU 加速嵌入

```python
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    device=device
)
```

### 2. 缓存常见查询

```python
query_cache = {}

if query in query_cache:
    return query_cache[query]
else:
    result = rag_query(query)
    query_cache[query] = result
    return result
```

## 📚 扩展阅读

- [Chroma Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [RAG 原理论文](https://arxiv.org/abs/2005.11401)

---

🎉 **现在你有了一个完整的 RAG 系统！**
