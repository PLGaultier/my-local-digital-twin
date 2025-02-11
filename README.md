# 🤖 AI-Powered Digital Twin (Local-First Edition)

A privacy-focused, locally-running AI assistant that acts as your digital twin using **RAG architecture** combined with cutting-edge open-source LLMs. Zero OpenAI dependencies!

![Demo Gif](./assets/demo.gif) *Example: User interacting with digital twin*

## 🚀 Key Innovations

### 🌐 **Core Architecture**
| Technique | Implementation | Benefit |
|-----------|----------------|---------|
| **Local LLMs** | Ollama + Mistral-7B/Llama 3 | No API costs, full data privacy |
| **RAG Pipeline** | FAISS + LangChain | Context-aware responses from your CV |
| **Open-Source Embeddings** | sentence-transformers/all-MiniLM-L6-v2 | Free alternative to OpenAI embeddings |
| **GPU Optimization** | Metal for M3 Pro (`num_gpu=35`) | 3-4x faster inference vs CPU-only |
| **Memory Caching** | `@st.cache_resource` | Instant document reloads between sessions |

### 💡 **Advanced Features**
- **Dynamic Context Injection**:  
  Automatically enhances prompts with relevant CV excerpts
  ```python
  relevant_docs = db.similarity_search(prompt, k=4)  # Semantic search
  ```
  
- **Personality Engineering**:
  ```python
  system="You are Pierre-Louis' twin. Use formal French professional tone..."
  ```

- **Anti-Hallucination Guardrails**:
  - Strict context filtering via RAG
  - Low-temperature parameter (`0.3`)
  - Negative prompting against speculation

---

## 🛠️ Tech Stack

| Category          | Tools                                                                 |
|-------------------|-----------------------------------------------------------------------|
| **LLM Runtime**   | Ollama (Mistral 7B)                                          |
| **Framework**      | LangChain (Document processing, RAG pipeline)                        |
| **Vector DB**      | FAISS (HNSW indexes for fast similarity search)                      |
| **Embeddings**     | HuggingFace all-MiniLM-L6-v2 (384-dim) × Open-source                 |
| **UI**            | Streamlit (Interactive web interface)                               |
| **Hardware**       | Apple M3 Pro (18GB unified memory)               |

---

