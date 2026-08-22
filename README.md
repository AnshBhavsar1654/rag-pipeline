# 📚 RAG Pipeline — Unified Chatbot

A state-of-the-art, modular Retrieval-Augmented Generation pipeline. **Swap your document, get a chatbot instantly.**

> Built from [RAG from Scratch](https://youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x) techniques (Parts 1–18), consolidated into a production-ready project.

![RAG Architecture](https://github.com/langchain-ai/rag-from-scratch/assets/122662504/54a2d76c-b07e-49e7-b4ce-fc45667360a1)

---

## ✨ Features

| Feature | Description |
|---|---|
| **🔄 Multi-Query Retrieval** | Generates query variants for broader recall |
| **⚡ RAG-Fusion + RRF** | Reciprocal Rank Fusion for superior ranking |
| **🔮 HyDE** | Hypothetical Document Embeddings for semantic search |
| **📝 Question Decomposition** | Breaks complex questions into sub-questions |
| **🔙 Step-Back Prompting** | Generates generic questions for broader context |
| **🏆 Re-ranking** | Cross-encoder or Cohere re-ranking layer |
| **💬 Conversation Memory** | Multi-turn dialogue with sliding window |
| **📄 Universal Loader** | PDF, TXT, MD, DOCX, CSV, HTML, URLs |
| **🎨 Web UI + CLI** | Gradio chatbot and terminal interface |
| **⚙️ Single Config** | One YAML file to control everything |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API Key

Create a `.env` file in the project root (auto-loaded on startup):

```env
GOOGLE_API_KEY=your-gemini-key
GROQ_API_KEY=your-groq-key
```

Or set it manually:

```bash
# Windows
set GOOGLE_API_KEY=your-key

# Linux/Mac
export GOOGLE_API_KEY=your-key
```

### 3. Add Your Documents

Place your documents in `sample_docs/` or edit `config/config.yaml`:

```yaml
documents:
  sources:
    - "./sample_docs/"           # Local directory
    - "./my_report.pdf"          # Single file
    - "https://example.com/page" # Web URL
```

### 4. Ingest Documents

```bash
python ingest.py
```

### 5. Start Chatting

**Web UI (Gradio):**
```bash
python app.py
```

**Terminal CLI:**
```bash
python cli.py
```

---

## ⚙️ Configuration

All settings are in **`config/config.yaml`**. Key options:

### Retrieval Strategy
```yaml
retrieval:
  strategy: "rag_fusion"  # simple | multi_query | rag_fusion | hyde
  top_k: 5
```

### LLM Provider
```yaml
llm:
  provider: "google"       # google | groq | openai | ollama
  model: "gemini-2.0-flash"
  temperature: 0.1
```

### Re-ranking (Optional)
```yaml
reranking:
  enabled: true
  provider: "cross_encoder"
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

---

## 🏗️ Architecture

```
RAG/
├── config/config.yaml          # ← Edit this file
├── src/
│   ├── config.py               # Pydantic config loader
│   ├── document_loaders/       # Universal document loader
│   ├── chunking/               # Text splitting
│   ├── embeddings/             # Embedding model factory
│   ├── vectorstore/            # Chroma with persistence
│   ├── retrieval/              # Strategy pattern
│   │   ├── multi_query.py      # Multi-query (Part 5)
│   │   ├── rag_fusion.py       # RAG-Fusion + RRF (Parts 6, 15)
│   │   ├── hyde.py             # HyDE (Part 9)
│   │   └── reranker.py         # Re-ranking layer
│   ├── query_transform/        # Query transformations
│   │   ├── decomposition.py    # Question decomposition (Part 7)
│   │   └── step_back.py        # Step-back prompting (Part 8)
│   ├── generation/             # LLM generation
│   ├── pipeline/               # Main orchestrator
│   └── chat/                   # Conversational memory
├── app.py                      # Gradio web UI
├── cli.py                      # Terminal chatbot
├── ingest.py                   # Document ingestion
└── notebooks/                  # Original reference notebooks
```

### Pipeline Flow

```
User Question
     │
     ▼
┌─────────────────────┐
│  Query Transform     │  (Multi-Query / RAG-Fusion / HyDE)
│  Generate variants   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Retrieval           │  (Vector similarity search per query)
│  Chroma DB           │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Re-ranking          │  (Optional: Cross-encoder / Cohere)
│  Score & filter      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Generation          │  (LLM with context + chat history)
│  OpenAI / Ollama     │
└──────────┬──────────┘
           │
           ▼
      Answer + Sources
```

---

## 📄 Supported Document Types

| Type | Extension | Loader |
|---|---|---|
| PDF | `.pdf` | PyPDFLoader |
| Plain Text | `.txt` | TextLoader |
| Markdown | `.md` | TextLoader |
| Word | `.docx` | Docx2txtLoader |
| CSV | `.csv` | CSVLoader |
| HTML | `.html`, `.htm` | BSHTMLLoader |
| Web URL | `http://...` | WebBaseLoader |

---

## 🔧 Advanced Usage

### Using Groq (Fast, Free Tier)

```yaml
llm:
  provider: "groq"
  model: "llama-3.3-70b-versatile"
```

> Note: Groq provides LLMs only — pair it with `huggingface` or `google` embeddings.

### Using Google Gemini

```yaml
llm:
  provider: "google"
  model: "gemini-3.6-flash"

embedding:
  provider: "google"
  model: "models/gemini-embedding-001"
```

### Using Ollama (Local LLMs)

```yaml
llm:
  provider: "ollama"
  model: "llama3"
  base_url: "http://localhost:11434"
```

### Using HuggingFace Embeddings (Free)

```yaml
embedding:
  provider: "huggingface"
  model: "sentence-transformers/all-MiniLM-L6-v2"
```

### Enabling Re-ranking

```yaml
reranking:
  enabled: true
  provider: "cross_encoder"
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  top_k: 3
```

---

## 📓 Reference Notebooks

The original "RAG from Scratch" notebooks are preserved in `notebooks/`:

| Notebook | Topics |
|---|---|
| Parts 1–4 | Indexing, Retrieval, Generation basics |
| Parts 5–9 | Multi-Query, RAG-Fusion, Decomposition, Step-Back, HyDE |
| Parts 10–11 | Routing, Query Construction |
| Parts 12–14 | Multi-representation Indexing, RAPTOR, ColBERT |
| Parts 15–18 | Re-ranking, CRAG, Long Context |