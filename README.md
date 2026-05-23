# Research Copilot AI

> A self-hosted, end-to-end AI-powered assistant that lets you query research papers and code repositories using natural language — powered entirely by open-source LLMs and a production-grade RAG pipeline. **No paid APIs. No data sent to the cloud.**

---

## Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [RAG Pipeline Deep Dive](#rag-pipeline-deep-dive)
- [API Reference](#api-reference)
- [Future Improvements](#future-improvements)
- [Skills Demonstrated](#skills-demonstrated)

---

## Overview

**Research Copilot AI** is a production-style Retrieval-Augmented Generation (RAG) system built for ML researchers and engineers. Upload any research paper (PDF) or code repository, then ask complex technical questions and receive grounded, citation-backed answers — all running locally on your own machine.

The system is built around advanced retrieval techniques including **Hierarchical Chunking**, **Hybrid BM25 + Semantic Search**, **Reciprocal Rank Fusion (RRF)**, and **Cross-Encoder Reranking**, making it significantly more accurate than naive vector-search chatbots.

---

## Demo Video

[![Watch the Demo](https://img.youtube.com/vi/dQxuZEU6Mhc/0.jpg)](https://youtu.be/dQxuZEU6Mhc)

---


## Features

### Research Paper Q&A
Upload any PDF and ask questions about architectures, training methods, equations, experimental setups, and more.
```
"How does LoRA reduce GPU memory usage?"
"What optimizer was used and what were the key hyperparameters?"
```

### Codebase Understanding
Index GitHub repositories and query functions, classes, and implementation logic.
```
"Where is the attention mechanism implemented in this repo?"
"How is the loss function calculated?"
```

### Hierarchical Retrieval
Instead of flat fixed-size chunking, the system builds a three-level chunk hierarchy:
- **Level 1** — 2048 tokens (document/section level)
- **Level 2** — 512 tokens (paragraph level)
- **Level 3** — 128 tokens (sentence level)

Only leaf nodes are retrieved, but parent context is preserved for generation.

### Hybrid Search with Reciprocal Rank Fusion (RRF)
Combines dense semantic search and sparse BM25 keyword retrieval, then fuses the ranked lists using RRF (`k=60`) for superior retrieval precision across both exact technical terms and semantic concepts.

### Cross-Encoder Reranking
Retrieved candidates are reranked using a `cross-encoder/ms-marco-MiniLM-L-6-v2` model before generation:
```
Retrieve Top 10 (Vector) + Top 10 (BM25)
         ↓ RRF Fusion
     Fused Ranked List
         ↓ Cross-Encoder Reranking
         Top 5 Final Chunks
         ↓ LLM Generation
```

### Citation-Grounded Responses
Every answer includes the source document filenames used to generate it, reducing hallucinations and improving trust.

### Fully Local Inference
All LLM inference runs via **Ollama** on your own machine:
- `gemma3:1b` (default — fast, memory-efficient)
- `deepseek-r1`
- `phi-3`

Zero API cost. Zero data leaving your machine.

---

## Architecture

```
  ┌─────────────────────────────────────────┐
  │         User Upload (PDF / Code)        │
  └────────────────────┬────────────────────┘
                       │
                       ▼
  ┌─────────────────────────────────────────┐
  │     Document Parsing (PyMuPDF /         │
  │     SimpleDirectoryReader / Tree-sitter)│
  └────────────────────┬────────────────────┘
                       │
                       ▼
  ┌─────────────────────────────────────────┐
  │   Hierarchical Chunking (2048/512/128)  │
  └────────────────────┬────────────────────┘
                       │
                       ▼
  ┌─────────────────────────────────────────┐
  │   Embedding (BAAI/bge-large-en-v1.5)   │
  └────────────────────┬────────────────────┘
                       │
                       ▼
  ┌─────────────────────────────────────────┐
  │        ChromaDB Vector Store            │
  └────────────────────┬────────────────────┘
                       │
           ┌───────────┴───────────┐
           ▼                       ▼
  ┌─────────────────┐   ┌─────────────────────┐
  │  Vector Search  │   │   BM25 Retrieval    │
  │   (Top 10)      │   │     (Top 10)        │
  └────────┬────────┘   └──────────┬──────────┘
           └───────────┬───────────┘
                       ▼
  ┌─────────────────────────────────────────┐
  │    Reciprocal Rank Fusion (RRF, k=60)  │
  └────────────────────┬────────────────────┘
                       │
                       ▼
  ┌─────────────────────────────────────────┐
  │  Cross-Encoder Reranker → Top 5 Chunks  │
  └────────────────────┬────────────────────┘
                       │
                       ▼
  ┌─────────────────────────────────────────┐
  │   Gemma3 / DeepSeek via Ollama (Local)  │
  └────────────────────┬────────────────────┘
                       │
                       ▼
  ┌─────────────────────────────────────────┐
  │    Citation-Grounded Answer to User     │
  └─────────────────────────────────────────┘
```

---

## Tech Stack

| Category | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **RAG Framework** | LlamaIndex |
| **LLM Inference** | Ollama (Gemma3, DeepSeek-R1, Phi-3) |
| **Embedding Model** | BAAI/bge-large-en-v1.5 (HuggingFace) |
| **Vector Database** | ChromaDB (persistent local store) |
| **Semantic Search** | VectorIndexRetriever (LlamaIndex) |
| **Keyword Search** | BM25Retriever (LlamaIndex) |
| **Retrieval Fusion** | Reciprocal Rank Fusion (custom implementation) |
| **Reranking** | cross-encoder/ms-marco-MiniLM-L-6-v2 (SBERT) |
| **Chunking Strategy** | HierarchicalNodeParser (LlamaIndex) |
| **PDF Parsing** | PyMuPDF, Unstructured.io |
| **Code Parsing** | Tree-sitter |
| **Backend API** | FastAPI |
| **Frontend UI** | Streamlit |
| **Deployment** | Hugging Face Spaces / Render |

---

## Project Structure

```
ai-copilot/
│
├── app.py                  # Streamlit frontend UI + model initialization
│
├── api/
│   └── server.py           # FastAPI backend with /upload_file and /query endpoints
│
├── core/
│   ├── ingestion.py        # Document loading + hierarchical chunking pipeline
│   ├── retrieval.py        # HybridRRFRetriever, ChromaDB setup, reranker
│   └── generation.py       # Prompt template + RetrieverQueryEngine assembly
│
├── chroma_db/              # Persistent ChromaDB vector store (auto-created)
│
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally
- At least 8GB RAM (16GB recommended for larger models)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/research-copilot-ai.git
cd research-copilot-ai
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull the LLM Model via Ollama

```bash
ollama pull gemma3:1b
```

> You can also use `deepseek-r1` or `phi3`. Update the model name in `app.py` accordingly.

### 5. Run the Streamlit App

```bash
streamlit run app.py
```

### 6. (Optional) Run the FastAPI Backend

```bash
uvicorn api.server:app --reload --port 8000
```

---

## Usage

### Streamlit UI

1. Open the app at `http://localhost:8501`
2. In the **sidebar**, upload one or more files (PDFs, `.py`, `.txt`, `.md`, etc.)
3. Click **"Index Documents"** and wait for the embedding pipeline to complete
4. Type your question in the chat input at the bottom
5. Receive a streamed, context-grounded answer with source citations

### FastAPI (Programmatic Access)

**Upload a document:**
```bash
curl -X POST http://localhost:8000/upload_file \
  -F "file=@paper.pdf"
```

**Query the system:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main contribution of this paper?"}'
```

**Example response:**
```json
{
  "answer": "The paper introduces LoRA, a method that reduces GPU memory usage by...",
  "sources_used": ["attention_is_all_you_need.pdf"]
}
```

---

## RAG Pipeline Deep Dive

### Hierarchical Chunking

Unlike naive fixed-size chunking, the system uses `HierarchicalNodeParser` to create a parent-child node tree. Only leaf nodes (128 tokens) are indexed and retrieved, but they maintain references to their parent nodes (512 and 2048 tokens), preserving broader context during generation.

```python
HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128],
    chunk_overlap=20
)
```

### Reciprocal Rank Fusion (RRF)

The custom `HybridRRFRetriever` fuses ranked results from both retrievers:

```
score(node) = 1/(rank_vector + k) + 1/(rank_bm25 + k)    where k = 60
```

This is the standard RRF formula from the original Cormack et al. paper. Using `k=60` prevents high-ranked documents from dominating and rewards consistent performance across both retrievers.

### Reranking

After fusion, the top candidates are reranked by a **cross-encoder** (`ms-marco-MiniLM-L-6-v2`), which jointly encodes the query and each chunk to produce a relevance score — far more accurate than bi-encoder similarity alone. The top 5 chunks are passed to the LLM.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/upload_file` | `POST` | Upload and index a file into the RAG system |
| `/query` | `POST` | Submit a natural language question and get an answer |

### `POST /upload_file`

| Parameter | Type | Description |
|---|---|---|
| `file` | `UploadFile` | Any supported file (PDF, .py, .md, .txt, etc.) |

### `POST /query`

| Parameter | Type | Description |
|---|---|---|
| `question` | `string` | Natural language question about the uploaded documents |

---

## Future Improvements

- [ ] **Ragas evaluation pipeline** — measure faithfulness, answer relevancy, and context precision
- [ ] **Multi-user session isolation** — scoped ChromaDB collections per session
- [ ] **Docker containerization** — one-command setup with `docker-compose`
- [ ] **CI/CD pipeline** — automated testing and deployment
- [ ] **Multi-modal retrieval** — image embeddings for figures and diagrams in papers
- [ ] **Graph-based retrieval** — knowledge graph integration for entity-level reasoning
- [ ] **Agentic RAG workflows** — multi-step reasoning with tool use
- [ ] **Fine-tuned embedding models** — domain-adapted embeddings for ML literature
- [ ] **Real-time repository indexing** — auto-sync with GitHub URLs
- [ ] **Authentication + multi-user support**

---

## Skills Demonstrated

- Retrieval-Augmented Generation (RAG)
- LLM Engineering & Prompt Design
- Semantic + Hybrid Information Retrieval
- Vector Databases (ChromaDB)
- Reciprocal Rank Fusion (RRF)
- Cross-Encoder Reranking
- Hierarchical Document Chunking
- Local LLM Deployment (Ollama)
- Open-Source AI Infrastructure
- FastAPI Backend Development
- Streamlit Frontend Development
- Document Parsing (PDF + Code)
- System Design for AI Applications
- Async & Streaming Response Handling
- Metadata-Aware Retrieval

---

## Author

Built by **[Aman Singh Budhala]** — feel free to connect on [LinkedIn](https://www.linkedin.com/in/aman-singh-budhala-a7037324a/).
