<!-- rag_bio_project/README.md -->

# rag_bio_project

A small, self-contained Retrieval-Augmented Generation (RAG) pipeline used for finance-related biography documents. It includes:

- Multi-source loading (local PDFs/TXT + optional URLs)
- Type-aware text splitting
- Embedding + ChromaDB vector store utilities
- A retriever with routing and strictness controls
- Prompt templates + LLM wrapper
- An end-to-end `run_pipeline` function and test utilities

This project is intended as a testbed / utility RAG pipeline that other apps in the repo can reuse.

---

## 1. Project structure (high level)

Key files:

- `loader.py` — load documents from:
  - `data_pdfs/` (PDFs)
  - `data_txt/` (TXT/Markdown)
  - Optional list of web URLs
- `splitter.py` — type-aware splitting logic:
  - `split_documents_type_aware(docs, default_type="pdf")`
- `embeddings.py` — build embedding models and vector stores
  - `EmbeddingConfig` + helpers
  - `build_embeddings(...)`
  - `build_embeddings_and_vectorstores(...)`
- `vectorstore.py` — ChromaDB utilities
  - `get_collection(...)`
  - `quick_query(...)`
- `retriever.py` — query orchestration over Chroma
  - high-threshold retrieval, MMR, routing over `{username}_{character}` collections
- `prompting.py` — prompt templates and auto mode selection
  - `build_prompt_messages_auto(...)`
- `llm.py` — LLM wrapper
  - `LLMConfig`
  - `build_chat_model(...)`
- `pipeline.py` — minimal end-to-end RAG call
  - `PipelineConfig`
  - `run_pipeline(question, cfg)`
- `middleware.py` — optional “character detection” + answer verification
- `tests.py` — small helpers to sanity-check retrieval and the pipeline
- `rag_bio_colab_e2e_continued_en.ipynb` — Colab-style notebook demo
- `.env` — local environment variables (API keys, etc.)
- `.env` — local environment variables (API keys, etc.)
- 'training_testing_pipeline' contains a pipeline used for training function: train_unified for implementation and a notebook showing how to use it.
---

## 2. Prerequisites

- Python ≥ 3.10
- A working virtual environment is recommended
- A vector DB directory (default: `index/`) will be created for Chroma
- API keys (set via environment variables or `.env`), for example:
  - `OPENAI_API_KEY` (if using OpenAI models)
  - optional others depending on your `provider` in `LLMConfig` / `PipelineConfig`

You can either:

- Install dependencies manually based on imports in the modules (`langchain-core`, `langchain-community`, `chromadb`, `openai`, `python-dotenv`, etc.),
- Or add your own `requirements.txt` and install with:

```bash
pip install -r requirements.txt
