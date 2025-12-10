# Finance Chatbot APP Folders

This repository contains a collection of experimental applications exploring document ingestion, retrieval, and AI-driven evaluation systems.
Each folder represents an independent project built around modern LLM workflows and data interaction pipelines.

---

## Projects Overview

### `rag_bio_project`

Implements a lightweight Retrieval-Augmented Generation (RAG) workflow for biography data.
Includes loaders, text splitters, embedding generators, and a Chroma-based vector store for multi-source documents.

### `custom-code` (PDF Crawler & Upload Menu)

A custom module integrated directly into Open WebUI that adds a floating PDF crawler button for uploading, crawling, and managing PDF documents within Knowledge Bases. Features automatic link extraction, thumbnail previews, and Knowledge Base integration.

**Quick Setup:**
1. Run `docker-compose up -d` (volume mounts handle the integration)
2. Add a browser bookmark with this URL:
   ```
   javascript:fetch('/api/v1/custom/inject-script?_='+Date.now()).then(r=>r.text()).then(eval)
   ```
3. Click the bookmark while on Open WebUI to load the floating button

📖 **[Full Documentation](custom-code/README.md)**

### `Grading_feature`

Including the page that can extract all users' chats and show them to admins.

### `scoring_page`

A web interface and backend for automatically scoring question quality using large language models.
Combines a **Node.js** API with a **Tailwind + Chart.js** frontend for visualization.

### `grading_feature`

A comprehensive chat extraction and analysis toolkit for OpenWebUI conversations.
Provides multiple methods to extract, organize, and analyze student-AI chat interactions from SQLite databases and exported JSON files.

---

## Tech Stack

* **Languages:** Python, JavaScript
* **Frameworks:** FastAPI, React, Node.js (Express)
* **Libraries/Tools:** Chroma, OpenAI-compatible APIs, TailwindCSS, Chart.js
* **Deployment:** Local or containerized environments (Docker-ready)

---

## Getting Started

1. Clone the repository

   ```bash
   git clone https://github.com/yourusername/ai-doc-projects.git
   cd ai-doc-projects
   ```
2. Navigate into any subfolder (`rag_bio_project/`, `upload_pdf_app/`, `scoring_page/`)
3. Follow that project’s individual README for setup and run instructions

---

## Environment

Each project uses a `.env` file to manage API keys and runtime variables such as:

* `OPENAI_API_KEY`
* `OPENWEBUI_BASE_URL`
* `QWEN_API_KEY`

---

## Purpose

These projects were created to:

* Experiment with **retrieval-based pipelines** and **embedding stores**
* Explore **document review and knowledge syncing** workflows
* Evaluate **question quality and reasoning metrics** with large models

---

## License

Open for research, experimentation, and personal development use.
