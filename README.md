# FamilyFinanceChat

Repository: https://github.com/mfedenia/FamilyFinanceChat

This repository contains several experimental applications exploring document ingestion, retrieval, and AI-driven evaluation workflows. Each project folder is an independent app with its own README and setup steps.

---

## Overview and Motivation

This project was developed for FIN 602 to modernize how students practice financial advising. While the core chatbot runs through OpenWebUI, this GitHub repository extends it with additional features tailored for the course. FIN 602 previously relied on human role-players, which often resulted in inconsistent experiences and scheduling challenges. By using AI to generate realistic family scenarios, the tool provides students with a reliable and engaging way to practice meaningful advising conversations. It supports wealth management students as they learn how to assess client needs, communicate recommendations, and apply course frameworks in a practical setting. The long-term goal is to expand this platform to support more courses within the program and ultimately other universities.

---


## Quick Links
- Repository: https://github.com/mfedenia/FamilyFinanceChat
- Per-project READMEs:
  - [RAG Bio](rag_bio_project/README.md)
  - [Upload and Web Crawler](custom-code/README.md)
  - [Chats Scoring Page](scoring_page/README.md)
  - [Professor Grading Dashboard](grading_feature/README.md)

---

## Extra Documentation (docs/)
The docs/ directory contains extra PDF guides and supporting documentation used for onboarding, deployment, and feature details. Current subfolders include:

- docs/openwebui_setup — step-by-step PDFs and notes for installing/configuring Open WebUI and integrating the custom-code upload/injection.
- docs/vm_setup — VM provisioning and GCP VM setup guides (SSH, firewall, VS Code Remote instructions).
- docs/features — feature-specific documentation and design notes for components and experiments.
- docs/extra — assorted reference PDFs, diagrams, and supplemental materials.

You can view these files directly on GitHub under the docs/ folder.

---

## High-level Setup

These applications have per-project setup instructions in their folders. High-level steps to get the environment running:

1. Clone repository
   ```bash
   git clone https://github.com/mfedenia/FamilyFinanceChat
   cd FamilyFinanceChat
   ```

2. Environment variables
   - Create a `.env` file in the project root or in each project folder as needed with keys such as:
     - OPENAI_API_KEY
     - OPENWEBUI_BASE_URL
     - QWEN_API_KEY
   - See each project README for any additional variables.

3. Running on the GCP VM (current setup)
   - custom-code / Open WebUI integration: typically run via Docker Compose on the VM:
     ```bash
     # from the project or repo root (see custom-code README)
     docker-compose up -d
     ```
     [custom-code](custom-code/README.md) is mounted/injected into your Open WebUI instance via volume mounts and a browser bookmarklet injection. See README

   - scoring_page & grading_feature: intentionally run locally (developer machine) via their run.sh scripts to avoid consuming VM resources.

   - rag_bio_project: follow that folder's README for ingestion and vector store setup (Chroma/embeddings).

4. Testing and verification
   - Follow each project's README for endpoints, web UI paths, or scripts to exercise functionality.

---

## Architecture & How the code works (overview)

- Each project is independent with its own frontend/backend or scripts:
  - custom-code: a module that integrates into Open WebUI (via injected client-side script) to provide a floating PDF upload/crawl UI and management of knowledge bases. Uses backend endpoints (mounted to the Open WebUI container) and persists crawled documents to the configured store.
  - rag_bio_project: lightweight RAG pipeline - loaders, splitters, embedding generation, Chroma vector store, and query paths to perform retrieval-augmented generation for biography documents.
  - scoring_page: Node.js API + Tailwind/Chart.js frontend used to evaluate and visualize question quality via LLM scoring. Intended to be run on-demand.
  - grading_feature: tools to extract Open WebUI conversation data (from SQLite OpenWebUI DB and constantly refreshed local db) and provide admin views / exports for grading and analysis.

- Deployment model in the current setup:
  - Primary services (custom-code/Open WebUI) are hosted on a GCP VM using Docker Compose and volume mounts.
  - Other utilities (scoring_page, grading_feature) are run locally when needed to avoid constant VM resource usage.

---

## What works

(Consolidated repo-level summary)

- custom-code (Upload feature)
  - PDF crawler / upload / extraction and integration into Open WebUI work and are deployed on the GCP VM.
  - The upload and extraction flow is functional and is used in production on the VM.

- rag_bio_project
  - Works fine and is used as a utility for lightweight RAG tasks (ingestion, embeddings, Chroma queries).

- scoring_page
  - Works when run locally via the provided run.sh script. API and frontend render and compute scores as expected for local use.

- grading_feature
  - Works when run locally via run.sh. The grading dashboard and extraction tools function for admin use.

---

## What doesn't work / Known issues & limitations

(Consolidated repo-level summary)

- OpenWebUI
  - Currently, sending 7+ messages into a single chat may cause the chatbot to get stuck
  - To mitigate this, the best solution at the moment is to create a new chat

- custom-code
  - Integration may break after upstream Open WebUI or container/image upgrades. Common causes include changes to Open WebUI API/DOM, container path changes, or volume mounting differences.
  - Suggested mitigations are in the roadmap below.

- scoring_page & grading_feature
  - Not hosted on the VM to avoid constant resource usage - they must be run manually via run.sh on a development machine or admin workstation.
  - grading dashboard usage currently requires an admin (professors / TAs) to SSH into the GCP VM using VS Code and run ./run_app.sh due to firewall restrictions. Documentation to set up this access will be provided.
  - No production-grade deployment, reverse proxying, or HTTPS by default for these local apps.

- rag_bio_project
  - Currently used as a utility and functioning; no major issues reported. If you encounter Chroma persistence or memory issues, report specifics for this README to be updated.

- General operational limitations
  - No multi-tenant packaging or per-tenant isolation; data and deployments are single-tenant in current setup.
  - Limited monitoring, auto-restart, and backup strategies for services running on the GCP VM.
  - No CI/CD safety checks or automated smoke tests run after upgrades (this contributes to upgrade breaks noted above).

---

## How to verify / quick smoke tests

- custom-code:
  - Ensure the Open WebUI container has the custom-code integration mounted per custom-code/README.
  - In browser, open Open WebUI, run bookmarklet or injected UI, upload a PDF, and confirm extraction and knowledge base persistence.

- scoring_page:
  - From scoring_page/ run:
    ```bash
    ./run.sh
    # visit http://localhost:<port> as described in scoring_page/README.md
    ```
  - Submit sample input and confirm scoring output and charts render.

- grading_feature:
  - For GCP-hosted grading dashboard access, SSH into the GCP VM with VS Code Remote and go to grading_feature/ and then run:
    ```bash
    ./run_app.sh
    ```
    Documentation for SSH/VS Code access is [here](docs/vm_setup/vscode_ssh_setup.pdf).

---

## What we'd work on next (roadmap / prioritized next steps)

Prioritized next steps and recommended actions:

1. Improve Open WebUI model latency (P0 - medium/large)
   - Add streaming output support so users see partial responses earlier.
   - Tune model settings, batching, and tokens to reduce perceived latency.
   - Consider using smaller/faster models or caching common responses where appropriate.

2. Explore multi-tenant feasibility (P0 - large)
   - Create a containerized image and deployment recipe (Docker image + docker-compose profiles or Helm) for per-tenant instances.
   - Design data isolation for per-tenant vector stores and configuration.
   - Plan authentication and quota controls.

3. Host scoring_page & grading_feature instead of run.sh (P1 - medium)
   - Containerize these apps and run them under process management (systemd / docker-compose).
   - Add reverse proxy (Nginx) with HTTPS, basic auth or OAuth for access control.
   - Provide an option to run them on-demand via a job scheduler or as on-call services.

4. Harden upgrade resilience for custom-code (P1 - small/medium)
   - Add smoke-test scripts that run post-upgrade to detect injection/API breaks.
   - Make the injection script tolerant to minor DOM/API changes in Open WebUI.
   - Automate re-injection or provide a small init script to fix common post-upgrade issues.

5. Upgrade VM & add monitoring (P1 - small)
   - Upgrade VM resources or consider scaling options if load warrants.
   - Add monitoring and alerting (GCP Monitoring or Prometheus + Grafana) and centralized logging.
   - Add backups for vector stores and SQLite/DB files.

6. CI/CD & testing (P2 - medium)
   - Add GitHub Actions for linting, unit tests, and a smoke test for custom-code integration.
   - Add image build/publish steps for containerized components.

7. Security & access control (P0/P1)
   - Add authentication for admin UIs, secure handling of API keys, and secrets rotation.
   - Implement per-institution authentication for multi-tenant deployments.

---

## Contact / Maintainers

- Primary maintainer: @mfedenia (repo owner)
- For grading dashboard access and run instructions: admins (professors / TAs) should use SSH + VS Code Remote to run the services as described above.
