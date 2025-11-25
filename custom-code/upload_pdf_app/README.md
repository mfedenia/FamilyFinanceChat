# PDF Review & Crawler Dashboard

A local web application for uploading PDFs, crawling linked documents, generating thumbnails, and syncing selected files with an OpenWebUI knowledge base.  
Built with **FastAPI (Python)** for the backend and **React** for the frontend.


## Features
- Upload multiple PDF files  
- Crawl links from uploaded PDFs  
- Auto-generate thumbnail previews  
- Include/exclude files before final upload  
- Sync final PDFs with an OpenWebUI Knowledge Base  
- CORS-enabled backend for easy local testing  

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/pdf-review-dashboard.git
cd pdf-review-dashboard
````

### 2. Make the script executable

```bash
chmod +x run_local.sh
```

### 3. Run locally

```bash
./run_local.sh
```

This will:

1. Create and activate a Python virtual environment
2. Install backend dependencies (`pip install -r requirements.txt` or run `requirements.sh`)
3. Launch the FastAPI backend at **[http://localhost:8000](http://localhost:8000)**
4. Install npm packages in the frontend directory
5. Start the React app at **[http://localhost:3000](http://localhost:3000)**

Use **Ctrl + C** to stop both servers cleanly.

---

## Environment Variables

The backend uses the following environment variables (you can set them in a `.env` file or system environment):

| Variable             | Description                          | Default                 |
| -------------------- | ------------------------------------ | ----------------------- |
| `OPENWEBUI_BASE_URL` | Base URL of your OpenWebUI instance  | `http://127.0.0.1:3000` |
| `OPENWEBUI_API_KEY`  | API key for OpenWebUI authentication | *(empty)*               |
| `OPENWEBUI_KB_ID`    | Knowledge Base ID for file uploads   | example ID              |

**Example `.env`**

```bash
OPENWEBUI_BASE_URL=http://127.0.0.1:3000
OPENWEBUI_API_KEY=your_api_key_here
OPENWEBUI_KB_ID=your_kb_id_here
```

---

## API Endpoints Overview

| Endpoint              | Method | Description                                                  |
| --------------------- | ------ | ------------------------------------------------------------ |
| `/api/upload`         | POST   | Upload and crawl PDF files                                   |
| `/api/pdfs`           | GET    | List all crawled PDFs                                        |
| `/api/pdfs/{name}`    | PATCH  | Toggle PDF exclusion status                                  |
| `/api/finalize`       | POST   | Move selected PDFs to knowledge base and upload to OpenWebUI |
| `/api/openwebui/test` | GET    | Test OpenWebUI connection                                    |
| `/api/reset`          | DELETE | Reset all state and data                                     |

---

## Cleaning Up

To remove old crawled data, thumbnails, and cached state:

```bash
curl -X DELETE http://localhost:8000/api/reset
```


## License

This project is for local development and testing purposes. You may freely modify or extend it for your own use.
