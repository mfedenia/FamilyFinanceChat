# PDF Review
 
## Backend
 
**OVERVIEW**
 
Powers the PDF Review system that lets professors view newely scraped PDFs, exclude unwanted ones, and finialize which files that are added to the chatbot's knowledge base
- Its built with FastAPI and designed to run with OpenWebUI
 
**Process**
 
This microservice gives info to display for frontend
- List all pdfs with their thumbnails
- Update exclusion of a pdf to the knowledge base
- Finalize uploads (move the uploads to KB)
 
**Flow**
 
1. PDFs are scraped and stored in /data/webscraped/
2. The backend reads them and generates thumbnails
3. The prof marks unwanted PDFs as X to exclude them
4. Clicking Finalize Upload moves approved PDFs to the /data/knowledge_base/ which then gets inputted to the kb in openwebui to retrai the model
 
 
 
## Run the backend server
 
uvicorn backend:app --host 0.0.0.0 --port 8081 --reload
 
API will be avalible at http://localhost:8081
 