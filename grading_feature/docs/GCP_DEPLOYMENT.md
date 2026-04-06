# GCP Deployment Guide

This guide helps you deploy the unified grading & scoring dashboard to Google Cloud Platform (GCP) alongside your OpenWebUI chatbot.

## Architecture Overview

```
GCP Instance
├── OpenWebUI (http://localhost:8000 or exposed port)
├── Grading Dashboard Frontend (served by backend on same port)
└── Grading Dashboard Backend (FastAPI)
```

## Deployment Steps

### 1. Build the Frontend

```bash
cd grading_feature/frontend
npm run build
```

This creates a `dist/` folder with production-ready files.

### 2. Serve Frontend + Backend from FastAPI

The easiest approach: FastAPI serves both the static React build and the API.

Update `grading_feature/backend/main.py`:

```python
from fastapi.staticfiles import StaticFiles
import os

# ... existing code ...

# Serve static frontend build
frontend_build_path = os.path.join(os.path.dirname(__file__), "../frontend/dist")
app.mount("/", StaticFiles(directory=frontend_build_path, html=True), name="static")
```

This makes FastAPI serve the React app at `/` and API endpoints at `/api/...`, `/users`, etc.

### 3. Deploy to GCP

#### Option A: GCP App Engine (Recommended for simplicity)

1. Create `app.yaml` in the root of `grading_feature/backend/`:

```yaml
runtime: python311
env: standard
entrypoint: uvicorn main:app --host 0.0.0.0 --port 8080

env_variables:
  OPENAI_API_KEY: "your-key-here"
  OPENAI_MODEL: "gpt-4o-mini"
  DB_PATH: "/mnt/sqlite/openwebui.db"  # adjust path based on your setup
  OUTPUT_PATH: "/tmp/grading_output.json"
  MOCK_SCORER: "0"

automatic_scaling:
  min_instances: 1
  max_instances: 5
```

2. Deploy:

```bash
cd grading_feature/backend
gcloud app deploy
```

3. Get your GCP URL:

```bash
gcloud app describe
```

Output will show: `serviceUrl: https://your-project-id.uc.r.appspot.com`

#### Option B: Cloud Run (More flexible)

1. Build and push Docker image:

```bash
# Create Dockerfile in grading_feature/backend/
gcloud builds submit --tag gcr.io/your-project-id/grading-dashboard
```

2. Deploy:

```bash
gcloud run deploy grading-dashboard --image gcr.io/your-project-id/grading-dashboard \
  --platform managed \
  --region us-central1 \
  --set-env-vars OPENAI_API_KEY=your-key-here,DB_PATH=/mount/path
```

#### Option C: Compute Engine VM (Most control)

1. SSH into your GCP VM where OpenWebUI runs.
2. Clone your repo:

```bash
git clone <your-repo>
cd FamilyFinanceChat/grading_feature
```

3. Install deps and start as a background service:

```bash
cd backend
pip install -r requirements.txt
nohup uvicorn main:app --host 0.0.0.0 --port 9500 > grading.log &
```

4. Update your WebUI environment to link students to:
   ```
   https://your-vm-public-ip:9500/student/feedback/{userId}
   ```

### 4. Environment Variables

Set these in your GCP deployment config:

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_MODEL`: Model name (default: `gpt-4o-mini`)
- `DB_PATH`: Path to OpenWebUI SQLite database
- `OUTPUT_PATH`: Temp path for JSON output
- `MOCK_SCORER`: Set to `1` for mock mode (no API calls)

### 5. Student Feedback Link Format

Once deployed, students access their feedback at:

```
https://your-gcp-domain/student/feedback/{userId}
```

For example, if deployed to App Engine:
```
https://your-project-id.uc.r.appspot.com/student/feedback/user-123
```

**To auto-link from OpenWebUI**: Add a button/link in OpenWebUI UI that redirects to this URL, passing the current user's ID.

### 6. CORS Considerations

If frontend and backend are on different domains, add CORS headers in `main.py`:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain", "https://your-openwebui-domain"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 7. Database Access

If your OpenWebUI SQLite DB is on your VM:

- **Local**: Set `DB_PATH` to the actual path (e.g., `/home/user/openwebui/data.db`)
- **Cloud SQL**: Move to Cloud SQL and update connection string in code
- **Mounted Volume**: Use Google Cloud Storage + mounted path

### Troubleshooting

**"API call fails with 500"**
- Check `DB_PATH` exists and backend can read it
- Check `OPENAI_API_KEY` is set correctly
- Check logs: `gcloud app logs read`

**"Frontend shows 'localhost' in console"**
- The API config auto-detects. If still issues, update `src/config.js` to hardcode your GCP domain

**"Students see CORS error"**
- Ensure both frontend and backend are served from same domain or CORS is configured

---

## Quick Recap: Student Flow

1. Student is logged into OpenWebUI (on GCP)
2. Clicks "View Feedback" → redirects to `https://your-gcp-domain/student/feedback/{userId}`
3. Page auto-loads their chats + scores them in realtime
4. Displays personalized feedback + ABI trust profile
5. No login required (uses their already-authenticated userId)
