# Custom PDF Router - Backend Integration Guide

## How `custom_pdf_router.py` Integrates with Open WebUI

**Last Updated:** December 11, 2025  
**Open WebUI Version:** 0.5.x+

---

## Table of Contents

1. [Integration Overview](#integration-overview)
2. [Import Dependencies](#import-dependencies)
3. [Integration Point 1: Router Registration](#integration-point-1-router-registration)
4. [Integration Point 2: Authentication](#integration-point-2-authentication)
5. [Integration Point 3: File Storage](#integration-point-3-file-storage)
6. [Integration Point 4: File Database Records](#integration-point-4-file-database-records)
7. [Integration Point 5: Content Processing (RAG)](#integration-point-5-content-processing-rag)
8. [Integration Point 6: Knowledge Base Association](#integration-point-6-knowledge-base-association)
9. [Upgrade Risk Assessment](#upgrade-risk-assessment)
10. [Mitigation Strategies](#mitigation-strategies)
11. [Pre-Upgrade Checklist](#pre-upgrade-checklist)
12. [Post-Upgrade Debugging](#post-upgrade-debugging)

---

## Integration Overview

`custom_pdf_router.py` integrates with **6 key parts** of the Open WebUI backend:

```
┌─────────────────────────────────────────────────────────────────┐
│                     custom_pdf_router.py                        │
└─────────────────────────────────────────────────────────────────┘
        │           │           │           │           │
        ▼           ▼           ▼           ▼           ▼
   ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
   │ main.py│  │  auth  │  │Storage │  │ Files  │  │Knowledge│
   │(router │  │(verify │  │Provider│  │ Model  │  │ Model   │
   │ mount) │  │ user)  │  │        │  │        │  │         │
   └────────┘  └────────┘  └────────┘  └────────┘  └────────┘
                               │           │           │
                               ▼           ▼           ▼
                          ┌────────┐  ┌────────┐  ┌────────┐
                          │  Disk  │  │ SQLite │  │ Vector │
                          │Storage │  │   DB   │  │   DB   │
                          └────────┘  └────────┘  └────────┘
```

---

## Import Dependencies

These are the exact imports from Open WebUI that `custom_pdf_router.py` uses:

```python
# Authentication
from open_webui.utils.auth import get_verified_user

# Database Models
from open_webui.models.knowledge import Knowledges
from open_webui.models.files import Files, FileForm

# Storage
from open_webui.storage.provider import Storage

# Content Processing
from open_webui.routers.retrieval import ProcessFileForm, process_file
```

### Import Risk Level

| Import | Risk Level | Reason |
|--------|------------|--------|
| `get_verified_user` | 🟢 Low | Core auth, rarely changes |
| `Files`, `FileForm` | 🟡 Medium | Schema may change |
| `Knowledges` | 🔴 High | Changed significantly in 0.5.x |
| `Storage` | 🟢 Low | Simple interface |
| `ProcessFileForm`, `process_file` | 🟡 Medium | Parameters may change |

---

## Integration Point 1: Router Registration

### Where It Happens

**File:** `/app/backend/open_webui/main.py`

### How It Works

```python
# Import the router
from open_webui.routers import custom_pdf_router

# Register it with the app
app.include_router(custom_pdf_router.router, prefix="/api/v1/custom", tags=["custom_pdf"])
```

### What Could Break

| Scenario | Impact | Likelihood |
|----------|--------|------------|
| `main.py` reorganized | Router not loaded | 🟡 Medium |
| Import path changes | Import error on startup | 🟢 Low |
| FastAPI version change | Incompatible router syntax | 🟢 Low |

### How to Detect

```bash
# Check if router is registered
curl http://localhost:8080/api/v1/custom/debug

# If you get HTML (404), router isn't registered
# If you get JSON, router is working
```

### How to Fix

1. Check the new `main.py` structure
2. Find where other routers are imported
3. Add your import in the same pattern
4. Find where `app.include_router()` is called
5. Add your router registration

---

## Integration Point 2: Authentication

### Where It Happens

Every endpoint in `custom_pdf_router.py`:

```python
@router.post("/pdf-upload")
async def upload_and_crawl(
    request: Request,
    user=Depends(get_verified_user)  # ← Here
):
```

### How It Works

`get_verified_user` is a FastAPI dependency that:
1. Extracts the JWT token from the request
2. Validates the token
3. Returns a `UserModel` object with `id`, `role`, `email`, etc.

### What Could Break

| Scenario | Impact | Likelihood |
|----------|--------|------------|
| `get_verified_user` renamed | Import error | 🟢 Low |
| `get_verified_user` moved | Import error | 🟢 Low |
| `UserModel` fields changed | Runtime errors accessing `user.id` | 🟡 Medium |

### How to Detect

```python
# In your endpoint, log the user object
log.info(f"User: {user}, type: {type(user)}, dir: {dir(user)}")
```

### How to Fix

```bash
# Find the current auth utility
grep -rn "def get_verified_user" /app/backend/open_webui/

# Check what it returns
grep -rn "class UserModel" /app/backend/open_webui/models/
```

---

## Integration Point 3: File Storage

### Where It Happens

In `finalize_upload()`:

```python
from open_webui.storage.provider import Storage

# Upload file content to storage
file_obj = io.BytesIO(content)
_, file_path = Storage.upload_file(file_obj, storage_filename, tags)
```

### How It Works

`Storage` is a facade that delegates to the configured storage provider:
- `LocalStorageProvider` - saves to `/app/backend/data/uploads/`
- `S3StorageProvider` - saves to S3/MinIO

### Current Method Signature

```python
# storage/provider.py
def upload_file(file: BinaryIO, filename: str, tags: Dict = {}) -> Tuple[bytes, str]:
    """
    Args:
        file: File-like object (must have .read())
        filename: Target filename (usually "{uuid}_{original_name}")
        tags: Optional metadata tags
    
    Returns:
        Tuple of (file_contents, file_path)
    """
```

### What Could Break

| Scenario | Impact | Likelihood |
|----------|--------|------------|
| Method signature changes | TypeError on call | 🟡 Medium |
| Return type changes | Code expecting wrong values | 🟡 Medium |
| New required parameters | TypeError on call | 🟡 Medium |
| Storage backend changes | Different path format | 🟢 Low |

### How to Detect

```bash
# Check current signature
grep -A 20 "def upload_file" /app/backend/open_webui/storage/provider.py
```

### How to Fix

```python
# Always check what the method expects
import inspect
print(inspect.signature(Storage.upload_file))
```

---

## Integration Point 4: File Database Records

### Where It Happens

In `finalize_upload()`:

```python
from open_webui.models.files import Files, FileForm

# Create file record
file_record = Files.insert_new_file(
    user.id,
    FileForm(
        id=file_id,
        filename=original_name,
        path=file_path,
        data={"status": "pending"},
        meta={
            "name": original_name,
            "content_type": "application/pdf",
            "size": len(content),
        }
    )
)
```

### Current FileForm Schema

```python
class FileForm(BaseModel):
    id: str                              # UUID
    hash: Optional[str] = None           # Content hash
    filename: str                        # Original filename
    path: str                            # Storage path
    data: dict = {}                      # Runtime data (status, content)
    meta: dict = {}                      # Metadata (name, content_type, size)
    access_control: Optional[dict] = None
```

### What Could Break

| Scenario | Impact | Likelihood |
|----------|--------|------------|
| `FileForm` fields added (required) | ValidationError | 🟡 Medium |
| `FileForm` fields renamed | ValidationError | 🟡 Medium |
| `Files.insert_new_file` signature changes | TypeError | 🟡 Medium |
| `data`/`meta` expected structure changes | Silent failures | 🔴 High |

### How to Detect

```bash
# Check FileForm schema
grep -A 20 "class FileForm" /app/backend/open_webui/models/files.py

# Check insert_new_file signature
grep -A 30 "def insert_new_file" /app/backend/open_webui/models/files.py
```

### How to Fix

1. Look at how the native file upload does it:
   ```bash
   grep -B 5 -A 20 "Files.insert_new_file" /app/backend/open_webui/routers/files.py
   ```
2. Copy the exact pattern used there

---

## Integration Point 5: Content Processing (RAG)

### Where It Happens

In `finalize_upload()`:

```python
from open_webui.routers.retrieval import ProcessFileForm, process_file

# Process file for RAG (extract text, create embeddings)
process_file(
    request,
    ProcessFileForm(file_id=file_id),  # ← CRITICAL: Only file_id!
    user=user,
)
```

### Current ProcessFileForm Schema

```python
class ProcessFileForm(BaseModel):
    file_id: str
    content: Optional[str] = None        # If provided, uses this content
    collection_name: Optional[str] = None # If provided, uses different path
```

### ⚠️ CRITICAL: Three Code Paths

The `process_file` function has **three different behaviors** based on what you pass:

| Parameters | Behavior | Use Case |
|------------|----------|----------|
| `file_id` only | Loads file from disk, extracts content | **New file uploads** |
| `file_id` + `content` | Uses provided content directly | Custom content injection |
| `file_id` + `collection_name` | Reuses existing vectors/content | Adding to knowledge base |

### What Could Break

| Scenario | Impact | Likelihood |
|----------|--------|------------|
| `ProcessFileForm` fields change | ValidationError | 🟡 Medium |
| `process_file` signature changes | TypeError | 🟡 Medium |
| Internal path logic changes | "Empty content" errors | 🔴 High |
| Async/sync behavior changes | Event loop errors | 🟡 Medium |

### How to Detect

```bash
# Check ProcessFileForm
grep -A 10 "class ProcessFileForm" /app/backend/open_webui/routers/retrieval.py

# Check process_file signature
grep -A 5 "def process_file" /app/backend/open_webui/routers/retrieval.py
```

### How to Fix

1. **Always check the native implementation first:**
   ```bash
   grep -B 5 -A 10 "process_file" /app/backend/open_webui/routers/files.py
   ```

2. **Match exactly what the native code does:**
   ```python
   # Native file upload does:
   process_file(request, ProcessFileForm(file_id=file_item.id), user=user)
   ```

3. **Never assume parameters** - if you're getting "empty content" errors, you're probably passing extra parameters that trigger the wrong code path.

---

## Integration Point 6: Knowledge Base Association

### Where It Happens

In `add_file_to_knowledge_base_async()`:

```python
from open_webui.models.knowledge import Knowledges

# Add file to knowledge base using junction table
result = Knowledges.add_file_to_knowledge_by_id(
    knowledge_id=knowledge_id,
    file_id=file_id,
    user_id=user.id
)
```

### How It Works (Post-0.5.x)

Files are associated with knowledge bases via a **junction table**:

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│   knowledge  │────<│  knowledge_file │>────│     file     │
│              │     │                 │     │              │
│ id           │     │ knowledge_id    │     │ id           │
│ name         │     │ file_id         │     │ filename     │
│ meta         │     │ user_id         │     │ data         │
└──────────────┘     │ created_at      │     └──────────────┘
                     └─────────────────┘
```

### ⚠️ MAJOR BREAKING CHANGE from Pre-0.5.x

**OLD way (BROKEN):**
```python
# This no longer works!
knowledge = Knowledges.get_knowledge_by_id(knowledge_id)
file_ids = knowledge.data.get("file_ids", [])  # ← 'data' doesn't exist anymore!
file_ids.append(file_id)
Knowledges.update_knowledge_data_by_id(knowledge_id, {"file_ids": file_ids})
```

**NEW way (CORRECT):**
```python
# Use the junction table method
Knowledges.add_file_to_knowledge_by_id(
    knowledge_id=knowledge_id,
    file_id=file_id,
    user_id=user.id
)
```

### What Could Break

| Scenario | Impact | Likelihood |
|----------|--------|------------|
| Method renamed | AttributeError | 🟡 Medium |
| Method signature changes | TypeError | 🟡 Medium |
| Junction table schema changes | Silent failures | 🟡 Medium |
| Return type changes | Code expecting wrong values | 🟢 Low |

### How to Detect

```bash
# Check available methods
grep -n "def.*knowledge" /app/backend/open_webui/models/knowledge.py

# Check the specific method
grep -A 20 "def add_file_to_knowledge_by_id" /app/backend/open_webui/models/knowledge.py
```

### How to Fix

1. **Look at the native knowledge router:**
   ```bash
   grep -B 5 -A 15 "add_file_to_knowledge" /app/backend/open_webui/routers/knowledge.py
   ```

2. **Copy the exact pattern used there**

---

## Upgrade Risk Assessment

### Risk Matrix

| Integration Point | Breaking Likelihood | Impact if Broken | Priority |
|-------------------|---------------------|------------------|----------|
| Router Registration | 🟢 Low | 🔴 Complete failure | P1 |
| Authentication | 🟢 Low | 🔴 Complete failure | P1 |
| File Storage | 🟡 Medium | 🟡 Upload failures | P2 |
| File Records | 🟡 Medium | 🟡 Upload failures | P2 |
| Content Processing | 🔴 High | 🟡 RAG not working | P2 |
| Knowledge Association | 🔴 High | 🟡 KB integration broken | P3 |

### Historical Breaking Changes

| Version | What Broke | How to Fix |
|---------|------------|------------|
| 0.5.x | `knowledge.data` → `knowledge.meta` | Use junction table methods |
| 0.5.x | `file_ids` stored in Knowledge | Use `KnowledgeFile` table |
| 0.5.x | `update_knowledge_data_by_id` | Use `add_file_to_knowledge_by_id` |

---

## Mitigation Strategies

### Strategy 1: Defensive Imports

Wrap imports in try/except to fail gracefully:

```python
try:
    from open_webui.models.knowledge import Knowledges
    KNOWLEDGE_AVAILABLE = True
except ImportError as e:
    log.warning(f"Knowledge module not available: {e}")
    KNOWLEDGE_AVAILABLE = False

# Later in code:
if KNOWLEDGE_AVAILABLE and knowledge_id:
    # ... add to KB
else:
    log.warning("Skipping KB integration - module not available")
```

### Strategy 2: Version Detection

Check the Open WebUI version at startup:

```python
def get_openwebui_version():
    try:
        from open_webui import __version__
        return __version__
    except:
        return "unknown"

def check_compatibility():
    version = get_openwebui_version()
    log.info(f"Open WebUI version: {version}")
    
    # Check for known incompatibilities
    if version.startswith("0.4"):
        log.warning("Version 0.4.x detected - KB integration may use old API")
```

### Strategy 3: Runtime Schema Validation

Check if expected attributes exist before using them:

```python
def add_to_knowledge_base(knowledge_id, file_id, user):
    knowledge = Knowledges.get_knowledge_by_id(knowledge_id)
    
    # Check which API is available
    if hasattr(Knowledges, 'add_file_to_knowledge_by_id'):
        # New API (0.5.x+)
        Knowledges.add_file_to_knowledge_by_id(knowledge_id, file_id, user.id)
    elif hasattr(knowledge, 'data'):
        # Old API (pre-0.5.x)
        file_ids = knowledge.data.get("file_ids", [])
        file_ids.append(file_id)
        Knowledges.update_knowledge_data_by_id(knowledge_id, {"file_ids": file_ids})
    else:
        raise RuntimeError("Unable to determine Knowledge API version")
```

### Strategy 4: Mirror Native Implementation

The safest approach is to **always copy what the native code does**:

```python
async def finalize_upload(...):
    # Instead of guessing, look at files.py and knowledge.py
    # and do EXACTLY what they do
    
    # 1. Upload like files.py does
    # 2. Create record like files.py does
    # 3. Process like files.py does
    # 4. Add to KB like knowledge.py does
```

### Strategy 5: Integration Tests

Create a test script that validates all integration points:

```python
# test_integrations.py
def test_imports():
    """Test that all required imports work"""
    from open_webui.utils.auth import get_verified_user
    from open_webui.models.knowledge import Knowledges
    from open_webui.models.files import Files, FileForm
    from open_webui.storage.provider import Storage
    from open_webui.routers.retrieval import ProcessFileForm, process_file
    print("✓ All imports successful")

def test_schemas():
    """Test that expected fields exist"""
    from open_webui.models.files import FileForm
    form = FileForm(id="test", filename="test.pdf", path="/tmp/test.pdf")
    assert hasattr(form, 'data')
    assert hasattr(form, 'meta')
    print("✓ FileForm schema OK")
    
    from open_webui.routers.retrieval import ProcessFileForm
    form = ProcessFileForm(file_id="test")
    assert hasattr(form, 'content')
    assert hasattr(form, 'collection_name')
    print("✓ ProcessFileForm schema OK")

def test_methods():
    """Test that expected methods exist"""
    from open_webui.models.knowledge import Knowledges
    assert hasattr(Knowledges, 'add_file_to_knowledge_by_id')
    assert hasattr(Knowledges, 'get_knowledge_by_id')
    print("✓ Knowledge methods OK")
    
    from open_webui.storage.provider import Storage
    assert hasattr(Storage, 'upload_file')
    print("✓ Storage methods OK")

if __name__ == "__main__":
    test_imports()
    test_schemas()
    test_methods()
    print("\n✓ All integration tests passed!")
```

Run after each upgrade:
```bash
docker exec -it open-webui python /app/custom-code/test_integrations.py
```

---

## Pre-Upgrade Checklist

Before upgrading Open WebUI:

### 1. Document Current State

```bash
# Save current versions of key files
mkdir -p /backup/openwebui_$(date +%Y%m%d)

# Models
cp /app/backend/open_webui/models/knowledge.py /backup/openwebui_$(date +%Y%m%d)/
cp /app/backend/open_webui/models/files.py /backup/openwebui_$(date +%Y%m%d)/

# Routers
cp /app/backend/open_webui/routers/retrieval.py /backup/openwebui_$(date +%Y%m%d)/
cp /app/backend/open_webui/routers/files.py /backup/openwebui_$(date +%Y%m%d)/
cp /app/backend/open_webui/routers/knowledge.py /backup/openwebui_$(date +%Y%m%d)/

# Main app
cp /app/backend/open_webui/main.py /backup/openwebui_$(date +%Y%m%d)/
```

### 2. Run Integration Tests

```bash
python /app/custom-code/test_integrations.py
```

### 3. Check Changelog

Look for breaking changes in:
- Official Open WebUI release notes
- GitHub commits to `models/` and `routers/`

### 4. Test in Staging First

Never upgrade production without testing custom routers first.

---

## Post-Upgrade Debugging

If something breaks after an upgrade:

### Step 1: Check Router Registration

```bash
# Does the endpoint respond?
curl http://localhost:8080/api/v1/custom/debug

# If HTML (404), check main.py for router registration
grep -n "custom_pdf" /app/backend/open_webui/main.py
```

### Step 2: Check Imports

```bash
# Start Python in container
docker exec -it open-webui python

# Try imports one by one
>>> from open_webui.utils.auth import get_verified_user
>>> from open_webui.models.knowledge import Knowledges
>>> from open_webui.models.files import Files, FileForm
>>> from open_webui.storage.provider import Storage
>>> from open_webui.routers.retrieval import ProcessFileForm, process_file
```

### Step 3: Compare with Native Code

```bash
# How does native file upload work now?
grep -B 5 -A 30 "def upload" /app/backend/open_webui/routers/files.py

# How does native KB file add work now?
grep -B 5 -A 30 "file/add" /app/backend/open_webui/routers/knowledge.py
```

### Step 4: Check for Schema Changes

```bash
# FileForm
diff /backup/openwebui_YYYYMMDD/files.py /app/backend/open_webui/models/files.py

# Knowledge
diff /backup/openwebui_YYYYMMDD/knowledge.py /app/backend/open_webui/models/knowledge.py

# ProcessFileForm
diff /backup/openwebui_YYYYMMDD/retrieval.py /app/backend/open_webui/routers/retrieval.py
```

### Step 5: Check Container Logs

```bash
# Look for Python errors
docker logs open-webui 2>&1 | grep -i "error\|exception\|traceback"

# Look for custom router logs
docker logs open-webui 2>&1 | grep "custom_pdf\|PDF"
```

---

## Quick Reference: What to Check in Each File

### `/app/backend/open_webui/models/files.py`

```bash
# FileForm schema
grep -A 15 "class FileForm" models/files.py

# Files.insert_new_file signature
grep -A 10 "def insert_new_file" models/files.py
```

### `/app/backend/open_webui/models/knowledge.py`

```bash
# KnowledgeModel schema
grep -A 15 "class KnowledgeModel" models/knowledge.py

# KnowledgeFile table (if it exists)
grep -A 15 "class KnowledgeFile" models/knowledge.py

# add_file_to_knowledge_by_id method
grep -A 20 "def add_file_to_knowledge_by_id" models/knowledge.py
```

### `/app/backend/open_webui/routers/retrieval.py`

```bash
# ProcessFileForm schema
grep -A 10 "class ProcessFileForm" routers/retrieval.py

# process_file function signature
grep -A 5 "def process_file" routers/retrieval.py
```

### `/app/backend/open_webui/storage/provider.py`

```bash
# Storage.upload_file signature
grep -A 15 "def upload_file" storage/provider.py
```

---

## Summary: Golden Rules

1. **Never assume API stability** - Always check before using
2. **Mirror native implementations** - If `files.py` does it one way, do it the same way
3. **Test after every upgrade** - Run integration tests immediately
4. **Keep backups** - Save key files before upgrading
5. **Check the source** - When in doubt, `grep` the actual code
6. **Watch for `data` vs `meta`** - This was the biggest breaking change
7. **ProcessFileForm paths matter** - Only pass `file_id` for new uploads
8. **Junction tables are the new way** - Don't try to store file_ids in knowledge.data
