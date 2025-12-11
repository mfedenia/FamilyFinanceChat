# Open WebUI Backend Architecture Guide

## For Custom Router Development and Version Migration

**Last Updated:** December 11, 2025  
**Relevant Open WebUI Version:** 0.5.x+

---

## Table of Contents

1. [Overview](#overview)
2. [Core Architecture](#core-architecture)
3. [Database Models](#database-models)
   - [Files Model](#files-model)
   - [Knowledge Model](#knowledge-model)
   - [KnowledgeFile Junction Table](#knowledgefile-junction-table)
4. [Key Routers](#key-routers)
   - [Files Router](#files-router)
   - [Knowledge Router](#knowledge-router)
   - [Retrieval Router](#retrieval-router)
5. [Storage Provider](#storage-provider)
6. [Content Processing Pipeline](#content-processing-pipeline)
7. [Vector Database Integration](#vector-database-integration)
8. [Custom PDF Router Integration Points](#custom-pdf-router-integration-points)
9. [Version Migration Guide](#version-migration-guide)
10. [Troubleshooting Common Issues](#troubleshooting-common-issues)

---

## Overview

Open WebUI's backend is built with **FastAPI** and uses a modular router-based architecture. The main application is defined in `/app/backend/open_webui/main.py`, which imports and registers all routers.

### Key Technologies
- **FastAPI** - Web framework
- **SQLAlchemy** - ORM for database operations
- **Pydantic** - Data validation and serialization
- **Qdrant/ChromaDB** - Vector database for embeddings
- **LangChain** - Document loading and text splitting

---

## Core Architecture

### Application Entry Point

**File:** `/app/backend/open_webui/main.py` (Lines 1-100)

```python
from open_webui.routers import (
    audio,
    images,
    ollama,
    openai,
    retrieval,
    # ... other routers
    knowledge,
    files,
    custom_pdf_router,  # Custom router added here
)
```

**Key Insight:** Custom routers must be imported here and registered using `app.include_router()`.

### Router Registration Pattern

```python
app.include_router(custom_pdf_router.router, prefix="/api/v1/custom", tags=["custom_pdf"])
```

---

## Database Models

### Files Model

**File:** `/app/backend/open_webui/models/files.py` (Lines 18-52)

```python
class File(Base):
    __tablename__ = "file"
    id = Column(String, primary_key=True, unique=True)
    user_id = Column(String)
    hash = Column(Text, nullable=True)
    filename = Column(Text)
    path = Column(Text, nullable=True)
    data = Column(JSON, nullable=True)      # Stores content, status, etc.
    meta = Column(JSON, nullable=True)      # Stores name, content_type, size
    access_control = Column(JSON, nullable=True)
    created_at = Column(BigInteger)
    updated_at = Column(BigInteger)
```

#### Important Fields

| Field | Purpose | Example Value |
|-------|---------|---------------|
| `id` | Unique file identifier (UUID) | `"f48db5b1-bb27-45a9-9fb8-cb5d05b2aa9d"` |
| `path` | Storage path | `"/app/backend/data/uploads/{id}_{filename}"` |
| `data` | Runtime data (content, status) | `{"content": "...", "status": "completed"}` |
| `meta` | File metadata | `{"name": "file.pdf", "content_type": "application/pdf", "size": 14586}` |

#### FileForm for Creating Records

**File:** `/app/backend/open_webui/models/files.py` (Lines 89-97)

```python
class FileForm(BaseModel):
    id: str
    hash: Optional[str] = None
    filename: str
    path: str
    data: dict = {}
    meta: dict = {}
    access_control: Optional[dict] = None
```

#### Key Methods in FilesTable

**File:** `/app/backend/open_webui/models/files.py` (Lines 100-290)

| Method | Purpose | Usage in custom_pdf_router.py |
|--------|---------|-------------------------------|
| `Files.insert_new_file(user_id, FileForm)` | Create a new file record | Creating uploaded PDF records |
| `Files.get_file_by_id(id)` | Retrieve file by ID | Checking file existence |
| `Files.update_file_data_by_id(id, data)` | Update `data` field | Setting status, content |
| `Files.update_file_hash_by_id(id, hash)` | Update hash | After content extraction |

---

### Knowledge Model

**File:** `/app/backend/open_webui/models/knowledge.py` (Lines 34-78)

```python
class Knowledge(Base):
    __tablename__ = "knowledge"
    id = Column(Text, unique=True, primary_key=True)
    user_id = Column(Text)
    name = Column(Text)
    description = Column(Text)
    meta = Column(JSON, nullable=True)           # NOTE: Was 'data' in older versions
    access_control = Column(JSON, nullable=True)
    created_at = Column(BigInteger)
    updated_at = Column(BigInteger)
```

#### ⚠️ BREAKING CHANGE: `data` → `meta`

In older versions, `KnowledgeModel` had a `data` attribute that stored file IDs:

```python
# OLD (pre-0.5.x):
knowledge.data = {"file_ids": ["id1", "id2"]}

# NEW (0.5.x+):
# File associations are stored in a separate junction table
# The 'data' field is renamed to 'meta'
knowledge.meta = {}  # No longer stores file_ids
```

#### KnowledgeModel Pydantic Schema

**File:** `/app/backend/open_webui/models/knowledge.py` (Lines 64-78)

```python
class KnowledgeModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    name: str
    description: str
    meta: Optional[dict] = None      # ← 'meta' not 'data'
    access_control: Optional[dict] = None
    created_at: int
    updated_at: int
```

---

### KnowledgeFile Junction Table

**File:** `/app/backend/open_webui/models/knowledge.py` (Lines 80-108)

This is the **new way** files are associated with knowledge bases:

```python
class KnowledgeFile(Base):
    __tablename__ = "knowledge_file"
    id = Column(Text, unique=True, primary_key=True)
    knowledge_id = Column(Text, ForeignKey("knowledge.id", ondelete="CASCADE"))
    file_id = Column(Text, ForeignKey("file.id", ondelete="CASCADE"))
    user_id = Column(Text, nullable=False)
    created_at = Column(BigInteger, nullable=False)
    updated_at = Column(BigInteger, nullable=False)

    __table_args__ = (
        UniqueConstraint("knowledge_id", "file_id", name="uq_knowledge_file_knowledge_file"),
    )
```

#### Key Methods for Knowledge-File Association

**File:** `/app/backend/open_webui/models/knowledge.py` (Lines 256-295)

```python
def add_file_to_knowledge_by_id(
    self, knowledge_id: str, file_id: str, user_id: str
) -> Optional[KnowledgeFileModel]:
    """Creates a KnowledgeFile record linking a file to a knowledge base"""
    with get_db() as db:
        knowledge_file = KnowledgeFileModel(
            id=str(uuid.uuid4()),
            knowledge_id=knowledge_id,
            file_id=file_id,
            user_id=user_id,
            created_at=int(time.time()),
            updated_at=int(time.time()),
        )
        result = KnowledgeFile(**knowledge_file.model_dump())
        db.add(result)
        db.commit()
        # ...
```

#### Other Useful Methods

| Method | Purpose |
|--------|---------|
| `Knowledges.get_knowledge_by_id(id)` | Get knowledge base by ID |
| `Knowledges.get_files_by_id(knowledge_id)` | Get all files in a knowledge base |
| `Knowledges.add_file_to_knowledge_by_id(knowledge_id, file_id, user_id)` | Add file to KB |
| `Knowledges.remove_file_from_knowledge_by_id(knowledge_id, file_id)` | Remove file from KB |

---

## Key Routers

### Files Router

**File:** `/app/backend/open_webui/routers/files.py`

#### Upload Flow (Lines 100-148)

```python
def process_uploaded_file(request, file, file_path, file_item, file_metadata, user):
    # ...
    if (not file.content_type.startswith(("image/", "video/"))) or (...):
        process_file(request, ProcessFileForm(file_id=file_item.id), user=user)
```

**Key Insight:** The native file upload only passes `file_id` to `process_file()`, NOT `collection_name`. This triggers the file loading path.

#### Upload Handler (Lines 170-280)

```python
def upload_file_handler(request, file, ...):
    # 1. Generate UUID for file
    id = str(uuid.uuid4())
    filename = f"{id}_{filename}"
    
    # 2. Upload to storage
    contents, file_path = Storage.upload_file(file.file, filename, tags)
    
    # 3. Create file record
    file_item = Files.insert_new_file(
        user.id,
        FileForm(
            id=id,
            filename=name,
            path=file_path,
            data={"status": "pending"},
            meta={"name": name, "content_type": file.content_type, "size": len(contents)},
        ),
    )
    
    # 4. Process file (extract content, create embeddings)
    process_uploaded_file(request, file, file_path, file_item, file_metadata, user)
```

---

### Knowledge Router

**File:** `/app/backend/open_webui/routers/knowledge.py`

#### Adding a File to Knowledge Base (Lines 280-340)

```python
@router.post("/{id}/file/add", response_model=Optional[KnowledgeFilesResponse])
def add_file_to_knowledge_by_id(request, id, form_data, user):
    # 1. Validate knowledge base exists and user has access
    knowledge = Knowledges.get_knowledge_by_id(id=id)
    
    # 2. Validate file exists and is processed
    file = Files.get_file_by_id(form_data.file_id)
    if not file.data:
        raise HTTPException(detail=ERROR_MESSAGES.FILE_NOT_PROCESSED)
    
    # 3. Create KnowledgeFile junction record
    Knowledges.add_file_to_knowledge_by_id(
        knowledge_id=id, file_id=form_data.file_id, user_id=user.id
    )
    
    # 4. Add content to vector database (using knowledge_id as collection)
    process_file(
        request,
        ProcessFileForm(file_id=form_data.file_id, collection_name=id),
        user=user,
    )
```

**Key Insight:** When adding to a knowledge base, `collection_name=id` (the knowledge base ID) is passed. This copies the file's vectors into the knowledge base's collection.

---

### Retrieval Router

**File:** `/app/backend/open_webui/routers/retrieval.py`

#### ProcessFileForm Schema (Lines 1438-1442)

```python
class ProcessFileForm(BaseModel):
    file_id: str
    content: Optional[str] = None
    collection_name: Optional[str] = None
```

#### process_file Function (Lines 1444-1670)

This is the **most critical function** for content processing. It has THREE code paths:

```python
def process_file(request, form_data, user):
    file = Files.get_file_by_id(form_data.file_id)
    
    collection_name = form_data.collection_name or f"file-{file.id}"
    
    # PATH 1: Content provided directly (Lines 1467-1492)
    if form_data.content:
        # Use provided content, create embeddings
        docs = [Document(page_content=form_data.content, metadata={...})]
        text_content = form_data.content
    
    # PATH 2: collection_name provided, no content (Lines 1493-1520)
    elif form_data.collection_name:
        # Try to get content from existing vectors or file.data.content
        result = VECTOR_DB_CLIENT.query(collection_name=f"file-{file.id}", ...)
        if result and len(result.ids[0]) > 0:
            # Use existing vectors
            docs = [...]
        else:
            # Use file.data.get("content", "")
            docs = [Document(page_content=file.data.get("content", ""), ...)]
        text_content = file.data.get("content", "")
    
    # PATH 3: No content, no collection_name (Lines 1521-1580)
    else:
        # Load from file using Loader
        file_path = Storage.get_file(file.path)
        loader = Loader(engine=..., ...)
        docs = loader.load(file.filename, file.meta.get("content_type"), file_path)
        text_content = " ".join([doc.page_content for doc in docs])
    
    # Save content and create embeddings
    Files.update_file_data_by_id(file.id, {"content": text_content})
    save_docs_to_vector_db(request, docs, collection_name, ...)
```

#### ⚠️ CRITICAL: Choosing the Right Path

| Scenario | What to Pass | Code Path |
|----------|--------------|-----------|
| New file upload | `ProcessFileForm(file_id=id)` | PATH 3 (Loader extracts content) |
| Adding file to KB | `ProcessFileForm(file_id=id, collection_name=kb_id)` | PATH 2 (reuses existing vectors) |
| Custom content | `ProcessFileForm(file_id=id, content="...")` | PATH 1 (uses provided content) |

**For `custom_pdf_router.py`:** Always use `ProcessFileForm(file_id=id)` for initial processing. The file will be loaded and content extracted automatically.

---

## Storage Provider

**File:** `/app/backend/open_webui/storage/provider.py`

### LocalStorageProvider (Lines 58-98)

```python
class LocalStorageProvider(StorageProvider):
    @staticmethod
    def upload_file(file: BinaryIO, filename: str, tags: Dict) -> Tuple[bytes, str]:
        contents = file.read()
        file_path = f"{UPLOAD_DIR}/{filename}"
        with open(file_path, "wb") as f:
            f.write(contents)
        return contents, file_path

    @staticmethod
    def get_file(file_path: str) -> str:
        return file_path
```

### Usage in custom_pdf_router.py

```python
from open_webui.storage.provider import Storage

# Upload file to storage
file_obj = io.BytesIO(content)
_, file_path = Storage.upload_file(file_obj, storage_filename, tags)

# file_path will be like: /app/backend/data/uploads/{uuid}_{filename}
```

---

## Content Processing Pipeline

### Complete Flow

```
1. User uploads PDF
        ↓
2. Storage.upload_file() saves to disk
        ↓
3. Files.insert_new_file() creates DB record
        ↓
4. process_file() is called with file_id
        ↓
5. Loader extracts text from PDF
        ↓
6. Text is split into chunks (RecursiveCharacterTextSplitter)
        ↓
7. Chunks are embedded (via embedding function)
        ↓
8. Embeddings saved to vector DB (VECTOR_DB_CLIENT.insert())
        ↓
9. File record updated with content and "completed" status
```

### Loader Class

**File:** `/app/backend/open_webui/retrieval/loaders/main.py`

The `Loader` class handles content extraction from various file types:

```python
loader = Loader(
    engine=CONTENT_EXTRACTION_ENGINE,  # "default", "tika", "docling", etc.
    TIKA_SERVER_URL=...,
    PDF_EXTRACT_IMAGES=...,
    # ... other config
)
docs = loader.load(filename, content_type, file_path)
```

---

## Vector Database Integration

### save_docs_to_vector_db Function

**File:** `/app/backend/open_webui/routers/retrieval.py` (Lines 1240-1435)

```python
def save_docs_to_vector_db(request, docs, collection_name, metadata=None, 
                           overwrite=False, split=True, add=False, user=None):
    # 1. Split documents into chunks
    if split:
        docs = text_splitter.split_documents(docs)
    
    # 2. Generate embeddings
    embedding_function = get_embedding_function(...)
    embeddings = asyncio.run(embedding_function(texts, ...))
    
    # 3. Insert into vector database
    VECTOR_DB_CLIENT.insert(
        collection_name=collection_name,
        items=[{"id": uuid, "text": text, "vector": embedding, "metadata": meta}, ...]
    )
```

### ⚠️ asyncio.run() Issue

Line 1410 uses `asyncio.run()` which **cannot be called from within an existing async context**:

```python
embeddings = asyncio.run(embedding_function(...))
```

This causes errors when `process_file` is called from an async endpoint. The native code handles this by calling `process_file` synchronously from sync endpoints or using `run_in_threadpool`.

---

## Custom PDF Router Integration Points

### Required Imports

```python
from open_webui.models.knowledge import Knowledges
from open_webui.models.files import Files, FileForm
from open_webui.storage.provider import Storage
from open_webui.routers.retrieval import ProcessFileForm, process_file
```

### File Upload Pattern (from custom_pdf_router.py)

```python
# 1. Read file content
with open(source, "rb") as f:
    content = f.read()

# 2. Generate unique ID and filename
file_id = str(uuid.uuid4())
storage_filename = f"{file_id}_{original_name}"

# 3. Upload to storage
file_obj = io.BytesIO(content)
_, file_path = Storage.upload_file(file_obj, storage_filename, tags)

# 4. Create file record
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

# 5. Process file (extract content, create embeddings)
process_file(
    request,
    ProcessFileForm(file_id=file_id),  # ← Only file_id, no collection_name!
    user=user,
)

# 6. Add to knowledge base (if specified)
Knowledges.add_file_to_knowledge_by_id(
    knowledge_id=knowledge_id,
    file_id=file_id,
    user_id=user.id
)
```

---

## Version Migration Guide

### Changes Between Versions

#### 1. Knowledge Base File Association

**Old Approach (pre-0.5.x):**
```python
# Files stored in knowledge.data.file_ids list
knowledge = Knowledges.get_knowledge_by_id(knowledge_id)
current_data = knowledge.data or {}
file_ids = current_data.get("file_ids", [])
file_ids.append(file_id)
Knowledges.update_knowledge_data_by_id(knowledge_id, {"file_ids": file_ids})
```

**New Approach (0.5.x+):**
```python
# Files stored in separate junction table
Knowledges.add_file_to_knowledge_by_id(
    knowledge_id=knowledge_id,
    file_id=file_id,
    user_id=user.id
)
```

#### 2. KnowledgeModel Schema Change

**Old:**
```python
class KnowledgeModel(BaseModel):
    data: Optional[dict] = None  # Stored file_ids here
```

**New:**
```python
class KnowledgeModel(BaseModel):
    meta: Optional[dict] = None  # No longer stores file_ids
```

#### 3. ProcessFileForm Usage

The `process_file` function's behavior depends on what parameters you provide:

| Parameter Combination | Behavior |
|-----------------------|----------|
| `file_id` only | Loads file from disk, extracts content |
| `file_id` + `content` | Uses provided content directly |
| `file_id` + `collection_name` | Tries to reuse existing vectors or file.data.content |

### Finding Breaking Changes

```bash
# Check KnowledgeModel schema
grep -n "class KnowledgeModel" /app/backend/open_webui/models/knowledge.py

# Check process_file signature
grep -n "def process_file" /app/backend/open_webui/routers/retrieval.py

# Check ProcessFileForm schema
grep -n "class ProcessFileForm" /app/backend/open_webui/routers/retrieval.py

# Check how native router adds files to KB
grep -n "add_file_to_knowledge" /app/backend/open_webui/routers/knowledge.py
```

---

## Troubleshooting Common Issues

### Error: "'KnowledgeModel' object has no attribute 'data'"

**Cause:** Using old `knowledge.data` attribute.

**Fix:** Use `knowledge.meta` or the new junction table methods.

### Error: "The content provided is empty"

**Cause:** Passing `collection_name` to `process_file()` for a new file that hasn't been processed yet.

**Fix:** Only pass `file_id` for new uploads:
```python
process_file(request, ProcessFileForm(file_id=file_id), user=user)
```

### Error: "asyncio.run() cannot be called from a running event loop"

**Cause:** `save_docs_to_vector_db` uses `asyncio.run()` internally, which conflicts with async contexts.

**Fix:** Call `process_file` from a synchronous context, or use `run_in_threadpool`:
```python
from fastapi.concurrency import run_in_threadpool
await run_in_threadpool(process_file, request, form_data, user)
```

### Files not appearing in Knowledge Base

**Cause:** Junction table record not created.

**Fix:** Ensure you call:
```python
Knowledges.add_file_to_knowledge_by_id(knowledge_id, file_id, user.id)
```

### Content not indexed (empty content field)

**Cause:** `process_file()` not called or failed.

**Fix:** Check logs for errors, ensure file exists at `file.path`, and call:
```python
process_file(request, ProcessFileForm(file_id=file_id), user=user)
```

---

## Summary

### Key Files to Monitor for Changes

| File | What to Check |
|------|---------------|
| `/app/backend/open_webui/models/knowledge.py` | `KnowledgeModel` schema, `KnowledgeFile` table, `add_file_to_knowledge_by_id()` |
| `/app/backend/open_webui/models/files.py` | `FileModel` schema, `FileForm`, `Files` methods |
| `/app/backend/open_webui/routers/retrieval.py` | `ProcessFileForm`, `process_file()`, `save_docs_to_vector_db()` |
| `/app/backend/open_webui/routers/knowledge.py` | How native code adds files to KB |
| `/app/backend/open_webui/routers/files.py` | How native code uploads files |
| `/app/backend/open_webui/storage/provider.py` | `Storage.upload_file()` signature |

### Golden Rules

1. **For new file uploads:** Use `ProcessFileForm(file_id=id)` only
2. **For KB association:** Use `Knowledges.add_file_to_knowledge_by_id()`
3. **Never access `knowledge.data`** - use `knowledge.meta` or junction table methods
4. **Always check native implementations** in `files.py` and `knowledge.py` as reference
