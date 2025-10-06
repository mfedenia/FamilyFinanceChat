
from chromadb import PersistentClient
import json

# We reuse the same embedder used for indexing
from embeddings import EmbeddingConfig, build_embeddings

def _embedder_from_coll_meta(meta: dict):
    info = meta.get("embedding")
    if isinstance(info, str):
        try:
            info = json.loads(info)
        except Exception:
            info = {}
    info = info or {}
    provider  = (info.get("provider") or "hf")
    model     = (info.get("model") or "BAAI/bge-small-zh-v1.5")
    normalize = bool(info.get("normalize", True))
    return build_embeddings(EmbeddingConfig(provider=provider, model=model, normalize=normalize))

def get_collection(persist_dir: str, collection_name: str):
    client = PersistentClient(path=persist_dir)
    return client.get_collection(collection_name)

def quick_query(persist_dir: str, collection_name: str, query_text: str, n_results: int = 5):
    client = PersistentClient(path=persist_dir)
    coll = client.get_collection(collection_name)
    embedder = _embedder_from_coll_meta(coll.metadata or {})
    # Compute query embedding with the SAME model/dim as the collection
    if hasattr(embedder, "embed_query"):
        qvec = embedder.embed_query(query_text)
    else:
        qvec = embedder.embed_documents([query_text])[0]
    return coll.query(query_embeddings=[qvec], n_results=n_results,
                      include=["documents","metadatas","distances"])
