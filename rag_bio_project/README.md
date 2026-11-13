# Retired standalone RAG pipeline (A22)

The canonical retrieval path is Open WebUI native Knowledge Base retrieval backed
by Qdrant. Upload documents through the native UI or `tools/kb_sync/` public-API CLI.
The standalone ChromaDB pipeline is retired; do not deploy or import it.

`data_pdfs/` is retained unchanged as source material. Its presence does not prove
that those documents are loaded in production; verify through Open WebUI.
Research notebooks are preserved in [archive/rag_bio_research](../archive/rag_bio_research/).
See [the decision](../docs/decisions/a22-retrieval.md) for rationale and recovery.
