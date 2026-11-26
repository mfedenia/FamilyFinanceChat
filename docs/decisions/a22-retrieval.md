# A22: retire the standalone ChromaDB retrieval pipeline

Date: 2026-11-12
Decision: retire, rather than promote behind a Pipelines server.

## Canonical path

Student chat -> Open WebUI native Knowledge Base retrieval -> Qdrant -> model.
Documents enter through the native UI or `tools/kb_sync/`, which uses the public
REST API. This is the single supported retrieval path.

## Evidence and reasoning

At repository revision `cebf04a0a2de17752fb6ba154a777db385f8c115`, production and test Compose configurations
select `VECTOR_DB=qdrant` and the respective Qdrant service. No Compose service
mounts or launches `rag_bio_project/`, and no active application imports its
standalone modules. Its README describes a local testbed with manually installed
ChromaDB dependencies. ARCHITECTURE.md and HANDOFF.md identify it as disconnected
from student chat. This is repository evidence, not a new inspection of the VM.

Promoting the experiment would require a supported Pipelines service, deployment,
authentication, dependency maintenance, retrieval evaluation and an explicit
migration of consumers. No demonstrated production requirement justifies that
second implementation. Keeping it runnable would preserve the ambiguity and
maintenance burden A22 is intended to remove. The prior citation bug illustrates
that testing one retrieval stack does not validate the other.

## Changes and preservation

- Remove `rag_bio_project/src/`, its generated Chroma index, and the standalone
  `training_testing_pipeline/unified_train.py` entry point.
- Preserve all five research notebooks byte-for-byte under
  `archive/rag_bio_research/`, clearly marked historical and unsupported.
- Retain `rag_bio_project/data_pdfs/` unchanged; documents are inputs, not a second
  retrieval implementation. No production Qdrant data or volumes are changed.
- Replace the old setup README with a retirement notice; update active architecture,
  handoff and compatibility documentation.

## Historical recovery

The full experiment, including its original notebook locations and index, remains
in Git at `cebf04a0a2de17752fb6ba154a777db385f8c115`:

```sh
git archive cebf04a0a2de17752fb6ba154a777db385f8c115 rag_bio_project/ > /tmp/ffc-rag-research.tar
```

Extract into an isolated research directory, not a production deployment. Archived
notebooks may require their historical dependencies and paths. They are not a
second supported service. A future proposal must demonstrate a missing capability
and use documented extension surfaces, with tests and a migration decision.

## Validation and limits

Verify unchanged notebook and PDF bytes against the parent revision; check that
no runnable standalone pipeline/index remains; search active application and
Compose files for references/imports; run compatibility guards and diff checks.
No model calls, production data migration or deployment is required. Existing CI
provides the native knowledge-base smoke test; this change does not claim a new
live VM validation. A22 is ready for review; project completion requires merge
with passing CI under the team's definition of done.
