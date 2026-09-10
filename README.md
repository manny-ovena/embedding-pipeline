# Embedding Pipeline

This crate builds a small document embedding pipeline for chunking text and generating vector embeddings with a local or hosted vLLM-compatible model backend.

It is designed for use cases like:
- indexing documents for semantic search
- generating embeddings for retrieval-augmented generation (RAG)
- turning unstructured text into normalized vector representations for downstream similarity or clustering workflows

The pipeline currently includes:
- a text chunker that splits documents into overlapping chunks
- a tokenizer-aware chunking flow
- an embeddder abstraction backed by a vLLM-compatible `/v1/embeddings` endpoint
- normalization and batch embedding support for pipeline-style processing

## Model download

```bash
hf download Qwen/Qwen3-Embedding-4B --local-dir "/srv/ai-models/Qwen/Qwen3-Embedding-4B"
```

## Typical usage

This application is intended to run a document through the pipeline and generate embeddings for each chunk, which can then be stored in a vector store, indexed for search, or used as features in an ML workflow.

The code is structured so the `EmbeddingBackend` can be swapped out for different providers or local deployments while keeping the rest of the pipeline consistent.
