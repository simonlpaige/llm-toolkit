#!/usr/bin/env python3
"""
Example: Simple RAG pipeline.

Demonstrates ingesting documents and querying with retrieval-augmented
generation using ChromaDB and OpenAI.

Usage:
    pip install chromadb
    export OPENAI_API_KEY=sk-...
    python examples/rag_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from llm_toolkit import SimpleRAG

# Create a RAG pipeline (in-memory by default)
rag = SimpleRAG(collection_name="demo")

# Add some documents
documents = [
    "Python 3.12 was released on October 2, 2023. It introduced improved error "
    "messages, a new type parameter syntax (PEP 695), and a per-interpreter GIL "
    "(PEP 684). Performance improved by about 5% over Python 3.11.",
    "FastAPI is a modern Python web framework for building APIs. It uses Python "
    "type hints for request validation and automatic OpenAPI documentation. "
    "It's built on Starlette and Pydantic, and is one of the fastest Python "
    "frameworks available.",
    "Docker containers package applications with their dependencies for consistent "
    "deployment. Docker Compose orchestrates multi-container applications. "
    "Images are built from Dockerfiles using layered filesystem snapshots.",
    "PostgreSQL is an advanced open-source relational database. It supports JSONB "
    "for document storage, full-text search, and extensions like PostGIS for "
    "geospatial data. Version 16 introduced logical replication from standbys.",
    "Redis is an in-memory data store used for caching, message brokering, and "
    "real-time analytics. It supports data structures like strings, hashes, lists, "
    "sets, and sorted sets. Redis 7.0 introduced Redis Functions.",
]

print("Ingesting documents...")
rag.add_many(documents)
print(f"Collection size: {rag.count} documents\n")

# Query
questions = [
    "What's new in Python 3.12?",
    "How does FastAPI handle request validation?",
    "What database supports JSONB?",
]

for q in questions:
    print(f"Q: {q}")

    # Show retrieved context
    results = rag.search(q, n_results=2)
    print(f"  Retrieved {len(results)} docs (distance: {results[0]['distance']:.3f})")

    # Get grounded answer
    answer = rag.query(q)
    print(f"  A: {answer}\n")
