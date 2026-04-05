"""
rag.py — Simple RAG pipeline (Retrieval-Augmented Generation).

Pipeline stages:
  1. Chunk   — split documents into overlapping text chunks
  2. Embed   — generate embeddings via OpenAI (text-embedding-3-small)
  3. Store   — keep vectors in memory (simple list; no external DB required)
  4. Retrieve — cosine-similarity search for top-k relevant chunks
  5. Generate — pass retrieved context + question to a chat model

Usage::

    from llm_toolkit.rag import RAGPipeline

    rag = RAGPipeline()
    rag.add_document("my_doc.txt", text)
    answer = rag.query("What is the main topic?")
    print(answer)
"""

from __future__ import annotations

import os
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Chunk:
    """A text chunk with its source and embedding."""
    text: str
    source: str = ""
    chunk_id: int = 0
    embedding: Optional[list[float]] = field(default=None, repr=False)


class RAGPipeline:
    """
    A self-contained RAG pipeline backed by an in-memory vector list.

    Args:
        embed_model: OpenAI embedding model to use.
        chat_model: Chat completion model to use for generation.
        chunk_size: Target character size of each chunk.
        chunk_overlap: Characters of overlap between adjacent chunks.
        top_k: Number of chunks to retrieve per query.
        openai_client: Pre-built OpenAI client instance. If None, created automatically.
        system_prompt: System prompt for the generation step.
    """

    DEFAULT_SYSTEM = textwrap.dedent("""
        You are a helpful assistant. Answer the user's question using ONLY the
        provided context. If the context does not contain enough information to
        answer, say so clearly. Do not make things up.
    """).strip()

    def __init__(
        self,
        embed_model: str = "text-embedding-3-small",
        chat_model: str = "gpt-4o-mini",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        top_k: int = 5,
        openai_client=None,
        system_prompt: Optional[str] = None,
    ):
        self.embed_model = embed_model
        self.chat_model = chat_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM
        self._chunks: list[Chunk] = []
        self._client = openai_client

    # ── Public API ────────────────────────────────────────────────────────────

    def add_text(self, text: str, source: str = "inline") -> int:
        """
        Chunk, embed, and store a raw text string.

        Returns:
            Number of chunks added.
        """
        chunks = self._chunk_text(text, source)
        embeddings = self._embed_texts([c.text for c in chunks])
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
        self._chunks.extend(chunks)
        return len(chunks)

    def add_document(self, path: str, encoding: str = "utf-8") -> int:
        """
        Load a file, chunk it, embed it, and store it.

        Returns:
            Number of chunks added.
        """
        p = Path(path)
        text = p.read_text(encoding=encoding)
        return self.add_text(text, source=p.name)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[Chunk]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query: The search query.
            top_k: Number of results to return. Defaults to ``self.top_k``.

        Returns:
            List of Chunk objects sorted by relevance (highest first).
        """
        if not self._chunks:
            return []

        k = top_k or self.top_k
        query_emb = self._embed_texts([query])[0]
        scored = [
            (self._cosine_similarity(query_emb, c.embedding), c)
            for c in self._chunks
            if c.embedding is not None
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:k]]

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_sources: bool = False,
    ) -> str | dict:
        """
        Full RAG query: retrieve relevant chunks and generate an answer.

        Args:
            question: The user's question.
            top_k: Number of chunks to retrieve.
            return_sources: If True, return a dict with 'answer' and 'sources'.

        Returns:
            Answer string, or dict with answer + sources if return_sources=True.
        """
        chunks = self.retrieve(question, top_k=top_k)

        if not chunks:
            answer = "No documents have been added to the knowledge base yet."
            if return_sources:
                return {"answer": answer, "sources": []}
            return answer

        context = self._build_context(chunks)
        prompt = f"Context:\n{context}\n\nQuestion: {question}"

        client = self._get_client()
        response = client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        answer = response.choices[0].message.content.strip()

        if return_sources:
            return {
                "answer": answer,
                "sources": [
                    {"source": c.source, "chunk_id": c.chunk_id, "text": c.text[:200]}
                    for c in chunks
                ],
            }
        return answer

    def clear(self) -> None:
        """Remove all stored chunks."""
        self._chunks = []

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def sources(self) -> list[str]:
        """List of unique source names in the knowledge base."""
        return list(dict.fromkeys(c.source for c in self._chunks))

    # ── Chunking ──────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, source: str) -> list[Chunk]:
        """Split text into overlapping character-level chunks."""
        # Normalize whitespace
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            # Try to split on sentence/paragraph boundary for cleaner chunks
            if end < len(text):
                # Look for a good break point within the last 20% of the chunk
                search_from = int(self.chunk_size * 0.8)
                boundary = self._find_boundary(chunk_text, search_from)
                if boundary:
                    chunk_text = chunk_text[:boundary]

            chunks.append(Chunk(
                text=chunk_text.strip(),
                source=source,
                chunk_id=chunk_id,
            ))
            chunk_id += 1

            # Advance with overlap
            start += len(chunk_text) - self.chunk_overlap

        return [c for c in chunks if c.text]

    @staticmethod
    def _find_boundary(text: str, from_pos: int) -> Optional[int]:
        """Find the last sentence/paragraph boundary at or after from_pos."""
        for pattern in (r"\n\n", r"\.\s", r"!\s", r"\?\s", r"\n"):
            match = None
            for m in re.finditer(pattern, text[from_pos:]):
                match = m
            if match:
                return from_pos + match.end()
        return None

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts using the OpenAI embeddings API."""
        client = self._get_client()
        # Batch to avoid single-call limits
        BATCH_SIZE = 100
        embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            response = client.embeddings.create(
                model=self.embed_model,
                input=batch,
            )
            embeddings.extend([item.embedding for item in response.data])
        return embeddings

    # ── Similarity ────────────────────────────────────────────────────────────

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        try:
            import numpy as np
            va, vb = np.array(a), np.array(b)
            denom = np.linalg.norm(va) * np.linalg.norm(vb)
            if denom == 0:
                return 0.0
            return float(np.dot(va, vb) / denom)
        except ImportError:
            # Pure-Python fallback (slower)
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

    # ── Context building ──────────────────────────────────────────────────────

    @staticmethod
    def _build_context(chunks: list[Chunk]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            header = f"[{i}] Source: {chunk.source or 'unknown'}"
            parts.append(f"{header}\n{chunk.text}")
        return "\n\n---\n\n".join(parts)

    # ── Client ────────────────────────────────────────────────────────────────

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai package required: pip install openai")
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return self._client
