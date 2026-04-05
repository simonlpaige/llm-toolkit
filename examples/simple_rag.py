"""
examples/simple_rag.py — RAG over local documents.

Demonstrates:
  - Loading text files into the RAG pipeline
  - Embedding and storing chunks
  - Querying with retrieved context
  - Displaying sources

Requirements:
  - OPENAI_API_KEY set in environment or .env
  - Sample documents are generated inline if none are provided

Run:
    python examples/simple_rag.py
    python examples/simple_rag.py path/to/your/doc.txt
"""

import os
import sys
import tempfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from llm_toolkit.rag import RAGPipeline


# ── Sample documents for demo purposes ───────────────────────────────────────

SAMPLE_DOCS = {
    "quantum_computing.txt": """
Quantum computing is a type of computation that harnesses quantum mechanical
phenomena such as superposition and entanglement to process information. Unlike
classical computers that use bits (0 or 1), quantum computers use quantum bits
or qubits, which can exist in multiple states simultaneously.

The fundamental advantage of quantum computing lies in its ability to perform
certain calculations exponentially faster than classical computers. For example,
Shor's algorithm can factor large integers in polynomial time, which would break
most current encryption systems. Grover's algorithm can search unsorted databases
quadratically faster than classical approaches.

Major companies investing in quantum computing include IBM, Google, Microsoft,
and Amazon. IBM's Quantum Experience provides cloud access to quantum computers.
Google claimed "quantum supremacy" in 2019 when their Sycamore processor performed
a specific calculation in 200 seconds that would take classical supercomputers
approximately 10,000 years.

Current quantum computers suffer from high error rates due to "decoherence" —
the tendency for qubits to lose their quantum properties when interacting with
the environment. Error correction is a major area of research. Fault-tolerant
quantum computing remains an engineering challenge for the decade ahead.

Applications being explored include: drug discovery and molecular simulation,
financial optimization problems, machine learning acceleration, cryptography
and post-quantum security, and logistics/supply chain optimization.
""",
    "machine_learning.txt": """
Machine learning is a branch of artificial intelligence that enables systems
to learn and improve from experience without being explicitly programmed.
It focuses on developing computer programs that can access data and use it
to learn for themselves.

The three main types of machine learning are:

1. Supervised Learning: The algorithm learns from labeled training data,
   making predictions or decisions based on that data. Examples include
   linear regression, decision trees, and neural networks. Common use cases
   are spam detection, image recognition, and price prediction.

2. Unsupervised Learning: The algorithm finds patterns in unlabeled data
   without predefined answers. Clustering algorithms like K-means group
   similar data points together. Applications include customer segmentation,
   anomaly detection, and dimensionality reduction.

3. Reinforcement Learning: An agent learns to make decisions by taking actions
   in an environment to maximize cumulative reward. This approach has achieved
   superhuman performance in games like Chess, Go, and Atari games. Real-world
   applications include robotics, autonomous vehicles, and recommendation systems.

Deep learning, a subset of machine learning using neural networks with many
layers, has revolutionized fields such as computer vision, natural language
processing, and speech recognition. The transformer architecture, introduced
in the "Attention Is All You Need" paper (2017), forms the basis for large
language models like GPT and Claude.

Key challenges in machine learning include overfitting (models that memorize
training data but generalize poorly), data quality and quantity requirements,
interpretability ("black box" problem), computational costs, and ensuring
fairness and avoiding bias in model outputs.
""",
}


def create_sample_docs(tmpdir: str) -> list[str]:
    """Write sample documents to a temp directory and return paths."""
    paths = []
    for filename, content in SAMPLE_DOCS.items():
        path = Path(tmpdir) / filename
        path.write_text(content.strip(), encoding="utf-8")
        paths.append(str(path))
    return paths


def main():
    print("=" * 65)
    print("LLM Toolkit — Simple RAG Demo")
    print("=" * 65)

    # Use provided paths or fall back to sample documents
    doc_paths = sys.argv[1:]
    using_samples = False

    if not doc_paths:
        print("\nNo documents provided — using built-in sample documents.")
        print("Usage: python examples/simple_rag.py [path/to/doc.txt ...]\n")
        tmpdir = tempfile.mkdtemp()
        doc_paths = create_sample_docs(tmpdir)
        using_samples = True

    # Build the RAG pipeline
    rag = RAGPipeline(
        embed_model=os.environ.get("LLM_EMBED_MODEL", "text-embedding-3-small"),
        chat_model=os.environ.get("LLM_DEFAULT_MODEL", "gpt-4o-mini"),
        chunk_size=600,
        chunk_overlap=80,
        top_k=3,
    )

    # Load documents
    print("Loading documents...")
    for path in doc_paths:
        count = rag.add_document(path)
        print(f"  ✓ {Path(path).name} → {count} chunks")

    print(f"\nKnowledge base: {rag.chunk_count} total chunks from {len(rag.sources)} source(s)\n")

    # Run demo queries
    queries = [
        "What is quantum supremacy and who achieved it?",
        "Explain the difference between supervised and unsupervised learning.",
        "What are the main challenges facing quantum computers today?",
        "How do transformers relate to large language models?",
    ] if using_samples else [
        "What is this document about?",
        "What are the key points?",
        "Summarize the main conclusions.",
    ]

    for question in queries:
        print(f"Q: {question}")
        print("-" * 60)

        result = rag.query(question, return_sources=True)

        print(f"A: {result['answer']}\n")
        print("Sources used:")
        for s in result["sources"]:
            preview = s["text"].replace("\n", " ")[:120]
            print(f"  [{s['source']} / chunk {s['chunk_id']}] {preview}...")
        print()

    # Interactive mode
    print("=" * 65)
    print("Interactive mode — type your question (or 'quit' to exit)")
    print("=" * 65)
    while True:
        try:
            q = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not q or q.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        answer = rag.query(q)
        print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    main()
