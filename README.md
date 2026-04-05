# llm-toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-Compatible-412991)](https://openai.com)
[![Anthropic](https://img.shields.io/badge/Anthropic-Compatible-D97757)](https://anthropic.com)
[![Google](https://img.shields.io/badge/Gemini-Compatible-4285F4)](https://ai.google.dev)

A practical Python utility library for LLM development. Prompt chaining, token counting, cost estimation, retry handling, streaming helpers, response caching, and a simple RAG pipeline — the tools you actually need.

## Features

- 📝 **Prompt Templates** — `{variable}` substitution, role-tagged messages, and multi-step chaining
- 🔢 **Token Counting** — exact counts via tiktoken for OpenAI; calibrated estimates for Claude, Gemini, Llama
- 💰 **Cost Estimation** — 2025/2026 pricing for OpenAI, Anthropic, Google, Meta, Mistral with cross-model comparison
- 🔄 **Retry with Backoff** — exponential backoff + jitter, handles 429/500/503, respects `Retry-After`
- 📡 **Streaming Helpers** — unified generators for OpenAI and Anthropic with token usage tracking
- 🔍 **RAG Pipeline** — chunk → embed → retrieve (cosine similarity) → generate — no external vector DB required
- 💾 **Response Caching** — disk-based JSON cache with TTL, decorator support, and hit/miss stats

## Quick Start

```bash
git clone https://github.com/simonlpaige/llm-toolkit.git
cd llm-toolkit
pip install -r requirements.txt

cp .env.example .env
# Add your API key(s) to .env

# Cost comparison works offline — no API key needed:
python examples/cost_compare.py
```

## Usage

### Prompt Templates & Chaining

```python
from llm_toolkit.prompts import PromptTemplate, PromptChain, build_messages

# Simple template
tpl = PromptTemplate("Summarize this in {language}: {text}")
prompt = tpl.render(language="French", text="Hello world")
msg = tpl.as_message(language="French", text="Hello world")
# → {"role": "user", "content": "Summarize this in French: Hello world"}

# Build message lists
messages = build_messages(
    system_prompt="You are a helpful assistant.",
    user_message="What is quantum computing?"
)

# Multi-step chain with LLM calls
chain = PromptChain(steps=[step1_fn, step2_fn, step3_fn], llm_fn=my_llm)
result = chain.run(topic="quantum computing")
print(chain.history)  # inspect each step's input/output
```

### Token Counting

```python
from llm_toolkit.tokens import count_tokens, count_tokens_messages

# Exact count for OpenAI (via tiktoken — works offline)
count_tokens("Hello, world!", model="gpt-4o")  # → 4

# Approximate count for other providers
count_tokens("Hello, world!", model="claude-3-5-sonnet")  # → 3
count_tokens("Hello, world!", model="llama-3.1-70b")      # → 3

# Count tokens in a full message list (includes per-message overhead)
count_tokens_messages([
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hi!"},
], model="gpt-4o")  # → 14
```

### Cost Estimation

```python
from llm_toolkit.costs import estimate_cost, compare_costs, format_cost

# Single model
result = estimate_cost("gpt-4o", input_tokens=2000, output_tokens=500)
print(format_cost(result["total_cost"]))  # → $0.0100

# Compare across providers (sorted cheapest → most expensive)
comparison = compare_costs(
    prompt_tokens=2000,
    completion_tokens=500,
    models=["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet", "gemini-2.0-flash"]
)
for r in comparison:
    print(f"{r['model']}: {format_cost(r['total_cost'])}")
```

### Retry with Backoff

```python
from llm_toolkit.retry import retry_llm, RetryBudget

# As a decorator
@retry_llm(max_attempts=5, base_delay=1.0)
def call_api(prompt):
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

# With a total budget (useful for parallel calls)
budget = RetryBudget(max_total_wait=120)

@retry_llm(max_attempts=5, on_retry=budget.check)
def call_with_budget(prompt):
    return client.chat.completions.create(...)
```

### Streaming

```python
from llm_toolkit.streaming import stream_openai, print_stream, make_openai_client

client = make_openai_client()
messages = [{"role": "user", "content": "Write a haiku about Python."}]

# Print to stdout as chunks arrive
result = print_stream(stream_openai(client, messages))
print(f"Tokens used: {result.input_tokens} in / {result.output_tokens} out")

# Or collect silently
from llm_toolkit.streaming import collect_stream
result = collect_stream(stream_openai(client, messages))
print(result.text)
```

### Response Caching

```python
from llm_toolkit.cache import DiskCache

cache = DiskCache(ttl=3600)  # 1-hour TTL

@cache.cached()
def get_embedding(text, model="text-embedding-3-small"):
    return client.embeddings.create(model=model, input=text)

# Manual usage
key = cache.make_key(model="gpt-4o", prompt="hello")
cache.set(key, {"response": "world"})
print(cache.get(key))   # → {"response": "world"}
print(cache.stats)       # → {"hits": 1, "misses": 0, ...}
```

### RAG Pipeline

```python
from llm_toolkit.rag import RAGPipeline

rag = RAGPipeline(
    embed_model="text-embedding-3-small",
    chat_model="gpt-4o-mini",
    chunk_size=800,
    top_k=5,
)

# Add documents
rag.add_text("Python 3.12 introduced improved error messages and typing.", source="release-notes")
rag.add_document("docs/architecture.txt")

# Query with sources
result = rag.query("What's new in Python 3.12?", return_sources=True)
print(result["answer"])
print(result["sources"])
```

## Supported Models & Pricing (April 2026)

| Provider | Model | Input $/1M | Output $/1M |
|----------|-------|-----------|-------------|
| OpenAI | gpt-4o | $2.50 | $10.00 |
| OpenAI | gpt-4o-mini | $0.15 | $0.60 |
| OpenAI | o3-mini | $1.10 | $4.40 |
| Anthropic | Claude 3.5 Sonnet | $3.00 | $15.00 |
| Anthropic | Claude 3.5 Haiku | $0.80 | $4.00 |
| Anthropic | Claude 3 Opus | $15.00 | $75.00 |
| Google | Gemini 2.0 Flash | $0.10 | $0.40 |
| Google | Gemini 2.5 Pro | $1.25 | $10.00 |
| Meta | Llama 3.1 70B | $0.88 | $0.88 |
| Mistral | Mistral Large | $2.00 | $6.00 |

Full pricing table in [`llm_toolkit/costs.py`](llm_toolkit/costs.py).

## Project Structure

```
llm-toolkit/
├── llm_toolkit/
│   ├── __init__.py       # Public API
│   ├── prompts.py        # Prompt templates, chaining, variable injection
│   ├── tokens.py         # Token counting (tiktoken + heuristics)
│   ├── costs.py          # Cost estimation across providers
│   ├── retry.py          # Retry with exponential backoff
│   ├── streaming.py      # Streaming helpers for OpenAI/Anthropic
│   ├── rag.py            # Simple RAG pipeline
│   └── cache.py          # Disk-based response caching
├── examples/
│   ├── prompt_chain.py   # Multi-step prompt chaining
│   ├── cost_compare.py   # Cross-provider cost comparison
│   └── simple_rag.py     # RAG over local documents
├── .env.example
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

```bash
# From source
pip install -e .

# Or just install deps
pip install -r requirements.txt
```

## License

MIT — see [LICENSE](LICENSE)
