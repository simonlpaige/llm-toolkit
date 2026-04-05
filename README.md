# llm-toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-Compatible-412991)](https://openai.com)
[![Anthropic](https://img.shields.io/badge/Anthropic-Compatible-D97757)](https://anthropic.com)

A practical Python utility library for working with LLMs. Prompt templates, token counting, cost estimation, retry handling, streaming helpers, and a simple RAG pipeline — tools a developer actually needs.

## Features

- 📝 **Prompt Templates** — reusable templates with `{{variable}}` substitution, partial application, and chaining
- 🔢 **Token Counting** — accurate counts via tiktoken (OpenAI) with fallback estimation for other providers
- 💰 **Cost Estimation** — per-model pricing for OpenAI + Anthropic, batch cost projections
- 🔄 **Retry & Rate Limiting** — exponential backoff, Retry-After parsing, token bucket rate limiter
- 📡 **Streaming** — helpers for streaming OpenAI and Anthropic responses with real-time output
- 🔍 **RAG Pipeline** — minimal retrieval-augmented generation with ChromaDB

## Quick Start

```bash
git clone https://github.com/simonlpaige/llm-toolkit.git
cd llm-toolkit
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your API key(s)

# Try the cost calculator (no API key needed)
python examples/cost_calculator.py
```

## Usage

### Prompt Templates

```python
from llm_toolkit import PromptTemplate, PromptChain

# Simple template
template = PromptTemplate(
    "Summarize this {{language}} code:\n\n{{code}}",
    defaults={"language": "Python"}
)
prompt = template.render(code="def fibonacci(n): ...")

# Chaining
chain = PromptChain()
chain.add("research", PromptTemplate("List 5 facts about {{topic}}."))
chain.add("summary", PromptTemplate("Summarize:\n{{research}}"))
results = chain.run(llm_fn=my_llm, topic="quantum computing")
```

### Token Counting & Cost Estimation

```python
from llm_toolkit import count_tokens, estimate_cost

tokens = count_tokens("Hello, world!", model="gpt-4o")  # 4

cost = estimate_cost(
    input_tokens=2000,
    output_tokens=500,
    model="claude-sonnet"
)
print(f"${cost['total_cost']:.4f}")  # $0.0135
```

### Retry & Rate Limiting

```python
from llm_toolkit import llm_retry, RateLimiter

@llm_retry(max_retries=3, initial_delay=1.0)
def call_api(prompt):
    return client.chat.completions.create(...)

# Rate limit to 60 RPM
limiter = RateLimiter(requests_per_minute=60)
for prompt in prompts:
    limiter.acquire()
    response = call_api(prompt)
```

### Streaming

```python
from llm_toolkit import stream_openai, print_stream

stream = stream_openai(
    messages=[{"role": "user", "content": "Write a haiku about Python."}],
    model="gpt-4o"
)
full_text = print_stream(stream)
```

### RAG Pipeline

```python
from llm_toolkit import SimpleRAG

rag = SimpleRAG(collection_name="docs")
rag.add("Python 3.12 introduced improved error messages.")
rag.add("FastAPI uses type hints for request validation.")

answer = rag.query("What's new in Python 3.12?")
```

## Supported Models & Pricing

Built-in pricing data for cost estimation:

| Provider | Models | Input $/1M | Output $/1M |
|----------|--------|-----------|-------------|
| OpenAI | gpt-4o | $2.50 | $10.00 |
| OpenAI | gpt-4o-mini | $0.15 | $0.60 |
| OpenAI | o1 | $15.00 | $60.00 |
| OpenAI | o3-mini | $1.10 | $4.40 |
| Anthropic | Claude Opus | $15.00 | $75.00 |
| Anthropic | Claude Sonnet | $3.00 | $15.00 |
| Anthropic | Claude Haiku | $0.80 | $4.00 |

## Project Structure

```
llm-toolkit/
├── llm_toolkit/
│   ├── __init__.py     # Public API
│   ├── prompts.py      # PromptTemplate + PromptChain
│   ├── tokens.py       # Token counting + cost estimation
│   ├── retry.py        # Retry decorator + RateLimiter
│   ├── streaming.py    # OpenAI/Anthropic streaming helpers
│   └── rag.py          # SimpleRAG pipeline
├── examples/
│   ├── cost_calculator.py  # Token counting demo (no API key needed)
│   ├── prompt_chain.py     # Template + chaining demo
│   └── rag_demo.py         # RAG pipeline demo
├── .env.example
├── requirements.txt
└── README.md
```

## License

MIT — see [LICENSE](LICENSE)
