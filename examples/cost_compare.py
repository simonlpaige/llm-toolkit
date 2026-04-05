"""
examples/cost_compare.py — Compare LLM costs across providers.

Demonstrates:
  - estimate_cost() for a single model
  - compare_costs() across all major models
  - Token counting to ground the estimates in a real prompt

Run:
    python examples/cost_compare.py
"""

from llm_toolkit.tokens import count_tokens
from llm_toolkit.costs import estimate_cost, compare_costs, format_cost, PRICING


SAMPLE_PROMPT = """
You are an expert software architect. Please review the following Python code
and provide detailed feedback on: (1) code quality, (2) performance, (3) security
concerns, (4) maintainability, and (5) suggested improvements.

```python
import sqlite3, os, pickle
from flask import Flask, request

app = Flask(__name__)
DB = sqlite3.connect('users.db')

@app.route('/login')
def login():
    username = request.args.get('user')
    password = request.args.get('pass')
    cur = DB.cursor()
    cur.execute(f"SELECT * FROM users WHERE name='{username}' AND pass='{password}'")
    user = cur.fetchone()
    if user:
        data = pickle.loads(user[3])
        return str(data)
    return "Login failed"
```
""".strip()

ESTIMATED_OUTPUT_TOKENS = 800  # typical code review response


def main():
    print("=" * 70)
    print("LLM Toolkit — Cost Comparison Demo")
    print("=" * 70)

    # Count tokens in our prompt
    input_tokens_gpt4o = count_tokens(SAMPLE_PROMPT, model="gpt-4o")
    input_tokens_claude = count_tokens(SAMPLE_PROMPT, model="claude-3-5-sonnet")
    input_tokens_gemini = count_tokens(SAMPLE_PROMPT, model="gemini-1.5-pro")

    print(f"\nSample prompt length: {len(SAMPLE_PROMPT)} characters")
    print(f"  GPT-4o tokens:   {input_tokens_gpt4o:,}")
    print(f"  Claude tokens:   {input_tokens_claude:,}  (approximated)")
    print(f"  Gemini tokens:   {input_tokens_gemini:,}  (approximated)")
    print(f"  Assumed output:  {ESTIMATED_OUTPUT_TOKENS:,} tokens")

    # Single model cost
    print("\n── Single Model Estimate: gpt-4o ────────────────────────────────────")
    result = estimate_cost("gpt-4o", input_tokens_gpt4o, ESTIMATED_OUTPUT_TOKENS)
    print(f"  Input:  {result['input_tokens']:,} tokens @ ${result['pricing_per_1m']['input']}/1M "
          f"= {format_cost(result['input_cost'])}")
    print(f"  Output: {result['output_tokens']:,} tokens @ ${result['pricing_per_1m']['output']}/1M "
          f"= {format_cost(result['output_cost'])}")
    print(f"  Total:  {format_cost(result['total_cost'])}")

    # Cross-provider comparison
    print("\n── Cross-Provider Comparison ────────────────────────────────────────")
    print(f"  (using {input_tokens_gpt4o} input tokens + {ESTIMATED_OUTPUT_TOKENS} output tokens)\n")

    models = [
        "gpt-4o-mini",
        "gpt-4o",
        "o3-mini",
        "claude-3-5-haiku",
        "claude-3-5-sonnet",
        "claude-3-opus",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-2.5-pro",
        "llama-3.1-8b",
        "llama-3.1-70b",
    ]

    results = compare_costs(input_tokens_gpt4o, ESTIMATED_OUTPUT_TOKENS, models=models)

    header = f"  {'Model':<30} {'Provider':<15} {'Input':<12} {'Output':<12} {'Total':<12}"
    print(header)
    print("  " + "-" * 78)

    for r in results:
        if "error" in r:
            print(f"  {r['model']:<30} {'N/A':<15} {'error':<12}")
            continue

        model_name = r["model"]
        provider = r.get("provider", "?")
        input_c = format_cost(r["input_cost"])
        output_c = format_cost(r["output_cost"])
        total_c = format_cost(r["total_cost"])
        print(f"  {model_name:<30} {provider:<15} {input_c:<12} {output_c:<12} {total_c:<12}")

    cheapest = results[0]
    expensive = results[-1]
    print(f"\n  Cheapest:  {cheapest['model']} @ {format_cost(cheapest['total_cost'])}")
    print(f"  Priciest:  {expensive['model']} @ {format_cost(expensive['total_cost'])}")

    if cheapest["total_cost"] > 0:
        ratio = expensive["total_cost"] / cheapest["total_cost"]
        print(f"  Ratio:     {ratio:.1f}x more expensive at the top")

    # Scale estimate
    print("\n── Scale Estimate (1M requests/month) ───────────────────────────────")
    for r in [results[0], results[len(results) // 2], results[-1]]:
        if "error" in r:
            continue
        monthly = r["total_cost"] * 1_000_000
        print(f"  {r['model']:<30} ${monthly:,.2f}/month")

    print()


if __name__ == "__main__":
    main()
