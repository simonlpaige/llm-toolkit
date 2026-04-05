#!/usr/bin/env python3
"""
Example: Token counting and cost estimation.

Demonstrates counting tokens across different models and estimating
API costs before making calls.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_toolkit import count_tokens, estimate_cost

sample_text = """
The transformer architecture, introduced in the seminal "Attention Is All You Need" 
paper by Vaswani et al. (2017), revolutionized natural language processing by replacing 
recurrent neural networks with self-attention mechanisms. This architecture forms the 
foundation of modern large language models including GPT-4, Claude, Gemini, and LLaMA.

Key innovations include multi-head attention, positional encoding, and the encoder-decoder 
structure. These models are trained on vast corpora of text data and can be fine-tuned 
for specific tasks or used directly through prompting.
"""

print("=" * 60)
print("Token Counting & Cost Estimation")
print("=" * 60)

# Count tokens across models
models = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "claude-sonnet"]
for model in models:
    tokens = count_tokens(sample_text, model)
    print(f"\n{model}: {tokens} tokens")

    # Estimate cost for a typical request
    cost = estimate_cost(
        input_tokens=tokens + 500,  # +500 for system prompt
        output_tokens=1000,         # ~750 words of output
        model=model,
    )
    print(f"  Estimated cost: ${cost['total_cost']:.4f}")
    print(f"    Input:  ${cost['input_cost']:.4f} ({cost['input_tokens']} tokens)")
    print(f"    Output: ${cost['output_cost']:.4f} ({cost['output_tokens']} tokens)")

# Batch cost estimation
print("\n" + "=" * 60)
print("Batch Processing Cost Estimate")
print("=" * 60)
num_requests = 1000
avg_input = 2000
avg_output = 500

for model in ["gpt-4o", "gpt-4o-mini", "claude-sonnet"]:
    cost = estimate_cost(avg_input * num_requests, avg_output * num_requests, model)
    print(f"\n{model} ({num_requests} requests):")
    print(f"  Total: ${cost['total_cost']:.2f}")
