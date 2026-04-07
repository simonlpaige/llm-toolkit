"""
costs.py — Cost estimation across LLM providers.

Pricing as of April 2026 (USD per 1M tokens).
Sources: OpenAI pricing page, Anthropic pricing page, Google AI Studio pricing.

Update PRICING when providers change rates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelPricing:
    """Per-model pricing in USD per 1M tokens."""
    input: float   # $/1M input tokens
    output: float  # $/1M output tokens
    provider: str
    notes: str = ""


# ── Current pricing table (USD / 1M tokens) ──────────────────────────────────
# Last updated: 2026-04
PRICING: dict[str, ModelPricing] = {
    # ── OpenAI ────────────────────────────────────────────────────────────────
    "gpt-4o": ModelPricing(input=2.50, output=10.00, provider="openai"),
    "gpt-4o-mini": ModelPricing(input=0.15, output=0.60, provider="openai"),
    "gpt-4o-audio-preview": ModelPricing(input=2.50, output=10.00, provider="openai"),
    "o1": ModelPricing(input=15.00, output=60.00, provider="openai", notes="reasoning model"),
    "o1-mini": ModelPricing(input=1.10, output=4.40, provider="openai", notes="reasoning model"),
    "o3": ModelPricing(input=10.00, output=40.00, provider="openai", notes="reasoning model"),
    "o3-mini": ModelPricing(input=1.10, output=4.40, provider="openai", notes="reasoning model"),
    "o4-mini": ModelPricing(input=1.10, output=4.40, provider="openai", notes="reasoning model"),
    "gpt-4-turbo": ModelPricing(input=10.00, output=30.00, provider="openai"),
    "gpt-4": ModelPricing(input=30.00, output=60.00, provider="openai"),
    "gpt-3.5-turbo": ModelPricing(input=0.50, output=1.50, provider="openai"),
    # Embeddings (output price not applicable — set to 0)
    "text-embedding-3-small": ModelPricing(input=0.02, output=0.0, provider="openai"),
    "text-embedding-3-large": ModelPricing(input=0.13, output=0.0, provider="openai"),
    "text-embedding-ada-002": ModelPricing(input=0.10, output=0.0, provider="openai"),

    # ── Anthropic ─────────────────────────────────────────────────────────────
    # Claude 4 (2025–2026)
    "claude-sonnet-4-5": ModelPricing(input=3.00, output=15.00, provider="anthropic"),
    "claude-sonnet-4": ModelPricing(input=3.00, output=15.00, provider="anthropic"),
    "claude-haiku-4": ModelPricing(input=0.80, output=4.00, provider="anthropic"),
    "claude-opus-4": ModelPricing(input=15.00, output=75.00, provider="anthropic"),
    # Claude 3.5
    "claude-3-5-sonnet-20241022": ModelPricing(input=3.00, output=15.00, provider="anthropic"),
    "claude-3-5-haiku-20241022": ModelPricing(input=0.80, output=4.00, provider="anthropic"),
    "claude-3-opus-20240229": ModelPricing(input=15.00, output=75.00, provider="anthropic"),
    "claude-3-haiku-20240307": ModelPricing(input=0.25, output=1.25, provider="anthropic"),
    "claude-3-sonnet-20240229": ModelPricing(input=3.00, output=15.00, provider="anthropic"),
    # Short aliases
    "claude-3-5-sonnet": ModelPricing(input=3.00, output=15.00, provider="anthropic"),
    "claude-3-5-haiku": ModelPricing(input=0.80, output=4.00, provider="anthropic"),
    "claude-3-opus": ModelPricing(input=15.00, output=75.00, provider="anthropic"),
    "claude-3-haiku": ModelPricing(input=0.25, output=1.25, provider="anthropic"),

    # ── Google ────────────────────────────────────────────────────────────────
    "gemini-1.5-pro": ModelPricing(
        input=3.50, output=10.50, provider="google",
        notes="<=128k prompt; >128k is $7/$21"
    ),
    "gemini-1.5-flash": ModelPricing(input=0.075, output=0.30, provider="google"),
    "gemini-1.5-flash-8b": ModelPricing(input=0.0375, output=0.15, provider="google"),
    "gemini-2.0-flash": ModelPricing(input=0.10, output=0.40, provider="google"),
    "gemini-2.0-flash-lite": ModelPricing(input=0.075, output=0.30, provider="google"),
    "gemini-2.5-pro": ModelPricing(input=1.25, output=10.00, provider="google"),

    # ── Meta / Open-source (via API providers) ────────────────────────────────
    "llama-3.1-405b": ModelPricing(input=3.00, output=3.00, provider="meta/third-party"),
    "llama-3.1-70b": ModelPricing(input=0.88, output=0.88, provider="meta/third-party"),
    "llama-3.1-8b": ModelPricing(input=0.18, output=0.18, provider="meta/third-party"),
    "llama-3.3-70b": ModelPricing(input=0.59, output=0.79, provider="meta/third-party"),

    # ── Mistral ───────────────────────────────────────────────────────────────
    "mistral-large": ModelPricing(input=2.00, output=6.00, provider="mistral"),
    "mistral-small": ModelPricing(input=0.20, output=0.60, provider="mistral"),
    "mixtral-8x22b": ModelPricing(input=1.20, output=1.20, provider="mistral"),
}


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int = 0,
) -> dict:
    """
    Estimate the cost of an API call.

    Args:
        model: Model name (partial match supported, e.g. "gpt-4o").
        input_tokens: Number of input/prompt tokens.
        output_tokens: Number of output/completion tokens.

    Returns:
        Dict with keys: model, input_tokens, output_tokens,
        input_cost, output_cost, total_cost, currency, pricing.

    Raises:
        KeyError: If the model is not found in the pricing table.
    """
    pricing = _resolve_pricing(model)

    input_cost = (input_tokens / 1_000_000) * pricing.input
    output_cost = (output_tokens / 1_000_000) * pricing.output
    total_cost = input_cost + output_cost

    return {
        "model": model,
        "provider": pricing.provider,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": round(input_cost, 8),
        "output_cost": round(output_cost, 8),
        "total_cost": round(total_cost, 8),
        "currency": "USD",
        "pricing_per_1m": {
            "input": pricing.input,
            "output": pricing.output,
        },
    }


def compare_costs(
    prompt_tokens: int,
    completion_tokens: int,
    models: Optional[list[str]] = None,
) -> list[dict]:
    """
    Compare estimated costs for multiple models side by side.

    Args:
        prompt_tokens: Input token count to use for all models.
        completion_tokens: Output token count to use for all models.
        models: List of model names. If None, uses a curated default set.

    Returns:
        List of cost dicts sorted by total_cost ascending.
    """
    if models is None:
        models = [
            "gpt-4o-mini",
            "gpt-4o",
            "o3-mini",
            "claude-3-5-haiku",
            "claude-3-5-sonnet",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
        ]

    results = []
    for model in models:
        try:
            results.append(estimate_cost(model, prompt_tokens, completion_tokens))
        except KeyError:
            results.append({"model": model, "error": "pricing not found"})

    results.sort(key=lambda x: x.get("total_cost", float("inf")))
    return results


def format_cost(cost: float, precision: int = 6) -> str:
    """Format a dollar cost for display."""
    if cost < 0.000001:
        return f"${cost:.2e}"
    if cost < 0.01:
        return f"${cost:.{precision}f}"
    return f"${cost:.4f}"


def _resolve_pricing(model: str) -> ModelPricing:
    """Find pricing for a model, supporting partial name matching."""
    # Exact match first
    if model in PRICING:
        return PRICING[model]

    # Case-insensitive exact
    model_lower = model.lower()
    for key, pricing in PRICING.items():
        if key.lower() == model_lower:
            return pricing

    # Substring match (longest key wins to prefer specific versions)
    candidates = [
        (key, p) for key, p in PRICING.items()
        if key.lower() in model_lower or model_lower in key.lower()
    ]
    if candidates:
        best_key, best_pricing = max(candidates, key=lambda x: len(x[0]))
        return best_pricing

    raise KeyError(
        f"No pricing found for model '{model}'. "
        f"Available models: {list(PRICING.keys())}"
    )
