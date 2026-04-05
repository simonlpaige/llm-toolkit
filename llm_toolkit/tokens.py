"""
Token counting and cost estimation across LLM providers.

Supports OpenAI (via tiktoken) and Anthropic models with accurate
per-model pricing.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Pricing per 1M tokens (as of 2025) ──────────────────────────────────────
# Format: (input_cost_per_1M, output_cost_per_1M)
PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o1": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
    "o3-mini": (1.10, 4.40),
    # Anthropic
    "claude-opus-4-20250514": (15.00, 75.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-3-opus-20240229": (15.00, 75.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    # Aliases
    "claude-opus": (15.00, 75.00),
    "claude-sonnet": (3.00, 15.00),
    "claude-haiku": (0.80, 4.00),
}

# ── Tiktoken encoding mappings ───────────────────────────────────────────────
_ENCODINGS: dict[str, str] = {
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "o1": "o200k_base",
    "o1-mini": "o200k_base",
    "o3-mini": "o200k_base",
}


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in text for a given model.

    Uses tiktoken for OpenAI models. For Anthropic models,
    uses a character-based estimate (~4 chars per token).

    Args:
        text: The text to count tokens for.
        model: Model name (e.g., "gpt-4o", "claude-sonnet").

    Returns:
        Estimated token count.
    """
    # Try tiktoken for OpenAI models
    encoding_name = _ENCODINGS.get(model)
    if encoding_name:
        try:
            import tiktoken

            enc = tiktoken.get_encoding(encoding_name)
            return len(enc.encode(text))
        except ImportError:
            logger.debug("tiktoken not installed, using estimate")

    # Fallback: character-based estimate
    return len(text) // 4


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-4o",
) -> dict[str, float]:
    """Estimate the cost of an LLM call.

    Args:
        input_tokens: Number of input (prompt) tokens.
        output_tokens: Number of output (completion) tokens.
        model: Model name.

    Returns:
        Dict with input_cost, output_cost, and total_cost in USD.
    """
    pricing = PRICING.get(model)
    if not pricing:
        # Try partial match
        for key, val in PRICING.items():
            if key in model or model in key:
                pricing = val
                break

    if not pricing:
        logger.warning("No pricing data for model '%s', using gpt-4o rates", model)
        pricing = PRICING["gpt-4o"]

    input_cost = (input_tokens / 1_000_000) * pricing[0]
    output_cost = (output_tokens / 1_000_000) * pricing[1]

    return {
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(input_cost + output_cost, 6),
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def truncate_to_tokens(
    text: str,
    max_tokens: int,
    model: str = "gpt-4o",
    suffix: str = "\n\n[truncated]",
) -> str:
    """Truncate text to fit within a token limit.

    Args:
        text: Text to truncate.
        max_tokens: Maximum token count.
        model: Model for token counting.
        suffix: Text to append if truncation occurs.

    Returns:
        Original text if within limit, otherwise truncated with suffix.
    """
    current = count_tokens(text, model)
    if current <= max_tokens:
        return text

    # Binary search for the right length
    ratio = max_tokens / current
    end = int(len(text) * ratio * 0.95)  # slightly under to account for suffix

    while count_tokens(text[:end] + suffix, model) > max_tokens and end > 0:
        end = int(end * 0.9)

    return text[:end] + suffix
