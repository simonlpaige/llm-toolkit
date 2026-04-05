"""
Streaming helpers for OpenAI and Anthropic APIs.

Process LLM responses token-by-token for real-time UIs,
progress indicators, and early termination.
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Generator, Optional, Callable

logger = logging.getLogger(__name__)


def stream_openai(
    messages: list[dict],
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Generator[str, None, None]:
    """Stream tokens from OpenAI's chat completions API.

    Yields individual text chunks as they arrive.

    Args:
        messages: Chat messages.
        model: Model name (default: gpt-4o).
        temperature: Sampling temperature.
        max_tokens: Max output tokens.
        api_key: API key override.
        base_url: Base URL override (for local/proxy endpoints).

    Yields:
        Text chunks (delta content).
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        base_url=base_url or os.getenv("LOCAL_LLM_BASE_URL"),
    )

    stream = client.chat.completions.create(
        model=model or os.getenv("OPENAI_MODEL", "gpt-4o"),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


def stream_anthropic(
    messages: list[dict],
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    system: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Generator[str, None, None]:
    """Stream tokens from Anthropic's messages API.

    Yields individual text chunks as they arrive.

    Args:
        messages: Chat messages (user/assistant only).
        model: Model name (default: claude-sonnet-4-20250514).
        temperature: Sampling temperature.
        max_tokens: Max output tokens.
        system: System prompt.
        api_key: API key override.

    Yields:
        Text chunks (delta content).
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    # Separate system messages
    chat_messages = []
    system_parts = [system] if system else []
    for m in messages:
        if m["role"] == "system":
            system_parts.append(m["content"])
        else:
            chat_messages.append(m)

    with client.messages.stream(
        model=model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        max_tokens=max_tokens,
        temperature=temperature,
        system="\n".join(system_parts) if system_parts else None,
        messages=chat_messages,
    ) as stream:
        for text in stream.text_stream:
            yield text


def print_stream(
    stream: Generator[str, None, None],
    end: str = "\n",
    file=None,
    on_token: Optional[Callable[[str], None]] = None,
) -> str:
    """Print a token stream to stdout and return the full text.

    Args:
        stream: A generator yielding text chunks.
        end: String to print after the stream completes.
        file: Output file (default: sys.stdout).
        on_token: Optional callback for each token.

    Returns:
        The complete accumulated text.
    """
    file = file or sys.stdout
    full_text = []

    for chunk in stream:
        print(chunk, end="", flush=True, file=file)
        full_text.append(chunk)
        if on_token:
            on_token(chunk)

    print(end, end="", file=file)
    return "".join(full_text)
