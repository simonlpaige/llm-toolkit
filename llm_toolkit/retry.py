"""
Retry and rate-limit handling for LLM API calls.

Provides decorators and utilities to gracefully handle rate limits,
transient errors, and API instability.
"""

from __future__ import annotations

import time
import logging
import threading
from typing import Optional, Callable, Any
from functools import wraps

logger = logging.getLogger(__name__)


def llm_retry(
    max_retries: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    retry_on: Optional[tuple] = None,
) -> Callable:
    """Decorator for retrying LLM API calls with exponential backoff.

    Automatically handles rate limits (429), server errors (500+),
    and connection issues.

    Args:
        max_retries: Maximum retry attempts.
        initial_delay: Starting delay in seconds.
        max_delay: Maximum delay between retries.
        backoff_factor: Multiplier for each retry.
        retry_on: Tuple of exception types to retry on.
                  Defaults to common API errors.

    Example:
        @llm_retry(max_retries=3)
        def call_api(prompt):
            return openai.chat.completions.create(...)
    """
    if retry_on is None:
        # Import common API error types lazily
        retry_exceptions = (
            ConnectionError,
            TimeoutError,
            OSError,
        )
    else:
        retry_exceptions = retry_on

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_error = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    error_str = str(e).lower()

                    # Check if it's a rate limit error (various providers)
                    is_rate_limit = (
                        "rate_limit" in error_str
                        or "429" in error_str
                        or "too many requests" in error_str
                        or "rate limit" in error_str
                    )

                    # Check if it's a retryable server error
                    is_server_error = (
                        "500" in error_str
                        or "502" in error_str
                        or "503" in error_str
                        or "overloaded" in error_str
                        or isinstance(e, retry_exceptions)
                    )

                    if not (is_rate_limit or is_server_error):
                        raise  # Non-retryable error

                    if attempt == max_retries:
                        logger.error(
                            "Max retries (%d) exhausted for %s: %s",
                            max_retries,
                            func.__name__,
                            e,
                        )
                        raise

                    # Parse Retry-After header if available
                    retry_after = _extract_retry_after(e)
                    wait_time = retry_after if retry_after else delay

                    logger.warning(
                        "Retry %d/%d for %s (waiting %.1fs): %s",
                        attempt + 1,
                        max_retries,
                        func.__name__,
                        wait_time,
                        str(e)[:100],
                    )

                    time.sleep(wait_time)
                    delay = min(delay * backoff_factor, max_delay)

            raise last_error  # Should never reach here

        return wrapper

    return decorator


def _extract_retry_after(error: Exception) -> Optional[float]:
    """Try to extract Retry-After from an API error response."""
    # OpenAI errors often have response headers
    response = getattr(error, "response", None)
    if response:
        headers = getattr(response, "headers", {})
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
    return None


class RateLimiter:
    """Token bucket rate limiter for controlling API call frequency.

    Thread-safe. Use as a context manager or call .acquire() directly.

    Example:
        limiter = RateLimiter(requests_per_minute=60)

        for prompt in prompts:
            limiter.acquire()
            response = call_api(prompt)
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        tokens_per_minute: Optional[int] = None,
    ):
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute
        self._interval = 60.0 / requests_per_minute
        self._last_request = 0.0
        self._lock = threading.Lock()
        self._token_count = 0
        self._token_reset_time = time.time()

    def acquire(self, tokens: int = 0) -> None:
        """Wait until a request is allowed under the rate limit.

        Args:
            tokens: Estimated tokens for this request (for TPM limiting).
        """
        with self._lock:
            now = time.time()

            # Request rate limiting
            elapsed = now - self._last_request
            if elapsed < self._interval:
                sleep_time = self._interval - elapsed
                time.sleep(sleep_time)

            # Token rate limiting
            if self.tpm and tokens > 0:
                if now - self._token_reset_time >= 60:
                    self._token_count = 0
                    self._token_reset_time = now

                if self._token_count + tokens > self.tpm:
                    sleep_time = 60 - (now - self._token_reset_time)
                    if sleep_time > 0:
                        logger.info("TPM limit reached, waiting %.1fs", sleep_time)
                        time.sleep(sleep_time)
                    self._token_count = 0
                    self._token_reset_time = time.time()

                self._token_count += tokens

            self._last_request = time.time()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        pass
