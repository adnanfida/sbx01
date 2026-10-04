"""
Production-grade batch text embedding utility with exponential backoff and jitter.
Supports any embedding API via dependency injection, concurrent batch execution,
and guarantees exact preservation of original input ordering.
"""

from __future__ import annotations

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class EmbeddingError(Exception):
    """Base exception for embedding pipeline errors."""
    pass


class PermanentAPIError(EmbeddingError):
    """Raised when an API error is non-retryable (e.g., 400 Bad Request, 401 Unauthorized)."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class MaxRetriesExceededError(EmbeddingError):
    """Raised when transient errors persist beyond max retries."""
    def __init__(self, message: str, last_exception: Optional[Exception] = None):
        super().__init__(message)
        self.last_exception = last_exception


@dataclass(frozen=True)
class RetryConfig:
    """Configuration for retry behavior with exponential backoff."""
    max_retries: int = 6
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    jitter_factor: float = 0.5  # random factor in [1 - jitter, 1 + jitter]
    retryable_status_codes: tuple[int, ...] = (429, 500, 502, 503, 504)


def extract_status_code_and_retry_after(exc: Exception) -> tuple[Optional[int], Optional[float]]:
    """
    Extracts HTTP status code and Retry-After delay (in seconds) from common
    exception patterns (requests, httpx, custom API exceptions, or SDK errors).
    """
    status_code: Optional[int] = None
    retry_after: Optional[float] = None

    # Check direct attributes on exception
    if hasattr(exc, "status_code"):
        status_code = getattr(exc, "status_code")
    elif hasattr(exc, "http_status"):
        status_code = getattr(exc, "http_status")
    elif hasattr(exc, "code"):
        status_code = getattr(exc, "code")

    # Check nested response object (e.g. requests.HTTPError, httpx.HTTPStatusError)
    response = getattr(exc, "response", None)
    if response is not None:
        if status_code is None and hasattr(response, "status_code"):
            status_code = response.status_code

        # Check headers for Retry-After
        headers = getattr(response, "headers", None)
        if headers and hasattr(headers, "get"):
            header_val = headers.get("retry-after") or headers.get("Retry-After")
            if header_val:
                try:
                    retry_after = float(header_val)
                except (ValueError, TypeError):
                    pass

    if hasattr(exc, "retry_after") and getattr(exc, "retry_after") is not None:
        try:
            retry_after = float(getattr(exc, "retry_after"))
        except (ValueError, TypeError):
            pass

    return status_code, retry_after


def is_retryable_error(exc: Exception, config: RetryConfig) -> bool:
    """
    Determines whether an exception is transient (retryable) or permanent.
    Network/connection timeouts and status codes in retryable_status_codes are retryable.
    4xx status codes (other than 429) are considered permanent.
    """
    # Explicit permanent error
    if isinstance(exc, PermanentAPIError):
        return False

    status_code, _ = extract_status_code_and_retry_after(exc)

    if status_code is not None:
        if status_code in config.retryable_status_codes:
            return True
        # 4xx errors other than 429 are client errors (Bad Request, Unauthorized, etc.)
        if 400 <= status_code < 500:
            return False
        # 5xx errors default to retryable
        if status_code >= 500:
            return True

    # Common transient connection / socket / timeout exceptions
    transient_exc_names = {
        "Timeout",
        "ConnectTimeout",
        "ReadTimeout",
        "ConnectionError",
        "RemoteDisconnected",
        "APIConnectionError",
        "RateLimitError",
        "InternalServerError",
    }
    exc_type_name = type(exc).__name__
    if any(name in exc_type_name for name in transient_exc_names):
        return True

    return False


def calculate_backoff(
    attempt: int,
    config: RetryConfig,
    retry_after: Optional[float] = None,
    sleeper_random: Callable[[float, float], float] = random.uniform,
) -> float:
    """
    Calculates backoff delay using exponential backoff with full jitter,
    respecting Retry-After header if provided.
    """
    if retry_after is not None and retry_after > 0:
        return min(config.max_delay, retry_after)

    # Exponential backoff: base_delay * (2 ** attempt)
    raw_delay = min(config.max_delay, config.base_delay * (2 ** attempt))

    # Apply jitter: e.g. uniform in [1 - factor, 1 + factor]
    low = max(0.0, 1.0 - config.jitter_factor)
    high = 1.0 + config.jitter_factor
    jittered_delay = raw_delay * sleeper_random(low, high)

    return min(config.max_delay, max(0.0, jittered_delay))


def embed_batch_with_retry(
    batch: list[str],
    embed_fn: Callable[[list[str]], list[list[float]]],
    config: RetryConfig,
    batch_idx: int = 0,
    sleeper: Callable[[float], None] = time.sleep,
) -> list[list[float]]:
    """
    Embeds a single batch of texts with exponential backoff and jitter on transient errors.
    """
    attempt = 0
    while True:
        try:
            vectors = embed_fn(batch)
            if len(vectors) != len(batch):
                raise ValueError(
                    f"API returned {len(vectors)} vectors for batch of size {len(batch)}."
                )
            return vectors

        except Exception as exc:
            status_code, retry_after = extract_status_code_and_retry_after(exc)

            if not is_retryable_error(exc, config):
                logger.error(
                    "Permanent error encountered on batch %d: %s (status=%s)",
                    batch_idx,
                    exc,
                    status_code,
                )
                raise PermanentAPIError(
                    f"Permanent error in batch {batch_idx}: {exc}",
                    status_code=status_code,
                ) from exc

            attempt += 1
            if attempt > config.max_retries:
                logger.error(
                    "Max retries (%d) exceeded on batch %d. Last error: %s",
                    config.max_retries,
                    batch_idx,
                    exc,
                )
                raise MaxRetriesExceededError(
                    f"Batch {batch_idx} failed after {config.max_retries} retries: {exc}",
                    last_exception=exc,
                ) from exc

            delay = calculate_backoff(attempt - 1, config, retry_after)
            logger.warning(
                "Batch %d: transient error %s (status=%s). Retrying in %.2fs (attempt %d/%d)...",
                batch_idx,
                type(exc).__name__,
                status_code,
                delay,
                attempt,
                config.max_retries,
            )
            sleeper(delay)


def chunk_list(items: Sequence[T], chunk_size: int) -> list[list[T]]:
    """Splits a sequence into chunks of maximum size `chunk_size`."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    return [list(items[i : i + chunk_size]) for i in range(0, len(items), chunk_size)]


def embed_texts(
    texts: Sequence[str],
    embed_batch_fn: Callable[[list[str]], list[list[float]]],
    batch_size: int = 100,
    max_concurrency: int = 4,
    retry_config: Optional[RetryConfig] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> list[list[float]]:
    """
    Embeds a sequence of texts in batches with retries, exponential backoff, and jitter.
    
    Guarantees:
      - Texts are processed in batches of size `batch_size`.
      - Transient failures (429, 5xx, network timeouts) are retried with exponential backoff.
      - Output vectors strictly preserve original input order.
      - Supports parallel workers (`max_concurrency > 1`) while maintaining order.

    Args:
      texts: Sequence of input text strings (e.g. 40,000 support article chunks).
      embed_batch_fn: Callable that accepts a list of texts and returns vectors.
      batch_size: Number of texts per API call.
      max_concurrency: Max concurrent requests (1 = sequential).
      retry_config: Optional RetryConfig (defaults to standard production settings).
      progress_callback: Optional callback(completed_batches, total_batches).
      sleeper: Sleep function used for backoff (customizable for unit tests).

    Returns:
      List of embedding vectors (list of floats) corresponding 1-to-1 to input texts.
    """
    if not texts:
        return []

    config = retry_config or RetryConfig()
    batches = chunk_list(texts, batch_size)
    total_batches = len(batches)

    # Pre-allocate results list indexed by batch index to guarantee original order
    results_by_batch: list[Optional[list[list[float]]]] = [None] * total_batches

    if max_concurrency <= 1 or total_batches == 1:
        # Sequential execution
        for idx, batch in enumerate(batches):
            vectors = embed_batch_with_retry(
                batch=batch,
                embed_fn=embed_batch_fn,
                config=config,
                batch_idx=idx,
                sleeper=sleeper,
            )
            results_by_batch[idx] = vectors
            if progress_callback:
                progress_callback(idx + 1, total_batches)
    else:
        # Concurrent execution with order preservation
        completed_count = 0
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            future_to_idx = {
                executor.submit(
                    embed_batch_with_retry,
                    batch,
                    embed_batch_fn,
                    config,
                    idx,
                    sleeper,
                ): idx
                for idx, batch in enumerate(batches)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                # If any batch raises PermanentAPIError or MaxRetriesExceededError,
                # future.result() will raise here immediately.
                vectors = future.result()
                results_by_batch[idx] = vectors
                completed_count += 1
                if progress_callback:
                    progress_callback(completed_count, total_batches)

    # Flatten results in strict batch order
    flattened: list[list[float]] = []
    for batch_vectors in results_by_batch:
        assert batch_vectors is not None, "Batch vector slot must be populated"
        flattened.extend(batch_vectors)

    return flattened
