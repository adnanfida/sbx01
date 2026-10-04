"""Split support-article text into overlapping chunks for embedding.

Tokens are whitespace-separated words (a toy tokenizer). Real subword
tokenizers produce more tokens than words, so callers should pick
``max_tokens`` well under the embedding model's limit.

Known limitations:
- Splits purely by token count, so chunks can cut mid-sentence or separate a
  heading from its content. Overlap reduces, but does not remove, that loss.
- Whitespace splitting does not suit Japanese (no spaces between words): a
  whole sentence or paragraph becomes one "token".
"""

import logging

logger = logging.getLogger(__name__)

# Recommended settings for a 512-token embedding model and a ~6,000-token
# retrieval budget: 300 words is about 400 real English tokens (about 15
# chunks per prompt), and 50 words of overlap is about 2-3 sentences.
DEFAULT_MAX_TOKENS = 300
DEFAULT_OVERLAP = 50

# The embedding model accepts 512 tokens, including its 2 special tokens.
DEFAULT_TOKEN_LIMIT = 510


class ChunkTooLongError(ValueError):
    """A single whitespace token exceeds the real token limit on its own."""


def _is_int(value):
    # bool is a subclass of int, but True/False are not valid sizes.
    return isinstance(value, int) and not isinstance(value, bool)


def chunk_text(text, max_tokens, overlap):
    """Return overlapping chunks of ``text``, in document order.

    Each chunk holds at most ``max_tokens`` tokens joined by single spaces,
    and consecutive chunks share exactly ``overlap`` tokens. Chunking stops
    once a chunk reaches the end of the text, so the last chunk always holds
    at least one token the previous chunk did not.

    Raises:
        ValueError: if ``text`` is not a string, ``max_tokens`` is not an
            integer >= 1, or ``overlap`` is not an integer in
            ``[0, max_tokens)``.
    """
    if not isinstance(text, str):
        raise ValueError(f"text must be a str, got {type(text).__name__}")
    if not _is_int(max_tokens) or max_tokens < 1:
        raise ValueError(f"max_tokens must be an integer >= 1, got {max_tokens!r}")
    if not _is_int(overlap) or overlap < 0:
        raise ValueError(f"overlap must be an integer >= 0, got {overlap!r}")
    if overlap >= max_tokens:
        raise ValueError(
            f"overlap ({overlap}) must be less than max_tokens ({max_tokens})"
        )

    tokens = text.split()
    step = max_tokens - overlap
    chunks = []
    for start in range(0, len(tokens), step):
        end = start + max_tokens
        chunks.append(" ".join(tokens[start:end]))
        if end >= len(tokens):
            break
    return chunks


def chunk_text_within_limit(
    text,
    max_tokens,
    overlap,
    count_tokens,
    token_limit=DEFAULT_TOKEN_LIMIT,
):
    """Chunk ``text`` so every chunk fits the embedding model's real limit.

    ``count_tokens`` is the embedding model's tokenizer (str -> int). Word
    counts underestimate real tokens, especially for error codes, URLs and
    Japanese, and the model truncates silently, so each chunk is checked with
    the real count. If any chunk is over ``token_limit``, the whole article is
    re-chunked with a smaller ``max_tokens`` (overlap scaled down in
    proportion), which keeps overlap uniform and chunk order stable.

    Raises:
        ValueError: for invalid arguments (see ``chunk_text``), a
            non-callable ``count_tokens``, or ``token_limit`` < 1.
        ChunkTooLongError: if a single whitespace token is over
            ``token_limit``, which word splitting cannot fix (typically
            Japanese text with no spaces).
    """
    if not callable(count_tokens):
        raise ValueError("count_tokens must be callable")
    if not _is_int(token_limit) or token_limit < 1:
        raise ValueError(f"token_limit must be an integer >= 1, got {token_limit!r}")

    while True:
        chunks = chunk_text(text, max_tokens, overlap)
        largest = max((count_tokens(c) for c in chunks), default=0)
        if largest <= token_limit:
            return chunks
        if max_tokens == 1:
            raise ChunkTooLongError(
                f"a single token counts as {largest} real tokens, over the "
                f"limit of {token_limit}; whitespace splitting cannot fix this"
            )
        # Shrink by the overshoot ratio, always by at least one word.
        new_max = max(1, min(max_tokens - 1, max_tokens * token_limit // largest))
        new_overlap = min(new_max - 1, overlap * new_max // max_tokens)
        logger.warning(
            "chunk of %d real tokens exceeds limit %d; re-chunking with "
            "max_tokens=%d, overlap=%d",
            largest, token_limit, new_max, new_overlap,
        )
        max_tokens, overlap = new_max, new_overlap
