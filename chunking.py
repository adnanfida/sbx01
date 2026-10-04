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

# Recommended settings for a 512-token embedding model and a ~6,000-token
# retrieval budget: 300 words is about 400 real English tokens (about 15
# chunks per prompt), and 50 words of overlap is about 2-3 sentences.
DEFAULT_MAX_TOKENS = 300
DEFAULT_OVERLAP = 50


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
