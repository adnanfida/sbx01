import pytest
from hypothesis import given, strategies as st

from chunking import DEFAULT_MAX_TOKENS, DEFAULT_OVERLAP, chunk_text


def words(n):
    return [f"w{i}" for i in range(n)]


def text_of(n):
    return " ".join(words(n))


def assert_invariants(text, max_tokens, overlap, chunks):
    tokens = text.split()
    split = [c.split() for c in chunks]

    for c in split:
        assert 1 <= len(c) <= max_tokens

    # Consecutive chunks share exactly `overlap` tokens.
    for prev, nxt in zip(split, split[1:]):
        if overlap:
            assert prev[-overlap:] == nxt[:overlap]
        assert len(nxt) > overlap  # every chunk adds new tokens

    # Dropping the overlaps rebuilds the original token list.
    rebuilt = split[0] + [t for c in split[1:] for t in c[overlap:]] if split else []
    assert rebuilt == tokens


# --- Empty input -----------------------------------------------------------

@pytest.mark.parametrize("text", ["", " ", "\n\t  \r\n"])
def test_empty_or_whitespace_only_returns_empty_list(text):
    assert chunk_text(text, 5, 2) == []


# --- Boundaries ------------------------------------------------------------

def test_shorter_than_one_chunk():
    assert chunk_text("reset the router", 5, 2) == ["reset the router"]


def test_exact_fit_gives_one_chunk_and_no_tail():
    assert chunk_text(text_of(5), 5, 2) == [text_of(5)]


def test_one_over_exact_fit_gives_two_chunks():
    chunks = chunk_text(text_of(6), 5, 2)
    assert chunks == ["w0 w1 w2 w3 w4", "w3 w4 w5"]


def test_stride_landing_on_end_emits_no_overlap_only_tail():
    # A naive loop would emit a fifth chunk "w8 w9" with no new tokens.
    chunks = chunk_text(text_of(10), 4, 2)
    assert chunks == ["w0 w1 w2 w3", "w2 w3 w4 w5", "w4 w5 w6 w7", "w6 w7 w8 w9"]


def test_zero_overlap_partitions_text():
    chunks = chunk_text(text_of(7), 3, 0)
    assert chunks == ["w0 w1 w2", "w3 w4 w5", "w6"]


def test_max_tokens_one():
    assert chunk_text("a b c", 1, 0) == ["a", "b", "c"]


def test_overlap_of_max_minus_one_slides_by_one():
    assert chunk_text("a b c d", 3, 2) == ["a b c", "b c d"]


# --- Large article ---------------------------------------------------------

def test_large_article_with_recommended_settings():
    text = text_of(8000)
    chunks = chunk_text(text, DEFAULT_MAX_TOKENS, DEFAULT_OVERLAP)
    assert_invariants(text, DEFAULT_MAX_TOKENS, DEFAULT_OVERLAP, chunks)
    assert chunks[0].split()[0] == "w0"
    assert chunks[-1].split()[-1] == "w7999"
    # stride 250: starts 0, 250, ..., 7750 -> 32 chunks
    assert len(chunks) == 32


def test_output_is_deterministic():
    text = text_of(1234)
    assert chunk_text(text, 300, 50) == chunk_text(text, 300, 50)


# --- Whitespace and Unicode ------------------------------------------------

def test_messy_whitespace_collapses_to_single_spaces():
    text = "  Step 1:\tOpen\n\nSettings   then\r\nclick  Save  "
    assert chunk_text(text, 10, 0) == ["Step 1: Open Settings then click Save"]


def test_accented_text_is_preserved():
    text = "Réinitialisez le mot de passe. Größe prüfen."
    assert chunk_text(text, 3, 1) == [
        "Réinitialisez le mot",
        "mot de passe.",
        "passe. Größe prüfen.",
    ]


def test_japanese_without_spaces_is_one_token():
    # Documents the limitation: a whole Japanese sentence counts as one
    # token here, though a real tokenizer would count many more.
    text = "パスワードをリセットするには設定を開いてください。"
    assert chunk_text(text, 5, 1) == [text]


# --- Invalid arguments -----------------------------------------------------

@pytest.mark.parametrize(
    "text, max_tokens, overlap",
    [
        ("a b", 0, 0),          # max_tokens too small
        ("a b", -1, 0),
        ("a b", 5, -1),         # negative overlap
        ("a b", 5, 5),          # overlap == max_tokens: stride 0
        ("a b", 5, 6),          # overlap > max_tokens
        ("a b", 5.0, 1),        # non-integer sizes
        ("a b", 5, 1.0),
        ("a b", "5", 1),
        ("a b", True, 0),       # bool is not a size
        ("a b", 5, False),
        (None, 5, 1),           # non-string text
        (b"a b", 5, 1),
        (["a", "b"], 5, 1),
    ],
)
def test_invalid_arguments_raise_value_error(text, max_tokens, overlap):
    with pytest.raises(ValueError):
        chunk_text(text, max_tokens, overlap)


def test_invalid_arguments_raise_even_for_empty_text():
    with pytest.raises(ValueError):
        chunk_text("", 5, 5)


# --- Property-based --------------------------------------------------------

@given(
    n=st.integers(min_value=0, max_value=400),
    max_tokens=st.integers(min_value=1, max_value=60),
    data=st.data(),
)
def test_invariants_hold_for_random_inputs(n, max_tokens, data):
    overlap = data.draw(st.integers(min_value=0, max_value=max_tokens - 1))
    text = text_of(n)
    chunks = chunk_text(text, max_tokens, overlap)
    assert_invariants(text, max_tokens, overlap, chunks)
    assert (chunks == []) == (n == 0)
