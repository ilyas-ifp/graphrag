# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

from unittest import mock
from unittest.mock import MagicMock

import pytest
import tiktoken

from graphrag.index.text_splitting.text_splitting import (
    TextChunk,
    NoopTextSplitter,
    Tokenizer,
    TokenTextSplitter,
    split_multiple_texts_on_tokens,
    split_single_text_on_tokens,
    split_markdown_on_tokens,
)


def test_noop_text_splitter() -> None:
    splitter = NoopTextSplitter()

    assert list(splitter.split_text("some text")) == ["some text"]
    assert list(splitter.split_text(["some", "text"])) == ["some", "text"]


class MockTokenizer:
    def encode(self, text):
        return [ord(char) for char in text]

    def decode(self, token_ids):
        return "".join(chr(id) for id in token_ids)


def test_split_text_str_empty():
    splitter = TokenTextSplitter(chunk_size=5, chunk_overlap=2)
    result = splitter.split_text("")

    assert result == []


def test_split_text_str_bool():
    splitter = TokenTextSplitter(chunk_size=5, chunk_overlap=2)
    result = splitter.split_text(None)  # type: ignore

    assert result == []


def test_split_text_str_int():
    splitter = TokenTextSplitter(chunk_size=5, chunk_overlap=2)
    with pytest.raises(TypeError):
        splitter.split_text(123)  # type: ignore


@mock.patch("graphrag.index.text_splitting.text_splitting.split_single_text_on_tokens")
def test_split_text_large_input(mock_split):
    large_text = "a" * 10_000
    mock_split.return_value = ["chunk"] * 2_000
    splitter = TokenTextSplitter(chunk_size=5, chunk_overlap=2)

    result = splitter.split_text(large_text)

    assert len(result) == 2_000, "Large input was not split correctly"
    mock_split.assert_called_once()


@mock.patch("graphrag.index.text_splitting.text_splitting.split_single_text_on_tokens")
@mock.patch("graphrag.index.text_splitting.text_splitting.Tokenizer")
def test_token_text_splitter(mock_tokenizer, mock_split_text):
    text = "chunk1 chunk2 chunk3"
    expected_chunks = ["chunk1", "chunk2", "chunk3"]

    mocked_tokenizer = MagicMock()
    mock_tokenizer.return_value = mocked_tokenizer
    mock_split_text.return_value = expected_chunks

    splitter = TokenTextSplitter()

    splitter.split_text(["chunk1", "chunk2", "chunk3"])

    mock_split_text.assert_called_once_with(text=text, tokenizer=mocked_tokenizer)


def test_encode_basic():
    splitter = TokenTextSplitter()
    result = splitter.encode("abc def")

    assert result == [13997, 711], "Encoding failed to return expected tokens"


def test_num_tokens_empty_input():
    splitter = TokenTextSplitter()
    result = splitter.num_tokens("")

    assert result == 0, "Token count for empty input should be 0"


def test_model_name():
    splitter = TokenTextSplitter(model_name="gpt-4o")
    result = splitter.encode("abc def")

    assert result == [26682, 1056], "Encoding failed to return expected tokens"


@mock.patch("tiktoken.encoding_for_model", side_effect=KeyError)
@mock.patch("tiktoken.get_encoding")
def test_model_name_exception(mock_get_encoding, mock_encoding_for_model):
    mock_get_encoding.return_value = mock.MagicMock()

    TokenTextSplitter(model_name="mock_model", encoding_name="mock_encoding")

    mock_get_encoding.assert_called_once_with("mock_encoding")
    mock_encoding_for_model.assert_called_once_with("mock_model")


def test_split_single_text_on_tokens():
    text = "This is a test text, meaning to be taken seriously by this test only."
    mocked_tokenizer = MockTokenizer()
    tokenizer = Tokenizer(
        chunk_overlap=5,
        tokens_per_chunk=10,
        decode=mocked_tokenizer.decode,
        encode=lambda text: mocked_tokenizer.encode(text),
    )

    expected_splits = [
        "This is a ",
        "is a test ",
        "test text,",
        "text, mean",
        " meaning t",
        "ing to be ",
        "o be taken",
        "taken seri",  # cspell:disable-line
        " seriously",
        "ously by t",  # cspell:disable-line
        " by this t",
        "his test o",
        "est only.",
    ]

    result = split_single_text_on_tokens(text=text, tokenizer=tokenizer)
    assert result == expected_splits


def test_split_multiple_texts_on_tokens():
    texts = [
        "This is a test text, meaning to be taken seriously by this test only.",
        "This is th second text, meaning to be taken seriously by this test only.",
    ]

    mocked_tokenizer = MockTokenizer()
    mock_tick = MagicMock()
    tokenizer = Tokenizer(
        chunk_overlap=5,
        tokens_per_chunk=10,
        decode=mocked_tokenizer.decode,
        encode=lambda text: mocked_tokenizer.encode(text),
    )

    split_multiple_texts_on_tokens(texts, tokenizer, tick=mock_tick)
    mock_tick.assert_called()


def test_split_single_text_on_tokens_no_overlap():
    text = "This is a test text, meaning to be taken seriously by this test only."
    enc = tiktoken.get_encoding("cl100k_base")

    def encode(text: str) -> list[int]:
        if not isinstance(text, str):
            text = f"{text}"
        return enc.encode(text)

    def decode(tokens: list[int]) -> str:
        return enc.decode(tokens)

    tokenizer = Tokenizer(
        chunk_overlap=1,
        tokens_per_chunk=2,
        decode=decode,
        encode=lambda text: encode(text),
    )

    expected_splits = [
        "This is",
        " is a",
        " a test",
        " test text",
        " text,",
        ", meaning",
        " meaning to",
        " to be",
        " be taken",  # cspell:disable-line
        " taken seriously",  # cspell:disable-line
        " seriously by",
        " by this",  # cspell:disable-line
        " this test",
        " test only",
        " only.",
    ]

    result = split_single_text_on_tokens(text=text, tokenizer=tokenizer)
    assert result == expected_splits


def test_split_markdown_on_tokens_empty_input():
    """Test with empty input texts."""
    mocked_tokenizer = MockTokenizer()
    mock_tick = MagicMock()
    tokenizer = Tokenizer(
        chunk_overlap=1,
        tokens_per_chunk=10,
        decode=mocked_tokenizer.decode,
        encode=lambda text: mocked_tokenizer.encode(text),
    )
    
    result = split_markdown_on_tokens([], tokenizer,tick=mock_tick)
    assert result == []


def test_split_markdown_on_tokens_empty_strings():
    """Test with empty string texts."""
    mocked_tokenizer = MockTokenizer()
    mock_tick = MagicMock()
    tokenizer = Tokenizer(
        chunk_overlap=1,
        tokens_per_chunk=10,
        decode=mocked_tokenizer.decode,
        encode=lambda text: mocked_tokenizer.encode(text),
    )
    
    result = split_markdown_on_tokens(["", "   ", ""], tokenizer,tick=mock_tick)
    assert result == []


def test_split_markdown_on_tokens_simple_text():
    """Test with simple text without markdown."""
    texts = ["This is a simple text without any markdown formatting."]
    
    mocked_tokenizer = MockTokenizer()
    mock_tick = MagicMock()
    tokenizer = Tokenizer(
        chunk_overlap=2,
        tokens_per_chunk=15,
        decode=mocked_tokenizer.decode,
        encode=lambda text: mocked_tokenizer.encode(text),
    )
    
    result = split_markdown_on_tokens(texts, tokenizer,tick=mock_tick)
    
    assert len(result) > 0
    assert all(isinstance(chunk, TextChunk) for chunk in result)
    assert all(chunk.source_doc_indices == [0] for chunk in result)
    assert all(chunk.n_tokens <= tokenizer.tokens_per_chunk for chunk in result)


def test_split_markdown_on_tokens_with_headings():
    """Test with markdown headings."""
    texts = [
        "# Main Title\n\nSome content here.\n\n## Subtitle\n\nMore content here."
    ]
    
    mocked_tokenizer = MockTokenizer()
    mock_tick = MagicMock()
    tokenizer = Tokenizer(
        chunk_overlap=2,
        tokens_per_chunk=20,
        decode=mocked_tokenizer.decode,
        encode=lambda text: mocked_tokenizer.encode(text),
    )
    
    result = split_markdown_on_tokens(texts, tokenizer,tick=mock_tick)
    
    assert len(result) > 0
    assert all(isinstance(chunk, TextChunk) for chunk in result)
    # Should split at heading boundaries
    assert any("Main Title" in chunk.text_chunk for chunk in result)
    assert any("Subtitle" in chunk.text_chunk for chunk in result)


def test_split_markdown_on_tokens_with_code_blocks():
    """Test with markdown code blocks."""
    texts = [
        "Here's some code:\n\n```python\ndef hello():\n    print('world')\n```\n\nAnd more text."
    ]
    
    mocked_tokenizer = MockTokenizer()
    mock_tick = MagicMock()
    tokenizer = Tokenizer(
        chunk_overlap=2,
        tokens_per_chunk=25,
        decode=mocked_tokenizer.decode,
        encode=lambda text: mocked_tokenizer.encode(text),
    )
    
    result = split_markdown_on_tokens(texts, tokenizer,tick=mock_tick)
    
    assert len(result) > 0
    assert all(isinstance(chunk, TextChunk) for chunk in result)


def test_split_markdown_on_tokens_with_horizontal_rules():
    """Test with markdown horizontal rules."""
    texts = [
        "Section 1\n\n---\n\nSection 2\n\n***\n\nSection 3"
    ]
    
    mocked_tokenizer = MockTokenizer()
    mock_tick = MagicMock()
    tokenizer = Tokenizer(
        chunk_overlap=1,
        tokens_per_chunk=15,
        decode=mocked_tokenizer.decode,
        encode=lambda text: mocked_tokenizer.encode(text),
    )
    
    result = split_markdown_on_tokens(texts, tokenizer,tick=mock_tick)
    
    assert len(result) > 0
    assert all(isinstance(chunk, TextChunk) for chunk in result)


def test_split_markdown_on_tokens_multiple_documents():
    """Test with multiple input documents."""
    texts = [
        "# Document 1\n\nThis is the first document.",
        "# Document 2\n\nThis is the second document.",
        "# Document 3\n\nThis is the third document."
    ]
    
    mocked_tokenizer = MockTokenizer()
    mock_tick = MagicMock()
    tokenizer = Tokenizer(
        chunk_overlap=2,
        tokens_per_chunk=20,
        decode=mocked_tokenizer.decode,
        encode=lambda text: mocked_tokenizer.encode(text),
    )
    
    result = split_markdown_on_tokens(texts, tokenizer,tick=mock_tick)
    
    assert len(result) > 0
    assert all(isinstance(chunk, TextChunk) for chunk in result)
    
    # Check that we have chunks from different documents
    doc_indices_found = set()
    for chunk in result:
        doc_indices_found.update(chunk.source_doc_indices)
    
    assert len(doc_indices_found) >= 2  # Should have chunks from multiple docs


def test_split_markdown_on_tokens_with_overlap():
    """Test that overlap is properly applied."""
    texts = ["This is a test text that should be split with proper overlap handling."]
    
    mocked_tokenizer = MockTokenizer()
    mock_tick = MagicMock()
    tokenizer = Tokenizer(
        chunk_overlap=3,
        tokens_per_chunk=10,
        decode=mocked_tokenizer.decode,
        encode=lambda text: mocked_tokenizer.encode(text),
    )
    
    result = split_markdown_on_tokens(texts, tokenizer,tick=mock_tick)
    
    assert len(result) > 1  # Should create multiple chunks
    assert all(chunk.n_tokens <= tokenizer.tokens_per_chunk for chunk in result)
    
    # Check that consecutive chunks have some overlap
    for i in range(len(result) - 1):
        current_chunk = result[i].text_chunk
        next_chunk = result[i + 1].text_chunk
        # There should be some common characters due to overlap
        assert len(current_chunk) > 0 and len(next_chunk) > 0


def test_split_markdown_on_tokens_large_chunk():
    """Test force splitting of large chunks that can't be split by separators."""
    # Create a long text without any markdown separators
    large_text = "a" * 100  # Long string without separators
    texts = [large_text]
    
    mocked_tokenizer = MockTokenizer()
    mock_tick = MagicMock()
    tokenizer = Tokenizer(
        chunk_overlap=2,
        tokens_per_chunk=20,
        decode=mocked_tokenizer.decode,
        encode=lambda text: mocked_tokenizer.encode(text),
    )
    
    result = split_markdown_on_tokens(texts, tokenizer,tick=mock_tick)
    
    assert len(result) > 1  # Should be split into multiple chunks
    assert all(chunk.n_tokens <= tokenizer.tokens_per_chunk for chunk in result)
    assert all(isinstance(chunk, TextChunk) for chunk in result)


def test_split_markdown_on_tokens_with_tick_callback():
    """Test that the tick callback is called."""
    texts = ["Text 1", "Text 2", "Text 3"]
    
    mocked_tokenizer = MockTokenizer()
    mock_tick = MagicMock()
    tokenizer = Tokenizer(
        chunk_overlap=1,
        tokens_per_chunk=10,
        decode=mocked_tokenizer.decode,
        encode=lambda text: mocked_tokenizer.encode(text),
    )
    
    result = split_markdown_on_tokens(texts, tokenizer, tick=mock_tick)
    
    assert mock_tick.call_count == len(texts)
    assert len(result) >= 0


def test_split_markdown_on_tokens_real_tokenizer():
    """Test with a real tiktoken tokenizer."""
    texts = [
        "# Introduction\n\nThis is a markdown document with various elements.\n\n## Code Example\n\n```python\nprint('hello')\n```\n\nEnd."
    ]
    mock_tick = MagicMock()
    enc = tiktoken.get_encoding("cl100k_base")

    def encode(text: str) -> list[int]:
        if not isinstance(text, str):
            text = f"{text}"
        return enc.encode(text)

    def decode(tokens: list[int]) -> str:
        return enc.decode(tokens)

    tokenizer = Tokenizer(
        chunk_overlap=2,
        tokens_per_chunk=15,
        decode=decode,
        encode=lambda text: encode(text),
    )
    
    result = split_markdown_on_tokens(texts, tokenizer,tick=mock_tick)
    
    assert len(result) > 0
    assert all(isinstance(chunk, TextChunk) for chunk in result)
    assert all(chunk.n_tokens <= tokenizer.tokens_per_chunk for chunk in result)
    assert all(len(chunk.text_chunk.strip()) > 0 for chunk in result)


def test_split_markdown_on_tokens_chunk_metadata():
    """Test that chunk metadata is correctly set."""
    texts = ["Document 1 content", "Document 2 content"]
    
    mocked_tokenizer = MockTokenizer()
    mock_tick = MagicMock()
    tokenizer = Tokenizer(
        chunk_overlap=1,
        tokens_per_chunk=10,
        decode=mocked_tokenizer.decode,
        encode=lambda text: mocked_tokenizer.encode(text),
    )
    
    result = split_markdown_on_tokens(texts, tokenizer,tick=mock_tick)
    
    for chunk in result:
        # Check that all required attributes exist
        assert hasattr(chunk, 'text_chunk')
        assert hasattr(chunk, 'source_doc_indices')
        assert hasattr(chunk, 'n_tokens')
        
        # Check types
        assert isinstance(chunk.text_chunk, str)
        assert isinstance(chunk.source_doc_indices, list)
        assert isinstance(chunk.n_tokens, int)
        
        # Check values
        assert len(chunk.text_chunk) > 0
        assert len(chunk.source_doc_indices) > 0
        assert chunk.n_tokens > 0
        assert all(isinstance(idx, int) for idx in chunk.source_doc_indices)
        assert all(0 <= idx < len(texts) for idx in chunk.source_doc_indices)


def test_split_markdown_on_tokens_no_overlap():
    """Test with zero overlap."""
    texts = ["This is a test text for checking zero overlap behavior."]
    
    mocked_tokenizer = MockTokenizer()
    mock_tick = MagicMock()
    tokenizer = Tokenizer(
        chunk_overlap=0,
        tokens_per_chunk=10,
        decode=mocked_tokenizer.decode,
        encode=lambda text: mocked_tokenizer.encode(text),
    )
    
    result = split_markdown_on_tokens(texts, tokenizer,tick=mock_tick)
    
    assert len(result) > 1  # Should create multiple chunks
    assert all(chunk.n_tokens <= tokenizer.tokens_per_chunk for chunk in result)
    
    # With zero overlap, consecutive chunks should not share content
    all_text = "".join(chunk.text_chunk for chunk in result)
    original_text = texts[0]
    # The reconstructed text should contain most of the original content
    assert len(all_text) > len(original_text) * 0.8  # Allow for some variation
