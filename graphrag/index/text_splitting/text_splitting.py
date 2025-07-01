# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing the 'Tokenizer', 'TextSplitter', 'NoopTextSplitter' and 'TokenTextSplitter' models."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass
from typing import Any, Literal, cast

import re
import pandas as pd
import tiktoken

import graphrag.config.defaults as defs
from graphrag.index.operations.chunk_text.typing import TextChunk
from graphrag.logger.progress import ProgressTicker

EncodedText = list[int]
DecodeFn = Callable[[EncodedText], str]
EncodeFn = Callable[[str], EncodedText]
LengthFn = Callable[[str], int]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Tokenizer:
    """Tokenizer data class."""

    chunk_overlap: int
    """Overlap in tokens between chunks"""
    tokens_per_chunk: int
    """Maximum number of tokens per chunk"""
    decode: DecodeFn
    """ Function to decode a list of token ids to a string"""
    encode: EncodeFn
    """ Function to encode a string to a list of token ids"""


class TextSplitter(ABC):
    """Text splitter class definition."""

    _chunk_size: int
    _chunk_overlap: int
    _length_function: LengthFn
    _keep_separator: bool
    _add_start_index: bool
    _strip_whitespace: bool

    def __init__(
        self,
        # based on text-ada-002-embedding max input buffer length
        # https://platform.openai.com/docs/guides/embeddings/second-generation-models
        chunk_size: int = 8191,
        chunk_overlap: int = 100,
        length_function: LengthFn = len,
        keep_separator: bool = False,
        add_start_index: bool = False,
        strip_whitespace: bool = True,
    ):
        """Init method definition."""
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._keep_separator = keep_separator
        self._add_start_index = add_start_index
        self._strip_whitespace = strip_whitespace

    @abstractmethod
    def split_text(self, text: str | list[str]) -> Iterable[str]:
        """Split text method definition."""


class NoopTextSplitter(TextSplitter):
    """Noop text splitter class definition."""

    def split_text(self, text: str | list[str]) -> Iterable[str]:
        """Split text method definition."""
        return [text] if isinstance(text, str) else text


class TokenTextSplitter(TextSplitter):
    """Token text splitter class definition."""

    _allowed_special: Literal["all"] | set[str]
    _disallowed_special: Literal["all"] | Collection[str]

    def __init__(
        self,
        encoding_name: str = defs.ENCODING_MODEL,
        model_name: str | None = None,
        allowed_special: Literal["all"] | set[str] | None = None,
        disallowed_special: Literal["all"] | Collection[str] = "all",
        **kwargs: Any,
    ):
        """Init method definition."""
        super().__init__(**kwargs)
        if model_name is not None:
            try:
                enc = tiktoken.encoding_for_model(model_name)
            except KeyError:
                log.exception("Model %s not found, using %s", model_name, encoding_name)
                enc = tiktoken.get_encoding(encoding_name)
        else:
            enc = tiktoken.get_encoding(encoding_name)
        self._tokenizer = enc
        self._allowed_special = allowed_special or set()
        self._disallowed_special = disallowed_special

    def encode(self, text: str) -> list[int]:
        """Encode the given text into an int-vector."""
        return self._tokenizer.encode(
            text,
            allowed_special=self._allowed_special,
            disallowed_special=self._disallowed_special,
        )

    def num_tokens(self, text: str) -> int:
        """Return the number of tokens in a string."""
        return len(self.encode(text))

    def split_text(self, text: str | list[str]) -> list[str]:
        """Split text method."""
        if isinstance(text, list):
            text = " ".join(text)
        elif cast("bool", pd.isna(text)) or text == "":
            return []
        if not isinstance(text, str):
            msg = f"Attempting to split a non-string value, actual is {type(text)}"
            raise TypeError(msg)

        tokenizer = Tokenizer(
            chunk_overlap=self._chunk_overlap,
            tokens_per_chunk=self._chunk_size,
            decode=self._tokenizer.decode,
            encode=lambda text: self.encode(text),
        )

        return split_single_text_on_tokens(text=text, tokenizer=tokenizer)


def split_single_text_on_tokens(text: str, tokenizer: Tokenizer) -> list[str]:
    """Split a single text and return chunks using the tokenizer."""
    result = []
    input_ids = tokenizer.encode(text)

    start_idx = 0
    cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
    chunk_ids = input_ids[start_idx:cur_idx]

    while start_idx < len(input_ids):
        chunk_text = tokenizer.decode(list(chunk_ids))
        result.append(chunk_text)  # Append chunked text as string
        if cur_idx == len(input_ids):
            break
        start_idx += tokenizer.tokens_per_chunk - tokenizer.chunk_overlap
        cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]

    return result


# Adapted from - https://github.com/langchain-ai/langchain/blob/77b359edf5df0d37ef0d539f678cf64f5557cb54/libs/langchain/langchain/text_splitter.py#L471
# So we could have better control over the chunking process
def split_multiple_texts_on_tokens(
    texts: list[str], tokenizer: Tokenizer, tick: ProgressTicker
) -> list[TextChunk]:
    """Split multiple texts and return chunks with metadata using the tokenizer."""
    print("\n\nUsing MD text splitting :) \n\n")
    result = []
    mapped_ids = []

    for source_doc_idx, text in enumerate(texts):
        encoded = tokenizer.encode(text)
        if tick:
            tick(1)  # Track progress if tick callback is provided
        mapped_ids.append((source_doc_idx, encoded))

    input_ids = [
        (source_doc_idx, id) for source_doc_idx, ids in mapped_ids for id in ids
    ]

    start_idx = 0
    cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
    chunk_ids = input_ids[start_idx:cur_idx]

    while start_idx < len(input_ids):
        chunk_text = tokenizer.decode([id for _, id in chunk_ids])
        doc_indices = list({doc_idx for doc_idx, _ in chunk_ids})
        result.append(TextChunk(chunk_text, doc_indices, len(chunk_ids)))
        if cur_idx == len(input_ids):
            break
        start_idx += tokenizer.tokens_per_chunk - tokenizer.chunk_overlap
        cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]

    return result

def split_markdown_on_tokens(
    texts: list[str], 
    tokenizer: Tokenizer, 
    tick: ProgressTicker,
) -> list[TextChunk]:
    """Split markdown texts using predefined regex patterns and return token-based chunks."""
    
    # Predefined markdown separators (regex patterns)
    MARKDOWN_SEPARATORS = [
        r"\n#{1,6} ",      # Markdown headings
        r"```\n",          # End of code block
        r"\n\*\*\*+\n",    # Horizontal lines with *
        r"\n---+\n",       # Horizontal lines with -
        r"\n___+\n",       # Horizontal lines with _
        "\n\n",            # Double newlines
        "\n",              # Single newlines
        " ",               # Spaces
        "",                # Character level
    ]
    
    def _split_with_separators(text: str, separators: list[str]) -> list[str]:
        """Recursively split text using separators."""
        if not text.strip():
            return []
        
        # Find the first separator that works
        separator = separators[-1]  # Default to last separator
        remaining_separators = []
        
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            
            # Use regex for non-empty separators
            if re.search(sep if len(sep) > 3 else re.escape(sep), text):
                separator = sep
                remaining_separators = separators[i + 1:]
                break
        
        # Split the text
        if separator == "":
            splits = list(text)
        else:
            pattern = separator if len(separator) > 3 else re.escape(separator)
            splits = [s for s in re.split(pattern, text) if s.strip()]
        
        # Process splits
        result = []
        current_group = []
        
        for split in splits:
            token_count = len(tokenizer.encode(split))
            
            if token_count <= tokenizer.tokens_per_chunk:
                current_group.append(split)
            else:
                # Process accumulated small splits
                if current_group:
                    merged = _merge_small_splits(current_group, separator)
                    result.extend(merged)
                    current_group = []
                
                # Handle large split
                if remaining_separators:
                    result.extend(_split_with_separators(split, remaining_separators))
                else:
                    # Force token-based splitting
                    result.extend(_force_split_by_tokens(split))
        
        # Process remaining small splits
        if current_group:
            merged = _merge_small_splits(current_group, separator)
            result.extend(merged)
        
        return result
    
    def _merge_small_splits(splits: list[str], separator: str) -> list[str]:
        """Merge small splits respecting token limits."""
        if not splits:
            return []
        
        result = []
        current_chunk = ""
        
        for split in splits:
            # Try adding this split to current chunk
            test_chunk = current_chunk + (separator if current_chunk else "") + split
            test_tokens = len(tokenizer.encode(test_chunk))
            
            if test_tokens <= tokenizer.tokens_per_chunk:
                current_chunk = test_chunk
            else:
                # Current chunk is full, save it and start new one
                if current_chunk:
                    result.append(current_chunk)
                current_chunk = split
        
        if current_chunk:
            result.append(current_chunk)
        
        return result
    
    def _force_split_by_tokens(text: str) -> list[str]:
        """Force split by tokens when no separators work."""
        if not text.strip():
            return []
        
        encoded = tokenizer.encode(text)
        chunks = []
        
        start_idx = 0
        while start_idx < len(encoded):
            end_idx = min(start_idx + tokenizer.tokens_per_chunk, len(encoded))
            chunk_tokens = encoded[start_idx:end_idx]
            chunk_text = tokenizer.decode(chunk_tokens)
            
            if chunk_text.strip():
                chunks.append(chunk_text)
            
            if end_idx == len(encoded):
                break
            
            start_idx = max(start_idx + 1, end_idx - tokenizer.chunk_overlap)
        
        return chunks
    
    # Main processing
    result = []
    all_text_chunks = []
    
    # First, split all texts using markdown separators
    for source_doc_idx, text in enumerate(texts):
        if tick:
            tick(1)
        
        if not text.strip():
            continue
            
        # Split using markdown separators
        text_chunks = _split_with_separators(text, MARKDOWN_SEPARATORS)
        
        # Tag each chunk with its source document
        for chunk in text_chunks:
            if chunk.strip():
                all_text_chunks.append((source_doc_idx, chunk))
    
    # Now create token-based chunks with metadata (similar to split_multiple_texts_on_tokens)
    if not all_text_chunks:
        return []
    
    # Encode all chunks and create input_ids with source tracking
    input_ids = []
    for source_doc_idx, chunk in all_text_chunks:
        encoded = tokenizer.encode(chunk)
        for token_id in encoded:
            input_ids.append((source_doc_idx, token_id))
    
    # Create final chunks with overlap
    start_idx = 0
    while start_idx < len(input_ids):
        end_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
        chunk_ids = input_ids[start_idx:end_idx]
        
        chunk_text = tokenizer.decode([token_id for _, token_id in chunk_ids])
        doc_indices = list({doc_idx for doc_idx, _ in chunk_ids})
        
        result.append(TextChunk(
            text_chunk=chunk_text,
            source_doc_indices=doc_indices,
            n_tokens=len(chunk_ids)
        ))
        
        if end_idx == len(input_ids):
            break
        
        start_idx += tokenizer.tokens_per_chunk - tokenizer.chunk_overlap
    
    return result