"""Tests for real tokenizer."""

import pytest
from memctrl.tokenizer import count_tokens, get_tokenizer


def test_count_tokens_basic():
    tokens = count_tokens("Hello world")
    assert isinstance(tokens, int)
    assert tokens > 0


def test_count_tokens_empty():
    tokens = count_tokens("")
    assert tokens == 0


def test_count_tokens_longer_text():
    short = count_tokens("Hello")
    long = count_tokens("Hello world, this is a much longer sentence with many more words")
    assert long > short


def test_get_tokenizer_singleton():
    t1 = get_tokenizer()
    t2 = get_tokenizer()
    assert t1 is t2


def test_get_tokenizer_different_model():
    t1 = get_tokenizer("distilbert-base-uncased")
    t2 = get_tokenizer("bert-base-uncased")
    assert t1 is not t2
