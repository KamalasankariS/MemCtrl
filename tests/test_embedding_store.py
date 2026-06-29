"""Tests for embedding store."""

import pytest
from memctrl.storage.embedding_store import EmbeddingStore


@pytest.fixture
def store():
    return EmbeddingStore()


def test_add_and_length(store):
    store.add("c1", "Hello world")
    assert len(store) == 1

    store.add("c2", "Goodbye world")
    assert len(store) == 2


def test_add_duplicate_updates(store):
    store.add("c1", "First version")
    store.add("c1", "Second version")
    assert len(store) == 1


def test_remove(store):
    store.add("c1", "Hello")
    store.remove("c1")
    assert len(store) == 0


def test_remove_nonexistent(store):
    store.remove("nonexistent")
    assert len(store) == 0


def test_search_returns_results(store):
    store.add("c1", "Python programming language")
    store.add("c2", "JavaScript web development")
    store.add("c3", "Python data science and machine learning")

    results = store.search("Python coding", top_k=2)
    assert len(results) <= 2
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    chunk_ids = [r[0] for r in results]
    assert "c1" in chunk_ids or "c3" in chunk_ids


def test_search_empty_store(store):
    results = store.search("anything")
    assert results == []


def test_search_scores_are_floats(store):
    store.add("c1", "Test content")
    results = store.search("Test", top_k=1)
    assert len(results) == 1
    _, score = results[0]
    assert isinstance(score, float)


def test_clear(store):
    store.add("c1", "Hello")
    store.add("c2", "World")
    store.clear()
    assert len(store) == 0
    assert store.search("Hello") == []


def test_search_relevance_ordering(store):
    store.add("c1", "The cat sat on the mat")
    store.add("c2", "Python programming tutorial for beginners")
    store.add("c3", "Advanced Python machine learning techniques")

    results = store.search("Python programming")
    assert len(results) >= 2

    chunk_ids = [r[0] for r in results]
    python_indices = [i for i, cid in enumerate(chunk_ids) if cid in ("c2", "c3")]
    cat_index = [i for i, cid in enumerate(chunk_ids) if cid == "c1"]

    if python_indices and cat_index:
        assert min(python_indices) < min(cat_index)
