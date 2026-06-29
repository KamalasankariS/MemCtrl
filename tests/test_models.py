import pytest
from memctrl.models import Chunk, ChunkPriority, ChunkType, Session, User


def test_chunk_creation():
    chunk = Chunk(
        id="chunk_001",
        content="User is allergic to penicillin",
        tokens=8,
        chunk_type=ChunkType.MEDICAL,
    )
    assert chunk.id == "chunk_001"
    assert chunk.tokens == 8
    assert chunk.chunk_type == ChunkType.MEDICAL
    assert chunk.priority == ChunkPriority.NORMAL
    assert not chunk.is_pinned


def test_chunk_pinning():
    chunk = Chunk(id="chunk_002", content="Important fact", tokens=3, is_pinned=True)
    assert chunk.is_pinned
    assert chunk.priority == ChunkPriority.USER_PINNED
    assert chunk.get_priority_value() == float("inf")


def test_chunk_importance():
    chunk = Chunk(id="chunk_003", content="Test", tokens=1)
    chunk.set_importance(0.85, "medical")
    assert chunk.importance_score == 0.85
    assert chunk.task_type == "medical"
    assert chunk.priority == ChunkPriority.POLICY_SUGGESTED


def test_chunk_serialization():
    chunk = Chunk(id="chunk_004", content="Test content", tokens=3, is_pinned=True)

    data = chunk.to_dict()
    assert data["id"] == "chunk_004"
    assert data["is_pinned"] is True

    chunk2 = Chunk.from_dict(data)
    assert chunk2.id == chunk.id
    assert chunk2.content == chunk.content
    assert chunk2.is_pinned == chunk.is_pinned


def test_session_management():
    session = Session(user_id="test_user")
    chunk1 = Chunk(id="c1", content="Hello", tokens=1)
    chunk2 = Chunk(id="c2", content="World", tokens=1)

    session.add_chunk(chunk1)
    session.add_chunk(chunk2)

    assert len(session.chunks) == 2
    assert session.get_total_tokens() == 2
    assert session.is_active


def test_session_temporary_chunks():
    session = Session(user_id="test_user")
    permanent = Chunk(id="p1", content="Keep", tokens=1, is_temporary=False)
    temporary = Chunk(id="t1", content="Remove", tokens=1, is_temporary=True)

    session.add_chunk(permanent)
    session.add_chunk(temporary)
    assert len(session.chunks) == 2

    session.close()
    assert len(session.chunks) == 1
    assert session.chunks[0].id == "p1"
    assert not session.is_active


def test_user_pinning():
    user = User(id="user_001")
    user.pin_chunk("chunk_001")
    assert user.is_pinned("chunk_001")
    assert not user.is_forgotten("chunk_001")

    user.forget_chunk("chunk_001")
    assert not user.is_pinned("chunk_001")
    assert user.is_forgotten("chunk_001")


def test_user_serialization():
    user = User(id="user_002", name="Kamala")
    user.pin_chunk("c1")
    user.pin_chunk("c2")

    data = user.to_dict()
    assert data["id"] == "user_002"
    assert data["name"] == "Kamala"
    assert len(data["pinned_chunk_ids"]) == 2

    user2 = User.from_dict(data)
    assert user2.id == user.id
    assert user2.is_pinned("c1")
    assert user2.is_pinned("c2")
