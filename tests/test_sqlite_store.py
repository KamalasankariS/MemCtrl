import pytest
import tempfile
from pathlib import Path

from memctrl.storage.sqlite_store import SQLiteStore
from memctrl.models import Chunk, Session, User, ChunkType


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    store = SQLiteStore(db_path)
    yield store
    Path(db_path).unlink(missing_ok=True)


def test_store_and_retrieve_chunk(temp_db):
    chunk = Chunk(id="test_001", content="Test content", tokens=2, chunk_type=ChunkType.FACT, is_pinned=True)
    chunk.metadata["user_id"] = "kamala"

    assert temp_db.store_chunk(chunk)

    retrieved = temp_db.retrieve_chunk("test_001")
    assert retrieved is not None
    assert retrieved.id == "test_001"
    assert retrieved.content == "Test content"
    assert retrieved.is_pinned is True


def test_search_chunks(temp_db):
    chunks = [
        Chunk(id="c1", content="Python programming", tokens=2),
        Chunk(id="c2", content="JavaScript coding", tokens=2),
        Chunk(id="c3", content="Python tutorial", tokens=2),
    ]
    for chunk in chunks:
        chunk.metadata["user_id"] = "kamala"
        temp_db.store_chunk(chunk)

    results = temp_db.search_chunks("Python")
    assert len(results) == 2
    assert all("Python" in r.content for r in results)


def test_pinned_chunks(temp_db):
    chunks = [
        Chunk(id="c1", content="Pinned", tokens=1, is_pinned=True),
        Chunk(id="c2", content="Not pinned", tokens=2, is_pinned=False),
        Chunk(id="c3", content="Also pinned", tokens=2, is_pinned=True),
    ]
    for chunk in chunks:
        chunk.metadata["user_id"] = "kamala"
        temp_db.store_chunk(chunk)

    pinned = temp_db.get_pinned_chunks("kamala")
    assert len(pinned) == 2
    assert all(c.is_pinned for c in pinned)


def test_session_storage(temp_db):
    session = Session(id="session_001", user_id="kamala")
    assert temp_db.store_session(session)

    retrieved = temp_db.retrieve_session("session_001")
    assert retrieved is not None
    assert retrieved.id == "session_001"
    assert retrieved.user_id == "kamala"


def test_user_storage(temp_db):
    user = User(id="kamala", name="Kamala")
    user.pin_chunk("c1")
    user.pin_chunk("c2")

    assert temp_db.store_user(user)

    retrieved = temp_db.retrieve_user("kamala")
    assert retrieved is not None
    assert retrieved.id == "kamala"
    assert retrieved.name == "Kamala"
    assert retrieved.is_pinned("c1")
    assert retrieved.is_pinned("c2")
