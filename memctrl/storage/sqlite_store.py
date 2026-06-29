import json
import sqlite3
from typing import List, Optional, Dict, Any
from pathlib import Path
from contextlib import contextmanager

from ..models import Chunk, Session, User
from ..config import get_config


class SQLiteStore:
    """SQLite-based persistent storage with full-text search."""

    def __init__(self, db_path: Optional[str] = None):
        config = get_config()
        self.db_path = db_path or config.sqlite_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    user_id TEXT,
                    content TEXT NOT NULL,
                    summary TEXT,
                    tokens INTEGER NOT NULL,
                    chunk_type TEXT,
                    priority TEXT,
                    is_pinned BOOLEAN DEFAULT 0,
                    is_temporary BOOLEAN DEFAULT 0,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    importance_score REAL,
                    task_type TEXT,
                    metadata TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    task_type TEXT,
                    started_at TEXT NOT NULL,
                    last_active TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    metadata TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    pinned_chunk_ids TEXT,
                    forgotten_chunk_ids TEXT,
                    preferences TEXT,
                    created_at TEXT NOT NULL,
                    last_active TEXT NOT NULL
                )
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_user ON chunks(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_pinned ON chunks(is_pinned)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)")

            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                USING fts5(id, content, content=chunks, content_rowid=rowid)
            """)

            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid, id, content)
                    VALUES (new.rowid, new.id, new.content);
                END
            """)

            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                    DELETE FROM chunks_fts WHERE rowid = old.rowid;
                END
            """)

            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                    UPDATE chunks_fts SET content = new.content WHERE rowid = old.rowid;
                END
            """)

    def _row_to_chunk(self, row: sqlite3.Row) -> Chunk:
        data = dict(row)
        data["timestamp"] = data["created_at"]
        data["metadata"] = json.loads(data["metadata"]) if data["metadata"] else {}
        return Chunk.from_dict(data)

    # -- Chunk operations --

    def store_chunk(self, chunk: Chunk) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO chunks
                (id, session_id, user_id, content, summary, tokens, chunk_type,
                 priority, is_pinned, is_temporary, created_at, last_accessed,
                 access_count, importance_score, task_type, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.id,
                    chunk.metadata.get("session_id"),
                    chunk.metadata.get("user_id"),
                    chunk.content,
                    chunk.summary,
                    chunk.tokens,
                    chunk.chunk_type.value,
                    chunk.priority.value,
                    chunk.is_pinned,
                    chunk.is_temporary,
                    chunk.timestamp.isoformat(),
                    chunk.last_accessed.isoformat(),
                    chunk.access_count,
                    chunk.importance_score,
                    chunk.task_type,
                    json.dumps(chunk.metadata),
                ),
            )
            return True

    def retrieve_chunk(self, chunk_id: str) -> Optional[Chunk]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_chunk(row)

    def delete_chunk(self, chunk_id: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
            return cursor.rowcount > 0

    def search_chunks(self, query: str, user_id: Optional[str] = None, limit: int = 10) -> List[Chunk]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if user_id:
                cursor.execute(
                    """
                    SELECT chunks.* FROM chunks
                    JOIN chunks_fts ON chunks.rowid = chunks_fts.rowid
                    WHERE chunks_fts MATCH ? AND chunks.user_id = ?
                    ORDER BY rank LIMIT ?
                    """,
                    (query, user_id, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT chunks.* FROM chunks
                    JOIN chunks_fts ON chunks.rowid = chunks_fts.rowid
                    WHERE chunks_fts MATCH ?
                    ORDER BY rank LIMIT ?
                    """,
                    (query, limit),
                )
            return [self._row_to_chunk(row) for row in cursor.fetchall()]

    def get_pinned_chunks(self, user_id: str) -> List[Chunk]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM chunks WHERE user_id = ? AND is_pinned = 1 ORDER BY created_at DESC",
                (user_id,),
            )
            return [self._row_to_chunk(row) for row in cursor.fetchall()]

    def get_chunks_by_session(self, session_id: str) -> List[Chunk]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM chunks WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            )
            return [self._row_to_chunk(row) for row in cursor.fetchall()]

    # -- Session operations --

    def store_session(self, session: Session) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO sessions
                (id, user_id, task_type, started_at, last_active, is_active, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.id,
                    session.user_id,
                    session.task_type,
                    session.started_at.isoformat(),
                    session.last_active.isoformat(),
                    session.is_active,
                    json.dumps(session.metadata),
                ),
            )
            return True

    def retrieve_session(self, session_id: str) -> Optional[Session]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            if not row:
                return None
            data = dict(row)
            data["metadata"] = json.loads(data["metadata"]) if data["metadata"] else {}
            chunks = self.get_chunks_by_session(session_id)
            return Session.from_dict(data, chunks)

    def get_user_sessions(self, user_id: str, active_only: bool = False) -> List[Session]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if active_only:
                cursor.execute(
                    "SELECT * FROM sessions WHERE user_id = ? AND is_active = 1 ORDER BY last_active DESC",
                    (user_id,),
                )
            else:
                cursor.execute(
                    "SELECT * FROM sessions WHERE user_id = ? ORDER BY last_active DESC",
                    (user_id,),
                )

            sessions = []
            for row in cursor.fetchall():
                data = dict(row)
                data["metadata"] = json.loads(data["metadata"]) if data["metadata"] else {}
                chunks = self.get_chunks_by_session(data["id"])
                sessions.append(Session.from_dict(data, chunks))
            return sessions

    # -- User operations --

    def store_user(self, user: User) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO users
                (id, name, pinned_chunk_ids, forgotten_chunk_ids, preferences,
                 created_at, last_active)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user.id,
                    user.name,
                    json.dumps(list(user.pinned_chunk_ids)),
                    json.dumps(list(user.forgotten_chunk_ids)),
                    json.dumps(user.preferences),
                    user.created_at.isoformat(),
                    user.last_active.isoformat(),
                ),
            )
            return True

    def retrieve_user(self, user_id: str) -> Optional[User]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            if not row:
                return None
            data = dict(row)
            data["pinned_chunk_ids"] = json.loads(data["pinned_chunk_ids"]) if data["pinned_chunk_ids"] else []
            data["forgotten_chunk_ids"] = json.loads(data["forgotten_chunk_ids"]) if data["forgotten_chunk_ids"] else []
            data["preferences"] = json.loads(data["preferences"]) if data["preferences"] else {}
            return User.from_dict(data)

    # -- Statistics --

    def get_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if user_id:
                cursor.execute("SELECT COUNT(*) FROM chunks WHERE user_id = ?", (user_id,))
                total_chunks = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM chunks WHERE user_id = ? AND is_pinned = 1", (user_id,))
                pinned_chunks = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM sessions WHERE user_id = ?", (user_id,))
                total_sessions = cursor.fetchone()[0]
            else:
                cursor.execute("SELECT COUNT(*) FROM chunks")
                total_chunks = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM chunks WHERE is_pinned = 1")
                pinned_chunks = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM sessions")
                total_sessions = cursor.fetchone()[0]

            return {
                "total_chunks": total_chunks,
                "pinned_chunks": pinned_chunks,
                "total_sessions": total_sessions,
            }
