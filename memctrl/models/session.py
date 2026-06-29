from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import uuid4

from .chunk import Chunk


@dataclass
class Session:
    id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    task_type: Optional[str] = None
    chunks: List[Chunk] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_chunk(self, chunk: Chunk):
        self.chunks.append(chunk)
        self.last_active = datetime.now()

    def get_total_tokens(self) -> int:
        return sum(chunk.tokens for chunk in self.chunks)

    def get_recent_chunks(self, n: int = 10) -> List[Chunk]:
        return sorted(self.chunks, key=lambda c: c.timestamp, reverse=True)[:n]

    def get_pinned_chunks(self) -> List[Chunk]:
        return [chunk for chunk in self.chunks if chunk.is_pinned]

    def close(self):
        self.is_active = False
        self.chunks = [chunk for chunk in self.chunks if not chunk.is_temporary]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "task_type": self.task_type,
            "chunk_ids": [chunk.id for chunk in self.chunks],
            "started_at": self.started_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "is_active": self.is_active,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], chunks: List[Chunk] = None) -> "Session":
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            task_type=data.get("task_type"),
            chunks=chunks or [],
            started_at=datetime.fromisoformat(data["started_at"]),
            last_active=datetime.fromisoformat(data["last_active"]),
            is_active=data["is_active"],
            metadata=data.get("metadata", {}),
        )

    def __repr__(self) -> str:
        return (
            f"Session(id={self.id}, user={self.user_id}, "
            f"chunks={len(self.chunks)}, active={self.is_active})"
        )
