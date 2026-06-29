from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum


class ChunkPriority(Enum):
    USER_PINNED = "user_pinned"
    SAFETY_CRITICAL = "safety_critical"
    POLICY_SUGGESTED = "policy_suggested"
    RECENT = "recent"
    NORMAL = "normal"
    LOW = "low"


class ChunkType(Enum):
    MEDICAL = "medical"
    CODE = "code"
    FACT = "fact"
    PREFERENCE = "preference"
    CONTEXT = "context"
    CONVERSATION = "conversation"
    OTHER = "other"


@dataclass
class Chunk:
    id: str
    content: str
    tokens: int
    priority: ChunkPriority = ChunkPriority.NORMAL
    chunk_type: ChunkType = ChunkType.OTHER
    timestamp: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    is_pinned: bool = False
    is_temporary: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    summary: Optional[str] = None
    compressed_content: Optional[bytes] = None
    compression_ratio: Optional[float] = None

    importance_score: Optional[float] = None
    task_type: Optional[str] = None

    def __post_init__(self):
        if self.tokens <= 0:
            raise ValueError(f"Tokens must be positive, got {self.tokens}")
        if self.is_pinned:
            self.priority = ChunkPriority.USER_PINNED

    def update_access(self):
        self.last_accessed = datetime.now()
        self.access_count += 1

    def set_importance(self, score: float, task_type: str):
        if not 0 <= score <= 1:
            raise ValueError(f"Importance score must be in [0,1], got {score}")
        self.importance_score = score
        self.task_type = task_type
        self.priority = ChunkPriority.POLICY_SUGGESTED

    def get_priority_value(self) -> float:
        priority_map = {
            ChunkPriority.USER_PINNED: float("inf"),
            ChunkPriority.SAFETY_CRITICAL: 100.0,
            ChunkPriority.POLICY_SUGGESTED: (
                self.importance_score * 99 if self.importance_score else 50.0
            ),
            ChunkPriority.RECENT: max(
                0, 50 - (datetime.now() - self.timestamp).total_seconds() / 3600,
            ),
            ChunkPriority.NORMAL: 25.0,
            ChunkPriority.LOW: 10.0,
        }
        return priority_map.get(self.priority, 25.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "tokens": self.tokens,
            "priority": self.priority.value,
            "chunk_type": self.chunk_type.value,
            "timestamp": self.timestamp.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "is_pinned": self.is_pinned,
            "is_temporary": self.is_temporary,
            "metadata": self.metadata,
            "summary": self.summary,
            "importance_score": self.importance_score,
            "task_type": self.task_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        return cls(
            id=data["id"],
            content=data["content"],
            tokens=data["tokens"],
            priority=ChunkPriority(data["priority"]),
            chunk_type=ChunkType(data["chunk_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            access_count=data["access_count"],
            is_pinned=bool(data["is_pinned"]),
            is_temporary=bool(data["is_temporary"]),
            metadata=data.get("metadata", {}),
            summary=data.get("summary"),
            importance_score=data.get("importance_score"),
            task_type=data.get("task_type"),
        )

    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return (
            f"Chunk(id={self.id}, tokens={self.tokens}, "
            f"priority={self.priority.value}, content='{preview}')"
        )
