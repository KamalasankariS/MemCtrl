from dataclasses import dataclass, field
from datetime import datetime
from typing import Set, Dict, Any
from uuid import uuid4


@dataclass
class User:
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    pinned_chunk_ids: Set[str] = field(default_factory=set)
    forgotten_chunk_ids: Set[str] = field(default_factory=set)
    preferences: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)

    def pin_chunk(self, chunk_id: str):
        self.pinned_chunk_ids.add(chunk_id)
        self.last_active = datetime.now()

    def forget_chunk(self, chunk_id: str):
        self.forgotten_chunk_ids.add(chunk_id)
        self.pinned_chunk_ids.discard(chunk_id)
        self.last_active = datetime.now()

    def is_pinned(self, chunk_id: str) -> bool:
        return chunk_id in self.pinned_chunk_ids

    def is_forgotten(self, chunk_id: str) -> bool:
        return chunk_id in self.forgotten_chunk_ids

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "pinned_chunk_ids": list(self.pinned_chunk_ids),
            "forgotten_chunk_ids": list(self.forgotten_chunk_ids),
            "preferences": self.preferences,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "User":
        return cls(
            id=data["id"],
            name=data.get("name", ""),
            pinned_chunk_ids=set(data.get("pinned_chunk_ids", [])),
            forgotten_chunk_ids=set(data.get("forgotten_chunk_ids", [])),
            preferences=data.get("preferences", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_active=datetime.fromisoformat(data["last_active"]),
        )
