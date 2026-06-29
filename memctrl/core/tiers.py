from typing import List, Optional, Dict
from collections import OrderedDict

from ..models import Chunk, ChunkPriority
from ..config import get_config
from ..storage.sqlite_store import SQLiteStore
from ..storage.embedding_store import EmbeddingStore


class Tier0_GPU:
    """Active GPU memory with LRU eviction."""

    def __init__(self, max_tokens: Optional[int] = None):
        config = get_config()
        self.max_tokens = max_tokens or config.get_tier0_tokens()
        self.current_tokens = 0
        self.storage: OrderedDict[str, Chunk] = OrderedDict()

    def add(self, chunk: Chunk, force: bool = False) -> bool:
        if chunk.id in self.storage:
            self.remove(chunk.id)

        if not force and self.current_tokens + chunk.tokens > self.max_tokens:
            return False

        while self.current_tokens + chunk.tokens > self.max_tokens:
            if not self._evict_lru():
                return False

        self.storage[chunk.id] = chunk
        self.current_tokens += chunk.tokens
        chunk.update_access()
        return True

    def get(self, chunk_id: str) -> Optional[Chunk]:
        chunk = self.storage.get(chunk_id)
        if chunk:
            chunk.update_access()
            self.storage.move_to_end(chunk_id)
        return chunk

    def remove(self, chunk_id: str) -> Optional[Chunk]:
        chunk = self.storage.pop(chunk_id, None)
        if chunk:
            self.current_tokens -= chunk.tokens
        return chunk

    def get_all(self) -> List[Chunk]:
        return list(self.storage.values())

    def get_pinned(self) -> List[Chunk]:
        return [c for c in self.storage.values() if c.is_pinned]

    def get_recent(self, n: int = 10) -> List[Chunk]:
        chunks = sorted(self.storage.values(), key=lambda c: c.timestamp, reverse=True)
        return chunks[:n]

    def is_full(self) -> bool:
        config = get_config()
        return self.current_tokens >= self.max_tokens * config.eviction_threshold

    def get_usage(self) -> Dict[str, float]:
        return {
            "current_tokens": self.current_tokens,
            "max_tokens": self.max_tokens,
            "utilization": self.current_tokens / self.max_tokens if self.max_tokens > 0 else 0,
            "num_chunks": len(self.storage),
        }

    def _evict_lru(self) -> bool:
        for chunk_id, chunk in list(self.storage.items()):
            if chunk.priority == ChunkPriority.USER_PINNED:
                continue
            self.remove(chunk_id)
            return True
        return False

    def clear(self):
        self.storage.clear()
        self.current_tokens = 0


class Tier1_RAM:
    """Compressed RAM storage with summarization."""

    def __init__(self, max_tokens: Optional[int] = None):
        config = get_config()
        self.max_tokens = max_tokens or config.get_tier1_tokens()
        self.current_tokens = 0
        self.storage: Dict[str, Chunk] = {}

    def add(self, chunk: Chunk, compressed: bool = False) -> bool:
        if not compressed and not chunk.summary:
            self._compress(chunk)

        compressed_tokens = self._get_compressed_tokens(chunk)

        if self.current_tokens + compressed_tokens > self.max_tokens:
            return False

        self.storage[chunk.id] = chunk
        self.current_tokens += compressed_tokens
        return True

    def get(self, chunk_id: str) -> Optional[Chunk]:
        return self.storage.get(chunk_id)

    def remove(self, chunk_id: str) -> Optional[Chunk]:
        chunk = self.storage.pop(chunk_id, None)
        if chunk:
            self.current_tokens -= self._get_compressed_tokens(chunk)
        return chunk

    def get_all(self) -> List[Chunk]:
        return list(self.storage.values())

    def decompress(self, chunk: Chunk) -> Chunk:
        # TODO: implement LLM-based decompression
        if chunk.summary and not chunk.content:
            chunk.content = chunk.summary
        return chunk

    def get_usage(self) -> Dict[str, float]:
        return {
            "current_tokens": self.current_tokens,
            "max_tokens": self.max_tokens,
            "utilization": self.current_tokens / self.max_tokens if self.max_tokens > 0 else 0,
            "num_chunks": len(self.storage),
        }

    def _compress(self, chunk: Chunk):
        config = get_config()
        if not chunk.summary:
            max_summary_tokens = int(chunk.tokens / config.compression_ratio)
            words = chunk.content.split()
            summary_words = words[: max_summary_tokens * 2]
            chunk.summary = " ".join(summary_words) + "..."
            chunk.compression_ratio = config.compression_ratio

    def _get_compressed_tokens(self, chunk: Chunk) -> int:
        if chunk.summary:
            config = get_config()
            return int(chunk.tokens / config.compression_ratio)
        return chunk.tokens

    def clear(self):
        self.storage.clear()
        self.current_tokens = 0


class Tier2_Disk:
    """Persistent disk storage backed by SQLite with semantic search."""

    def __init__(self, db_path: Optional[str] = None, embedding_model: Optional[str] = None):
        config = get_config()
        self.store = SQLiteStore(db_path)
        self.embeddings = EmbeddingStore(embedding_model or config.embedding_model)

    def add(self, chunk: Chunk) -> bool:
        stored = self.store.store_chunk(chunk)
        if stored:
            self.embeddings.add(chunk.id, chunk.content)
        return stored

    def get(self, chunk_id: str) -> Optional[Chunk]:
        return self.store.retrieve_chunk(chunk_id)

    def remove(self, chunk_id: str) -> bool:
        self.embeddings.remove(chunk_id)
        return self.store.delete_chunk(chunk_id)

    def search(self, query: str, user_id: Optional[str] = None, limit: int = 10) -> List[Chunk]:
        """Semantic search using embeddings, falling back to FTS."""
        if len(self.embeddings) > 0:
            results = self.embeddings.search(query, top_k=limit * 2)
            chunks = []
            for chunk_id, score in results:
                chunk = self.store.retrieve_chunk(chunk_id)
                if chunk and (user_id is None or chunk.metadata.get("user_id") == user_id):
                    chunks.append(chunk)
                if len(chunks) >= limit:
                    break
            if chunks:
                return chunks

        return self.store.search_chunks(query, user_id, limit)

    def get_pinned(self, user_id: str) -> List[Chunk]:
        return self.store.get_pinned_chunks(user_id)

    def get_by_session(self, session_id: str) -> List[Chunk]:
        return self.store.get_chunks_by_session(session_id)

    def get_stats(self, user_id: Optional[str] = None) -> Dict:
        stats = self.store.get_stats(user_id)
        stats["embedding_count"] = len(self.embeddings)
        return stats


class TierManager:
    """Manages promotion and demotion across all three memory tiers."""

    def __init__(
        self,
        tier0: Optional[Tier0_GPU] = None,
        tier1: Optional[Tier1_RAM] = None,
        tier2: Optional[Tier2_Disk] = None,
    ):
        self.tier0 = tier0 or Tier0_GPU()
        self.tier1 = tier1 or Tier1_RAM()
        self.tier2 = tier2 or Tier2_Disk()

    def add_chunk(self, chunk: Chunk, user_id: str, session_id: str) -> bool:
        chunk.metadata["user_id"] = user_id
        chunk.metadata["session_id"] = session_id

        self.tier2.add(chunk)

        if chunk.is_pinned or chunk.get_priority_value() > 50:
            if self.tier0.add(chunk, force=chunk.is_pinned):
                return True

        return self.tier1.add(chunk)

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        chunk = self.tier0.get(chunk_id)
        if chunk:
            return chunk

        chunk = self.tier1.get(chunk_id)
        if chunk:
            self.promote_to_tier0(chunk)
            return chunk

        chunk = self.tier2.get(chunk_id)
        if chunk:
            if chunk.is_pinned or chunk.get_priority_value() > 50:
                self.promote_to_tier0(chunk)
            else:
                self.promote_to_tier1(chunk)
            return chunk

        return None

    def remove_chunk(self, chunk_id: str):
        self.tier0.remove(chunk_id)
        self.tier1.remove(chunk_id)
        self.tier2.remove(chunk_id)

    def promote_to_tier0(self, chunk: Chunk) -> bool:
        if self.tier0.add(chunk, force=chunk.is_pinned):
            self.tier1.remove(chunk.id)
            return True
        return False

    def promote_to_tier1(self, chunk: Chunk) -> bool:
        return self.tier1.add(chunk)

    def demote_to_tier1(self, chunk_id: str) -> bool:
        chunk = self.tier0.remove(chunk_id)
        if chunk:
            return self.tier1.add(chunk)
        return False

    def demote_to_tier2(self, chunk_id: str):
        self.tier0.remove(chunk_id)
        self.tier1.remove(chunk_id)

    def get_all_stats(self) -> Dict:
        return {
            "tier0": self.tier0.get_usage(),
            "tier1": self.tier1.get_usage(),
            "tier2": self.tier2.get_stats(),
        }
