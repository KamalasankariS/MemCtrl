"""In-memory embedding store using sentence-transformers for semantic search."""

import logging
from typing import List, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """Stores embeddings in memory and supports cosine-similarity search."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self.ids: List[str] = []
        self.embeddings: List[np.ndarray] = []

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                logger.warning("sentence-transformers not installed; embedding search disabled")
                return None
        return self._model

    def __len__(self) -> int:
        return len(self.ids)

    def add(self, chunk_id: str, content: str):
        if self.model is None:
            return
        if chunk_id in self.ids:
            idx = self.ids.index(chunk_id)
            self.embeddings[idx] = self.model.encode(content, normalize_embeddings=True)
            return
        embedding = self.model.encode(content, normalize_embeddings=True)
        self.ids.append(chunk_id)
        self.embeddings.append(embedding)

    def remove(self, chunk_id: str):
        if chunk_id in self.ids:
            idx = self.ids.index(chunk_id)
            self.ids.pop(idx)
            self.embeddings.pop(idx)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if self.model is None or len(self.ids) == 0:
            return []

        query_embedding = self.model.encode(query, normalize_embeddings=True)

        matrix = np.stack(self.embeddings)

        # Filter out any NaN/inf embeddings
        valid_mask = np.all(np.isfinite(matrix), axis=1)
        if not np.any(valid_mask):
            return []

        valid_ids = [self.ids[i] for i in range(len(self.ids)) if valid_mask[i]]
        valid_matrix = matrix[valid_mask]

        similarities = valid_matrix @ query_embedding
        similarities = np.nan_to_num(similarities, nan=0.0)

        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(valid_ids[i], float(similarities[i])) for i in top_indices]

    def clear(self):
        self.ids.clear()
        self.embeddings.clear()
