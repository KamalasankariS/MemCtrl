import random
from typing import List, Dict, Tuple

from ..models import Chunk, Session


def create_hindsight_labels(session: Session) -> List[Dict]:
    """Create training labels from a completed session by checking future references."""
    chunks = session.chunks
    labels = []

    for i, chunk in enumerate(chunks):
        if i >= len(chunks) - 2:
            continue

        was_important = _check_if_important(chunk, chunks[i + 1 :])

        from .policy_network import PolicyNetwork

        dummy_policy = PolicyNetwork(task_type="general")
        features = dummy_policy._extract_features(chunk, chunks[:i])

        labels.append({
            "chunk_id": chunk.id,
            "features": features,
            "label": 1.0 if was_important else 0.0,
            "task_type": session.task_type or "general",
        })

    return labels


def _check_if_important(chunk: Chunk, future_chunks: List[Chunk]) -> bool:
    if chunk.is_pinned:
        return True

    if chunk.chunk_type.value == "medical":
        return True

    chunk_words = set(chunk.content.lower().split())
    key_terms = {w for w in chunk_words if len(w) > 4}

    if not key_terms:
        return False

    overlap_count = 0
    for future_chunk in future_chunks:
        future_words = set(future_chunk.content.lower().split())
        if len(key_terms & future_words) >= 2:
            overlap_count += 1

    return overlap_count >= 2


def generate_training_data_from_dataset(
    dataset_name: str,
    num_examples: int = 1000,
) -> Tuple[List[Dict], List[Dict]]:
    """Placeholder: generate dummy training data. Replace with real dataset loading."""
    all_data = [
        {"features": [random.random() for _ in range(128)], "label": random.choice([0.0, 1.0])}
        for _ in range(num_examples)
    ]

    split_idx = int(0.8 * len(all_data))
    return all_data[:split_idx], all_data[split_idx:]
