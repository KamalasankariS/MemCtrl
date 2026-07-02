"""Benchmark: prove task-aware retention keeps important chunks longer.

Compares task-aware eviction vs naive FIFO eviction under memory pressure.
Measures what percentage of medical/code chunks survive vs general chat.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import random

sys.path.append(str(Path(__file__).parent.parent))

from memctrl.core.tiers import (
    Tier0_Active,
    Tier1_RAM,
    TierManager,
    compute_task_aware_priority,
)
from memctrl.models import Chunk


SEED = 42


def make_chunks(n_per_type=20):
    """Create a realistic mix of chunks with different task types."""
    random.seed(SEED)
    chunks = []
    types = ["medical", "code", "tutoring", "writing", "general"]
    sample_content = {
        "medical": "Patient reports chest pain with radiating to left arm. BP 140/90.",
        "code": "TypeError in parse_config: expected dict got NoneType at line 42.",
        "tutoring": "To solve 3x + 7 = 22, subtract 7 from both sides, then divide.",
        "writing": "The dragon unfurled its wings and gazed at the sunset below.",
        "general": "What are the best restaurants near downtown?",
    }

    for task_type in types:
        for i in range(n_per_type):
            age_hours = random.uniform(0, 72)
            chunk = Chunk(
                id=f"{task_type}_{i}",
                content=sample_content[task_type],
                tokens=random.randint(10, 30),
            )
            chunk.task_type = task_type
            chunk.timestamp = datetime.now() - timedelta(hours=age_hours)
            chunk.access_count = random.randint(0, 5)
            chunks.append(chunk)

    random.shuffle(chunks)
    return chunks


def run_naive_eviction(chunks, capacity_tokens):
    """FIFO eviction: oldest chunks get evicted first, regardless of type."""
    tier0 = Tier0_Active(max_tokens=capacity_tokens)

    for chunk in chunks:
        if not tier0.add(chunk):
            # Evict oldest non-pinned chunk
            all_chunks = sorted(tier0.get_all(), key=lambda c: c.timestamp)
            for old in all_chunks:
                if not old.is_pinned:
                    tier0.remove(old.id)
                    break
            tier0.add(chunk)

    return tier0


def run_task_aware_eviction(chunks, capacity_tokens):
    """Task-aware eviction: uses compute_task_aware_priority to decide."""
    tier0 = Tier0_Active(max_tokens=capacity_tokens)
    tier1 = Tier1_RAM(max_tokens=10000)
    manager = TierManager(tier0, tier1)

    for chunk in chunks:
        if not tier0.add(chunk):
            manager.task_aware_evict(num_to_evict=1)
            tier0.add(chunk)

    return tier0


def count_by_type(tier0):
    counts = {}
    for chunk in tier0.get_all():
        t = chunk.task_type or "unknown"
        counts[t] = counts.get(t, 0) + 1
    return counts


def main():
    print("=" * 70)
    print("BENCHMARK: Task-Aware vs Naive Retention")
    print("=" * 70)

    chunks = make_chunks(n_per_type=20)
    total_tokens = sum(c.tokens for c in chunks)
    capacity = total_tokens // 3  # Only 1/3 fits — forces eviction

    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Total tokens: {total_tokens}")
    print(f"Tier0 capacity: {capacity} tokens (~33%)")
    print()

    # Run both strategies
    naive_tier0 = run_naive_eviction(chunks.copy(), capacity)
    aware_tier0 = run_task_aware_eviction(chunks.copy(), capacity)

    naive_counts = count_by_type(naive_tier0)
    aware_counts = count_by_type(aware_tier0)

    print(f"{'Task Type':12s} | {'Naive':>6s} | {'Task-Aware':>10s} | {'Improvement':>11s}")
    print("-" * 50)

    for task_type in ["medical", "code", "tutoring", "writing", "general"]:
        n = naive_counts.get(task_type, 0)
        a = aware_counts.get(task_type, 0)
        diff = a - n
        sign = "+" if diff > 0 else ""
        print(f"{task_type:12s} | {n:6d} | {a:10d} | {sign}{diff:10d}")

    naive_critical = naive_counts.get("medical", 0) + naive_counts.get("code", 0)
    aware_critical = aware_counts.get("medical", 0) + aware_counts.get("code", 0)

    print("-" * 50)
    print(f"{'CRITICAL':12s} | {naive_critical:6d} | {aware_critical:10d} | "
          f"+{aware_critical - naive_critical:10d}")
    print()

    if aware_critical > naive_critical:
        pct = (aware_critical - naive_critical) / max(naive_critical, 1) * 100
        print(f"Task-aware retention keeps {pct:.0f}% more critical chunks (medical+code).")
    elif aware_critical == naive_critical:
        print("Both strategies retained the same number of critical chunks.")
    else:
        print("Naive strategy retained more critical chunks (unexpected).")

    print()

    # Show priority scores for a sample
    print("Sample priority scores (higher = retained longer):")
    print(f"{'Chunk':20s} | {'Base':>6s} | {'Task-Aware':>10s}")
    print("-" * 42)
    samples = sorted(chunks[:10], key=lambda c: compute_task_aware_priority(c), reverse=True)
    for chunk in samples:
        base = chunk.get_priority_value()
        aware = compute_task_aware_priority(chunk)
        print(f"{chunk.id:20s} | {base:6.1f} | {aware:10.1f}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
