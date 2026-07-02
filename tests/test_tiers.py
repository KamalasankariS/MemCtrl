import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

from memctrl.core.tiers import (
    Tier0_Active,
    Tier0_GPU,
    Tier1_RAM,
    Tier2_Disk,
    TierManager,
    compute_task_aware_priority,
)
from memctrl.models import Chunk


def test_tier0_basic():
    tier0 = Tier0_Active(max_tokens=100)
    chunk = Chunk(id="c1", content="Test", tokens=10)

    assert tier0.add(chunk)
    assert tier0.current_tokens == 10

    retrieved = tier0.get("c1")
    assert retrieved is not None
    assert retrieved.id == "c1"

    removed = tier0.remove("c1")
    assert removed is not None
    assert tier0.current_tokens == 0


def test_tier0_capacity():
    tier0 = Tier0_Active(max_tokens=100)

    for i in range(10):
        chunk = Chunk(id=f"c{i}", content=f"Test {i}", tokens=10)
        tier0.add(chunk)

    assert tier0.current_tokens == 100
    assert tier0.is_full()

    extra = Chunk(id="extra", content="Extra", tokens=10)
    assert not tier0.add(extra, force=False)
    assert tier0.add(extra, force=True)
    assert tier0.current_tokens <= 100


def test_tier0_pinned_never_evicted():
    tier0 = Tier0_Active(max_tokens=50)

    pinned = Chunk(id="pinned", content="Important", tokens=30, is_pinned=True)
    tier0.add(pinned, force=True)

    normal = Chunk(id="normal", content="Normal", tokens=20)
    tier0.add(normal, force=True)

    extra = Chunk(id="extra", content="Extra", tokens=10)
    tier0.add(extra, force=True)

    assert tier0.get("pinned") is not None
    assert tier0.get("normal") is None


def test_tier0_gpu_alias():
    """Tier0_GPU is a backward-compat alias for Tier0_Active."""
    assert Tier0_GPU is Tier0_Active


def test_tier1_compression():
    tier1 = Tier1_RAM(max_tokens=100)
    chunk = Chunk(id="c1", content="This is a long test content " * 10, tokens=50)

    assert tier1.add(chunk)
    assert tier1.current_tokens < 50

    retrieved = tier1.get("c1")
    assert retrieved is not None
    assert retrieved.summary is not None


@pytest.fixture
def temp_tier2():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    tier2 = Tier2_Disk(db_path)
    yield tier2
    Path(db_path).unlink(missing_ok=True)


def test_tier2_persistence(temp_tier2):
    chunk = Chunk(id="c1", content="Persistent", tokens=5)
    chunk.metadata["user_id"] = "test_user"

    assert temp_tier2.add(chunk)

    retrieved = temp_tier2.get("c1")
    assert retrieved is not None
    assert retrieved.content == "Persistent"


def test_tier2_search(temp_tier2):
    chunks = [
        Chunk(id="c1", content="Python programming", tokens=2),
        Chunk(id="c2", content="JavaScript coding", tokens=2),
    ]
    for chunk in chunks:
        chunk.metadata["user_id"] = "test_user"
        temp_tier2.add(chunk)

    results = temp_tier2.search("Python", user_id="test_user")
    assert len(results) >= 1
    assert any(r.content == "Python programming" for r in results)


def test_tier_manager_flow(temp_tier2):
    tier0 = Tier0_Active(max_tokens=50)
    tier1 = Tier1_RAM(max_tokens=100)
    manager = TierManager(tier0, tier1, temp_tier2)

    important = Chunk(id="imp", content="Important", tokens=10)
    important.set_importance(0.9, "medical")
    manager.add_chunk(important, user_id="test", session_id="s1")
    assert tier0.get("imp") is not None

    normal = Chunk(id="norm", content="Normal", tokens=10)
    normal.set_importance(0.3, "general")
    manager.add_chunk(normal, user_id="test", session_id="s1")
    assert tier1.get("norm") is not None

    retrieved = manager.get_chunk("norm")
    assert retrieved is not None


def test_tier_manager_stats(temp_tier2):
    manager = TierManager(tier2=temp_tier2)
    chunk = Chunk(id="c1", content="Test", tokens=5)
    manager.add_chunk(chunk, user_id="test", session_id="s1")

    stats = manager.get_all_stats()
    assert "tier0" in stats
    assert "tier1" in stats
    assert "tier2" in stats


# -- Task-aware priority scoring --

def test_task_aware_priority_medical_higher_than_general():
    med = Chunk(id="m1", content="Medical info", tokens=10)
    med.task_type = "medical"
    med.set_importance(0.5, "medical")

    gen = Chunk(id="g1", content="General chat", tokens=10)
    gen.task_type = "general"
    gen.set_importance(0.5, "general")

    assert compute_task_aware_priority(med) > compute_task_aware_priority(gen)


def test_task_aware_priority_pinned_is_infinite():
    chunk = Chunk(id="p1", content="Pinned", tokens=10, is_pinned=True)
    chunk.task_type = "general"
    assert compute_task_aware_priority(chunk) == float("inf")


def test_task_aware_priority_decays_with_age():
    fresh = Chunk(id="f1", content="Fresh", tokens=10)
    fresh.task_type = "general"

    old = Chunk(id="o1", content="Old", tokens=10)
    old.task_type = "general"
    old.timestamp = datetime.now() - timedelta(hours=24)

    assert compute_task_aware_priority(fresh) > compute_task_aware_priority(old)


def test_task_aware_priority_medical_decays_slower():
    """Medical chunks should retain priority longer than general."""
    age = timedelta(hours=24)

    med = Chunk(id="m1", content="Medical", tokens=10)
    med.task_type = "medical"
    med.timestamp = datetime.now() - age

    gen = Chunk(id="g1", content="General", tokens=10)
    gen.task_type = "general"
    gen.timestamp = datetime.now() - age

    # After 24 hours, general has fully decayed but medical still has recency
    med_pri = compute_task_aware_priority(med)
    gen_pri = compute_task_aware_priority(gen)
    assert med_pri > gen_pri


def test_task_aware_priority_access_bonus():
    accessed = Chunk(id="a1", content="Accessed", tokens=10)
    accessed.task_type = "general"
    accessed.access_count = 5

    fresh = Chunk(id="f1", content="Fresh", tokens=10)
    fresh.task_type = "general"
    fresh.access_count = 0

    assert compute_task_aware_priority(accessed) > compute_task_aware_priority(fresh)


# -- Task-aware eviction --

def test_task_aware_evict_general_before_medical(temp_tier2):
    tier0 = Tier0_Active(max_tokens=100)
    tier1 = Tier1_RAM(max_tokens=1000)
    manager = TierManager(tier0, tier1, temp_tier2)

    med = Chunk(id="med", content="Medical info", tokens=20)
    med.task_type = "medical"
    tier0.add(med)

    gen = Chunk(id="gen", content="General chat", tokens=20)
    gen.task_type = "general"
    tier0.add(gen)

    evicted = manager.task_aware_evict(num_to_evict=1)
    assert evicted == 1
    # General should be evicted first (lower task-aware priority)
    assert tier0.get("med") is not None
    assert tier0.get("gen") is None
    assert tier1.get("gen") is not None


def test_task_aware_evict_pinned_never_evicted(temp_tier2):
    tier0 = Tier0_Active(max_tokens=100)
    tier1 = Tier1_RAM(max_tokens=1000)
    manager = TierManager(tier0, tier1, temp_tier2)

    pinned = Chunk(id="pin", content="Pinned", tokens=20, is_pinned=True)
    tier0.add(pinned, force=True)

    gen = Chunk(id="gen", content="General", tokens=20)
    gen.task_type = "general"
    tier0.add(gen)

    evicted = manager.task_aware_evict(num_to_evict=2)
    assert tier0.get("pin") is not None
    assert evicted == 1


# -- Task-aware promotion thresholds --

def test_medical_promotes_easier(temp_tier2):
    """Medical chunks have a lower promotion threshold (30 vs 50 for general)."""
    tier0 = Tier0_Active(max_tokens=100)
    tier1 = Tier1_RAM(max_tokens=1000)
    manager = TierManager(tier0, tier1, temp_tier2)

    # importance 0.4 => priority_value = 0.4 * 99 = 39.6
    # medical threshold: 30 (promotes), general threshold: 50 (doesn't)
    med = Chunk(id="med", content="Medical", tokens=10)
    med.set_importance(0.4, "medical")
    manager.add_chunk(med, user_id="test", session_id="s1")
    assert tier0.get("med") is not None

    gen = Chunk(id="gen", content="General", tokens=10)
    gen.set_importance(0.4, "general")
    manager.add_chunk(gen, user_id="test", session_id="s1")
    assert tier0.get("gen") is None
    assert tier1.get("gen") is not None


# -- LLM-powered compression --

def test_tier1_llm_compression():
    mock_llm = MagicMock()
    mock_llm.provider_name = "anthropic"
    mock_llm.generate.return_value = "LLM summary of the content"

    tier1 = Tier1_RAM(max_tokens=1000, llm=mock_llm)
    chunk = Chunk(id="llm1", content="A very long piece of text " * 20, tokens=100)
    tier1.add(chunk)

    assert chunk.summary == "LLM summary of the content"
    mock_llm.generate.assert_called_once()
    call_msgs = mock_llm.generate.call_args[0][0]
    assert "Summarize" in call_msgs[0]["content"]


def test_tier1_llm_compression_with_task_type():
    mock_llm = MagicMock()
    mock_llm.provider_name = "openai"
    mock_llm.generate.return_value = "Medical summary"

    tier1 = Tier1_RAM(max_tokens=1000, llm=mock_llm)
    chunk = Chunk(id="med1", content="Patient has symptoms " * 20, tokens=100)
    chunk.task_type = "medical"
    tier1.add(chunk)

    call_msgs = mock_llm.generate.call_args[0][0]
    assert "medical terms" in call_msgs[0]["content"]


def test_tier1_llm_compression_fallback():
    mock_llm = MagicMock()
    mock_llm.provider_name = "anthropic"
    mock_llm.generate.side_effect = Exception("API error")

    tier1 = Tier1_RAM(max_tokens=1000, llm=mock_llm)
    chunk = Chunk(
        id="fall1",
        content="First sentence here. Second sentence here. Third one too.",
        tokens=50,
    )
    tier1.add(chunk)

    assert chunk.summary is not None
    assert len(chunk.summary) > 0


def test_tier1_echo_llm_skipped():
    mock_llm = MagicMock()
    mock_llm.provider_name = "echo"

    tier1 = Tier1_RAM(max_tokens=1000, llm=mock_llm)
    chunk = Chunk(
        id="echo1",
        content="Some content. Another sentence. More text here.",
        tokens=50,
    )
    tier1.add(chunk)

    mock_llm.generate.assert_not_called()
    assert chunk.summary is not None


def test_tier1_no_llm_uses_extractive():
    tier1 = Tier1_RAM(max_tokens=1000, llm=None)
    chunk = Chunk(
        id="noLLM",
        content="Important fact one. Less important. Critical detail here.",
        tokens=50,
    )
    tier1.add(chunk)

    assert chunk.summary is not None
    assert len(chunk.summary) > 0


# -- LLM-powered decompression --

def test_tier1_llm_decompression():
    mock_llm = MagicMock()
    mock_llm.provider_name = "anthropic"
    mock_llm.generate.return_value = "Expanded detailed content from summary"

    tier1 = Tier1_RAM(max_tokens=1000, llm=mock_llm)
    chunk = Chunk(id="dec1", content="", tokens=10)
    chunk.summary = "Short summary"
    chunk.content = chunk.summary  # content was lost, only summary remains
    chunk.task_type = "code"

    tier1.decompress(chunk)

    assert chunk.content == "Expanded detailed content from summary"
    call_msgs = mock_llm.generate.call_args[0][0]
    assert "Expand" in call_msgs[0]["content"]
    assert "technical details" in call_msgs[0]["content"]


def test_tier1_decompression_fallback_without_llm():
    tier1 = Tier1_RAM(max_tokens=1000, llm=None)
    chunk = Chunk(id="dec2", content="", tokens=10)
    chunk.summary = "Just the summary"
    chunk.content = ""

    tier1.decompress(chunk)

    assert chunk.content == "Just the summary"


def test_tier1_decompression_skips_if_content_exists():
    mock_llm = MagicMock()
    mock_llm.provider_name = "anthropic"

    tier1 = Tier1_RAM(max_tokens=1000, llm=mock_llm)
    chunk = Chunk(id="dec3", content="Full original content", tokens=10)
    chunk.summary = "Short summary"

    tier1.decompress(chunk)

    # Should not call LLM since content != summary (original still intact)
    mock_llm.generate.assert_not_called()
    assert chunk.content == "Full original content"


# -- Task classification in TierManager --

def test_tier_manager_classifies_task(temp_tier2):
    manager = TierManager(tier2=temp_tier2)
    chunk = Chunk(id="cls1", content="What is the dosage for ibuprofen?", tokens=10)
    manager.add_chunk(chunk, user_id="test", session_id="s1")
    assert chunk.metadata["user_id"] == "test"


def test_tier_manager_passes_llm_to_tier1(temp_tier2):
    mock_llm = MagicMock()
    mock_llm.provider_name = "anthropic"
    manager = TierManager(tier2=temp_tier2, llm=mock_llm)

    assert manager.tier1.llm is mock_llm
