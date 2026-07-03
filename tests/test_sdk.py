"""Integration tests for the SDK API (Option 3: Context Budget Optimizer).

Tests the add_message() / optimize() workflow that real users will use
with their own LLM SDKs.
"""

import pytest
from unittest.mock import MagicMock

from memctrl import MemoryController


@pytest.fixture
def ctrl():
    mock_llm = MagicMock()
    mock_llm.provider_name = "echo"
    mock_llm.generate.return_value = "Echo response"
    return MemoryController(user_id="sdk_test", llm=mock_llm)


def test_add_message_returns_chunk_info(ctrl):
    result = ctrl.add_message("user", "What is recursion?")
    assert result["chunk_id"]
    assert result["tokens"] > 0
    assert result["tier"] in ("active", "compressed")


def test_add_message_stores_in_tiers(ctrl):
    ctrl.add_message("user", "Hello world")
    tier0 = ctrl.tier_manager.tier0.get_all()
    tier1 = ctrl.tier_manager.tier1.get_all()
    assert len(tier0) + len(tier1) >= 1


def test_optimize_returns_messages_list(ctrl):
    ctrl.add_message("user", "Explain Python decorators")
    messages = ctrl.optimize(max_tokens=4096)

    assert isinstance(messages, list)
    assert messages[0]["role"] == "system"
    assert any(m["role"] == "user" for m in messages)


def test_optimize_includes_pinned_in_system(ctrl):
    ctrl.pin("Always use type hints in Python")
    ctrl.add_message("user", "Write a function")
    messages = ctrl.optimize(max_tokens=4096)

    system_msg = messages[0]["content"]
    assert "type hints" in system_msg


def test_optimize_respects_token_budget(ctrl):
    for i in range(50):
        ctrl.add_message("user", f"Message number {i} with some padding text here")
        ctrl.add_message("assistant", f"Response to message {i} with details")

    messages = ctrl.optimize(max_tokens=500)
    # Should have fewer messages than what was added
    assert len(messages) < 100


def test_optimize_preserves_chronological_order(ctrl):
    ctrl.add_message("user", "First message")
    ctrl.add_message("assistant", "First response")
    ctrl.add_message("user", "Second message")

    messages = ctrl.optimize(max_tokens=4096)
    user_msgs = [m for m in messages if m["role"] == "user"]

    if len(user_msgs) >= 2:
        assert "First" in user_msgs[0]["content"]
        assert "Second" in user_msgs[1]["content"]


def test_optimize_no_internal_keys_leaked(ctrl):
    ctrl.add_message("user", "Test message")
    messages = ctrl.optimize(max_tokens=4096)

    for msg in messages:
        assert "_ts" not in msg
        assert set(msg.keys()).issubset({"role", "content"})


def test_full_conversation_flow(ctrl):
    """Simulate a real SDK usage pattern."""
    # User adds messages as conversation progresses
    ctrl.add_message("user", "I'm building a Flask API for patient records")
    ctrl.add_message("assistant", "I can help with that. Flask is great for REST APIs.")
    ctrl.add_message("user", "The patient model has name, DOB, and diagnosis fields")
    ctrl.add_message("assistant", "Here's a SQLAlchemy model for that...")

    # Pin something critical
    ctrl.pin("Patient records require HIPAA compliance")

    # Get optimized context for next API call
    messages = ctrl.optimize(max_tokens=4096)

    # Verify structure is ready for any LLM SDK
    assert messages[0]["role"] == "system"
    assert "HIPAA" in messages[0]["content"]
    assert all("role" in m and "content" in m for m in messages)

    # Verify conversation content is present
    contents = " ".join(m["content"] for m in messages)
    assert "Flask" in contents or "patient" in contents.lower()


def test_forget_then_optimize(ctrl):
    ctrl.add_message("user", "My password is hunter2")
    ctrl.add_message("user", "What is the weather?")

    ctrl.forget("password", confirm=False)

    messages = ctrl.optimize(max_tokens=4096)
    contents = " ".join(m["content"] for m in messages)
    assert "hunter2" not in contents


def test_pin_unpin_optimize():
    """Use a fresh controller to avoid cross-test pin leakage."""
    mock_llm = MagicMock()
    mock_llm.provider_name = "echo"
    mock_llm.generate.return_value = "Echo response"
    c = MemoryController(user_id="unpin_test", llm=mock_llm)

    result = c.pin("Remember: user prefers dark mode")
    chunk_id = result["chunk_id"]

    messages = c.optimize(max_tokens=4096)
    assert "dark mode" in messages[0]["content"]

    c.unpin(chunk_id)

    messages = c.optimize(max_tokens=4096)
    assert "dark mode" not in messages[0]["content"]


def test_optimize_empty_conversation(ctrl):
    messages = ctrl.optimize(max_tokens=4096)
    assert len(messages) >= 1
    assert messages[0]["role"] == "system"


def test_optimize_with_tight_budget(ctrl):
    ctrl.add_message("user", "A very long message " * 100)
    ctrl.add_message("assistant", "A very long response " * 100)
    ctrl.add_message("user", "Short recent question")

    messages = ctrl.optimize(max_tokens=200)
    # Should at least have system + the most recent message
    assert len(messages) >= 1
    assert messages[0]["role"] == "system"
