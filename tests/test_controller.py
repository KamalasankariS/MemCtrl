import pytest
import tempfile

from memctrl import MemoryController
from memctrl.config import MemCtrlConfig, set_config


@pytest.fixture
def temp_controller():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = MemCtrlConfig(
            data_dir=tmpdir,
            sqlite_path=f"{tmpdir}/test.db",
            duckdb_path=f"{tmpdir}/test.duckdb",
        )
        set_config(config)
        controller = MemoryController(user_id="test_user")
        yield controller
        controller.close_session()


def test_controller_init(temp_controller):
    assert temp_controller.user_id == "test_user"
    assert temp_controller.control_mode == "hybrid"
    assert temp_controller.user is not None


def test_chat_basic(temp_controller):
    response = temp_controller.chat("Hello")
    assert response is not None
    assert "Hello" in response
    assert temp_controller.current_session is not None
    assert len(temp_controller.current_session.chunks) > 0


def test_pin(temp_controller):
    result = temp_controller.pin("Important fact", note="User preference")
    assert result["success"] is True
    assert "chunk_id" in result
    assert temp_controller.user.is_pinned(result["chunk_id"])

    memory = temp_controller.show_memory(category="pinned")
    assert len(memory["pinned"]) == 1
    assert memory["pinned"][0]["content"] == "Important fact"


def test_forget(temp_controller):
    temp_controller.chat("Python programming")
    temp_controller.chat("JavaScript coding")

    result = temp_controller.forget("Python", confirm=True)
    assert result["success"] is True
    assert result["confirm_required"] is True
    assert len(result["matches"]) > 0

    chunk_ids = [m["chunk_id"] for m in result["matches"]]
    result2 = temp_controller.forget_confirmed(chunk_ids)
    assert result2["success"] is True
    assert result2["num_deleted"] > 0


def test_temporary(temp_controller):
    result = temp_controller.temporary("Meeting at 3pm")
    assert result["success"] is True

    session = temp_controller.current_session
    assert any(c.is_temporary for c in session.chunks)

    temp_controller.close_session()
    temp_controller.chat("New session")


def test_show_memory(temp_controller):
    temp_controller.pin("Pinned content")
    temp_controller.chat("Regular chat")
    temp_controller.temporary("Temp content")

    memory = temp_controller.show_memory(category="all")
    assert "pinned" in memory
    assert "session" in memory
    assert len(memory["pinned"]) == 1
    assert len(memory["session"]) > 0


def test_get_stats(temp_controller):
    temp_controller.chat("Test message")
    temp_controller.pin("Important")

    stats = temp_controller.get_stats()
    assert "user_id" in stats
    assert "tiers" in stats
    assert "tier0" in stats["tiers"]
    assert "tier1" in stats["tiers"]
    assert "tier2" in stats["tiers"]


def test_export_data(temp_controller):
    temp_controller.pin("Export test")
    temp_controller.chat("Hello")

    json_export = temp_controller.export_data(format="json")
    assert "export_timestamp" in json_export
    assert "pinned_chunks" in json_export

    text_export = temp_controller.export_data(format="text")
    assert "MemCtrl Data Export" in text_export


def test_memory_pressure_handling(temp_controller):
    for i in range(100):
        temp_controller.chat(f"Message {i}")

    stats = temp_controller.get_stats()
    tier0_util = stats["tiers"]["tier0"]["utilization"]
    assert tier0_util < 1.0
