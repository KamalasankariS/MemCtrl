from memctrl import MemoryController


def test_basic_usage_flow():
    controller = MemoryController(user_id="kamala")

    response = controller.chat("Help me with Python")
    assert response is not None
    assert "Python" in response

    result = controller.pin("I prefer TypeScript", note="Language preference")
    assert result["success"] is True

    memory = controller.show_memory()
    assert "pinned" in memory

    stats = controller.get_stats()
    assert "tiers" in stats

    controller.close_session()
