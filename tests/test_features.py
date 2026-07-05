"""Tests for the 5 new features:
1. Auto-pin suggestions
2. Stale memory cleanup suggestions
3. Transparent SDK wrapper
4. Cross-session memory persistence
5. Context budget debugger
"""
import pytest
import tempfile
import time

from memctrl import MemoryController, wrap, WrappedClient
from memctrl.config import MemCtrlConfig, set_config


@pytest.fixture
def ctrl():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = MemCtrlConfig(
            data_dir=tmpdir,
            sqlite_path=f"{tmpdir}/test.db",
            duckdb_path=f"{tmpdir}/test.duckdb",
        )
        set_config(config)
        controller = MemoryController(user_id="feat_test_user", provider="echo")
        yield controller
        controller.close_session()


# -- Feature 1: Auto-pin suggestions --

class TestSuggestPins:
    def test_no_suggestions_for_normal_text(self, ctrl):
        ctrl.add_message("user", "Hello, how are you?")
        suggestions = ctrl.suggest_pins()
        assert suggestions == []

    def test_detects_api_key(self, ctrl):
        ctrl.add_message("user", "My api_key is sk-abcdefghijklmnopqrstuvwxyz1234")
        suggestions = ctrl.suggest_pins()
        assert len(suggestions) >= 1
        assert any(s["category"] == "api_key" for s in suggestions)

    def test_detects_password(self, ctrl):
        ctrl.add_message("user", "The password is hunter2")
        suggestions = ctrl.suggest_pins()
        assert len(suggestions) >= 1
        assert any(s["category"] == "credential" for s in suggestions)

    def test_detects_dosage(self, ctrl):
        ctrl.add_message("user", "Take 500 mg daily")
        suggestions = ctrl.suggest_pins()
        assert len(suggestions) >= 1
        assert any(s["category"] == "dosage" for s in suggestions)

    def test_detects_connection_string(self, ctrl):
        ctrl.add_message("user", "Use postgresql://user:pass@localhost:5432/db")
        suggestions = ctrl.suggest_pins()
        assert len(suggestions) >= 1
        assert any(s["category"] == "connection_string" for s in suggestions)

    def test_accept_pin_suggestion(self, ctrl):
        ctrl.add_message("user", "My secret is sk-abcdefghijklmnopqrstuvwxyz1234")
        suggestions = ctrl.suggest_pins()
        assert len(suggestions) >= 1

        chunk_id = suggestions[0]["chunk_id"]
        result = ctrl.accept_pin_suggestion(chunk_id)
        assert result["success"] is True

        # Verify it's pinned
        memory = ctrl.show_memory(category="pinned")
        assert any(p["chunk_id"] == chunk_id for p in memory["pinned"])

    def test_skips_already_pinned(self, ctrl):
        ctrl.pin("password is hunter2")
        suggestions = ctrl.suggest_pins()
        # Pinned chunks should not appear in suggestions
        for s in suggestions:
            pinned = ctrl.show_memory(category="pinned")
            assert s["chunk_id"] not in [p["chunk_id"] for p in pinned["pinned"]]

    def test_suggestion_has_required_fields(self, ctrl):
        ctrl.add_message("user", "PORT is 8080")
        suggestions = ctrl.suggest_pins()
        if suggestions:
            s = suggestions[0]
            assert "chunk_id" in s
            assert "category" in s
            assert "matched_text" in s
            assert "content_preview" in s
            assert "reason" in s


# -- Feature 2: Stale memory cleanup suggestions --

class TestSuggestCleanup:
    def test_no_suggestions_for_fresh_data(self, ctrl):
        ctrl.add_message("user", "Fresh message")
        suggestions = ctrl.suggest_cleanup(stale_hours=24.0)
        assert suggestions == []

    def test_detects_stale_chunks(self, ctrl):
        ctrl.add_message("user", "Old message")

        # Make chunks appear stale by backdating last_accessed
        from datetime import datetime, timedelta
        all_chunks = ctrl.tier_manager.tier0.get_all() + ctrl.tier_manager.tier1.get_all()
        for chunk in all_chunks:
            chunk.last_accessed = datetime.now() - timedelta(hours=48)

        suggestions = ctrl.suggest_cleanup(stale_hours=24.0)
        assert len(suggestions) > 0
        assert suggestions[0]["hours_stale"] > 24

    def test_cleanup_preserves_pinned(self, ctrl):
        ctrl.pin("Important pinned data")

        from datetime import datetime, timedelta
        for chunk in ctrl.tier_manager.tier0.get_all():
            chunk.last_accessed = datetime.now() - timedelta(hours=48)

        suggestions = ctrl.suggest_cleanup(stale_hours=1.0)
        suggestion_ids = [s["chunk_id"] for s in suggestions]
        pinned = ctrl.show_memory(category="pinned")
        for p in pinned["pinned"]:
            assert p["chunk_id"] not in suggestion_ids

    def test_accept_cleanup(self, ctrl):
        ctrl.add_message("user", "Will be cleaned up")

        from datetime import datetime, timedelta
        all_chunks = ctrl.tier_manager.tier0.get_all() + ctrl.tier_manager.tier1.get_all()
        for chunk in all_chunks:
            if not chunk.is_pinned:
                chunk.last_accessed = datetime.now() - timedelta(hours=48)

        suggestions = ctrl.suggest_cleanup(stale_hours=24.0)
        assert len(suggestions) > 0

        chunk_ids = [s["chunk_id"] for s in suggestions]
        result = ctrl.accept_cleanup(chunk_ids)
        assert result["success"] is True
        assert result["num_deleted"] > 0
        assert result["tokens_saved"] > 0

    def test_cleanup_suggestions_sorted_by_staleness(self, ctrl):
        ctrl.add_message("user", "Older message")
        ctrl.add_message("user", "Newer message")

        from datetime import datetime, timedelta
        all_chunks = ctrl.tier_manager.tier0.get_all()
        unpinned = [c for c in all_chunks if not c.is_pinned]
        if len(unpinned) >= 2:
            unpinned[0].last_accessed = datetime.now() - timedelta(hours=72)
            unpinned[1].last_accessed = datetime.now() - timedelta(hours=48)

            suggestions = ctrl.suggest_cleanup(stale_hours=24.0)
            if len(suggestions) >= 2:
                assert suggestions[0]["hours_stale"] >= suggestions[1]["hours_stale"]

    def test_cleanup_shows_token_savings(self, ctrl):
        ctrl.add_message("user", "Delete me for savings")

        from datetime import datetime, timedelta
        for chunk in ctrl.tier_manager.tier0.get_all():
            if not chunk.is_pinned:
                chunk.last_accessed = datetime.now() - timedelta(hours=48)

        suggestions = ctrl.suggest_cleanup(stale_hours=24.0)
        if suggestions:
            assert "tokens" in suggestions[0]
            assert "total_recoverable_tokens" in suggestions[0]
            assert suggestions[0]["total_recoverable_tokens"] > 0


# -- Feature 3: Transparent SDK wrapper --

class TestWrap:
    def test_wrap_creates_wrapped_client(self):
        class MockClient:
            pass

        wrapped = wrap(MockClient(), max_tokens=2048)
        assert isinstance(wrapped, WrappedClient)

    def test_wrapped_client_has_memory_controller(self):
        class MockClient:
            pass

        wrapped = wrap(MockClient(), max_tokens=2048, user_id="wrap_test")
        assert wrapped.mc is not None
        assert wrapped.mc.user_id == "wrap_test"

    def test_detect_openai_provider(self):
        class FakeOpenAI:
            __module__ = "openai._client"

        assert WrappedClient._detect_provider(FakeOpenAI()) == "openai"

    def test_detect_anthropic_provider(self):
        class FakeAnthropic:
            __module__ = "anthropic._client"

        assert WrappedClient._detect_provider(FakeAnthropic()) == "anthropic"

    def test_wrapped_exposes_pin(self):
        class MockClient:
            pass

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemCtrlConfig(
                data_dir=tmpdir,
                sqlite_path=f"{tmpdir}/wrap.db",
                duckdb_path=f"{tmpdir}/wrap.duckdb",
            )
            set_config(config)
            wrapped = wrap(MockClient(), max_tokens=2048, user_id="wrap_pin_test")
            result = wrapped.pin("Remember this")
            assert result["success"] is True

    def test_wrapped_exposes_budget_report(self):
        class MockClient:
            pass

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemCtrlConfig(
                data_dir=tmpdir,
                sqlite_path=f"{tmpdir}/wrap2.db",
                duckdb_path=f"{tmpdir}/wrap2.duckdb",
            )
            set_config(config)
            wrapped = wrap(MockClient(), max_tokens=2048, user_id="wrap_budget_test")
            report = wrapped.budget_report()
            assert "max_tokens" in report
            assert report["max_tokens"] == 2048

    def test_wrapped_exposes_suggest_pins(self):
        class MockClient:
            pass

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemCtrlConfig(
                data_dir=tmpdir,
                sqlite_path=f"{tmpdir}/wrap3.db",
                duckdb_path=f"{tmpdir}/wrap3.duckdb",
            )
            set_config(config)
            wrapped = wrap(MockClient(), max_tokens=2048, user_id="wrap_suggest_test")
            suggestions = wrapped.suggest_pins()
            assert isinstance(suggestions, list)


# -- Feature 4: Cross-session persistence --

class TestCrossSessionPersistence:
    def test_trash_persists_across_controllers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemCtrlConfig(
                data_dir=tmpdir,
                sqlite_path=f"{tmpdir}/persist.db",
                duckdb_path=f"{tmpdir}/persist.duckdb",
            )
            set_config(config)

            # Session 1: create and delete data
            ctrl1 = MemoryController(user_id="persist_user", provider="echo")
            ctrl1.add_message("user", "Delete this later")
            ctrl1.forget("Delete", confirm=False)
            assert len(ctrl1.trash) > 0
            trash_content = ctrl1.trash[0]["content"]
            ctrl1.close_session()

            # Session 2: trash should still be there
            ctrl2 = MemoryController(user_id="persist_user", provider="echo")
            assert len(ctrl2.trash) > 0
            assert any(t["content"] == trash_content for t in ctrl2.trash)
            ctrl2.close_session()

    def test_audit_log_persists_across_controllers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemCtrlConfig(
                data_dir=tmpdir,
                sqlite_path=f"{tmpdir}/audit.db",
                duckdb_path=f"{tmpdir}/audit.duckdb",
            )
            set_config(config)

            # Session 1: perform some actions
            ctrl1 = MemoryController(user_id="audit_user", provider="echo")
            ctrl1.add_message("user", "Hello")
            ctrl1.pin("Important")
            assert len(ctrl1.audit_log) >= 2
            ctrl1.close_session()

            # Session 2: audit log should be loaded
            ctrl2 = MemoryController(user_id="audit_user", provider="echo")
            assert len(ctrl2.audit_log) >= 2
            actions = [e["action"] for e in ctrl2.audit_log]
            assert "add_message" in actions
            assert "pin" in actions
            ctrl2.close_session()

    def test_restore_from_trash_removes_from_sqlite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemCtrlConfig(
                data_dir=tmpdir,
                sqlite_path=f"{tmpdir}/restore.db",
                duckdb_path=f"{tmpdir}/restore.duckdb",
            )
            set_config(config)

            ctrl = MemoryController(user_id="restore_user", provider="echo")
            ctrl.add_message("user", "Restore me")
            ctrl.forget("Restore", confirm=False)
            assert len(ctrl.trash) > 0
            chunk_id = ctrl.trash[0]["chunk_id"]

            ctrl.restore_from_trash(chunk_id)
            assert not any(t["chunk_id"] == chunk_id for t in ctrl.trash)

            # Verify it's gone from SQLite too
            from_db = ctrl.tier_manager.tier2.store.get_trash(ctrl.user_id)
            assert not any(t["chunk_id"] == chunk_id for t in from_db)
            ctrl.close_session()

    def test_pins_persist_across_sessions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemCtrlConfig(
                data_dir=tmpdir,
                sqlite_path=f"{tmpdir}/pins.db",
                duckdb_path=f"{tmpdir}/pins.duckdb",
            )
            set_config(config)

            ctrl1 = MemoryController(user_id="pin_persist_user", provider="echo")
            ctrl1.pin("My API key is abc123")
            ctrl1.close_session()

            ctrl2 = MemoryController(user_id="pin_persist_user", provider="echo")
            memory = ctrl2.show_memory(category="pinned")
            assert len(memory["pinned"]) == 1
            assert "abc123" in memory["pinned"][0]["content"]
            ctrl2.close_session()


# -- Feature 5: Context budget debugger --

class TestBudgetReport:
    def test_empty_budget_report(self, ctrl):
        report = ctrl.budget_report(max_tokens=4096)
        assert report["max_tokens"] == 4096
        assert "breakdown" in report
        assert "recommendations" in report
        assert report["usage_pct"] < 100

    def test_budget_breakdown_has_all_categories(self, ctrl):
        ctrl.add_message("user", "Test message")
        ctrl.pin("Pinned item")

        report = ctrl.budget_report(max_tokens=4096)
        breakdown = report["breakdown"]
        assert "system_prompt" in breakdown
        assert "pinned" in breakdown
        assert "active" in breakdown
        assert "compressed" in breakdown

    def test_budget_tokens_sum(self, ctrl):
        ctrl.add_message("user", "Test message")
        report = ctrl.budget_report(max_tokens=4096)
        breakdown = report["breakdown"]
        component_sum = (
            breakdown["system_prompt"]["tokens"]
            + breakdown["pinned"]["tokens"]
            + breakdown["active"]["tokens"]
            + breakdown["compressed"]["tokens"]
        )
        assert report["total_used"] == component_sum

    def test_budget_remaining_correct(self, ctrl):
        ctrl.add_message("user", "Some content")
        report = ctrl.budget_report(max_tokens=4096)
        assert report["remaining"] == 4096 - report["total_used"]

    def test_budget_recommendations_low_usage(self, ctrl):
        report = ctrl.budget_report(max_tokens=4096)
        # With minimal data, should recommend "plenty of budget"
        assert any("budget" in r.lower() for r in report["recommendations"])

    def test_budget_with_custom_max_tokens(self, ctrl):
        report = ctrl.budget_report(max_tokens=2048)
        assert report["max_tokens"] == 2048

    def test_pinned_count_in_report(self, ctrl):
        ctrl.pin("Pin 1")
        ctrl.pin("Pin 2")
        report = ctrl.budget_report()
        assert report["breakdown"]["pinned"]["count"] == 2


# -- Feature 6: Entity-preserving compression --

class TestEntityPreservingCompression:
    def test_extract_entities_url(self):
        from memctrl.core.tiers import Tier1_RAM
        entities = Tier1_RAM._extract_entities("Visit https://example.com/webhook for details")
        assert any("https://example.com/webhook" in e for e in entities)

    def test_extract_entities_version(self):
        from memctrl.core.tiers import Tier1_RAM
        entities = Tier1_RAM._extract_entities("Using Python 3.11.4 with Flask")
        assert any("3.11.4" in e for e in entities)

    def test_extract_entities_measurement(self):
        from memctrl.core.tiers import Tier1_RAM
        entities = Tier1_RAM._extract_entities("Patient takes 500 mg daily")
        assert any("500 mg" in e for e in entities)

    def test_extract_entities_camelcase(self):
        from memctrl.core.tiers import Tier1_RAM
        entities = Tier1_RAM._extract_entities("The ConvNeXt model outperforms ResNet")
        assert any("ConvNeXt" in e for e in entities)
        assert any("ResNet" in e for e in entities)

    def test_extract_entities_empty(self):
        from memctrl.core.tiers import Tier1_RAM
        entities = Tier1_RAM._extract_entities("hello world")
        assert entities == []

    def test_entities_appended_to_summary(self, ctrl):
        # Add a message with entities that will go to tier1 (compressed)
        ctrl.add_message("user", "Deploy Flask app on port 8080 at https://api.example.com/webhook using version 3.11.4")
        tier1_chunks = ctrl.tier_manager.tier1.get_all()
        if tier1_chunks:
            chunk = tier1_chunks[0]
            if chunk.summary:
                # Entities should be appended in brackets
                assert "[" in chunk.summary
                assert "8080" in chunk.summary or "3.11.4" in chunk.summary


# -- Feature 7: Auto-pin critical values during compression --

class TestAutoPinDuringCompression:
    def test_auto_pins_password(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemCtrlConfig(
                data_dir=tmpdir,
                sqlite_path=f"{tmpdir}/autopin.db",
                duckdb_path=f"{tmpdir}/autopin.duckdb",
            )
            set_config(config)
            ctrl = MemoryController(user_id="autopin_user", provider="echo")

            # This message goes to tier1 (not tier0), triggering auto-pin
            ctrl.add_message("user", "The password is supersecret123")

            pinned = ctrl.tier_manager.tier2.get_pinned("autopin_user")
            pinned_contents = " ".join(p.content for p in pinned)
            assert "password" in pinned_contents.lower() or "supersecret123" in pinned_contents
            ctrl.close_session()

    def test_auto_pins_dosage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemCtrlConfig(
                data_dir=tmpdir,
                sqlite_path=f"{tmpdir}/autopin2.db",
                duckdb_path=f"{tmpdir}/autopin2.duckdb",
            )
            set_config(config)
            ctrl = MemoryController(user_id="autopin_med_user", provider="echo")

            ctrl.add_message("user", "Patient takes 500 mg daily for hypertension")

            pinned = ctrl.tier_manager.tier2.get_pinned("autopin_med_user")
            pinned_contents = " ".join(p.content for p in pinned)
            assert "500 mg" in pinned_contents
            ctrl.close_session()

    def test_auto_pin_no_duplicates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemCtrlConfig(
                data_dir=tmpdir,
                sqlite_path=f"{tmpdir}/autopin3.db",
                duckdb_path=f"{tmpdir}/autopin3.duckdb",
            )
            set_config(config)
            ctrl = MemoryController(user_id="autopin_dup_user", provider="echo")

            ctrl.add_message("user", "The password is hunter2")
            ctrl.add_message("user", "Reminder: the password is hunter2")

            pinned = ctrl.tier_manager.tier2.get_pinned("autopin_dup_user")
            password_pins = [p for p in pinned if "password" in p.content.lower()]
            assert len(password_pins) == 1
            ctrl.close_session()

    def test_auto_pin_marked_as_auto(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemCtrlConfig(
                data_dir=tmpdir,
                sqlite_path=f"{tmpdir}/autopin4.db",
                duckdb_path=f"{tmpdir}/autopin4.duckdb",
            )
            set_config(config)
            ctrl = MemoryController(user_id="autopin_meta_user", provider="echo")

            ctrl.add_message("user", "Use postgresql://admin:pass@db.host:5432/mydb")

            pinned = ctrl.tier_manager.tier2.get_pinned("autopin_meta_user")
            auto_pinned = [p for p in pinned if p.metadata.get("auto_pinned")]
            assert len(auto_pinned) >= 1
            assert auto_pinned[0].metadata["category"] == "connection_string"
            ctrl.close_session()

    def test_manually_pinned_not_duplicated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemCtrlConfig(
                data_dir=tmpdir,
                sqlite_path=f"{tmpdir}/autopin5.db",
                duckdb_path=f"{tmpdir}/autopin5.duckdb",
            )
            set_config(config)
            ctrl = MemoryController(user_id="autopin_manual_user", provider="echo")

            # Manually pin first
            ctrl.pin("password is hunter2")
            # Then add message with same content — should not create duplicate auto-pin
            ctrl.add_message("user", "Remember, password is hunter2")

            pinned = ctrl.tier_manager.tier2.get_pinned("autopin_manual_user")
            password_pins = [p for p in pinned if "hunter2" in p.content]
            assert len(password_pins) == 1
            ctrl.close_session()
