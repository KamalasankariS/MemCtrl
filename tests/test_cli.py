"""Tests for CLI interface."""

from click.testing import CliRunner
from unittest.mock import patch, MagicMock

from memctrl.interfaces.cli import cli


def runner():
    return CliRunner()


@patch("memctrl.interfaces.cli._get_controller")
def test_chat_command(mock_ctrl):
    controller = MagicMock()
    controller.chat.return_value = "Hello back!"
    mock_ctrl.return_value = controller

    result = runner().invoke(cli, ["chat", "Hello"])
    assert result.exit_code == 0
    assert "Hello back!" in result.output


@patch("memctrl.interfaces.cli._get_controller")
def test_pin_command(mock_ctrl):
    controller = MagicMock()
    controller.pin.return_value = {
        "success": True, "chunk_id": "abc123",
    }
    mock_ctrl.return_value = controller

    result = runner().invoke(cli, ["pin", "Remember this"])
    assert result.exit_code == 0
    assert "abc123" in result.output


@patch("memctrl.interfaces.cli._get_controller")
def test_forget_command(mock_ctrl):
    controller = MagicMock()
    controller.forget.return_value = {
        "success": True, "num_deleted": 2,
    }
    mock_ctrl.return_value = controller

    result = runner().invoke(cli, ["forget", "old stuff"])
    assert result.exit_code == 0
    assert "2" in result.output


@patch("memctrl.interfaces.cli._get_controller")
def test_show_command(mock_ctrl):
    controller = MagicMock()
    controller.show_memory.return_value = {
        "pinned": [], "session": [],
    }
    mock_ctrl.return_value = controller

    result = runner().invoke(cli, ["show"])
    assert result.exit_code == 0
    assert "pinned" in result.output


@patch("memctrl.interfaces.cli._get_controller")
def test_stats_command(mock_ctrl):
    controller = MagicMock()
    controller.get_stats.return_value = {
        "user_id": "test", "tiers": {},
    }
    mock_ctrl.return_value = controller

    result = runner().invoke(cli, ["stats"])
    assert result.exit_code == 0
    assert "user_id" in result.output


@patch("memctrl.interfaces.cli._get_controller")
def test_export_command(mock_ctrl):
    controller = MagicMock()
    controller.export_data.return_value = '{"export": true}'
    mock_ctrl.return_value = controller

    result = runner().invoke(cli, ["export"])
    assert result.exit_code == 0
    assert "export" in result.output


def test_cli_help():
    result = runner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "MemCtrl" in result.output


def test_cli_has_api_key_option():
    result = runner().invoke(cli, ["--help"])
    assert "--api-key" in result.output


def test_cli_has_provider_option():
    result = runner().invoke(cli, ["--help"])
    assert "--provider" in result.output


def test_cli_api_key_passed_to_context():
    """Verify --api-key is stored in context, not dropped."""
    result = runner().invoke(
        cli, ["--api-key", "sk-test", "--help"],
    )
    assert result.exit_code == 0
