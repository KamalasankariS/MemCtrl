"""Tests for Gradio web interface."""

import pytest
from unittest.mock import patch, MagicMock
import gradio as gr


def test_create_app_returns_blocks():
    from memctrl.interfaces.web import create_app
    app = create_app(provider="echo")
    assert isinstance(app, gr.Blocks)


def test_create_app_has_tabs():
    from memctrl.interfaces.web import create_app
    app = create_app(provider="echo")

    tab_names = []
    for block in app.blocks.values():
        if isinstance(block, gr.Tab):
            tab_names.append(block.label)

    assert "Chat" in tab_names
    assert "Memory Control" in tab_names
    assert "Inspect" in tab_names
