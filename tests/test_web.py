"""Tests for Gradio web interface."""

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

    assert "Settings" in tab_names
    assert "Chat" in tab_names
    assert "Memory Control" in tab_names
    assert "Inspect" in tab_names


def test_create_app_has_password_field():
    from memctrl.interfaces.web import create_app
    app = create_app(provider="echo")

    has_password = False
    for block in app.blocks.values():
        if isinstance(block, gr.Textbox) and block.type == "password":
            has_password = True
            break

    assert has_password
