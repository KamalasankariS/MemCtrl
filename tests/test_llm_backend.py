"""Tests for LLM backend."""

import pytest
from unittest.mock import patch, MagicMock

from memctrl.llm.backend import (
    EchoLLM,
    AnthropicLLM,
    HuggingFaceLLM,
    LLMBackend,
    create_llm_backend,
)


def test_echo_llm_returns_user_message():
    llm = EchoLLM()
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello there"},
    ]
    response = llm.generate(messages)
    assert "Hello there" in response
    assert "[Echo]" in response


def test_echo_llm_no_user_message():
    llm = EchoLLM()
    messages = [{"role": "system", "content": "System only"}]
    response = llm.generate(messages)
    assert "[Echo]" in response


def test_echo_llm_uses_last_user_message():
    llm = EchoLLM()
    messages = [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Response"},
        {"role": "user", "content": "Second"},
    ]
    response = llm.generate(messages)
    assert "Second" in response


def test_create_llm_backend_echo():
    llm = create_llm_backend("echo")
    assert isinstance(llm, EchoLLM)


@patch.dict("os.environ", {}, clear=True)
def test_create_llm_backend_auto_no_key():
    llm = create_llm_backend("auto")
    assert isinstance(llm, EchoLLM)


@patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
@patch("memctrl.llm.backend.AnthropicLLM.__init__", return_value=None)
def test_create_llm_backend_auto_with_key(mock_init):
    llm = create_llm_backend("auto")
    assert isinstance(llm, AnthropicLLM)


def test_create_llm_backend_unknown():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        create_llm_backend("nonexistent")


def test_llm_backend_is_abstract():
    with pytest.raises(TypeError):
        LLMBackend()


def test_echo_llm_max_tokens_ignored():
    llm = EchoLLM()
    messages = [{"role": "user", "content": "Test"}]
    r1 = llm.generate(messages, max_tokens=10)
    r2 = llm.generate(messages, max_tokens=1000)
    assert r1 == r2


# -- HuggingFace LLM mock tests --


def test_huggingface_llm_generate():
    """Test HuggingFaceLLM.generate builds prompt and calls model correctly."""
    import torch

    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    llm = HuggingFaceLLM.__new__(HuggingFaceLLM)
    llm.model_name = "test-model"
    llm.tokenizer = mock_tokenizer
    llm.model = mock_model

    mock_input_ids = torch.tensor([[1, 2, 3]])
    mock_inputs = MagicMock()
    mock_inputs.__getitem__ = lambda self, key: {"input_ids": mock_input_ids}[key]
    mock_inputs.to.return_value = mock_inputs
    mock_tokenizer.return_value = mock_inputs

    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    mock_model.device = "cpu"
    mock_tokenizer.decode.return_value = "Generated response"

    messages = [
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "Hello"},
    ]

    response = llm.generate(messages, max_tokens=50)
    assert response == "Generated response"
    mock_tokenizer.decode.assert_called_once()


# -- Anthropic LLM mock tests --


def test_anthropic_llm_generate():
    """Test AnthropicLLM.generate calls client correctly."""
    llm = AnthropicLLM.__new__(AnthropicLLM)
    llm.model = "claude-sonnet-4-20250514"

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Anthropic response")]
    mock_client.messages.create.return_value = mock_response
    llm.client = mock_client

    messages = [
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "Hello"},
    ]

    response = llm.generate(messages, max_tokens=100)
    assert response == "Anthropic response"

    call_kwargs = mock_client.messages.create.call_args[1]
    assert call_kwargs["system"] == "Be helpful"
    assert call_kwargs["messages"] == [{"role": "user", "content": "Hello"}]
    assert call_kwargs["max_tokens"] == 100


def test_anthropic_llm_no_system():
    """Test AnthropicLLM without system message."""
    llm = AnthropicLLM.__new__(AnthropicLLM)
    llm.model = "claude-sonnet-4-20250514"

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Response")]
    mock_client.messages.create.return_value = mock_response
    llm.client = mock_client

    messages = [{"role": "user", "content": "Hello"}]
    llm.generate(messages)

    call_kwargs = mock_client.messages.create.call_args[1]
    assert "system" not in call_kwargs
