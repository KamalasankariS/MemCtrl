"""Tests for LLM backend with BYOK support."""

from unittest.mock import patch, MagicMock

import pytest

from memctrl.llm.backend import (
    EchoLLM,
    AnthropicLLM,
    OpenAILLM,
    OllamaLLM,
    HuggingFaceLLM,
    LLMBackend,
    create_llm_backend,
)


# -- Echo --

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


def test_echo_llm_max_tokens_ignored():
    llm = EchoLLM()
    messages = [{"role": "user", "content": "Test"}]
    r1 = llm.generate(messages, max_tokens=10)
    r2 = llm.generate(messages, max_tokens=1000)
    assert r1 == r2


def test_echo_provider_name():
    assert EchoLLM().provider_name == "echo"


# -- Factory --

def test_create_llm_backend_echo():
    llm = create_llm_backend("echo")
    assert isinstance(llm, EchoLLM)


@patch.dict("os.environ", {}, clear=True)
def test_create_auto_no_keys_no_ollama():
    with patch.object(OllamaLLM, "is_available", return_value=False):
        llm = create_llm_backend("auto")
        assert isinstance(llm, EchoLLM)


@patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"})
@patch("memctrl.llm.backend.AnthropicLLM.__init__", return_value=None)
def test_create_auto_with_anthropic_key(mock_init):
    llm = create_llm_backend("auto")
    assert isinstance(llm, AnthropicLLM)


@patch.dict("os.environ", {}, clear=True)
def test_create_auto_ollama_available():
    with patch.object(OllamaLLM, "is_available", return_value=True):
        llm = create_llm_backend("auto")
        assert isinstance(llm, OllamaLLM)


def test_create_unknown_provider():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        create_llm_backend("nonexistent")


def test_llm_backend_is_abstract():
    with pytest.raises(TypeError):
        LLMBackend()


# -- BYOK: api_key parameter --

@patch("memctrl.llm.backend.AnthropicLLM.__init__", return_value=None)
def test_create_anthropic_with_byok(mock_init):
    llm = create_llm_backend("anthropic", api_key="sk-my-key")
    assert isinstance(llm, AnthropicLLM)
    mock_init.assert_called_once_with(api_key="sk-my-key")


@patch("memctrl.llm.backend.OpenAILLM.__init__", return_value=None)
def test_create_openai_with_byok(mock_init):
    llm = create_llm_backend("openai", api_key="sk-openai-key")
    assert isinstance(llm, OpenAILLM)
    mock_init.assert_called_once_with(api_key="sk-openai-key")


@patch.dict("os.environ", {}, clear=True)
def test_anthropic_requires_key():
    with pytest.raises((ValueError, ImportError)):
        AnthropicLLM(api_key=None)


@patch.dict("os.environ", {}, clear=True)
def test_openai_requires_key():
    with pytest.raises((ValueError, ImportError)):
        OpenAILLM(api_key=None)


# -- Ollama --

def test_ollama_provider_name():
    llm = OllamaLLM()
    assert llm.provider_name == "ollama"


def test_ollama_default_host():
    llm = OllamaLLM()
    assert "localhost" in llm.host
    assert "11434" in llm.host


def test_ollama_custom_model():
    llm = OllamaLLM(model="mistral")
    assert llm.model == "mistral"


# -- Anthropic mock --

def test_anthropic_generate():
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
    assert call_kwargs["messages"] == [
        {"role": "user", "content": "Hello"},
    ]
    assert call_kwargs["max_tokens"] == 100


def test_anthropic_no_system():
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


# -- OpenAI mock --

def test_openai_generate():
    llm = OpenAILLM.__new__(OpenAILLM)
    llm.model = "gpt-4o-mini"

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "OpenAI response"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    llm.client = mock_client

    messages = [{"role": "user", "content": "Hello"}]
    response = llm.generate(messages, max_tokens=50)
    assert response == "OpenAI response"


# -- HuggingFace mock --

def test_huggingface_generate():
    import torch

    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    llm = HuggingFaceLLM.__new__(HuggingFaceLLM)
    llm.model_name = "test-model"
    llm.tokenizer = mock_tokenizer
    llm.model = mock_model

    mock_input_ids = torch.tensor([[1, 2, 3]])
    mock_inputs = MagicMock()
    mock_inputs.__getitem__ = (
        lambda self, key: {"input_ids": mock_input_ids}[key]
    )
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
