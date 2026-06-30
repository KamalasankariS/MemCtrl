"""Pluggable LLM backend with BYOK support.

Supports Anthropic, OpenAI, Ollama (local), HuggingFace, and Echo (testing).
API keys are passed in memory only — never logged, never persisted.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Sanitize: never log API keys
_REDACTED = "[REDACTED]"


class LLMBackend(ABC):
    """Base class for all LLM backends."""

    provider_name: str = "base"

    @abstractmethod
    def generate(
        self, messages: List[Dict[str, str]], max_tokens: int = 512,
    ) -> str:
        pass

    def is_available(self) -> bool:
        return True


class EchoLLM(LLMBackend):
    """Returns the user message back. For testing without an API key."""

    provider_name = "echo"

    def generate(
        self, messages: List[Dict[str, str]], max_tokens: int = 512,
    ) -> str:
        for msg in reversed(messages):
            if msg["role"] == "user":
                return f"[Echo] {msg['content']}"
        return "[Echo] (no user message)"


class AnthropicLLM(LLMBackend):
    """Anthropic Claude backend. Requires an API key."""

    provider_name = "anthropic"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "pip install anthropic"
            )

        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "Anthropic API key required. Pass api_key= or "
                "set ANTHROPIC_API_KEY env var."
            )
        self.model = model
        self.client = anthropic.Anthropic(api_key=key)

    def generate(
        self, messages: List[Dict[str, str]], max_tokens: int = 512,
    ) -> str:
        system_msg = None
        chat_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                chat_messages.append(msg)

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": chat_messages,
        }
        if system_msg:
            kwargs["system"] = system_msg

        response = self.client.messages.create(**kwargs)
        return response.content[0].text


class OpenAILLM(LLMBackend):
    """OpenAI-compatible backend. Works with OpenAI, Azure, and
    any API that follows the OpenAI chat completions format."""

    provider_name = "openai"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
    ):
        try:
            import openai
        except ImportError:
            raise ImportError(
                "pip install openai"
            )

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key required. Pass api_key= or "
                "set OPENAI_API_KEY env var."
            )
        self.model = model
        kwargs: Dict = {"api_key": key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = openai.OpenAI(**kwargs)

    def generate(
        self, messages: List[Dict[str, str]], max_tokens: int = 512,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


class OllamaLLM(LLMBackend):
    """Ollama backend. Fully local, no API key needed.
    Requires Ollama running at localhost:11434."""

    provider_name = "ollama"

    def __init__(
        self,
        model: str = "llama3.2",
        host: str = "http://localhost:11434",
    ):
        self.model = model
        self.host = host.rstrip("/")

    def is_available(self) -> bool:
        try:
            import urllib.request
            req = urllib.request.Request(
                f"{self.host}/api/tags", method="GET",
            )
            urllib.request.urlopen(req, timeout=2)
            return True
        except Exception:
            return False

    def generate(
        self, messages: List[Dict[str, str]], max_tokens: int = 512,
    ) -> str:
        import urllib.request
        import json

        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["message"]["content"]
        except Exception as e:
            raise ConnectionError(
                f"Ollama request failed. Is Ollama running at "
                f"{self.host}? Error: {e}"
            )


class HuggingFaceLLM(LLMBackend):
    """Local HuggingFace model. Requires GPU + model download."""

    provider_name = "huggingface"

    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError(
                "pip install transformers torch"
            )

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto",
        )

    def generate(
        self, messages: List[Dict[str, str]], max_tokens: int = 512,
    ) -> str:
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"<<SYS>>{content}<</SYS>>\n"
            elif role == "user":
                prompt += f"[INST] {content} [/INST]\n"
            elif role == "assistant":
                prompt += f"{content}\n"

        inputs = self.tokenizer(
            prompt, return_tensors="pt",
        ).to(self.model.device)
        outputs = self.model.generate(
            **inputs, max_new_tokens=max_tokens,
            do_sample=True, temperature=0.7,
        )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(
            generated, skip_special_tokens=True,
        )


def create_llm_backend(
    provider: str = "auto",
    api_key: Optional[str] = None,
    **kwargs,
) -> LLMBackend:
    """Create an LLM backend. API key is passed in memory only.

    Args:
        provider: "anthropic", "openai", "ollama", "huggingface",
                  "echo", or "auto" (tries each in order).
        api_key: API key for the provider. Never logged or persisted.
        **kwargs: Additional provider-specific arguments
                  (model, base_url, host, etc).
    """
    if provider == "anthropic":
        return AnthropicLLM(api_key=api_key, **kwargs)
    elif provider == "openai":
        return OpenAILLM(api_key=api_key, **kwargs)
    elif provider == "ollama":
        return OllamaLLM(**kwargs)
    elif provider == "huggingface":
        return HuggingFaceLLM(**kwargs)
    elif provider == "echo":
        return EchoLLM()
    elif provider == "auto":
        return _auto_detect(api_key, **kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def _auto_detect(
    api_key: Optional[str] = None, **kwargs,
) -> LLMBackend:
    """Try providers in order: Anthropic -> OpenAI -> Ollama -> Echo."""
    # Anthropic
    anthropic_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        logger.info("Using Anthropic backend")
        return AnthropicLLM(api_key=anthropic_key, **kwargs)

    # OpenAI
    openai_key = api_key or os.getenv("OPENAI_API_KEY")
    if openai_key:
        logger.info("Using OpenAI backend")
        return OpenAILLM(api_key=openai_key, **kwargs)

    # Ollama (local, no key needed)
    ollama = OllamaLLM()
    if ollama.is_available():
        logger.info("Using Ollama backend (local)")
        return ollama

    # Fallback
    logger.info("No LLM backend available, using Echo mode")
    return EchoLLM()
