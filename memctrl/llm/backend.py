"""Pluggable LLM backend with support for Anthropic, HuggingFace, and Echo (testing)."""

import os
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 512) -> str:
        pass


class EchoLLM(LLMBackend):
    """Returns the user message back. Useful for testing without an API key."""

    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 512) -> str:
        for msg in reversed(messages):
            if msg["role"] == "user":
                return f"[Echo] {msg['content']}"
        return "[Echo] (no user message)"


class AnthropicLLM(LLMBackend):
    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: Optional[str] = None):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install the anthropic package: pip install anthropic")

        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 512) -> str:
        system_msg = None
        chat_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                chat_messages.append(msg)

        kwargs = {"model": self.model, "max_tokens": max_tokens, "messages": chat_messages}
        if system_msg:
            kwargs["system"] = system_msg

        response = self.client.messages.create(**kwargs)
        return response.content[0].text


class HuggingFaceLLM(LLMBackend):
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError("Install transformers and torch: pip install transformers torch")

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )

    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 512) -> str:
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

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


def create_llm_backend(provider: str = "auto", **kwargs) -> LLMBackend:
    """Factory function to create the appropriate LLM backend."""
    if provider == "anthropic":
        return AnthropicLLM(**kwargs)
    elif provider == "huggingface":
        return HuggingFaceLLM(**kwargs)
    elif provider == "echo":
        return EchoLLM()
    elif provider == "auto":
        if os.getenv("ANTHROPIC_API_KEY"):
            logger.info("Auto-detected Anthropic API key, using AnthropicLLM")
            return AnthropicLLM(**kwargs)
        logger.info("No API key found, falling back to EchoLLM")
        return EchoLLM()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
