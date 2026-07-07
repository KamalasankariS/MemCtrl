"""Token counting with optional transformers backend.

Falls back to word-based estimation when transformers is not installed.
"""

import logging

logger = logging.getLogger(__name__)

_tokenizer = None
_tokenizer_model = None
_use_fallback = False


def get_tokenizer(model_name: str = "distilbert-base-uncased"):
    global _tokenizer, _tokenizer_model, _use_fallback
    if _use_fallback:
        return None
    if _tokenizer is None or _tokenizer_model != model_name:
        try:
            from transformers import AutoTokenizer
            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            _tokenizer_model = model_name
        except ImportError:
            logger.info(
                "transformers not installed; using word-based token estimation. "
                "Install with: pip install memctrl-llm[ml]"
            )
            _use_fallback = True
            return None
    return _tokenizer


def count_tokens(text: str, model_name: str = "distilbert-base-uncased") -> int:
    tokenizer = get_tokenizer(model_name)
    if tokenizer is None:
        # ~1.3 tokens per word is a reasonable estimate for English text
        return int(len(text.split()) * 1.3)
    return len(tokenizer.encode(text, add_special_tokens=False))
