"""Real token counting using transformers AutoTokenizer."""

from transformers import AutoTokenizer

_tokenizer = None
_tokenizer_model = None


def get_tokenizer(model_name: str = "distilbert-base-uncased") -> AutoTokenizer:
    global _tokenizer, _tokenizer_model
    if _tokenizer is None or _tokenizer_model != model_name:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _tokenizer_model = model_name
    return _tokenizer


def count_tokens(text: str, model_name: str = "distilbert-base-uncased") -> int:
    tokenizer = get_tokenizer(model_name)
    return len(tokenizer.encode(text, add_special_tokens=False))
