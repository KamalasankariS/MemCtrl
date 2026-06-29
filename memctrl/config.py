import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
import yaml


@dataclass
class MemCtrlConfig:
    tier0_budget_gb: float = 4.0
    tier1_budget_gb: float = 4.0
    tier2_budget_gb: float = float("inf")

    llm_model: str = "meta-llama/Llama-2-7b-chat-hf"
    llm_provider: str = "auto"  # "auto", "anthropic", "huggingface", "echo"
    task_classifier_path: str = "models/task_classifier.pt"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer_model: str = "distilbert-base-uncased"

    policy_medical_path: str = "models/policy_medical.pt"
    policy_code_path: str = "models/policy_code.pt"
    policy_writing_path: str = "models/policy_writing.pt"
    policy_tutoring_path: str = "models/policy_tutoring.pt"
    policy_general_path: str = "models/policy_general.pt"

    data_dir: str = "data/user_data"
    sqlite_path: str = "data/user_data/memory.db"
    duckdb_path: str = "data/user_data/embeddings.duckdb"

    compression_ratio: float = 4.0
    eviction_threshold: float = 0.9
    importance_threshold: float = 0.5

    control_mode: str = "hybrid"

    max_tokens_per_chunk: int = 512
    max_context_tokens: int = 4096

    task_types: list = field(default_factory=lambda: ["medical", "code", "writing", "tutoring", "general"])

    def ensure_directories(self):
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path("models").mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, path: str) -> "MemCtrlConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str):
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    def get_tier0_tokens(self) -> int:
        bytes_per_token = 2
        tokens = int(self.tier0_budget_gb * 1024**3 / bytes_per_token)
        return min(tokens, self.max_context_tokens)

    def get_tier1_tokens(self) -> int:
        return int(self.get_tier0_tokens() * self.compression_ratio)


_config: Optional[MemCtrlConfig] = None


def get_config() -> MemCtrlConfig:
    global _config
    if _config is None:
        config_path = os.getenv("MEMCTRL_CONFIG")
        if config_path and os.path.exists(config_path):
            _config = MemCtrlConfig.from_yaml(config_path)
        else:
            _config = MemCtrlConfig()
    return _config


def set_config(config: MemCtrlConfig):
    global _config
    _config = config
