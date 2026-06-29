[![CI](https://github.com/KamalasankariS/MemCtrl/actions/workflows/ci.yml/badge.svg)](https://github.com/KamalasankariS/MemCtrl/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

# MemCtrl

**Task-aware memory management for long-context language models.**

MemCtrl gives LLMs a 3-tier memory hierarchy (GPU, RAM, disk) so conversations can run indefinitely on commodity hardware. A task classifier learns what to keep in active context, while users retain explicit control to pin, forget, or mark information as temporary.

## Features

- **3-tier memory hierarchy** — active GPU context (LRU), compressed RAM (extractive summarization), persistent disk (SQLite + FTS5)
- **Semantic search** — sentence-transformer embeddings for context retrieval across sessions
- **Task-aware policies** — DistilBERT classifier routes memory decisions by domain (medical, code, writing, tutoring, general)
- **User control** — pin important facts, forget sensitive data, mark things as session-only
- **Pluggable LLM backend** — Anthropic Claude, HuggingFace models, or Echo mode for testing
- **CLI and web UI** — Click CLI with interactive mode, Gradio web interface
- **Privacy-first** — all data stored locally in SQLite, no external telemetry

## Installation

```bash
git clone https://github.com/KamalasankariS/MemCtrl.git
cd MemCtrl
pip install -e .
```

## Quick Start

### Python API

```python
from memctrl import MemoryController

controller = MemoryController(user_id="kamala")

# Chat (auto-manages memory across tiers)
response = controller.chat("Help me debug this Python error")

# Pin to permanent memory
controller.pin("I prefer TypeScript over JavaScript", note="Language preference")

# Temporary (session-only, auto-deleted on close)
controller.temporary("Meeting at 3pm today")

# Forget matching content
controller.forget("old project notes", confirm=False)

# Inspect memory state
memory = controller.show_memory(category="all")
stats = controller.get_stats()

# Export all user data
data = controller.export_data(format="json")

controller.close_session()
```

### CLI

```bash
# Single message
memctrl chat "Hello" --user kamala

# Interactive session
memctrl interactive --user kamala

# Memory operations
memctrl pin "Allergic to penicillin" --user kamala --note "Medical"
memctrl forget "old notes" --user kamala
memctrl show --user kamala --category pinned
memctrl stats --user kamala

# Export data
memctrl export --user kamala --format json
```

### Web UI

```bash
memctrl start --port 7860
# Opens Gradio interface at http://localhost:7860
```

The web UI has three tabs:
- **Chat** — conversation with automatic memory management
- **Memory Control** — pin, forget, or add temporary content
- **Inspect** — view stored memory and tier statistics

## LLM Backend

MemCtrl auto-detects available backends:

| Provider | Setup | Use case |
|----------|-------|----------|
| **Anthropic** | Set `ANTHROPIC_API_KEY` env var | Production |
| **HuggingFace** | Requires GPU + model download | Local/research |
| **Echo** | No setup needed | Testing/development |

Override with the `--provider` flag or `llm_provider` in config:

```bash
memctrl chat "Hello" --provider echo
```

```python
from memctrl.llm.backend import create_llm_backend
llm = create_llm_backend("anthropic")
controller = MemoryController(user_id="kamala", llm=llm)
```

## Architecture

```
                    ┌─────────────────┐
                    │ MemoryController│
                    │   (Public API)  │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │   TierManager   │
                    │ promote/demote  │
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
   ┌──────┴──────┐   ┌──────┴──────┐   ┌──────┴──────┐
   │  Tier 0     │   │  Tier 1     │   │  Tier 2     │
   │  GPU/Active │   │  RAM/Compr. │   │  Disk/Perm. │
   │  LRU evict  │   │  Extractive │   │  SQLite+FTS │
   │  OrderedDict│   │  summarize  │   │  Embeddings │
   └─────────────┘   └─────────────┘   └─────────────┘
```

**Tier 0 (GPU)** — Active context window. LRU eviction with pinned-chunk protection. Fastest access.

**Tier 1 (RAM)** — Compressed storage. Extractive summarization picks the most informative sentences. Chunks promoted back to Tier 0 on access.

**Tier 2 (Disk)** — Persistent SQLite with FTS5 full-text search and sentence-transformer embeddings for semantic retrieval. All chunks are stored here as ground truth.

## Project Structure

```
memctrl/
├── __init__.py              # Public API exports
├── config.py                # Configuration (dataclass + YAML)
├── controller.py            # MemoryController — main entry point
├── tokenizer.py             # Real token counting (AutoTokenizer)
├── core/
│   └── tiers.py             # Tier0_GPU, Tier1_RAM, Tier2_Disk, TierManager
├── llm/
│   └── backend.py           # LLMBackend ABC + Anthropic/HuggingFace/Echo
├── ml/
│   ├── task_classifier.py   # DistilBERT 5-class task classifier
│   ├── policy_network.py    # Chunk importance prediction
│   └── hindsight_labeler.py # Training label generation
├── models/
│   ├── chunk.py             # Chunk dataclass (content, tokens, priority)
│   ├── session.py           # Session tracking
│   └── user.py              # User preferences and pinned chunks
├── storage/
│   ├── sqlite_store.py      # SQLite + FTS5 persistent storage
│   └── embedding_store.py   # Sentence-transformer semantic search
├── interfaces/
│   ├── cli.py               # Click CLI
│   └── web.py               # Gradio web UI
scripts/
├── download_real_datasets.py
├── prepare_task_classifier_data.py
├── train_task_classifier.py
└── train_task_classifier_phased.py   # 3-phase training
tests/                        # 76 tests
```

## Training

To train the task classifier on real + synthetic data, see [TRAINING.md](TRAINING.md).

## Configuration

Default config is built in. Override with a YAML file:

```bash
export MEMCTRL_CONFIG=path/to/config.yaml
```

```yaml
tier0_budget_gb: 4.0
tier1_budget_gb: 4.0
llm_provider: "auto"
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
tokenizer_model: "distilbert-base-uncased"
control_mode: "hybrid"       # automatic, hybrid, or manual
compression_ratio: 4.0
max_tokens_per_chunk: 512
max_context_tokens: 4096
```

## Testing

```bash
pip install -e .
pytest tests/ -o addopts="" -q
```

76 tests covering models, storage, tiers, controller, tokenizer, LLM backends, embedding search, CLI, and web UI.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- transformers, sentence-transformers
- SQLite (built-in)
- Optional: `anthropic` (for Claude backend), `gradio` (for web UI)

## License

MIT License
