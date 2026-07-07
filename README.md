[![CI](https://github.com/KamalasankariS/MemCtrl/actions/workflows/ci.yml/badge.svg)](https://github.com/KamalasankariS/MemCtrl/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/memctrl-llm.svg)](https://pypi.org/project/memctrl-llm/)

# MemCtrl

**Stop paying for context window upgrades. MemCtrl extends your LLM conversations 5.7x longer with 88% recall accuracy — for free.**

MemCtrl is a context budget optimizer SDK that sits between your app and any LLM API. It manages a 3-tier memory hierarchy, compresses old messages, preserves critical facts, and returns an optimized message list that fits your token budget.

**No API calls on your behalf. No hosted service. No fees. Your keys stay in your memory — never logged, never stored.**

## Why MemCtrl?

| Problem | Without MemCtrl | With MemCtrl |
|---------|----------------|--------------|
| Long conversations | Context window fills up, old messages lost | 5.7x longer conversations, same budget |
| Token costs | Pay for full context every call | 8-13% token savings via compression |
| Critical facts | Passwords, dosages, IDs vanish after compression | Auto-pinned, 100% preserved |
| Medical/code context | Treated same as casual chat | Task-aware retention (medical kept 7 days, code 3 days) |

### Eval Results

Tested with Ollama (llama3) across 3 real-world scenarios:

| Scenario | Messages | Recall | Token Savings |
|----------|----------|--------|---------------|
| Flask API Debugging | 40 | **100%** | 11% |
| ML Tutoring Session | 30 | **80%** | 13% |
| Medical Consultation | 24 | **80%** | — |
| **Overall** | **94** | **88%** | **5.7x endurance** |

## Install

```bash
pip install memctrl-llm            # Lightweight (~2MB), uses extractive compression
```

With neural compression (distilbart + embeddings):

```bash
pip install memctrl-llm[ml]          # Adds torch, transformers, sentence-transformers
```

With LLM backends:

```bash
pip install memctrl-llm[anthropic]   # For Claude
pip install memctrl-llm[openai]      # For GPT
pip install memctrl-llm[all]         # Everything
```

## Quick Start

### SDK API (recommended)

```python
import memctrl
from openai import OpenAI

client = OpenAI(api_key="sk-...")
mc = memctrl.MemoryController(user_id="dev1")

# Record messages as they happen
mc.add_message("user", "My DB connection is postgresql://admin:pass@localhost:5432/mydb")
mc.add_message("assistant", "Got it. I'll remember your connection string.")
mc.add_message("user", "Now help me write a migration script")

# Get optimized context for your next API call
messages = mc.optimize(max_tokens=4096)

# Pass directly to any LLM
response = client.chat.completions.create(model="gpt-4o", messages=messages)
```

### Zero-Code Wrapper

```python
import memctrl
from openai import OpenAI

client = OpenAI(api_key="sk-...")
wrapped = memctrl.wrap(client, max_tokens=4096)

# Use normally — memctrl manages memory automatically
response = wrapped.chat("Help me debug my Flask app")
response = wrapped.chat("The error is on line 42")
response = wrapped.chat("What was my DB connection string?")  # Still remembers
```

### Pin Critical Facts

```python
mc.pin("Patient is allergic to penicillin")
mc.pin("API key: sk-abc123", note="Production key")

# Pinned facts are always included in optimize() output
messages = mc.optimize(max_tokens=4096)
```

### Auto-Pin Suggestions

MemCtrl detects critical information and suggests pinning — it never auto-pins user-facing data without asking.

```python
mc.add_message("user", "The password is hunter2")

suggestions = mc.suggest_pins()
# [{"chunk_id": "...", "category": "credential",
#   "reason": "Detected credential — this looks important. Pin it?"}]

# User decides
mc.accept_pin_suggestion(suggestions[0]["chunk_id"])
```

During compression, critical values (credentials, medical data, connection strings) are automatically preserved so they survive summarization.

### Stale Memory Cleanup

```python
suggestions = mc.suggest_cleanup(stale_hours=24.0)
# [{"chunk_id": "...", "tokens": 45, "hours_stale": 36.2,
#   "reason": "Not accessed in 36.2 hours. Deleting saves 45 tokens."}]

# User decides what to delete
mc.accept_cleanup([s["chunk_id"] for s in suggestions])
```

### Context Budget Debugger

```python
report = mc.budget_report(max_tokens=4096)
# {
#   "total_used": 2847, "remaining": 1249, "usage_pct": 69.5,
#   "breakdown": {
#     "system_prompt": {"tokens": 12},
#     "pinned": {"tokens": 89, "count": 3},
#     "active": {"tokens": 1820, "count": 14},
#     "compressed": {"tokens": 926, "count": 8},
#   },
#   "recommendations": ["Plenty of budget remaining."]
# }
```

### Forget and Temporary

```python
# Soft-delete with confirmation
result = mc.forget("old project notes")
mc.forget_confirmed(result["matches"])

# Session-only memory (auto-deleted on close)
mc.temporary("Meeting at 3pm today")

# Restore from trash
mc.restore_from_trash(chunk_id)
```

### Cross-Session Persistence

Pins, trash, and audit logs persist to SQLite across sessions:

```python
# Session 1
mc = memctrl.MemoryController(user_id="dev1")
mc.pin("Deploy to us-east-1")
mc.close_session()

# Session 2 — pin is still there
mc = memctrl.MemoryController(user_id="dev1")
messages = mc.optimize(max_tokens=4096)  # Includes "Deploy to us-east-1"
```

## How It Works

```
User App ──► add_message() ──► TierManager ──► optimize() ──► LLM API
                                    │
                 ┌──────────────────┼──────────────────┐
                 │                  │                  │
          ┌──────┴──────┐   ┌──────┴──────┐   ┌──────┴──────┐
          │   Tier 0    │   │   Tier 1    │   │   Tier 2    │
          │   Active    │   │  Compressed │   │  Persistent │
          │  LRU evict  │   │  distilbart │   │  SQLite+FTS │
          │  OrderedDict│   │  + entities │   │  Embeddings │
          └─────────────┘   └─────────────┘   └─────────────┘
```

**Tier 0 (Active)** — Recent messages in full. LRU eviction, pinned chunks protected.

**Tier 1 (Compressed)** — Older messages summarized by local distilbart model (free, no API calls). Named entities (numbers, URLs, field names, medical values) are extracted and appended to summaries so they survive compression.

**Tier 2 (Persistent)** — All chunks stored in SQLite with FTS5 full-text search and sentence-transformer embeddings for semantic retrieval.

### Task-Aware Retention

MemCtrl classifies each message by domain and adjusts retention:

| Task Type | Retention Weight | Decay Period | Promotion Threshold |
|-----------|-----------------|-------------|-------------------|
| Medical | 1.5x | 7 days | 30 (easiest) |
| Code | 1.3x | 3 days | 35 |
| Tutoring | 1.2x | 2 days | 40 |
| Writing | 1.0x | 1 day | 45 |
| General | 0.8x | 12 hours | 50 (hardest) |

Medical data is kept 14x longer than general chat before eviction.

### Entity-Preserving Compression

When messages are compressed, MemCtrl extracts and preserves:
- Version numbers, formatted numbers, percentages
- URLs, connection strings, API keys
- Medical measurements, lab results, vitals
- Code identifiers (snake_case, CamelCase, function calls)
- Math notation, X-Headers

These are appended to summaries in brackets so the LLM can still reference them.

## BYOK (Bring Your Own Key)

MemCtrl supports multiple LLM backends. Keys are passed in memory only — never logged, never persisted.

```python
# Anthropic
mc = memctrl.MemoryController(provider="anthropic", api_key="sk-...")

# OpenAI
mc = memctrl.MemoryController(provider="openai", api_key="sk-...")

# Ollama (local, no key needed)
mc = memctrl.MemoryController(provider="ollama")

# Auto-detect (tries Anthropic → OpenAI → Ollama → Echo)
mc = memctrl.MemoryController(provider="auto")
```

## CLI

```bash
memctrl chat "Hello" --user dev1
memctrl pin "Allergic to penicillin" --user dev1
memctrl forget "old notes" --user dev1
memctrl show --user dev1 --category pinned
memctrl stats --user dev1
memctrl export --user dev1 --format json
```

## Configuration

Override defaults with a YAML file:

```yaml
# config.yaml
tier0_budget_gb: 4.0
tier1_budget_gb: 4.0
llm_provider: "auto"
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
tokenizer_model: "distilbert-base-uncased"
control_mode: "hybrid"
compression_ratio: 4.0
max_tokens_per_chunk: 512
max_context_tokens: 4096
```

```bash
export MEMCTRL_CONFIG=path/to/config.yaml
```

## Testing

```bash
pip install memctrl-llm[dev]
pytest tests/ -o addopts="" -q
```

160+ tests covering models, storage, tiers, controller, SDK, features, tokenizer, LLM backends, CLI, and web UI.

## License

MIT License
