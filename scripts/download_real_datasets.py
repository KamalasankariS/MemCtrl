"""Download real datasets from HuggingFace for task classifier training.

All data is real — no synthetic generation. Sources:
- medical:  lavita/ChatDoctor-HealthCareMagic-100k
- code:     pacovaldez/stackoverflow-questions
- writing:  euclaise/writingprompts
- tutoring: camel-ai/math
- general:  HuggingFaceH4/ultrachat_200k
"""

import json
import random
import hashlib
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset


N_PER_CLASS = 5000
SEED = 42


def _hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def _truncate(text: str, max_chars: int = 500) -> str:
    return text[:max_chars].strip()


def load_medical(n: int = N_PER_CLASS):
    print(f"\n[1/5] Loading medical (ChatDoctor-HealthCareMagic)...")
    try:
        ds = load_dataset(
            "lavita/ChatDoctor-HealthCareMagic-100k",
            split="train",
        )
        items = []
        for row in tqdm(ds, total=min(n * 2, len(ds)), desc="Medical"):
            text = (row.get("input") or "").strip()
            if len(text) > 30:
                items.append({
                    "text": _truncate(text),
                    "label": "medical",
                    "source": "ChatDoctor-HealthCareMagic",
                    "id": _hash(text),
                })
            if len(items) >= n:
                break
        print(f"  Loaded {len(items)} medical examples")
        return items
    except Exception as e:
        print(f"  Failed: {e}")
        return []


def load_code(n: int = N_PER_CLASS):
    print(f"\n[2/5] Loading code (StackOverflow questions)...")
    try:
        ds = load_dataset(
            "pacovaldez/stackoverflow-questions",
            split="train", streaming=True,
        )
        items = []
        for row in tqdm(ds, total=n, desc="Code"):
            text = (row.get("title") or "").strip()
            if len(text) > 20:
                items.append({
                    "text": _truncate(text),
                    "label": "code",
                    "source": "StackOverflow",
                    "id": _hash(text),
                })
            if len(items) >= n:
                break
        print(f"  Loaded {len(items)} code examples")
        return items
    except Exception as e:
        print(f"  Failed: {e}")
        return []


def load_writing(n: int = N_PER_CLASS):
    print(f"\n[3/5] Loading writing (WritingPrompts)...")
    try:
        ds = load_dataset(
            "euclaise/writingprompts",
            split="train", streaming=True,
        )
        items = []
        for row in tqdm(ds, total=n, desc="Writing"):
            text = (
                row.get("prompt")
                or row.get("title")
                or ""
            ).strip()
            if len(text) > 30:
                items.append({
                    "text": _truncate(text),
                    "label": "writing",
                    "source": "WritingPrompts",
                    "id": _hash(text),
                })
            if len(items) >= n:
                break
        print(f"  Loaded {len(items)} writing examples")
        return items
    except Exception as e:
        print(f"  Failed: {e}")
        return []


def load_tutoring(n: int = N_PER_CLASS):
    print(f"\n[4/5] Loading tutoring (math_qa)...")
    try:
        ds = load_dataset("allenai/math_qa", split="train")
        items = []
        for row in tqdm(ds, total=min(n * 2, len(ds)), desc="Tutoring"):
            text = (row.get("Problem") or "").strip()
            if len(text) > 30:
                items.append({
                    "text": _truncate(text),
                    "label": "tutoring",
                    "source": "MathQA",
                    "id": _hash(text),
                })
            if len(items) >= n:
                break
        print(f"  Loaded {len(items)} tutoring examples")
        return items
    except Exception as e:
        print(f"  Failed: {e}")
        # Fallback: GSM8K
        print("  Trying GSM8K fallback...")
        try:
            ds = load_dataset("gsm8k", "main", split="train")
            items = []
            for row in tqdm(ds, total=min(n, len(ds)), desc="GSM8K"):
                text = (row.get("question") or "").strip()
                if len(text) > 30:
                    items.append({
                        "text": _truncate(text),
                        "label": "tutoring",
                        "source": "GSM8K",
                        "id": _hash(text),
                    })
                if len(items) >= n:
                    break
            print(f"  Loaded {len(items)} tutoring examples")
            return items
        except Exception as e2:
            print(f"  Fallback also failed: {e2}")
            return []


def load_general(n: int = N_PER_CLASS):
    print(f"\n[5/5] Loading general (UltraChat)...")
    try:
        ds = load_dataset(
            "HuggingFaceH4/ultrachat_200k",
            split="train_sft", streaming=True,
        )
        items = []
        for row in tqdm(ds, total=n, desc="General"):
            text = (row.get("prompt") or "").strip()
            if len(text) > 20:
                items.append({
                    "text": _truncate(text),
                    "label": "general",
                    "source": "UltraChat",
                    "id": _hash(text),
                })
            if len(items) >= n:
                break
        print(f"  Loaded {len(items)} general examples")
        return items
    except Exception as e:
        print(f"  Failed: {e}")
        return []


def deduplicate(data):
    seen = set()
    unique = []
    for item in data:
        if item["id"] not in seen:
            seen.add(item["id"])
            unique.append(item)
    removed = len(data) - len(unique)
    if removed > 0:
        print(f"\nRemoved {removed} duplicates")
    return unique


def main():
    print("=" * 70)
    print("DOWNLOADING REAL DATASETS FOR TASK CLASSIFIER")
    print("=" * 70)

    random.seed(SEED)

    all_data = []
    all_data.extend(load_medical())
    all_data.extend(load_code())
    all_data.extend(load_writing())
    all_data.extend(load_tutoring())
    all_data.extend(load_general())

    all_data = deduplicate(all_data)
    random.shuffle(all_data)

    total = len(all_data)
    print(f"\nTotal: {total:,} examples")

    labels = {}
    for item in all_data:
        labels[item["label"]] = labels.get(item["label"], 0) + 1
    for label, count in sorted(labels.items()):
        print(f"  {label:12s}: {count:6,d}")

    train_end = int(0.8 * total)
    val_end = int(0.9 * total)

    dataset = {
        "train": all_data[:train_end],
        "val": all_data[train_end:val_end],
        "test": all_data[val_end:],
    }

    output_file = "data/task_classifier_data.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\nSaved to: {output_file}")
    print(
        f"Split: {len(dataset['train']):,} train / "
        f"{len(dataset['val']):,} val / "
        f"{len(dataset['test']):,} test"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
