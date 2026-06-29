"""Check dataset for issues: overlaps, label leakage, template patterns."""

import json
from collections import Counter

with open("data/task_classifier_data.json", "r") as f:
    data = json.load(f)

print("=" * 60)
print("DATASET DIAGNOSTIC")
print("=" * 60)

train_texts = set(item["text"] for item in data["train"])
val_texts = set(item["text"] for item in data["val"])
test_texts = set(item["text"] for item in data["test"])

print(f"\nSplit sizes:")
print(f"  Train: {len(data['train'])}")
print(f"  Val:   {len(data['val'])}")
print(f"  Test:  {len(data['test'])}")

print(f"\nOverlaps (should be 0):")
print(f"  Train/Val:  {len(train_texts & val_texts)}")
print(f"  Train/Test: {len(train_texts & test_texts)}")
print(f"  Val/Test:   {len(val_texts & test_texts)}")

train_labels = Counter(item["label"] for item in data["train"])
print(f"\nTrain label distribution:")
for label, count in sorted(train_labels.items()):
    print(f"  {label:12s}: {count:6,d}")

print(f"\nChecking for label leakage (first 100)...")
leakage_count = 0
for item in data["train"][:100]:
    if item["label"] in item["text"].lower():
        leakage_count += 1
        print(f"  LEAK: '{item['label']}' in: {item['text'][:80]}...")
print(f"Label leakage: {leakage_count}/100")

train_synthetic = sum(1 for item in data["train"] if item.get("is_synthetic", False))
train_real = len(data["train"]) - train_synthetic
print(f"\nComposition:")
print(f"  Real:      {train_real:6,d} ({train_real / len(data['train']) * 100:.1f}%)")
print(f"  Synthetic: {train_synthetic:6,d} ({train_synthetic / len(data['train']) * 100:.1f}%)")

print(f"\nTop start patterns:")
for label_name in ("medical", "code"):
    starts = Counter()
    for item in data["train"]:
        if item["label"] == label_name:
            starts[" ".join(item["text"].split()[:3])] += 1
    print(f"  {label_name}:")
    for pattern, count in starts.most_common(3):
        print(f"    '{pattern}...': {count}")

print("=" * 60)
