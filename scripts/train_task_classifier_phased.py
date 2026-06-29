"""3-phase training for task classifier: real-only, weighted mixing, final evaluation."""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))

from memctrl.ml.task_classifier import TaskClassifier

LABEL_NAMES = ["medical", "code", "writing", "tutoring", "general"]


class TaskDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = {label: i for i, label in enumerate(LABEL_NAMES)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(self.label2id[item["label"]]),
            "is_synthetic": item.get("is_synthetic", False),
        }


def load_data(data_file="data/task_classifier_data.json"):
    with open(data_file, "r") as f:
        data = json.load(f)
    return data["train"], data["val"], data["test"]


def filter_real_only(data):
    return [item for item in data if not item.get("is_synthetic", False)]


def compute_sample_weights(data, synthetic_weight=0.3):
    return [synthetic_weight if item.get("is_synthetic", False) else 1.0 for item in data]


def evaluate(model, dataloader, device, split_name="Val"):
    model.eval()
    all_preds = []
    all_labels = []
    all_is_synthetic = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {split_name}", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_is_synthetic.extend(batch["is_synthetic"].numpy())

    avg_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_is_synthetic = np.array(all_is_synthetic)

    overall_acc = (all_preds == all_labels).mean()

    real_mask = ~all_is_synthetic
    real_acc = (all_preds[real_mask] == all_labels[real_mask]).mean() if real_mask.sum() > 0 else 0.0

    synthetic_mask = all_is_synthetic
    synthetic_acc = (all_preds[synthetic_mask] == all_labels[synthetic_mask]).mean() if synthetic_mask.sum() > 0 else 0.0

    return {
        "loss": avg_loss,
        "overall_acc": overall_acc,
        "real_acc": real_acc,
        "synthetic_acc": synthetic_acc,
        "all_preds": all_preds.tolist(),
        "all_labels": all_labels.tolist(),
    }


def _train_epoch(model, train_loader, optimizer, criterion, device, desc):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=desc)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(train_loader)


def train_phase1_real_only(model, train_data, val_data, tokenizer, device, epochs=3, lr=2e-5):
    print("\n" + "=" * 70)
    print("PHASE 1: Training on REAL data only")
    print("=" * 70)

    train_real = filter_real_only(train_data)
    val_real = filter_real_only(val_data)
    print(f"Train: {len(train_real)} real examples")

    train_loader = DataLoader(TaskDataset(train_real, tokenizer), batch_size=16, shuffle=True)
    val_loader = DataLoader(TaskDataset(val_real, tokenizer), batch_size=32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        avg_loss = _train_epoch(model, train_loader, optimizer, criterion, device, f"Phase 1 - Epoch {epoch + 1}/{epochs}")
        val_results = evaluate(model, val_loader, device, "Val (Real)")
        print(f"Epoch {epoch + 1}: Train Loss={avg_loss:.4f}, Val Loss={val_results['loss']:.4f}, Val Acc={val_results['overall_acc']:.4f}")

    print("\nPhase 1 complete: Model anchored on real data")
    return model


def train_phase2_weighted_mixing(model, train_data, val_data, tokenizer, device, epochs=3, lr=1e-5, synthetic_weight=0.3):
    print("\n" + "=" * 70)
    print(f"PHASE 2: Weighted mixing (synthetic_weight={synthetic_weight})")
    print("=" * 70)

    sample_weights = compute_sample_weights(train_data, synthetic_weight)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(TaskDataset(train_data, tokenizer), batch_size=16, sampler=sampler)
    val_loader = DataLoader(TaskDataset(val_data, tokenizer), batch_size=32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        avg_loss = _train_epoch(model, train_loader, optimizer, criterion, device, f"Phase 2 - Epoch {epoch + 1}/{epochs}")
        val_results = evaluate(model, val_loader, device, "Val (Mixed)")
        print(f"Epoch {epoch + 1}: Train Loss={avg_loss:.4f}, Overall={val_results['overall_acc']:.4f}, Real={val_results['real_acc']:.4f}, Synthetic={val_results['synthetic_acc']:.4f}")

    print("\nPhase 2 complete: Coverage expanded")
    return model


def train_phase3_final_validation(model, test_data, tokenizer, device):
    print("\n" + "=" * 70)
    print("PHASE 3: Final Test Set Evaluation")
    print("=" * 70)

    test_loader = DataLoader(TaskDataset(test_data, tokenizer), batch_size=32)
    results = evaluate(model, test_loader, device, "Test")

    print(f"\nFINAL RESULTS:")
    print(f"  Test Loss:      {results['loss']:.4f}")
    print(f"  Test Overall:   {results['overall_acc']:.4f}")
    print(f"  Test Real:      {results['real_acc']:.4f}")
    print(f"  Test Synthetic: {results['synthetic_acc']:.4f}")

    print("\nPer-class Performance:")
    print(classification_report(results["all_labels"], results["all_preds"], target_names=LABEL_NAMES, digits=3))

    cm = confusion_matrix(results["all_labels"], results["all_preds"])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.title("Confusion Matrix - Test Set")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    Path("results").mkdir(exist_ok=True)
    plt.savefig("results/confusion_matrix.png", dpi=150)
    print("Confusion matrix saved to results/confusion_matrix.png")

    return results


def main():
    print("=" * 70)
    print("3-PHASE TASK CLASSIFIER TRAINING")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    train_data, val_data, test_data = load_data()

    train_real = sum(1 for item in train_data if not item.get("is_synthetic", False))
    train_synthetic = len(train_data) - train_real
    print(f"\nDataset: {train_real} real, {train_synthetic} synthetic ({train_synthetic / len(train_data) * 100:.1f}% synthetic)")

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TaskClassifier(model_name=model_name, num_classes=5)
    model.to(device)

    print(f"Model: {model_name}, Parameters: {sum(p.numel() for p in model.parameters()):,}")

    model = train_phase1_real_only(model, train_data, val_data, tokenizer, device, epochs=3, lr=2e-5)
    model = train_phase2_weighted_mixing(model, train_data, val_data, tokenizer, device, epochs=3, lr=1e-5, synthetic_weight=0.3)
    results = train_phase3_final_validation(model, test_data, tokenizer, device)

    save_path = "models/task_classifier.pt"
    model.save(save_path, tokenizer_name=model_name)

    print(f"\nTraining complete. Final test accuracy (real): {results['real_acc']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
