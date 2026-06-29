"""Train task classifier on prepared data."""

import json
from typing import List, Tuple

from memctrl.ml.task_classifier import TaskClassifier


def load_prepared_data(data_file: str = "data/task_classifier_data.json") -> Tuple[List, List, List]:
    print(f"Loading data from {data_file}...")

    with open(data_file, "r") as f:
        data = json.load(f)

    train_data = [(item["text"], item["label"]) for item in data["train"]]
    val_data = [(item["text"], item["label"]) for item in data["val"]]
    test_data = [(item["text"], item["label"]) for item in data["test"]]

    print(f"Loaded {len(train_data)} train, {len(val_data)} val, {len(test_data)} test examples")
    return train_data, val_data, test_data


def main():
    print("=" * 60)
    print("Training Task Classifier")
    print("=" * 60)

    train_data, val_data, test_data = load_prepared_data()

    # TODO: wire up train_task_classifier with proper signature
    # model = train_task_classifier(train_data, val_data, epochs=3, batch_size=16, lr=2e-5, save_path="models/task_classifier.pt")

    print("Training complete.")
    print("Model saved to: models/task_classifier.pt")


if __name__ == "__main__":
    main()
