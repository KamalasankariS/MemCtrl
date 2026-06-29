import torch
import torch.nn as nn
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

from ..models import Chunk


class PolicyNetwork(nn.Module):
    """Predicts importance score for memory chunks based on task type."""

    def __init__(self, task_type: str, input_dim: int = 128):
        super().__init__()
        self.task_type = task_type
        self.input_dim = input_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.scorer = nn.Linear(128, 1)

    def forward(self, features):
        encoded = self.encoder(features)
        return torch.sigmoid(self.scorer(encoded))

    def predict_importance(self, chunk: Chunk, context: Optional[List[Chunk]] = None) -> float:
        self.eval()
        with torch.no_grad():
            features = self._extract_features(chunk, context)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            score = self.forward(features_tensor)
            return score.item()

    def _extract_features(self, chunk: Chunk, context: Optional[List[Chunk]] = None) -> List[float]:
        features = []

        age_seconds = (datetime.now() - chunk.timestamp).total_seconds()
        features.append(1.0 / (1.0 + age_seconds / 3600))

        chunk_types = ["medical", "code", "fact", "preference", "context", "conversation", "other"]
        features.extend(1.0 if chunk.chunk_type.value == t else 0.0 for t in chunk_types)

        features.append(min(chunk.tokens / 100.0, 1.0))

        medical_terms = [
            "allergy", "allergic", "medication", "diagnosis",
            "symptom", "doctor", "patient",
        ]
        has_medical = any(t in chunk.content.lower() for t in medical_terms)
        features.append(1.0 if has_medical else 0.0)

        code_patterns = ["def ", "class ", "import ", "function", "error", "bug", "```"]
        features.append(1.0 if any(p in chunk.content.lower() for p in code_patterns) else 0.0)

        features.append(min(chunk.access_count / 10.0, 1.0))
        features.append(1.0 if chunk.is_pinned else 0.0)

        # Pad to input_dim
        features.extend(0.0 for _ in range(self.input_dim - len(features)))
        return features[: self.input_dim]

    def save(self, path: str):
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "task_type": self.task_type,
                "input_dim": self.input_dim,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "PolicyNetwork":
        checkpoint = torch.load(path, map_location="cpu")
        model = cls(task_type=checkpoint["task_type"], input_dim=checkpoint["input_dim"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model


def train_policy_network(
    task_type: str,
    train_data: List[Dict],
    val_data: List[Dict],
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    save_path: str = None,
) -> PolicyNetwork:
    from torch.utils.data import DataLoader, TensorDataset

    train_features = torch.tensor(
        [d["features"] for d in train_data], dtype=torch.float32,
    )
    train_labels = torch.tensor(
        [d["label"] for d in train_data], dtype=torch.float32,
    ).unsqueeze(1)
    val_features = torch.tensor(
        [d["features"] for d in val_data], dtype=torch.float32,
    )
    val_labels = torch.tensor(
        [d["label"] for d in val_data], dtype=torch.float32,
    ).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(TensorDataset(val_features, val_labels), batch_size=batch_size)

    input_dim = train_features.shape[1]
    model = PolicyNetwork(task_type=task_type, input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for features, labels in train_loader:
            optimizer.zero_grad()
            scores = model(features)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in val_loader:
                scores = model(features)
                preds = (scores > 0.5).float()
                correct += (preds == labels).sum().item()
                total += len(labels)

        val_acc = correct / total
        print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)
        print(f"Model saved to {save_path}")

    return model
