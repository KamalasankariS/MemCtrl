import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Dict, Tuple, Optional
from pathlib import Path


class TaskClassifier(nn.Module):
    TASK_TYPES = ["medical", "code", "writing", "tutoring", "general"]

    def __init__(self, model_name: str = "distilbert-base-uncased", num_classes: int = 5):
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes

        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

        self.label2id = {label: i for i, label in enumerate(self.TASK_TYPES)}
        self.id2label = {i: label for i, label in enumerate(self.TASK_TYPES)}

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled_output)

    def predict(
        self,
        text: str,
        tokenizer: Optional[AutoTokenizer] = None,
        device: str = "cpu",
        return_probs: bool = False,
    ) -> Tuple[str, Optional[Dict[str, float]]]:
        self.eval()

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        with torch.no_grad():
            encoding = tokenizer(
                text,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            logits = self.forward(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)[0]
            pred_idx = torch.argmax(probs).item()
            pred_label = self.id2label[pred_idx]

            if return_probs:
                prob_dict = {label: probs[idx].item() for idx, label in self.id2label.items()}
                return pred_label, prob_dict

            return pred_label, None

    def save(self, path: str, tokenizer_name: Optional[str] = None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "model_name": self.model_name,
                "tokenizer_name": tokenizer_name or self.model_name,
                "num_classes": self.num_classes,
                "label2id": self.label2id,
                "id2label": self.id2label,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "TaskClassifier":
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            model_name=checkpoint["model_name"],
            num_classes=checkpoint["num_classes"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model


def load_task_classifier(
    path: str = "models/task_classifier.pt", device: str = "cpu",
) -> TaskClassifier:
    return TaskClassifier.load(path, device)
