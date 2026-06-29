import pytest
import torch
from memctrl.ml.task_classifier import TaskClassifier
from transformers import AutoTokenizer


def test_task_classifier_init():
    model = TaskClassifier(model_name="distilbert-base-uncased", num_classes=5)
    assert model.num_classes == 5
    assert model.model_name == "distilbert-base-uncased"
    assert len(model.label2id) == 5
    assert "medical" in model.label2id
    assert "code" in model.label2id


def test_task_classifier_forward():
    model = TaskClassifier()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    encoding = tokenizer(
        "I have chest pain and fever",
        return_tensors="pt",
        padding="max_length",
        max_length=128,
        truncation=True,
    )
    logits = model(encoding["input_ids"], encoding["attention_mask"])
    assert logits.shape == (1, 5)


def test_task_classifier_predict():
    model = TaskClassifier()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    pred_label, probs = model.predict("I have chest pain and fever", tokenizer, return_probs=True)
    assert pred_label in ["medical", "code", "writing", "tutoring", "general"]
    assert isinstance(probs, dict)
    assert len(probs) == 5
    assert all(0 <= v <= 1 for v in probs.values())
    assert abs(sum(probs.values()) - 1.0) < 0.01


def test_task_classifier_save_load(tmp_path):
    model = TaskClassifier()
    save_path = tmp_path / "test_model.pt"
    model.save(str(save_path))
    assert save_path.exists()

    loaded_model = TaskClassifier.load(str(save_path))
    assert loaded_model.num_classes == model.num_classes
    assert loaded_model.model_name == model.model_name
