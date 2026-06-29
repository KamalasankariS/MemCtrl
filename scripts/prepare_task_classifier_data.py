"""Prepare training data for task classifier from downloaded datasets."""

import json
import random
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm


def load_meddialog(data_dir: Path, max_examples: int = 10000) -> List[Tuple[str, str]]:
    print("Loading MedDialog...")
    data = []
    train_file = data_dir / "meddialog" / "train.txt"

    if not train_file.exists():
        print(f"  {train_file} not found, skipping")
        return data

    with open(train_file, "r", encoding="utf-8") as f:
        for line in tqdm(f.readlines()[:max_examples], desc="MedDialog"):
            line = line.strip()
            if len(line) > 50:
                data.append((line, "medical"))

    print(f"  Loaded {len(data)} medical examples")
    return data


def load_ubuntu(data_dir: Path, max_examples: int = 10000) -> List[Tuple[str, str]]:
    print("Loading Ubuntu Dialogue...")
    data = []
    ubuntu_dir = data_dir / "ubuntu"

    if not ubuntu_dir.exists():
        print(f"  {ubuntu_dir} not found, skipping")
        return data

    tsv_files = list(ubuntu_dir.glob("*.tsv"))
    if not tsv_files:
        print("  No .tsv files found")
        return data

    code_keywords = {"install", "error", "command", "terminal", "sudo"}
    for tsv_file in tsv_files[:3]:
        with open(tsv_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in tqdm(f.readlines()[: max_examples // 3], desc=f"Ubuntu {tsv_file.name}"):
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    text = parts[-1]
                    if len(text) > 50 and any(kw in text.lower() for kw in code_keywords):
                        data.append((text, "code"))

    print(f"  Loaded {len(data)} code examples")
    return data


def load_dailydialog(data_dir: Path, max_examples: int = 10000) -> List[Tuple[str, str]]:
    print("Loading DailyDialog...")
    data = []
    daily_dir = data_dir / "dailydialog"

    if not daily_dir.exists():
        print(f"  {daily_dir} not found, skipping")
        return data

    train_dir = daily_dir / "train"
    dialog_files = list(train_dir.glob("*.txt")) if train_dir.exists() else list(daily_dir.glob("*.txt"))

    if not dialog_files:
        print("  No dialog files found")
        return data

    for dialog_file in tqdm(dialog_files[:50], desc="DailyDialog"):
        with open(dialog_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
            if len(content) > 50:
                data.append((content, "general"))

    data = data[:max_examples]
    print(f"  Loaded {len(data)} general examples")
    return data


def load_from_huggingface(max_examples: int = 5000) -> List[Tuple[str, str]]:
    print("Loading from Hugging Face...")
    data = []

    try:
        from datasets import load_dataset

        print("  Loading WritingPrompts...")
        try:
            dataset = load_dataset("writingPrompts/writingPrompts", split="train")
            for i, example in enumerate(tqdm(dataset, total=min(max_examples, len(dataset)), desc="WritingPrompts")):
                if i >= max_examples:
                    break
                prompt = example.get("prompt", "")
                if len(prompt) > 50:
                    data.append((prompt, "writing"))
        except Exception as e:
            print(f"  Could not load WritingPrompts: {e}")

        print(f"  Loaded {len(data)} examples from Hugging Face")
    except ImportError:
        print("  datasets package not installed, skipping")

    return data


def create_synthetic_examples(num_examples: int = 2000) -> List[Tuple[str, str]]:
    print("Creating synthetic examples...")
    data = []

    medical_templates = [
        "Patient presents with {symptom} and {symptom2}",
        "Allergic to {medication}, prescribed {medication2}",
        "Diagnosis: {condition}, treatment plan includes {treatment}",
        "Medical history shows {condition} managed with {medication}",
    ]
    symptoms = ["headache", "fever", "cough", "chest pain", "nausea", "fatigue"]
    medications = ["penicillin", "aspirin", "ibuprofen", "amoxicillin", "metformin"]
    conditions = ["hypertension", "diabetes", "asthma", "arthritis", "migraine"]
    treatments = ["medication", "physical therapy", "surgery", "lifestyle changes"]

    for _ in range(num_examples // 4):
        template = random.choice(medical_templates)
        text = template.format(
            symptom=random.choice(symptoms),
            symptom2=random.choice(symptoms),
            medication=random.choice(medications),
            medication2=random.choice(medications),
            condition=random.choice(conditions),
            treatment=random.choice(treatments),
        )
        data.append((text, "medical"))

    code_templates = [
        "Error: {error} in {language} when {action}",
        "How to {action} in {language}? Getting {error}",
        "Debugging {error} in {framework}",
        "{language} {action} not working, shows {error}",
    ]
    errors = ["syntax error", "null pointer", "index out of bounds", "connection timeout", "permission denied"]
    languages = ["Python", "JavaScript", "Java", "C++", "TypeScript"]
    frameworks = ["React", "Django", "Flask", "Node.js", "Spring"]
    actions = ["import module", "install package", "run script", "compile code", "deploy app"]

    for _ in range(num_examples // 4):
        template = random.choice(code_templates)
        text = template.format(
            error=random.choice(errors),
            language=random.choice(languages),
            action=random.choice(actions),
            framework=random.choice(frameworks),
        )
        data.append((text, "code"))

    tutoring_templates = [
        "Can you explain {concept} in {subject}?",
        "I don't understand {concept}, can you help?",
        "What is the difference between {concept} and {concept2}?",
        "How does {concept} work in {subject}?",
    ]
    concepts = ["derivatives", "integrals", "matrices", "probability", "algebra", "momentum", "energy", "forces", "electricity", "waves"]
    subjects = ["mathematics", "physics", "chemistry", "biology", "computer science"]

    for _ in range(num_examples // 4):
        template = random.choice(tutoring_templates)
        text = template.format(
            concept=random.choice(concepts),
            concept2=random.choice(concepts),
            subject=random.choice(subjects),
        )
        data.append((text, "tutoring"))

    writing_templates = [
        "Write a {genre} story about {topic}",
        "Create a {type} describing {topic}",
        "Draft a {type} for {purpose}",
        "Compose {type} about {topic}",
    ]
    genres = ["sci-fi", "mystery", "romance", "fantasy", "horror"]
    types = ["poem", "essay", "article", "blog post", "short story"]
    topics = ["adventure", "friendship", "courage", "loss", "discovery"]
    purposes = ["a blog", "a magazine", "a presentation", "social media", "a newsletter"]

    for _ in range(num_examples // 4):
        template = random.choice(writing_templates)
        text = template.format(
            genre=random.choice(genres),
            type=random.choice(types),
            topic=random.choice(topics),
            purpose=random.choice(purposes),
        )
        data.append((text, "writing"))

    print(f"  Created {len(data)} synthetic examples")
    return data


def count_labels(data: List[Tuple[str, str]]) -> dict:
    counts = {}
    for _, label in data:
        counts[label] = counts.get(label, 0) + 1
    return counts


def prepare_task_classifier_data(output_file: str = "data/task_classifier_data.json"):
    print("=" * 60)
    print("Preparing Task Classifier Training Data")
    print("=" * 60)

    data_dir = Path("data/datasets")
    all_data = []

    all_data.extend(load_meddialog(data_dir, max_examples=10000))
    all_data.extend(load_ubuntu(data_dir, max_examples=10000))
    all_data.extend(load_dailydialog(data_dir, max_examples=10000))
    all_data.extend(load_from_huggingface(max_examples=5000))
    all_data.extend(create_synthetic_examples(num_examples=5000))

    random.shuffle(all_data)

    total = len(all_data)
    train_end = int(0.8 * total)
    val_end = int(0.9 * total)

    train_data = all_data[:train_end]
    val_data = all_data[train_end:val_end]
    test_data = all_data[val_end:]

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data_dict = {
        "train": [{"text": text, "label": label} for text, label in train_data],
        "val": [{"text": text, "label": label} for text, label in val_data],
        "test": [{"text": text, "label": label} for text, label in test_data],
        "label_counts": {
            "train": count_labels(train_data),
            "val": count_labels(val_data),
            "test": count_labels(test_data),
        },
    }

    with open(output_path, "w") as f:
        json.dump(data_dict, f, indent=2)

    print(f"\nSaved to: {output_file}")
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    for label, count in sorted(data_dict["label_counts"]["train"].items()):
        print(f"  {label:12s}: {count:6d} ({count / len(train_data) * 100:.1f}%)")


if __name__ == "__main__":
    prepare_task_classifier_data()
