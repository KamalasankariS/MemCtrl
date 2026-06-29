# Training Guide

## Task Classifier

The task classifier is a DistilBERT model fine-tuned to classify user queries into 5 categories: `medical`, `code`, `writing`, `tutoring`, `general`.

### 1. Download datasets

```bash
# Option A: Download raw datasets (MedDialog, Ubuntu Dialogue, DailyDialog)
bash scripts/download_datasets.sh

# Option B: Use HuggingFace datasets + synthetic data (recommended, no manual download)
python scripts/download_real_datasets.py
```

### 2. Prepare training data

```bash
python scripts/prepare_task_classifier_data.py
```

This creates `data/task_classifier_data.json` with train/val/test splits.

### 3. Verify data quality

```bash
python scripts/check_data.py
```

Checks for overlaps between splits, label leakage, and class distribution.

### 4. Train

```bash
# Simple training
python scripts/train_task_classifier.py

# 3-phase training (recommended): real-only → weighted mixing → evaluation
python scripts/train_task_classifier_phased.py
```

The 3-phase approach:
1. **Phase 1**: Train only on real data to anchor the model
2. **Phase 2**: Mix in synthetic data with lower sampling weight (0.3x)
3. **Phase 3**: Evaluate on held-out test set with per-class metrics

Model is saved to `models/task_classifier.pt`.

## Policy Networks

Policy networks predict chunk importance for each task type. They are trained via hindsight labeling.

```python
from memctrl.ml.hindsight_labeler import HindsightLabeler
from memctrl.ml.policy_network import PolicyNetwork

# Label training data from session logs
labeler = HindsightLabeler()
labeled_data = labeler.label_session(session)

# Train policy for a specific task type
policy = PolicyNetwork(input_dim=768, hidden_dim=256)
# ... training loop with labeled_data
policy.save("models/policy_medical.pt")
```

## Hardware Requirements

- **CPU**: All training works on CPU (slower but functional)
- **GPU**: CUDA-compatible GPU recommended for faster training
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: ~2GB for datasets, ~500MB for model weights
