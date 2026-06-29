#!/bin/bash
# Download datasets for MemCtrl training.
# Run from project root: bash scripts/download_datasets.sh

set -e

echo "Downloading datasets for MemCtrl..."

mkdir -p data/datasets
cd data/datasets

# LongBench (Primary Evaluation)
echo ""
echo "[1/4] Downloading LongBench..."
if [ ! -d "LongBench" ]; then
    git clone https://github.com/THUDM/LongBench.git
    cd LongBench
    wget https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip
    unzip -q data.zip
    rm data.zip
    cd ..
    echo "  LongBench downloaded"
else
    echo "  LongBench already exists"
fi

# MedDialog (Medical conversations)
echo ""
echo "[2/4] Downloading MedDialog..."
if [ ! -d "meddialog" ]; then
    mkdir -p meddialog
    cd meddialog
    wget https://huggingface.co/datasets/medical_dialog/resolve/main/en/train.txt
    wget https://huggingface.co/datasets/medical_dialog/resolve/main/en/val.txt
    cd ..
    echo "  MedDialog downloaded"
else
    echo "  MedDialog already exists"
fi

# Ubuntu Dialogue (Code/Tech support)
echo ""
echo "[3/4] Downloading Ubuntu Dialogue Corpus..."
if [ ! -d "ubuntu" ]; then
    mkdir -p ubuntu
    cd ubuntu
    wget http://cs.mcgill.ca/~jpineau/datasets/ubuntu-corpus-1.0/ubuntu_dialogs.tgz
    tar -xzf ubuntu_dialogs.tgz
    rm ubuntu_dialogs.tgz
    cd ..
    echo "  Ubuntu Dialogue downloaded"
else
    echo "  Ubuntu Dialogue already exists"
fi

# DailyDialog (General conversation)
echo ""
echo "[4/4] Downloading DailyDialog..."
if [ ! -d "dailydialog" ]; then
    mkdir -p dailydialog
    cd dailydialog
    wget http://yanran.li/files/ijcnlp_dailydialog.zip
    unzip -q ijcnlp_dailydialog.zip
    rm ijcnlp_dailydialog.zip
    cd ..
    echo "  DailyDialog downloaded"
else
    echo "  DailyDialog already exists"
fi

cd ../..

echo ""
echo "Dataset download complete."
echo ""
echo "Next steps:"
echo "  1. python scripts/prepare_task_classifier_data.py"
echo "  2. python scripts/train_task_classifier.py"
