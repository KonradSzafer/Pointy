# Pointy

This repository contains the official model and benchmarking system implementation of our paper "Pointy – A Lightweight Transformer for Point Cloud Foundation Models".

## Repository structure

```bash
├── configs/            # Configuration files for experiments
├── scripts/            # Scripts for running experiments
├── src/
│   ├── data/           # Data processing
│   ├── datasets/       # Dataset implementations
│   ├── metrics/        # Evaluation metrics
│   ├── models/
│   │   ├── baselines/  # Benchmark models
│   │   └── pointy*.py  # Pointy model
│   ├── trainers/       # Training and evaluation pipelines
│   └── *.py            # Utility functions
└── requirements.txt    # Dependencies
```

## Installation

1. Clone the repository

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Environment setup

1. Create a `.env` file in the root directory with the following paths:

```bash
DATA_PATH=/path/to/datasets/
CHECKPOINTS_PATH=/path/to/checkpoints/
RESULTS_PATH=/path/to/results/
```

## Pre-trained Models

Pre-trained model weights and configurations will be released soon. Once available, you'll be able to download and place them in your `CHECKPOINTS_PATH` directory for zero-shot experiments.

## Running Experiments

### Training

To train the model from scratch:
```bash
python -m scripts.train_classifier --config configs/classification_pointy.yaml
```

### Evaluation

Update the checkpoint path in `configs/zero_shot_evaluation_*.yaml` to point to your downloaded weights.

Run evaluation:

```bash
python -m scripts.evaluate_zero_shot \
    --config configs/zero_shot_evaluation_classification_modelnet40.yaml
```
